"""
sangeet/data/delay_pattern.py

MusicGen-style delay pattern for multi-codebook generation.

Motivation
----------
In the flat (current) encoding, all 8 codebooks for frame t appear
contiguously. Each high-codebook token has only K-1 tokens of context from
the previous codebooks of the *same* frame before it must be predicted.

With the delay pattern, codebook k is shifted forward by k steps:

    Step:  0    1    2    3    4    5    6    7  ...
    cb0:  t=0  t=1  t=2  t=3  t=4  t=5  t=6  t=7
    cb1:  PAD  t=0  t=1  t=2  t=3  t=4  t=5  t=6
    cb2:  PAD  PAD  t=0  t=1  t=2  t=3  t=4  t=5
    cb3:  PAD  PAD  PAD  t=0  t=1  t=2  t=3  t=4

Now when the model predicts cb3 at step 3+4=7, it has already seen
cb0 at t=4, cb1 at t=3, cb2 at t=2 — the *same audio frame's* companion
codebooks from this and recent frames.  This dramatically increases
the mutual information between codebooks.

Expected improvements vs flat:
- Tabla / percussion stays coherent beyond the context window boundary
- High-codebook (cb4-cb7) static noise greatly reduced
- Codebook dropout eliminated

Usage
-----
    from sangeet.data.delay_pattern import codes_to_delay_tokens, delay_tokens_to_codes

    # Encode (for tokenization / training data)
    tokens = codes_to_delay_tokens(codes, pad_token_id=PAD_ID)

    # Decode (for inference)
    codes = delay_tokens_to_codes(tokens, n_codebooks=8, codebook_size=1024, token_offset=2)
"""

from __future__ import annotations

import numpy as np


def codes_to_delay_tokens(
    codes: np.ndarray,
    *,
    pad_token_id: int = 0,
    token_offset: int = 2,
) -> np.ndarray:
    """
    Convert Encodec codes [K, T] → delay-pattern flat token sequence [T * K].

    The output sequence has the same length as the flat sequence: T * K tokens.
    The first k positions of codebook k are filled with pad_token_id.

    The token id for code value c in codebook k is:
        token_offset + k * codebook_size + c

    Parameters
    ----------
    codes       : np.ndarray [K, T], dtype int16 or int32
    pad_token_id: token id used for delayed (missing) positions
    token_offset: number of special tokens before codebook tokens (PAD=0, BOS=1)

    Returns
    -------
    np.ndarray [T * K], dtype int64
    """
    codes = np.asarray(codes, dtype=np.int64)
    K, T = codes.shape
    codebook_size = int(codes.max()) + 1  # inferred; will be overridden by caller if needed

    out = np.full(T * K, fill_value=int(pad_token_id), dtype=np.int64)

    for k in range(K):
        cb_tokens = token_offset + k * codebook_size + codes[k]  # [T]
        # Codebook k is delayed by k steps: cb_k[t] goes to position (t + k) * K + k
        for t in range(T):
            dest = (t + k) * K + k
            if dest < T * K:
                out[dest] = int(cb_tokens[t])

    return out


def codes_to_delay_tokens_v2(
    codes: np.ndarray,
    *,
    codebook_size: int,
    pad_token_id: int = 0,
    token_offset: int = 2,
) -> np.ndarray:
    """
    Vectorised version — use this for bulk preprocessing (much faster).
    codebook_size must be provided explicitly (1024 for Encodec 6kbps).
    """
    codes = np.asarray(codes, dtype=np.int64)
    K, T = codes.shape

    out = np.full(T * K, fill_value=int(pad_token_id), dtype=np.int64)

    # Token ids for every (k, t): shape [K, T]
    cb_offsets = np.arange(K, dtype=np.int64)[:, None]  # [K, 1]
    token_ids  = token_offset + cb_offsets * codebook_size + codes  # [K, T]

    for k in range(K):
        # Destination positions for cb k: (0+k)*K+k, (1+k)*K+k, ..., (T-1+k)*K+k
        # but clipped to [0, T*K)
        t_range = np.arange(T, dtype=np.int64)
        dests   = (t_range + k) * K + k
        valid   = dests < T * K
        out[dests[valid]] = token_ids[k, t_range[valid]]

    return out


def delay_tokens_to_codes(
    tokens: np.ndarray,
    *,
    n_codebooks: int,
    codebook_size: int,
    token_offset: int = 2,
    pad_token_id: int = 0,
) -> np.ndarray:
    """
    Invert the delay pattern: flat token sequence [T * K] → codes [K, T].

    Positions that were PAD (pad_token_id) decode to 0 — they won't appear
    in the final usable audio frames (the first K-1 frames are warmup).

    Parameters
    ----------
    tokens        : np.ndarray [N] of delay-pattern token ids (N must be divisible by K)
    n_codebooks   : K
    codebook_size : 1024 for Encodec 6kbps
    token_offset  : 2 (PAD=0, BOS=1)
    pad_token_id  : value used for padded positions

    Returns
    -------
    np.ndarray [K, T], dtype int16
        First K-1 frames contain zeros (warmup) and should be discarded.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    K = int(n_codebooks)
    N = tokens.size
    if N % K != 0:
        raise ValueError(f"Token count {N} not divisible by K={K}")
    T = N // K

    codes = np.zeros((K, T), dtype=np.int16)

    for k in range(K):
        # cb k is stored at positions k, K+k, 2K+k, ... i.e. step*K+k for step in 0..T-1
        # step = t + k  →  t = step - k
        for step in range(T):
            pos = step * K + k
            tok = int(tokens[pos])
            if tok == pad_token_id:
                continue
            code_val = tok - token_offset - k * codebook_size
            t = step - k
            if 0 <= t < T and 0 <= code_val < codebook_size:
                codes[k, t] = code_val

    return codes


def delay_tokens_to_codes_v2(
    tokens: np.ndarray,
    *,
    n_codebooks: int,
    codebook_size: int,
    token_offset: int = 2,
    pad_token_id: int = 0,
) -> np.ndarray:
    """Vectorised inverse — use during inference."""
    tokens = np.asarray(tokens, dtype=np.int64)
    K = int(n_codebooks)
    N = tokens.size
    T = N // K

    codes = np.zeros((K, T), dtype=np.int16)
    steps = np.arange(T, dtype=np.int64)

    for k in range(K):
        pos      = steps * K + k          # positions in flat token array
        tok      = tokens[pos]            # [T]
        valid    = tok != pad_token_id
        t_vals   = steps - k              # [T]  (may be negative)
        in_range = (t_vals >= 0) & (t_vals < T) & valid
        code_vals = tok - token_offset - k * codebook_size
        good = in_range & (code_vals >= 0) & (code_vals < codebook_size)
        codes[k, t_vals[good]] = code_vals[good].astype(np.int16)

    return codes
