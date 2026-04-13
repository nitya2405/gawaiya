from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """
    Minimal byte-level tokenizer for optional text prompting without extra deps.

    - Encodes UTF-8 bytes.
    - Reserves 0 for PAD, 1 for BOS, 2 for EOS.
    - Actual bytes are offset by +3 => 3..258.
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3
    vocab_size: int = 259

    def encode(self, text: str, *, max_len: int) -> list[int]:
        if max_len <= 0:
            return []
        b = text.encode("utf-8", errors="ignore")[:max_len]
        return [self.bos_id, *[x + self.byte_offset for x in b], self.eos_id]

