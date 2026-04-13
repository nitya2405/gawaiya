from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sangeet.config import load_yaml, resolve_path
from sangeet.data.dataset import CarnaticTokenDataset, TokenSpec, collate_lm
from sangeet.data.vocab import build_vocab, load_vocab, save_vocab
from sangeet.model.transformer_lm import CarnaticLMConfig, CarnaticTransformerLM
from sangeet.utils.jsonl import read_jsonl
from sangeet.utils.runtime import find_repo_root


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Encodec-token Transformer LM (Carnatic/Hindustani)"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/train_carnatic_small.yaml",
        help="Path to YAML config.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_wrapper(batch, *, token_spec):
    """Windows-safe collate wrapper."""
    return collate_lm(batch, token_spec=token_spec)


# ---------------------------------------------------------------------
# Token spec loader
# ---------------------------------------------------------------------


def load_token_spec(
    manifest_path: Path, *, repo_root: Path
) -> tuple[TokenSpec, dict[str, Any]]:

    first = next(iter(read_jsonl(manifest_path)))

    tok_path = first.get("tokens_path")
    if not tok_path:
        raise KeyError("tokens_path missing in manifest")

    p = Path(tok_path)
    if not p.is_absolute():
        p = (repo_root / p).resolve()

    with np.load(p, allow_pickle=False) as z:

        n_codebooks = int(z["n_codebooks"][0])
        codebook_size = int(z["codebook_size"][0])
        frame_rate = float(z["frame_rate"][0])
        encodec_sr = int(z["sample_rate"][0])
        bandwidth = float(z["bandwidth"][0])

    spec = TokenSpec(
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
    )

    meta = {
        "n_codebooks": n_codebooks,
        "codebook_size": codebook_size,
        "frame_rate": frame_rate,
        "encodec_sample_rate": encodec_sr,
        "encodec_bandwidth": bandwidth,
    }

    return spec, meta


# ---------------------------------------------------------------------
# Vocab handling
# ---------------------------------------------------------------------


def load_or_build_vocabs(
    manifest_path: Path,
    vocabs_dir: Path,
):

    raga_p = vocabs_dir / "raga.json"
    tala_p = vocabs_dir / "tala.json"
    artist_p = vocabs_dir / "artist.json"

    if raga_p.exists() and tala_p.exists() and artist_p.exists():

        return (
            load_vocab(raga_p),
            load_vocab(tala_p),
            load_vocab(artist_p),
        )

    raga_vals = []
    tala_vals = []
    artist_vals = []

    for row in tqdm(read_jsonl(manifest_path), desc="Building vocabs"):

        raga_vals.append(str(row.get("raga", "unknown")))
        tala_vals.append(str(row.get("tala", "unknown")))
        artist_vals.append(str(row.get("artist", "unknown")))

    raga_vocab = build_vocab(raga_vals)
    tala_vocab = build_vocab(tala_vals)
    artist_vocab = build_vocab(artist_vals)

    vocabs_dir.mkdir(parents=True, exist_ok=True)

    save_vocab(raga_p, raga_vocab)
    save_vocab(tala_p, tala_vocab)
    save_vocab(artist_p, artist_vocab)

    return raga_vocab, tala_vocab, artist_vocab


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: dict[str, Any],
    token_meta: dict[str, Any],
):

    path.parent.mkdir(parents=True, exist_ok=True)

    obj = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "cfg": cfg,
        "token_meta": token_meta,
    }

    tmp = path.with_suffix(".tmp")

    torch.save(obj, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    args = parse_args()

    repo_root = find_repo_root()

    cfg = load_yaml(resolve_path(args.config, base_dir=repo_root))

    # ------------------------------------------------
    # Performance flags
    # ------------------------------------------------

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------
    # Paths
    # ------------------------------------------------

    tokens_manifest = resolve_path(
        cfg["data"]["tokens_manifest"], base_dir=repo_root
    )

    output_dir = resolve_path(
        cfg["training"]["output_dir"], base_dir=repo_root
    )

    vocabs_dir = resolve_path(
        cfg["data"].get("vocabs_dir", Path(output_dir) / "vocabs"),
        base_dir=repo_root,
    )

    # ------------------------------------------------
    # Seed
    # ------------------------------------------------

    seed = int(cfg["training"].get("seed", 42))

    set_seed(seed)

    # ------------------------------------------------
    # Token spec + vocab
    # ------------------------------------------------

    token_spec, token_meta = load_token_spec(
        tokens_manifest,
        repo_root=repo_root,
    )

    raga_vocab, tala_vocab, artist_vocab = load_or_build_vocabs(
        tokens_manifest,
        Path(vocabs_dir),
    )

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    max_seq_len = cfg["model"].get("max_seq_len")
    max_seq_len = int(max_seq_len) if max_seq_len else None

    dataset = CarnaticTokenDataset(
        tokens_manifest,
        repo_root=repo_root,
        token_spec=token_spec,
        raga_vocab=raga_vocab,
        tala_vocab=tala_vocab,
        artist_vocab=artist_vocab,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    # ------------------------------------------------
    # DataLoader
    # ------------------------------------------------

    batch_size = int(cfg["training"].get("batch_size", 2))
    num_workers = int(cfg["training"].get("num_workers", 0))

    collate_fn = partial(collate_wrapper, token_spec=token_spec)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows-safe
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=False,
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    mcfg = CarnaticLMConfig(
        d_model=int(cfg["model"]["d_model"]),
        n_layers=int(cfg["model"]["n_layers"]),
        n_heads=int(cfg["model"]["n_heads"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        ff_mult=int(cfg["model"].get("ff_mult", 4)),
        cross_attention=bool(cfg["model"].get("cross_attention", True)),
        max_seq_len=int(cfg["model"].get("max_seq_len", 4096)),
        cfg_dropout=float(cfg["model"].get("cfg_dropout", 0.0)),
    )

    model = CarnaticTransformerLM(
        mcfg,
        token_spec=token_spec,
        raga_vocab_size=raga_vocab.size,
        tala_vocab_size=tala_vocab.size,
        artist_vocab_size=artist_vocab.size,
    )

    device_str = str(cfg["training"].get("device", "cuda"))
    device_str = device_str if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    model.to(device)

    # ------------------------------------------------
    # Optimizer
    # ------------------------------------------------

    lr = float(cfg["training"].get("lr", 3e-4))
    wd = float(cfg["training"].get("weight_decay", 0.01))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    # ------------------------------------------------
    # AMP
    # ------------------------------------------------

    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ------------------------------------------------
    # Training params
    # ------------------------------------------------

    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_steps = int(cfg["training"].get("max_steps", 100_000))
    log_every = int(cfg["training"].get("log_every", 50))
    ckpt_every = int(cfg["training"].get("ckpt_every", 2000))
    warmup = int(cfg["training"].get("warmup_steps", 2000))

    # Per-codebook loss weights: upweight early codebooks (melody/harmony)
    # and downweight late ones (fine acoustic detail).  None = uniform.
    _raw_weights = cfg["training"].get("cb_loss_weights", None)
    cb_loss_weights: list[float] | None = [float(w) for w in _raw_weights] if _raw_weights else None
    if cb_loss_weights:
        print(f"[INFO] Per-codebook loss weights: {cb_loss_weights}")

    # ------------------------------------------------
    # Resume
    # ------------------------------------------------

    start_step = 0

    if args.resume:

        ckpt = torch.load(
            resolve_path(args.resume, base_dir=repo_root),
            map_location="cpu",
            weights_only=False,
        )

        # Use non-strict load to handle new parameters (e.g. null_cond_emb for
        # CFG fine-tuning from a pre-CFG checkpoint).
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        # Only warn on keys that are not the known new CFG parameter.
        _expected_new = {"null_cond_emb"}
        real_missing    = [k for k in missing    if k not in _expected_new]
        real_unexpected = [k for k in unexpected if k not in _expected_new]
        if real_missing:
            print(f"[WARN] Keys missing from checkpoint: {real_missing}")
        if real_unexpected:
            print(f"[WARN] Unexpected keys in checkpoint: {real_unexpected}")
        if "null_cond_emb" in missing:
            print("[INFO] null_cond_emb not in checkpoint — initialised from scratch (CFG fine-tune from pre-CFG model).")

        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler") and amp:
                scaler.load_state_dict(ckpt["scaler"])
        except ValueError as e:
            print(f"[WARN] Optimizer state skipped ({e}). Starting optimizer fresh — normal for fine-tunes that add new parameters.")

        start_step = int(ckpt.get("step", 0))

        print(f"Resumed from step {start_step}")

    # ------------------------------------------------
    # Checkpoints
    # ------------------------------------------------

    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # Tensorboard
    # ------------------------------------------------

    writer = None

    if bool(cfg["training"].get("tensorboard", True)):

        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(Path(output_dir) / "tb"))

        except Exception:
            writer = None

    # ------------------------------------------------
    # Train loop
    # ------------------------------------------------

    model.train()

    t0 = time.time()
    running_loss = 0.0

    pbar = tqdm(
        total=max_steps,
        initial=start_step,
        desc="Training",
        dynamic_ncols=True,
    )

    step = start_step

    while step < max_steps:

        for batch in loader:

            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)

            raga_id = batch["raga_id"].to(device, non_blocking=True)
            tala_id = batch["tala_id"].to(device, non_blocking=True)
            artist_id = batch["artist_id"].to(device, non_blocking=True)

            # LR schedule
            def lr_schedule(s: int) -> float:

                if warmup <= 0:
                    return lr

                if s < warmup:
                    return lr * (s / warmup)

                progress = (s - warmup) / max(1, max_steps - warmup)

                return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule(step)

            amp_ctx = (
                torch.cuda.amp.autocast(enabled=amp)
                if device.type == "cuda"
                else nullcontext()
            )

            with amp_ctx:

                out = model(
                    input_ids,
                    target_ids=target_ids,
                    raga_id=raga_id,
                    tala_id=tala_id,
                    artist_id=artist_id,
                    cb_loss_weights=cb_loss_weights,
                )

                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()

            running_loss += float(loss.item()) * grad_accum

            # Step
            if (step + 1) % grad_accum == 0:

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg["training"].get("grad_clip", 1.0)),
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)

            # Log
            if (step + 1) % log_every == 0:

                dt = time.time() - t0

                avg = running_loss / log_every

                running_loss = 0.0

                t0 = time.time()

                pbar.set_postfix(
                    loss=f"{avg:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    sec=f"{dt:.1f}",
                )

                if writer:

                    writer.add_scalar("train/loss", avg, step + 1)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step + 1)

            # Checkpoint
            if (step + 1) % ckpt_every == 0:

                ckpt_path = ckpt_dir / f"step_{step+1}.pt"
                latest_path = ckpt_dir / "latest.pt"

                save_checkpoint(
                    ckpt_path,
                    step=step + 1,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler if amp else None,
                    cfg=cfg,
                    token_meta=token_meta,
                )

                save_checkpoint(
                    latest_path,
                    step=step + 1,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler if amp else None,
                    cfg=cfg,
                    token_meta=token_meta,
                )

            step += 1
            pbar.update(1)

    pbar.close()

    if writer:
        writer.close()

    print(f"Training done. Checkpoints: {ckpt_dir}")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
