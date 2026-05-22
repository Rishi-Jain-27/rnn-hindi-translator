# Fetch and stage the IITB Hindi-English parallel corpus.

# Downloads via HuggingFace `datasets`
# writes each split to `DataConfig.raw_dir` as line-aligned parallel text


from __future__ import annotations

import os

from ..config import DataConfig

# HF dataset id for the IIT Bombay Hindi-English parallel corpus.
_IITB_HF_ID = "cfilt/iitb-english-hindi"

# Our split names -> HF split names.
_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


def _pair_paths(raw_dir: str, split: str) -> tuple[str, str]:
    # (hi_path, en_path) for a split, e.g. data/raw/train.hi, data/raw/train.en.
    return (os.path.join(raw_dir, f"{split}.hi"),
            os.path.join(raw_dir, f"{split}.en"))


def _load_split(hf_split: str):
    # load_dataset with a fallback for `datasets` versions that require trust_remote_code."""
    from datasets import load_dataset  # lazy: only needed at download time on Colab
    try:
        return load_dataset(_IITB_HF_ID, split=hf_split)
    except Exception:
        # Newer `datasets` requires opt-in to run a dataset's loading script.
        return load_dataset(_IITB_HF_ID, split=hf_split, trust_remote_code=True)


def _write_pairs(rows, hi_path: str, en_path: str) -> int:
    """Write one sentence per line to .hi/.en, keeping the two files line-aligned.

    Collapses internal whitespace/newlines (they'd desync the line-per-sentence
    invariant) and skips rows with an empty side. Format integrity only — real cleaning
    is clean.py.
    """
    n = 0
    with open(hi_path, "w", encoding="utf-8") as fh, \
         open(en_path, "w", encoding="utf-8") as fe:
        for row in rows:
            t = row["translation"]
            hi = " ".join(t["hi"].split())
            en = " ".join(t["en"].split())
            if not hi or not en:
                continue
            fh.write(hi + "\n")
            fe.write(en + "\n")
            n += 1
    return n


def download_iitb(cfg: DataConfig, force: bool = False) -> dict[str, tuple[str, str]]:
    """Download IITB and stage each split to cfg.raw_dir as parallel .hi/.en files.

    Returns {split: (hi_path, en_path)}. Idempotent: skips a split whose files already
    exist unless force=True.
    """
    os.makedirs(cfg.raw_dir, exist_ok=True)
    paths: dict[str, tuple[str, str]] = {}

    for split, hf_split in _SPLITS.items():
        hi_path, en_path = _pair_paths(cfg.raw_dir, split)
        paths[split] = (hi_path, en_path)

        if not force and os.path.exists(hi_path) and os.path.exists(en_path):
            print(f"[download] {split}: already staged -> {hi_path}, {en_path}")
            continue

        print(f"[download] {split}: loading {_IITB_HF_ID} [{hf_split}] ...")
        rows = _load_split(hf_split)
        n = _write_pairs(rows, hi_path, en_path)
        print(f"[download] {split}: wrote {n} pairs -> {hi_path}, {en_path}")

    return paths


def download(cfg: DataConfig, force: bool = False) -> dict[str, tuple[str, str]]:
    """Dispatch on cfg.corpus. Currently IITB only; Samanantar is a future add."""
    if cfg.corpus not in ("iitb", "iitb+samanantar"):
        raise ValueError(f"unknown corpus {cfg.corpus!r}")
    paths = download_iitb(cfg, force=force)
    if cfg.corpus == "iitb+samanantar":
        raise NotImplementedError("Samanantar staging not implemented yet (IITB only).")
    return paths
