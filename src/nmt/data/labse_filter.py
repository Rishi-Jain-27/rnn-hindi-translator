# Drop noisy pairs by LaBSE cross-lingual cosine similarity.

# Runs on the cleaned TRAIN pairs (after clean.py, before tokenizer training)
# embeds both sides with LaBSE and keeps only pairs whose hi/en cosine similarity >= cfg.labse_threshold.
# dev/test are passed through untouched
# LaBSE is a pretrained data- quality filter, not part of the from-scratch translator. Heavy one-time GPU pass; cached.


from __future__ import annotations

import os

from ..config import DataConfig
from .clean import _read_pairs, _write_pairs  # reuse line-aligned I/O

_LABSE_ID = "sentence-transformers/LaBSE"
_labse_model = None  # cached SentenceTransformer


def _get_labse(device: str | None = None):
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer  # lazy: Colab only
        _labse_model = SentenceTransformer(_LABSE_ID, device=device)  # device=None -> auto (GPU if present)
    return _labse_model


def labse_filter(cfg: DataConfig, force: bool = False, device: str | None = None,
                 batch: int | None = None, chunk: int = 50_000) -> dict[str, tuple[str, str]]:
    # Filter cleaned train pairs by LaBSE cosine; return {split: (hi_path, en_path)}.

    # Reads train.clean.* (from clean.py), writes train.labse.*; dev/test pass through as their cleaned paths.
    # Embeddings are computed in chunks so memory stays bounded on ~1.6M pairs. Idempotent unless force=True.
    batch = batch or cfg.labse_batch
    cache = cfg.cache_dir

    in_hi = os.path.join(cache, "train.clean.hi")
    in_en = os.path.join(cache, "train.clean.en")
    o_hi = os.path.join(cache, "train.labse.hi")
    o_en = os.path.join(cache, "train.labse.en")

    out = {
        "train": (o_hi, o_en),
        "dev": (os.path.join(cache, "dev.clean.hi"), os.path.join(cache, "dev.clean.en")),
        "test": (os.path.join(cache, "test.clean.hi"), os.path.join(cache, "test.clean.en")),
    }

    if not force and os.path.exists(o_hi) and os.path.exists(o_en):
        print(f"[labse] train: already filtered -> {o_hi}, {o_en}")
        return out

    pairs = _read_pairs(in_hi, in_en)
    model = _get_labse(device)
    from tqdm.auto import tqdm  # progress bar (tqdm ships with sentence-transformers)

    kept: list[tuple[str, str]] = []
    sim_sum = 0.0
    n = len(pairs)
    # process in big slices: encode each side once per slice (sentence-transformers batches
    # internally at batch_size), then score the slice in one vectorized numpy op. avoids the
    # per-256 GPU->CPU sync of the old loop; tqdm shows progress; memory bounded by `chunk`.
    for i in tqdm(range(0, n, chunk), desc="[labse] encoding", unit="slice"):
        sub = pairs[i:i + chunk]
        his = [h for h, _ in sub]
        ens = [e for _, e in sub]
        # normalize_embeddings=True -> cosine similarity is the row-wise dot product.
        he = model.encode(his, batch_size=batch, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
        ee = model.encode(ens, batch_size=batch, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)
        sims = (he * ee).sum(axis=1)                     # (len(sub),) cosine per pair
        sim_sum += float(sims.sum())
        kept.extend(p for p, keep in zip(sub, sims >= cfg.labse_threshold) if keep)

    _write_pairs(kept, o_hi, o_en)
    mean = sim_sum / n if n else 0.0
    print(f"[labse] train: kept {len(kept)}/{n} (thr={cfg.labse_threshold}, mean_sim={mean:.3f}) "
          f"-> {o_hi}, {o_en}")
    return out
