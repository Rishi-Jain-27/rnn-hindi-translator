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


def labse_filter(cfg: DataConfig, force: bool = False,
                 device: str | None = None, batch: int | None = None) -> dict[str, tuple[str, str]]:
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

    kept: list[tuple[str, str]] = []
    sim_sum = 0.0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        his = [h for h, _ in chunk]
        ens = [e for _, e in chunk]
        # normalize_embeddings=True -> cosine similarity is just the row-wise dot product.
        he = model.encode(his, convert_to_tensor=True, normalize_embeddings=True,
                          batch_size=batch, show_progress_bar=False)
        ee = model.encode(ens, convert_to_tensor=True, normalize_embeddings=True,
                          batch_size=batch, show_progress_bar=False)
        sims = (he * ee).sum(dim=1)                      # (len(chunk),) cosine per pair
        for (h, e), s in zip(chunk, sims.tolist()):
            sim_sum += s
            if s >= cfg.labse_threshold:
                kept.append((h, e))

    _write_pairs(kept, o_hi, o_en)
    n = len(pairs)
    mean = sim_sum / n if n else 0.0
    print(f"[labse] train: kept {len(kept)}/{n} (thr={cfg.labse_threshold}, mean_sim={mean:.3f}) "
          f"-> {o_hi}, {o_en}")
    return out
