# Cleaning: NFC normalization, length/ratio filters, dedupe, dev/test leak removal.

# Reads the raw parallel files staged by download.py and writes cleaned ones to `DataConfig.cache_dir` (`train.clean.hi`/`.en`, etc.)
# # Only `train` is filtered;`dev`/`test` are normalized but never dropped and are used to build the leak set.
# Exact subword-token length filtering (cfg.min_len/max_len) happens later in dataset.py — here lengths are coarse word counts.


from __future__ import annotations

import os
import unicodedata
from collections import Counter

from ..config import DataConfig


# ---------------------------------------------------------------- normalization

def nfc(s: str) -> str:
    # Unicode NFC: canonical compose, so visually identical Hindi collapses to one code-point sequence.
    return unicodedata.normalize("NFC", s)


_indic_norm = None  # cached indic-nlp Hindi normalizer (built lazily)


def _normalize_hi(s: str, cfg: DataConfig) -> str:
    # NFC + indic-nlp Hindi normalization, gated by cfg.nfc.
    s = s.strip()
    if cfg.nfc:
        s = nfc(s)
        global _indic_norm
        if _indic_norm is None:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            _indic_norm = IndicNormalizerFactory().get_normalizer("hi")
        s = _indic_norm.normalize(s)
    return s


def _normalize_en(s: str, cfg: DataConfig) -> str:
    # NFC + strip (case is preserved — SentencePiece/the model handle casing).
    s = s.strip()
    return nfc(s) if cfg.nfc else s


# ----------------------------------------------------------------------- langid

_lid_model = None  # cached fastText lid.176 model


def _get_lid(lid_path: str):
    global _lid_model
    if _lid_model is None:
        import fasttext  # lazy: installed in the notebook, not pinned in pyproject
        _lid_model = fasttext.load_model(lid_path)
    return _lid_model


def _detect(model, text: str) -> str:
    # Top-1 language code; fastText labels look like '__label__en' and need newline-free text.
    label = model.predict(text.replace("\n", " "), k=1)[0][0]
    return label.replace("__label__", "")


# -------------------------------------------------------------------------- I/O

def _read_pairs(hi_path: str, en_path: str) -> list[tuple[str, str]]:
    with open(hi_path, encoding="utf-8") as fh, open(en_path, encoding="utf-8") as fe:
        his, ens = fh.read().splitlines(), fe.read().splitlines()
    if len(his) != len(ens):
        raise ValueError(f"line-count mismatch: {hi_path} ({len(his)}) vs {en_path} ({len(ens)})")
    return list(zip(his, ens))


def _write_pairs(pairs, hi_path: str, en_path: str) -> None:
    with open(hi_path, "w", encoding="utf-8") as fh, open(en_path, "w", encoding="utf-8") as fe:
        for hi, en in pairs:
            fh.write(hi + "\n")
            fe.write(en + "\n")


def _ratio_ok(hi: str, en: str, max_ratio: float) -> bool:
    """Coarse src/tgt length-ratio gate on whitespace word counts."""
    a, b = len(hi.split()), len(en.split())
    if a == 0 or b == 0:
        return False
    return max(a, b) / min(a, b) <= max_ratio


# ------------------------------------------------------------------ orchestrator

def clean(cfg: DataConfig, force: bool = False, lid_path: str = "lid.176.bin") -> dict[str, tuple[str, str]]:
    # Normalize+filter the corpus. Returns {split: (hi_path, en_path)} of cleaned files.
    os.makedirs(cfg.cache_dir, exist_ok=True)
    out: dict[str, tuple[str, str]] = {}

    def raw(split):
        return (os.path.join(cfg.raw_dir, f"{split}.hi"), os.path.join(cfg.raw_dir, f"{split}.en"))

    def cln(split):
        return (os.path.join(cfg.cache_dir, f"{split}.clean.hi"), os.path.join(cfg.cache_dir, f"{split}.clean.en"))

    # --- dev/test: normalize only (never dropped); collect leak sets ---
    leak_hi: set[str] = set()
    leak_en: set[str] = set()
    for split in ("dev", "test"):
        o_hi, o_en = cln(split)
        out[split] = (o_hi, o_en)
        if not force and os.path.exists(o_hi) and os.path.exists(o_en):
            pairs = _read_pairs(o_hi, o_en)                      # already normalized
        else:
            r_hi, r_en = raw(split)
            pairs = [(_normalize_hi(h, cfg), _normalize_en(e, cfg)) for h, e in _read_pairs(r_hi, r_en)]
            _write_pairs(pairs, o_hi, o_en)
        for h, e in pairs:
            leak_hi.add(h)
            leak_en.add(e)

    # --- train: full clean ---
    o_hi, o_en = cln("train")
    out["train"] = (o_hi, o_en)
    if not force and os.path.exists(o_hi) and os.path.exists(o_en):
        print(f"[clean] train: already cleaned -> {o_hi}, {o_en}")
        return out

    model = _get_lid(lid_path) if cfg.langid_filter else None
    seen: set[tuple[str, str]] = set()
    kept: list[tuple[str, str]] = []
    stats: Counter = Counter()

    r_hi, r_en = raw("train")
    for h, e in _read_pairs(r_hi, r_en):
        h, e = _normalize_hi(h, cfg), _normalize_en(e, cfg)
        stats["total"] += 1
        if not h or not e:
            stats["empty"] += 1
            continue
        if model is not None and (_detect(model, h) != "hi" or _detect(model, e) != "en"):
            stats["langid"] += 1
            continue
        if not _ratio_ok(h, e, cfg.max_len_ratio):
            stats["ratio"] += 1
            continue
        if cfg.dedupe and (h, e) in seen:
            stats["dup"] += 1
            continue
        seen.add((h, e))
        if h in leak_hi or e in leak_en:
            stats["leak"] += 1
            continue
        kept.append((h, e))

    _write_pairs(kept, o_hi, o_en)
    stats["kept"] = len(kept)
    print(f"[clean] train: {dict(stats)} -> {o_hi}, {o_en}")
    return out
