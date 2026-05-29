# Learning Phase — Initial Machine Translation work

This folder archives the **initial learning phase** of the project, frozen for reference.
It is *not* part of the active Transformer pipeline (which lives in `../src/nmt/`).

## Contents
- `experiments.ipynb` — the original notebook covering:
  - **Phase A — Vanilla Seq2Seq** (unidirectional GRU encoder, context-vector bottleneck) — BLEU **1.91**
  - **Phase B — Bahdanau Attention** (bidirectional encoder + additive attention) — BLEU **4.78**
  - **Phase B + AdamW + dropout 0.3** — BLEU **5.63**
- `Hindi_English_Truncated_Corpus.csv` — the small (~13k filtered pairs) Kaggle corpus used throughout.

## Why it's archived
These were built to *feel* the encoder/decoder/attention machinery by hand — word-level tokenization and all.
The current build is a clean rebuild: from-scratch Transformer, SentencePiece subwords, IITB-scale data, and
the full set of modeling / training / inference / evaluation / analysis improvements (see `../CLAUDE.md`).
SentencePiece changes the vocabulary, so none of the Phase B checkpoints transfer — this notebook stays as
the baseline that motivated the rebuild.
