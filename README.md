# Hindi → English Machine Translation, from scratch

A from-scratch encoder–decoder **Transformer** for Hindi → English translation, built end-to-end
as a learning project — every sublayer hand-built, no pretrained weights anywhere (no mBART /
NLLB / IndicTrans2 / HF checkpoints, no fine-tuning). Uses `nn.Linear / nn.Embedding / nn.Dropout`
as primitives but **not** `nn.Transformer` / `nn.MultiheadAttention`.

## Results

Trained on **IITB** (~913k pairs after cleaning + LaBSE filtering), joint SentencePiece unigram 16k,
single-GPU (Colab Pro / Kaggle T4).

| Model                          | Decode      | BLEU  | chrF++ | TER   |
| ------------------------------ | ----------- | ----- | ------ | ----- |
| RNN baseline (`learning-phase/`) | greedy      |  5.63 |   —    |   —   |
| Transformer base (sinusoidal)  | greedy + KV-cache | 19.25 | 48.70  | 69.75 |
| Transformer base (sinusoidal)  | beam (k=5, lp=0.6) | **20.07** | **49.08** | **68.28** |

Best dev-nll **1.937** (token perplexity ≈ 6.9). RoPE retrain and back-translation in progress.

## Layout

Installable Python package — code lives in `src/nmt/` on the dev box, runs on Colab / Kaggle
via `pip install -e .` driven from `notebooks/`.

```
hindi-translator/
  learning-phase/      # ARCHIVED RNN learning phase (frozen baseline, BLEU 5.63)
  src/nmt/             # the package (import as `nmt`)
    config.py          #   ModelConfig / DataConfig / TrainConfig / DecodeConfig
    model/             #   embeddings, positional (sin/learned/RoPE), attention (MHA + KV-cache),
                       #     feedforward (ReLU/GeLU/SwiGLU/GeGLU), norm (LN/RMSNorm, pre/post),
                       #     encoder, decoder, transformer
    data/              #   download (IITB), clean (NFC + script-check langid + ratio + dedupe +
                       #     leak), labse_filter, tokenizer (SentencePiece), dataset (token-batch)
    train/             #   optim (AdamW), schedule (inv-sqrt + cosine), loss (label-smoothed NLL),
                       #     ema (EMA + SWA), checkpoint (resumable + averaging), tracking
                       #     (TB/wandb), profiling (torch.profiler), loop (AMP + grad-accum + clip)
    decode/            #   greedy (batched + KV-cache), beam (length + coverage penalty), cache,
                       #     translate, plus mbr / ensemble stubs
    eval/              #   metrics (BLEU / chrF++ / TER via sacrebleu), evaluate driver
    analysis/          #   attention viz, head importance, alignments, embeddings, probing
  notebooks/           # thin drivers: 01_data, 02_train (Colab + Kaggle), 03_backtranslation,
                       #   04_decode_eval (Colab + Kaggle), 05_analysis, 00_run_tests
  tests/               # shape/sanity tests (run via `PYTHONPATH=src pytest`)
  pyproject.toml
```

## Status

- **Model (T2):** walking skeleton complete (sinusoidal · LayerNorm · pre-norm · ReLU · MHA · 3-way tied).
  Circle-backs: KV-cache and RoPE done; ALiBi / MQA / GQA / Shaw / drop-path deferred.
- **Data (T1):** download → clean → LaBSE → tokenizer → dataset done; back-translation in progress.
- **Train (T3):** AdamW + inv-sqrt/cosine schedule + label-smoothed NLL + AMP (bf16/fp16 auto) +
  exact token-normalised grad-accum + clip + EMA + SWA + resumable + averaging + TB tracking + profiler.
- **Decode (T4):** greedy + KV-cache + beam (length penalty, coverage penalty); MBR / ensemble / diverse beam ⬜.
- **Eval (T6):** BLEU / chrF++ / TER (sacrebleu); manual eval ⬜.
- **Analysis (T7):** stubs only.

Tracked feature-by-feature in `CLAUDE.md` (T1–T7).

## Quickstart

```bash
git clone https://github.com/Rishi-Jain-27/hindi-translator
cd hindi-translator
pip install -e .

# tests (torch needed for most; a few are torch-free)
PYTHONPATH=src pytest

# end-to-end pipeline (run on a GPU box — Colab Pro or Kaggle T4)
#   notebooks/01_data.ipynb       — IITB download → clean → LaBSE filter → train tokenizer
#   notebooks/02_train.ipynb      — train (forward hi→en) with AMP + EMA + resumable checkpoints
#   notebooks/02b_train_reverse.ipynb — train reverse en→hi (for back-translation)
#   notebooks/04_decode_eval.ipynb — translate test set, report BLEU / chrF++ / TER
```

Data is **not** committed — IITB is downloaded on the GPU box; cached corpus / checkpoints live on
Google Drive (Colab) or a Kaggle Dataset (Kaggle).

## Constraints

- **From-scratch only.** No pretrained weights, no fine-tuning. (Phase D will add pretrained ASR / TTS
  for a speech-to-speech app — but the translator itself stays hand-built.)
- **Single GPU.** No DDP. Designed for Colab Pro / Kaggle T4.
- **No heavy neural metrics.** BLEU + chrF++ + TER + manual; no COMET / BLEURT (VRAM at eval).

## License & credits

Educational project. Corpus credit: [IITB parallel corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi).
Tokenizer: SentencePiece. Metrics: sacrebleu. Pair filtering: LaBSE.
