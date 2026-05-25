"""Typed config dataclasses — the single source of truth for the whole NMT system.

Pure-Python (no torch import) so it imports anywhere, including the Mac. Runtime-only
choices (e.g. AMP dtype "auto") are stored as strings and resolved on Colab at train time.

Single source of truth: vocab_size and the special-token ids live in DataConfig/the
tokenizer; build_model copies them into ModelConfig so they are never set twice by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters (read by everything in nmt.model)."""

    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    attn_dropout: float = 0.1
    activation: str = "relu"           # relu | gelu | swiglu | geglu
    norm_type: str = "layernorm"       # layernorm | rmsnorm
    norm_position: str = "pre"         # pre | post
    pos_encoding: str = "sinusoidal"   # sinusoidal | learned | rope | alibi
    max_len: int = 512                 # caps positional table + mask size
    attn_variant: str = "mha"          # mha | mqa | gqa
    n_kv_heads: Optional[int] = None   # gqa only; None -> n_heads (mha) / 1 (mqa)
    rel_pos: bool = False              # Shaw relative positions
    droppath: float = 0.0             # stochastic depth (drop-path) rate
    tie_embeddings: bool = True        # FIXED: src emb = tgt emb = output projection

    # Filled from the tokenizer at build_model time (defaults match the fixed id contract).
    vocab_size: int = 16000
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class DataConfig:
    """Corpus, tokenizer, cleaning/filtering, and back-translation settings."""

    # Paths point at Colab-local disk or a Google Drive mount — data is never on the Mac.
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    corpus: str = "iitb"               # iitb | iitb+samanantar

    # Tokenizer
    tokenizer_model: str = "unigram"   # unigram | bpe
    vocab_size: int = 16000            # single source of truth; copied into ModelConfig
    joint_vocab: bool = True           # FIXED: one shared SentencePiece vocab

    # Cleaning / filtering
    nfc: bool = True                   # Unicode NFC normalization (Hindi)
    min_len: int = 1                   # min tokens after tokenization
    max_len: int = 100                 # max tokens after tokenization
    max_len_ratio: float = 2.5         # drop pairs whose src/tgt length ratio exceeds this
    langid_filter: bool = True
    dedupe: bool = True
    labse_threshold: float = 0.70      # drop pairs below this cross-lingual cosine similarity
    labse_batch: int = 256
    seed: int = 1337

    # Back-translation (flipped on for the BT run)
    bt_enabled: bool = False
    bt_tag: str = "<bt>"               # prefixed to synthetic source (tagged BT)
    bt_mono_path: Optional[str] = None
    bt_sample: bool = True             # sampled (not beam) decode for BT diversity


@dataclass
class TrainConfig:
    """Optimization, schedule, AMP, checkpointing, and eval cadence."""

    out_dir: str = "checkpoints"

    # Batching (token-based)
    max_tokens: int = 8192             # per micro-batch
    grad_accum: int = 4                # effective ~32k tokens

    # Steps / schedule
    max_steps: int = 100_000
    warmup_steps: int = 4000
    lr: float = 7e-4                   # peak LR
    schedule: str = "inv_sqrt"         # inv_sqrt | cosine
    betas: tuple[float, float] = (0.9, 0.98)   # FIXED (Transformer)
    eps: float = 1e-9
    weight_decay: float = 0.01         # excluded from norm/bias params (see train/optim.py)
    label_smoothing: float = 0.1
    loss_norm: str = "token"           # token | sentence
    grad_clip: float = 1.0

    # Mixed precision
    amp: bool = True
    amp_dtype: str = "auto"            # auto | bf16 | fp16 ; "auto" resolves at train time
    grad_checkpoint: bool = False      # enable if OOM at depth
    compile: bool = False              # torch.compile (off by default — flaky w/ dynamic shapes)

    # Weight averaging
    ema_decay: Optional[float] = 0.999   # None disables EMA
    use_swa: bool = False
    swa_start: int = 80_000              # step to begin collecting SWA snapshots
    swa_period: int = 1000               # collect an SWA snapshot every N steps once started
    ckpt_avg_last_n: int = 5             # average last N checkpoints at the end

    # Cadence
    eval_every: int = 2000
    ckpt_every: int = 2000
    ckpt_keep_last_n: int = 2           # rotation: keep last N step_*.pt (best.pt always kept)
    patience: int = 10                  # early stop after N evals without improvement
    curriculum: bool = False            # short->long curriculum warmup
    seed: int = 1337

    # Observability (tracking.py)
    tracker: str = "tensorboard"        # none | tensorboard | wandb | both
    wandb_project: Optional[str] = None
    log_every: int = 50                 # log train metrics every N optimizer steps


@dataclass
class DecodeConfig:
    """Inference settings (greedy / beam / mbr / ensemble)."""

    mode: str = "beam"                  # greedy | beam | mbr | ensemble
    beam_size: int = 5
    length_penalty: float = 0.6         # GNMT-style
    coverage_penalty: float = 0.0       # try ~0.2 to reduce dropped/repeated content
    diverse_groups: int = 1             # >1 enables diverse beam search
    diverse_strength: float = 0.5
    nbest: int = 1                      # return top-k beams
    max_decode_len: int = 256

    # MBR
    mbr_samples: int = 32
    mbr_utility: str = "chrf"           # chrf | bleu
    sampling_temp: float = 1.0
    top_k: int = 0
    top_p: float = 0.0

    # Ensembling
    ensemble_ckpts: tuple[str, ...] = ()
    batch_size: int = 64
