# joint sentencepiece tokenizer: train, load, encode, decode
# one shared vocab over hi+en (joint), unigram 16k, fixed id contract pad/unk/bos/eos = 0/1/2/3
# byte_fallback so corpus text never produces <unk>. exposes ids + vocab_size for build_model.

from __future__ import annotations

import os

from ..config import DataConfig


class Tokenizer:
    # thin wrapper over a trained sentencepiece model

    def __init__(self, sp):
        self.sp = sp

    # --- ids/size that build_model + dataset read ---

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @classmethod
    def load(cls, model_file: str) -> "Tokenizer":
        # load a trained .model from disk
        import sentencepiece as spm
        return cls(spm.SentencePieceProcessor(model_file=model_file))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        # text -> piece ids; dataset adds eos to src and bos to the decoder input
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # ids -> text; control symbols (pad/bos/eos) are dropped by sp
        return self.sp.decode(ids)


def _paths(cfg: DataConfig, model_prefix: str | None):
    # default name encodes type+size so unigram/bpe runs don't clobber each other
    prefix = model_prefix or os.path.join(cfg.cache_dir, f"spm_{cfg.tokenizer_model}_{cfg.vocab_size}")
    return prefix, prefix + ".model"


def train_tokenizer(cfg: DataConfig, train_hi: str, train_en: str,
                    model_prefix: str | None = None, force: bool = False,
                    sample_size: int = 2_000_000, **spm_kwargs) -> Tokenizer:
    # train joint sentencepiece on the labse-filtered train (both sides -> one vocab)
    # extra **spm_kwargs pass straight to the trainer (e.g. hard_vocab_limit=False for tiny test corpora)
    import sentencepiece as spm
    prefix, model_file = _paths(cfg, model_prefix)

    # idempotent: reuse an existing model unless force
    if not force and os.path.exists(model_file):
        print(f"[tokenizer] already trained -> {model_file}")
        return Tokenizer.load(model_file)

    spm.SentencePieceTrainer.train(
        input=f"{train_hi},{train_en}",         # comma-separated -> one joint vocab over hi+en
        model_prefix=prefix,
        model_type=cfg.tokenizer_model,         # unigram (default) | bpe
        vocab_size=cfg.vocab_size,              # 16k
        character_coverage=1.0,                 # small mixed charset (devanagari + latin)
        byte_fallback=True,                     # oov -> raw bytes, so corpus text never hits <unk>
        normalization_rule_name="identity",     # already nfc/indic-normalized in clean.py
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,  # fixed id contract
        user_defined_symbols=[cfg.bt_tag],      # <bt> kept as a single token for tagged back-translation
        input_sentence_size=sample_size,        # cap + shuffle to bound train time/memory
        shuffle_input_sentence=True,
        **spm_kwargs,
    )
    print(f"[tokenizer] trained {cfg.tokenizer_model} vocab={cfg.vocab_size} -> {model_file}")
    return Tokenizer.load(model_file)
