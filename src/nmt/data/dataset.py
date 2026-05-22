# translationdataset + token-batch sampler + padding collate
# turns cleaned/labse'd parallel text into the exact tensors Transformer.forward wants:
# src_ids, src_pad_mask, tgt_in (bos-shifted decoder input), tgt_pad_mask, labels (eos-suffixed)
# masks are boolean keep-masks (true = real token), matching the model convention.

from __future__ import annotations

import random

from ..config import DataConfig
from .clean import _read_pairs


class TranslationDataset:
    # map-style dataset of encoded (src, tgt_in, labels) triples

    def __init__(self, cfg: DataConfig, tokenizer, hi_path: str, en_path: str, train: bool = True):
        self.examples: list[tuple[list[int], list[int], list[int]]] = []
        self.lengths: list[int] = []   # max(len(src), len(tgt_in)) per example, for bucketing

        for hi, en in _read_pairs(hi_path, en_path):
            src = tokenizer.encode(hi, add_eos=True)   # source ids + eos
            tgt = tokenizer.encode(en)                 # target core (no specials)
            # exact subword length filter (the bit deferred from clean.py); train only
            if train and not (cfg.min_len <= len(src) <= cfg.max_len
                              and cfg.min_len <= len(tgt) <= cfg.max_len):
                continue
            tgt_in = [tokenizer.bos_id] + tgt          # decoder input: bos + y
            labels = tgt + [tokenizer.eos_id]          # gold next-tokens: y + eos (== len(tgt_in))
            self.examples.append((src, tgt_in, labels))
            self.lengths.append(max(len(src), len(tgt_in)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class TokenBatchSampler:
    # yields lists of indices; each batch holds ~max_tokens tokens INCLUDING padding.
    # sorts by length first so a batch packs similar-length seqs (less padding waste).

    def __init__(self, lengths, max_tokens: int, shuffle: bool = True, seed: int = 1337):
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = self._build(lengths, max_tokens)

    def _build(self, lengths, max_tokens):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            new_max = max(cur_max, lengths[i])
            # padded cost if we append i = (count+1) * new_max; flush first if it blows the budget
            if cur and (len(cur) + 1) * new_max > max_tokens:
                batches.append(cur)
                cur, cur_max = [i], lengths[i]
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            # reshuffle batch order each epoch (lengths fixed, so batches are stable)
            random.Random(self.seed + self.epoch).shuffle(order)
            self.epoch += 1
        for j in order:
            yield self.batches[j]


class Collate:
    # pads a list of (src, tgt_in, labels) into batched long tensors + bool keep-masks

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        srcs, tgts, labs = zip(*batch)
        S = max(len(s) for s in srcs)
        T = max(len(t) for t in tgts)

        def pad(seqs, L):
            return torch.tensor([list(s) + [self.pad_id] * (L - len(s)) for s in seqs], dtype=torch.long)

        src_ids = pad(srcs, S)
        tgt_in = pad(tgts, T)
        labels = pad(labs, T)                          # pad with pad_id; loss ignores it (ignore_index=pad)
        return {
            "src_ids": src_ids,
            "src_pad_mask": src_ids != self.pad_id,    # bool keep-mask (true = real)
            "tgt_in": tgt_in,
            "tgt_pad_mask": tgt_in != self.pad_id,
            "labels": labels,
        }


def make_dataloader(cfg: DataConfig, tokenizer, hi_path: str, en_path: str,
                    max_tokens: int, train: bool = True, shuffle=None, seed=None):
    # wire dataset + token-batch sampler + collate into a torch DataLoader.
    # max_tokens comes from TrainConfig; min_len/max_len/seed from DataConfig.
    from torch.utils.data import DataLoader
    ds = TranslationDataset(cfg, tokenizer, hi_path, en_path, train=train)
    shuffle = train if shuffle is None else shuffle
    sampler = TokenBatchSampler(ds.lengths, max_tokens, shuffle=shuffle,
                                seed=cfg.seed if seed is None else seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=Collate(tokenizer.pad_id))
