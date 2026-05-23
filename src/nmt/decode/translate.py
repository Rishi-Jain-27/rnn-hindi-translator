# High-level translate(model, sentences, tokenizer, cfg)
# tokenize(+eos) -> pad to (B,S) -> greedy_decode -> detokenize ; returns list[str] in input order
# tokenizer carries pad/bos/eos ids (passed to greedy_decode as its cfg) and does encode/decode
# cfg = DecodeConfig (max_decode_len now; mode greedy|beam|mbr|ensemble deferred)
# Used by eval/evaluate.py and the notebooks.

import torch
from .greedy import greedy_decode

# translate in batches of cfg.batch_size
def translate(model, sentences, tokenizer, cfg):
    # Edge case: empty input
    if len(sentences) == 0:
        return []

    results = []
    
    # Loop through sentences in strides of cfg.batch_size (slice i + batch_size every iteration)
    for i in range(0, len(sentences), cfg.batch_size):
        batch_sentences = sentences[i : i + cfg.batch_size]

        # Call helper on each slice, and extend results
        results.extend(_translate(model=model,
                                  batch_sentences=batch_sentences,
                                  tokenizer=tokenizer,
                                  cfg=cfg))
    
    # return
    return results


# small, per-batch helper
def _translate(model, batch_sentences, tokenizer, cfg):
    # 1. Encode each source sentence to ids with end marker and no begin marker

    # loop input strings, call tokenizer.encode on each with add_eos=True and add_bos=False
    encoded = []
    for string in batch_sentences:
        encoded.append(tokenizer.encode(string, add_bos=False, add_eos=True))
    # that gives us [BATCH_SIZE] id lists of different lengths within the encoded list

    # 2. pad them into a rectangle for greedy_decode

    # Find the length of the longest list of ids
    len_longest = len(max(encoded, key=len))

    # pad the rest to that length
    for idx, ids in enumerate(encoded):
        # right pads to each list of ids until the reach the longest length
        while len(ids) < len_longest:
           encoded[idx].append(tokenizer.pad_id)

    # stack into (B, S) into tensor
    src = torch.tensor(encoded)

    # 3. Put tensor to the model device
    device = next(model.parameters()).device
    src = src.to(device)

    # 4. call greedy decode
    en_ids = greedy_decode(model, src, cfg.max_decode_len, tokenizer)

    # 5. Decode the returned lists to strings
    en_strs = []
    # loop, call tokenizer.decode on each
    # sentencepiece stitches subword pieces together, collect strings
    for en_id in en_ids:
        en_strs.append(tokenizer.decode(en_id))

    # 6. Return the list of translated strings in the same order as input
    return en_strs
