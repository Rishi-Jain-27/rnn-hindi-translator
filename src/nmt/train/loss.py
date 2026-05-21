# Label-smoothed cross-entropy with ignore_index=pad_id
# supports token- vs sentence-level normalization
# for token norm under gradient accumulation, sum the loss over the whole accum window and divide by total tokens (not per-micro-batch mean).

import torch

def label_smoothed_nll(logits, labels, pad_id, smoothing):
    # returns (loss_sum, nll_sum, n_tokens)
    # logits straight from Transformer.forward
    # Labels are the y + eos targets padded
    # loss sum is the smoothed loss summed over all non-pad tokens - this gets backpropped
    # nll_sum is the -log p_gold over no-pad tokens
    # n_tokens is the number of non-pad tokens

    # label smoothing for the sake of preventing overconfidence and overfitting, also not smoothing doesn't make sense with translation
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    eps = smoothing
    V = logits.size(-1)
    eps_i = eps/(V-1)
    smooth = -logp.sum(dim=-1)
    loss_tok = (1 - eps - eps_i) * nll + eps_i * smooth
    keep = labels != pad_id
    loss_tok = loss_tok * keep
    nll = nll * keep
    loss_sum = loss_tok.sum()
    nll_sum = nll.sum()
    n_tokens = keep.sum()
    return (loss_sum, nll_sum, n_tokens)
