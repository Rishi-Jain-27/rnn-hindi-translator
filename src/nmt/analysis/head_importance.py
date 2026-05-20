"""Attention-head importance via ablation (Michel et al., "Are Sixteen Heads Really
Better Than One?"): mask each head, measure the BLEU/loss delta, rank heads, and
optionally prune low-importance heads and re-evaluate."""
