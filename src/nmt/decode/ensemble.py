"""Decode-time logit averaging across multiple checkpoints/models. Prefer cheap
checkpoint-averaging (train/ema.py) over true N-model ensembling on a single GPU;
this is for the final report-quality comparison."""
