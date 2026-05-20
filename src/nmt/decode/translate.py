"""High-level translate(model, sentences, cfg) entry point: tokenize -> encode ->
decode (greedy | beam | mbr | ensemble per DecodeConfig.mode) -> detokenize.
Used by eval/evaluate.py and the notebooks."""
