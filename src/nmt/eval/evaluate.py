'''
End-to-end evaluation:
load a checkpoint -> decode dev/test -> report BLEU + chrF/chrF++ + TER as a single table (sacrebleu, consistent tokenization).
'''

from ..decode.translate import translate
from .metrics import compute_metrics

def evaluate(model, tokenizer, srcs, references, cfg):
    # Sources (srcs) and references are already loaded lists of strings
    # cfg is DecodeConfig

    # 1. Translate sources into hypotheses
    hypotheses = translate(model=model,
                           sentences=srcs,
                           tokenizer=tokenizer,
                           cfg=cfg)

    # 2. Call compute metrics
    results = compute_metrics(hypotheses=hypotheses,
                              references=references)

    # 3. Print a readable table
    print("\n".join([f"{k}: {v}" for k, v in list(results.items())]))

    # 4. Return the dict of results
    return results

def read_lines(path_hi, path_en):
    with open(path_hi, 'r', encoding="utf-8") as f1:
        srcs = f1.read().split('\n')
    
    with open(path_en, 'r', encoding="utf-8") as f2:
        references = f2.read().split('\n')
    
    if srcs[-1] == "":
        srcs.pop()
    if references[-1] == "":
        references.pop()
    
    if len(srcs) != len(references):
        raise ValueError(f"{len(srcs)} != {len(references)}")
    
    return srcs, references