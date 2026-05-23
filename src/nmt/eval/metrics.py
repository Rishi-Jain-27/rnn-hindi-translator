# Corpus BLEU, chrF, TER via sacrebleu.

import sacrebleu

def compute_metrics(hypotheses, references):
    # hypotheses are the model's English for each sentence
    # references is the correct English

    # BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score

    # chrF++
    chrFpp = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score

    # TER
    ter = sacrebleu.corpus_ter(hypotheses=hypotheses, references=[references]).score


    results = {"bleu":bleu,
               "chrF":chrFpp,
               "TER":ter}
    
    return results
