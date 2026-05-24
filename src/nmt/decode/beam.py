'''
Beam search: length & coverage penalties, diverse beam, n-best.
- keep the k best partial hypotheses alive at all times (beam_size, default 5)
- extend each live beam by each possible token
- score them all, keep the global top k
- Scoring is additive in log space.
(Hypothesis score is the sum of log probabilities of tokens.)
- Descending survivors need to continue the previous beam's context 
- survivor carries parent beam and what token it added
- Recover both from the flat index (parent = index / V, token = index mod V)
- Need parent index to rebuild the full sequence and for KV cache
'''

def beam_search(kv_cache, beam_index, n_best=1):
    pass

