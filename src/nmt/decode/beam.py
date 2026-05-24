'''
Beam search
- keep the k best partial hypotheses alive at all times (beam_size, default 5)
- extend each live beam by each possible token
- score them all, keep the global top k
- Scoring is additive in log space.
(Hypothesis score is the sum of log probabilities of tokens.)
- Descending survivors need to continue the previous beam's context 
- survivor carries parent beam and what token it added
- Recover both from the flat index (parent = index / V, token = index mod V)
- Need parent index to rebuild the full sequence and for KV cache

For the future: coverage penalty, diverse beam search, early eos suppression, no-repeat n-gram blocking, repetition penalty, tighter early-stop
'''

from ..config import ModelConfig
from .cache import new_cache, reorder_cache
import torch
import torch.nn.functional

def beam_search(model, tokenizer, src, cfg):
    device = next(model.parameters()).device
    V = tokenizer.vocab_size

    # 1. Setup
    model.eval()
    with torch.inference_mode():
        # Create source pad mask
        src_pad_mask = (src != tokenizer.pad_id).expand(cfg.beam_size, -1)

        src = src.expand(cfg.beam_size, -1)
        # Encode the source once
        memory = model.encoder(src, src_pad_mask)

        # both ^^ are expanded for k beams where k = cfg.beam_size

        # Create cache
        kv_cache = new_cache(len(model.decoder.layers))

        # Seed all beams at BOS (a (k, 1) token tensor)
        input = torch.full((cfg.beam_size, 1), tokenizer.bos_id, device=device)

        # Create a length-k vector of running scores, init as [0, -inf,..., -inf]
        init = [0] + ([float("-inf")] * (cfg.beam_size - 1))
        running_scores = torch.tensor(init, device=device)

        # Create storage for each token sequence so far
        token_sequences = [[] for _ in range(cfg.beam_size)]

        # Create finished
        finished = []
        
        # 2. Loop
        for t in range(cfg.max_decode_len):
            # decode into logits
            logits = model.decode_step(input, memory, src_pad_mask, kv_cache)

            # create log probs over V
            log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)

            # Add each beam's running score across its row
            # Broadcast the length-k score vector over the V axis
            # to create a (k, V) grid of candidate cumulative scores
            # (k, 1) + (k, V) -> (k, V) grid
            candidate_scores = running_scores.unsqueeze(1) + log_probs

            # Flatten to (k*V,) and get the best candidate of the top 2k in case k finish
            cumulative_scores, top_indices = torch.topk(torch.flatten(candidate_scores), 2 * cfg.beam_size, dim=0, largest=True, sorted=True)

            # For each flat index recover
            # parent = index / V and
            # token = index mod V
            parents = top_indices // V
            tokens = top_indices % V

            # Walk 2k survivors
            live_parents = []
            live_tokens = []
            live_scores = []
            for i, token in enumerate(tokens):
                # if token is EOS, add to hypothesis
                if token == tokenizer.eos_id:
                    finished.append((token_sequences[parents[i]], cumulative_scores[i]))
                else:
                    live_parents.append(parents[i])
                    live_tokens.append(tokens[i])
                    live_scores.append(cumulative_scores[i])
                    if len(live_parents) == cfg.beam_size:
                        break

            # Update token_sequences
            new = []
            for j, _ in enumerate(live_parents):
                new.append(token_sequences[live_parents[j]] + [live_tokens[j].item()])
            token_sequences = new
            
            # reorder cache
            kv_cache = reorder_cache(kv_cache=kv_cache,
                                     beam_index=torch.tensor(live_parents, dtype=torch.long, device=device))
            
            # build next (k, 1) token tensor from live beams' tokens
            input = torch.tensor(live_tokens, dtype=torch.long, device=device).unsqueeze(dim=1)

            # update running score
            running_scores = torch.tensor(live_scores, device=device)

            # check if finished
            if len(finished) >= cfg.beam_size:
                break
        
        # Choose which pool to rank
        if len(finished) > 0:
            # if finished has anything in it, that's the pool
            pool = [(seq, sco.item()) for (seq, sco) in finished]
        else:
            # else, it's the live beams found from token_sequences and running_scores
            pool = [(token_sequences[i], running_scores[i].item()) for i in range(len(running_scores))]
        
        # Apply length penalty to each candidate score
        length_penalty_pool = []
        a = cfg.length_penalty
        for i, (sequence, score) in enumerate(pool):
            L = len(sequence)
            penalized_score = score / ((5 + L) / 6) ** a
            length_penalty_pool.append((sequence, penalized_score))

        # Choose the candidate with the highest penalized score
        # Scores are negative, so the best is closest to zero (the max)
        best_sequence_idx = 0
        best_score = float("-inf")
        for i, (_, penal_score) in enumerate(length_penalty_pool):
            if penal_score > best_score:
                best_score = penal_score
                best_sequence_idx = i
        # get the sequence of the best scoring candidate
        best_sequence = length_penalty_pool[best_sequence_idx][0]

        return best_sequence

