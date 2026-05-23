# Batched greedy decoding.
from ..config import ModelConfig
from .cache import new_cache
import torch

def greedy_decode(model, src, max_len, cfg):
    device = next(model.parameters()).device
    # 1. Start eval mode
    model.eval()
    with torch.inference_mode():
        # Create src pad mask
        src_pad_mask = (src != cfg.pad_id)
        
        # get batch size
        B = src.shape[0]

        # 2. Encode the source once
        memory = model.encoder(src, src_pad_mask)

        # Create new cache
        kv_cache = new_cache(len(model.decoder.layers))

        # Seed the target with BOS (beginning of sentence) for every sentence in the batch (B,1) tensor
        input = torch.full((B, 1), cfg.bos_id, device=device)
        predicted_ids = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # 3. repeat until max length
        for t in range(max_len):
            # Run the decoder
            logits = model.decode_step(input, memory, src_pad_mask, kv_cache)
            
            # argmax
            id = torch.argmax(logits[:, -1, :], dim=-1)

            # Force finished rows to pad
            finish = torch.masked_fill(id, finished, cfg.pad_id)

            # add to predicted_ids
            predicted_ids.append(finish)
            finished = finished | (id == cfg.eos_id)

            # update input
            input = finish.unsqueeze(dim=1)

            # check for EOS
            if finished.all():
                break
        
        # remove end of sentence marker
        predicted_ids = torch.stack(predicted_ids, dim=1)
        result = []
        # Loop over each row in predicted_ids
        for row in predicted_ids:
            assemble = []
            # Loop over each value
            for val in row:
                if val != cfg.eos_id:
                    assemble.append(val.item())
                else:
                    break
            result.append(assemble)
        return result
