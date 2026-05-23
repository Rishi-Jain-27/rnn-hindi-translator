# Batched greedy decoding.
'''
encode the source once, then loop to grow output one token at a time,
build prefix, run decoder, take the last position's logits, argmax, append, stop at eos
'''

from ..config import ModelConfig
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

        # Seed the target with BOS (beginning of sentence) for every sentence in the batch (B,1) tensor
        decoder_input = torch.full((B, 1), cfg.bos_id, device=device)
        predicted_ids = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # 3. repeat until max length
        for t in range(max_len):
            # Run the decoder
            logits = model.decoder(decoder_input, memory, decoder_input != cfg.pad_id, src_pad_mask)
            
            # argmax
            id = torch.argmax(logits[:, -1, :], dim=-1)

            # Force finished rows to pad
            finish = torch.masked_fill(id, finished, cfg.pad_id)

            # add to predicted_ids
            predicted_ids.append(finish)
            finished = finished | (id == cfg.eos_id)
            decoder_input = torch.cat((decoder_input, finish.reshape(B, 1)), dim=1) # dim=1 is time axis, so we add more across time!

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
