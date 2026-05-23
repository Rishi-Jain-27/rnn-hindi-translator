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
        # add a batch dim to src
        src = src.unsqueeze(0).to(device)

        # Create src pad mask
        src_pad_mask = (src != cfg.pad_id)

        # 2. Encode the source once
        encoded = model.encoder(src, src_pad_mask)

        # Seed the target with BOS (beginning of sentence) for every sentence in the batch (B,1) tensor
        decoder_input = torch.tensor([[cfg.bos_id]], device=device)
        predicted_ids = []

        # 3. repeat until max length
        for t in range(max_len):
            # Run the decoder
            logits = model.decoder(decoder_input, encoded, decoder_input != cfg.pad_id, src_pad_mask)
            
            # argmax
            id = torch.argmax(logits[:, -1, :], dim=-1)

            # add to predicted_ids
            predicted_ids.append(id)
            decoder_input = torch.cat((decoder_input, id.reshape(1, 1)), dim=1) # dim=1 is time axis, so we add more across time!

            # check for EOS
            if id == cfg.eos_id:
                break;

        return [t.item() for t in predicted_ids]
