import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = d_model ** 0.5
        self.init_std = d_model** (-0.5)
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.init_std)
        
        # rezero the padded rows
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()

    def forward(self, ids):
        # return B t d_model for encoder and decoder inputs
        # multipies looked up vectors by sqrt(d_model) only input lookup is scaled
        return self.scale * self.embed(ids)

    def project(self, hidden):
        # return logits B T V using hidden @ W T with funct linear
        return nn.functional.linear(hidden, self.embed.weight)
