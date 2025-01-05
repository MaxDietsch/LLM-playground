from torch import nn 
import torch
from torch.nn import functional as F

from models.blocks.decoder_block import DecoderBlock
from models.embedding.positional_embedding import PositionalEmbedding
from models.embedding.token_embedding import TokenEmbedding
from dataclasses import dataclass

@dataclass
class TransformerDecoderConfig(): 
    d_model = 384
    num_heads = 6 
    context_length = 256 
    batch_size = 64 
    n_layer = 6
    dropout = 0.2
    vocab_size = 65
    num_kv_heads = None


class TransformerDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding = nn.Embedding(vocab_size, d_model)
        # self.position_embedding = nn.Embedding(block_size, d_model
        self.context_length = config.context_length
        self.d_model=config.d_model
        self.vocab_size = config.vocab_size
        self.num_heads = config.num_heads
        self.n_layer = config.n_layer
        self.token_embedding = TokenEmbedding(self.vocab_size, self.d_model)
        self.position_embedding = PositionalEmbedding(self.context_length, self.d_model)
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.d_model) # final layer norm
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding(x) # (B,T,C)
        pos_emb = self.position_embedding(x) #self.position_embedding(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop x to the last block_size tokens
            x_cond = x[:, -self.context_length:]
            # get the predictions
            logits, loss = self(x_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # (B, T+1)
        return x