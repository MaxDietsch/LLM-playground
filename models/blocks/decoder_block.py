from torch import nn
import torch
from models.layers.feed_forward import FeedFoward
from models.layers.multi_head_attention import MultiHeadAttention
from models.model.config import TransformerDecoderConfig



class DecoderBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: TransformerDecoderConfig) -> None:
        # d_model: embedding dimension, d_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
