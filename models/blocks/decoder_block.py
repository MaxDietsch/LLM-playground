from torch import nn
import torch
from models.layers.RMSNorm import RMSNorm
from models.layers.feed_forward import FeedFoward
from models.layers.multi_head_attention import MultiHeadAttention
from models.model.config import TransformerDecoderConfig


class DecoderBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: TransformerDecoderConfig) -> None:
        # d_model: embedding dimension, d_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffwd = FeedFoward(config)
        self.ffn_norm = RMSNorm(config.d_model)
        # self.ln1 = nn.LayerNorm(config.d_model)
        # self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.attention_norm(x))
        x = x + self.ffwd(self.ffn_norm(x))
        return x
