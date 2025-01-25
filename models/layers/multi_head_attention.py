from torch import nn
import torch
from torch.nn import functional as F
import math

from models.model.config import TransformerDecoderConfig


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, config: TransformerDecoderConfig) -> None:
        super().__init__()

        # Determine the number of key-value heads (defaults to num_heads if not specified)
        self.num_kv_heads = (
            config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        )

        # Set the number of query heads and the number of repetitions for K and V
        self.num_heads_q = config.num_heads
        self.num_rep = self.num_heads_q // self.num_kv_heads

        # Calculate the head d_modelension
        self.d_head = config.d_model // config.num_heads

        # calculate freqs matrix for RoPE
        self.apply_rope = config.apply_rope
        if self.apply_rope:
            self.freqs_complex = self.precompute_theta_pos_frequencies(config.context_length)

        # Linear transformations for queries, keys, values, and output
        self.Wq = nn.Linear(config.d_model, config.num_heads * self.d_head, bias=False)
        self.Wk = nn.Linear(config.d_model, self.num_kv_heads * self.d_head, bias=False)
        self.Wv = nn.Linear(config.d_model, self.num_kv_heads * self.d_head, bias=False)
        self.Wo = nn.Linear(config.num_heads * self.d_head, config.d_model, bias=False)

        # Initialize key and value caches with zeros
        self.cache_k = torch.zeros(
            (config.batch_size, config.context_length, self.num_kv_heads, self.d_head)
        )
        self.cache_v = torch.zeros(
            (config.batch_size, config.context_length, self.num_kv_heads, self.d_head)
        )

    @staticmethod
    def repeat_heads(x: torch.Tensor, num_rep: int) -> torch.Tensor:
        # Repeat the heads of K and V to match the number of heads in Q
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        if num_rep == 1:
            return x
        else:
            return (
                x[:, :, :, None, :]
                .expand(batch_size, seq_len, num_kv_heads, num_rep, head_dim)
                .reshape(batch_size, seq_len, num_kv_heads * num_rep, head_dim)
            )

    def precompute_theta_pos_frequencies(self, context_length: int, theta: float = 10000.0):
        assert self.d_head % 2 == 0, "Dimension must be divisible by 2 for applying RoPE"

        theta_numerator = torch.arange(0, self.d_head, 2).float()

        # (d_head / 2)
        theta = 1.0 / (theta ** (theta_numerator / self.d_head))
        # (T)
        m = torch.arange(context_length)
        # (T, 1) * (q, d_head / 2) -> (T, d_head / 2)
        freqs = torch.outer(m, theta).float()
        # (T, d_head / 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex

    def apply_RoPE(self, x: torch.Tensor):

        # (B, T, num_heads, d_head) -> (B, T, num_heads, d_head / 2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # (T, d_head) -> (1, T, 1, d_head / 2)
        freqs_complex = self.freqs_complex.unsqueeze(0).unsqueeze(2)

        # (B, T, num_heads, d_head / 2) * (1, T, 1, d_head / 2) -> (B, T, num_heads, d_head / 2)
        x_rotated = x_complex * freqs_complex

        # (B, T, num_heads, d_head / 2) -> (B, T, num_heads, d_head / 2, 2)
        x_rotated = torch.view_as_real(x_rotated)

        # (B, T, num_heads, d_head / 2, 2) -> (B, T, num_heads, d_head)
        x = x.reshape(*x.shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape  # (B, T, dim)

        # (B, T, dim) -> (B, T, num_heads_q * d_head)
        xq = self.Wq(x)

        # (B, T, dim) -> (B, T, num_kv_heads * d_head)
        xk = self.Wk(x)

        # (B, T, dim) -> (B, T, num_kv_heads * d_head)
        xv = self.Wv(x)

        # (B, T, num_heads_q * d_head) -> (B, T, num_heads_q, d_head)
        xq = xq.view(batch_size, seq_len, self.num_heads_q, self.d_head)

        # (B, T, num_kv_heads * d_head) -> (B, T, num_kv_heads, d_head)
        xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.d_head)

        if self.apply_rope:
            xq = self.apply_RoPE(xq)
            xk = self.apply_RoPE(xk)

        # Repeat the heads of K and V to match the number of heads in Q
        keys = self.repeat_heads(xk, self.num_rep)
        values = self.repeat_heads(xv, self.num_rep)

        # (B, T, num_heads_q, d_head) -> (B, num_heads_q, T, d_head)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, num_heads_q, T, d_head) * (B, num_heads_q, d_head, T) -> (B, num_heads_q, T, T)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, num_heads_q, T, T) * (B, num_heads_q, T, d_head) -> (B, num_heads_q, T, d_head)
        context = torch.matmul(scores, values)

        # (B, num_heads_q, T, d_head) -> (B, T, num_heads_q, d_head) -> (B, T, d_head * num_heads_q)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # (B, T, d_head * num_heads_q) -> (B, T, d_model)
        x = self.Wo(context)

        return x
