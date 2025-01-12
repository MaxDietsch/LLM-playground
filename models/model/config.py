from typing import Optional
from pydantic import BaseModel


class Config(BaseModel):
    model: str


class TransformerDecoderConfig(Config):
    d_model: int
    num_heads: int
    context_length: int
    batch_size: int
    n_layer: int
    dropout: float
    vocab_size: int
    num_kv_heads: Optional[int] = None