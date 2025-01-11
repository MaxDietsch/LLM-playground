from models.model.transformer_decoder import TransformerDecoderConfig


ModelConfig = TransformerDecoderConfig(
    model="Transformer",
    d_model=384,
    num_heads=6,
    context_length=256,
    batch_size=64,
    n_layer=6,
    dropout=0.2,
    vocab_size=65,
    num_kv_heads=None,
)
