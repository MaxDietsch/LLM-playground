from torch import nn


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
