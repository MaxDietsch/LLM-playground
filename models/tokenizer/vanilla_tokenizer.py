import torch
from models.tokenizer.tokenizer import Tokenizer


class VanillaTokenizer(Tokenizer):

    def __init__(self, tokens: list[str]):
        super().__init__()

        self.str_to_token = {ch: i for i, ch in enumerate(tokens)}

        self.token_to_str = {i: ch for i, ch in enumerate(tokens)}

    def encode(self, string: str) -> torch.Tensor:
        return torch.tensor([self.str_to_token[c] for c in string])

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(self.token_to_str[token] for token in tokens)
