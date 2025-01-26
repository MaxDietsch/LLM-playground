from abc import ABC, abstractmethod
import torch


class Tokenizer(ABC):

    @abstractmethod
    def encode(self, string: str) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        pass
