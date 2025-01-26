from abc import ABC, abstractmethod
from typing import Literal
import torch


class Dataloader(ABC):

    def __init__(self, context_length: int, batch_size: int):
        self.context_length = context_length
        self.batch_size = batch_size

    @abstractmethod
    def set_train_data(self, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def set_val_data(self, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def get_batch(self, split: Literal["train", "val"]) -> tuple[torch.Tensor, torch.Tensor]:
        pass
