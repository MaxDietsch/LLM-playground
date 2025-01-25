from abc import ABC, abstractmethod
from typing import Literal
import torch


class Dataloader(ABC):

    def __init__(
        self, train_data: torch.Tensor, val_data: torch.Tensor, context_length: int, batch_size: int
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.context_length = context_length
        self.batch_size = batch_size

    @abstractmethod
    def get_batch(self, split: Literal["train", "val"]) -> tuple[torch.Tensor, torch.Tensor]:
        pass
