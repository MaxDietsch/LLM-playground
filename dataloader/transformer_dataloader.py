from typing import Literal
from dataloader.dataloader import Dataloader
import torch


class SimpleTokenDataloader(Dataloader):

    def __init__(self, context_length: int, batch_size: int):
        super().__init__(
            context_length=context_length,
            batch_size=batch_size,
        )

    def set_train_data(self, train_data: torch.Tensor) -> None:
        self.train_data = train_data

    def set_val_data(self, val_data: torch.Tensor) -> None:
        self.val_data = val_data

    def get_batch(self, split: Literal["train", "val"]) -> tuple[torch.Tensor, torch.Tensor]:
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
        x = torch.stack([data[i : i + self.context_length] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_length + 1] for i in ix])
        # x, y = x.to(device), y.to(device)
        return x, y
