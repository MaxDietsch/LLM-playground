from typing import Literal
from dataloader.dataloader import Dataloader
import torch


class SimpleTokenDataloader(Dataloader):

    def __init__(self, 
                 train_data: torch.Tensor, 
                 val_data: torch.Tensor, 
                 context_length: int, 
                 batch_size: int):
        super().__init__( 
            train_data=train_data, 
            val_data=val_data, 
            context_length=context_length, 
            batch_size=batch_size
        )

    def get_batch(self, split: Literal["train", "val"]) -> tuple[torch.Tensor, torch.Tensor]:
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
        x = torch.stack([data[i : i + self.context_length] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_length + 1] for i in ix])
        # x, y = x.to(device), y.to(device)
        return x, y
