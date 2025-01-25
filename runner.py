from typing import Literal
import torch
from dataloader.dataloader import Dataloader
from models.model.model import AIModel


class Runner:

    def __init__(
        self,
        dataloader: Dataloader,
        model: AIModel,
        optimizer: torch.optim.Optimizer,
        device: Literal["cuda", "cpu"],
        train_epochs: int = 10,
        eval_interval: int = 1,
        eval_iters: int = 10,
    ):

        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_epochs = train_epochs
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        self.model.to(self.device)

    def train(self) -> None:
        for epoch in range(self.train_epochs):

            # every once in a while evaluate the loss on train and val sets
            if epoch % self.eval_interval == 0 or epoch == self.train_epochs - 1:
                losses = self.eval()
                print(
                    f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

            # sample a batch of data
            xb, yb = self.dataloader.get_batch("train")
            xb, yb = xb.to(self.device), yb.to(self.device)

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def eval(self) -> dict[str, torch.Tensor]:
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.dataloader.get_batch(split)  # type: ignore[arg-type]
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def generate(self) -> None:
        # TODO
        pass
