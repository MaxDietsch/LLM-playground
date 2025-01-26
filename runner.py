from typing import Literal
import torch
from dataloader.dataloader import Dataloader
from models.model.model import AIModel
from models.tokenizer.tokenizer import Tokenizer


class Runner:

    def __init__(
        self,
        dataloader: Dataloader,
        tokenizer: Tokenizer,
        model: AIModel,
        optimizer: torch.optim.Optimizer,
        train_data: str,
        val_data: str,
        device: Literal["cuda", "cpu"],
        train_epochs: int = 10,
        eval_interval: int = 1,
        eval_iters: int = 10,
    ):

        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_epochs = train_epochs
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        self.model.to(self.device)

        self.dataloader.set_train_data(self.tokenize_dataset(train_data))
        self.dataloader.set_val_data(self.tokenize_dataset(val_data))

    def tokenize_dataset(self, data: str) -> torch.Tensor:
        return self.tokenizer.encode(data)

    def train(self) -> None:
        for epoch in range(self.train_epochs):

            # every once in a while evaluate the loss on train and val sets
            if epoch % self.eval_interval == 0 or epoch == self.train_epochs - 1:
                losses = self.eval()
                print(
                    f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                output = self.generate("", 100)
                print(f"generated output for empty string:\n {output}")

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
                xb, yb = self.dataloader.get_batch(split)  # type: ignore[arg-type]
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, loss = self.model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def generate(self, start_str: str, new_tokens: int) -> str:

        padding_length = self.model.context_length - len(start_str)
        start_str = " " * padding_length + start_str
        tokenized_start = self.tokenizer.encode(start_str).unsqueeze(0)
        tokenized_start = tokenized_start.to(self.device)
        output = self.model.generate(tokenized_start, new_tokens)

        output = output.squeeze(0).tolist()

        output_string = self.tokenizer.decode(output)

        return output_string
