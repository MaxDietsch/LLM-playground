from typing import Literal
from dataloader.transformer_dataloader import SimpleTokenDataloader
from models.model.transformer_decoder import TransformerDecoder, TransformerDecoderConfig
import torch

from runner import Runner

context_length = 256
batch_size = 64
train_epochs = 5000
eval_interval = 500
learning_rate = 3e-4
device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(string: str) -> list[int]:
    return [stoi[c] for c in string]  # encoder: take a string, output a list of integers


def decode(token: list[int]) -> str:
    return "".join([itos[i] for i in token])  # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model_config = TransformerDecoderConfig(
    model="Transformer",
    d_model=384,
    num_heads=6,
    context_length=context_length,
    batch_size=64,
    n_layer=6,
    dropout=0.2,
    vocab_size=65,
    num_kv_heads=None,
)
dataloader = SimpleTokenDataloader(
    train_data=train_data, val_data=val_data, context_length=context_length, batch_size=batch_size
)
model = TransformerDecoder(model_config)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

runner = Runner(
    dataloader=dataloader,
    model=model,
    optimizer=optimizer,
    device=device,
    train_epochs=train_epochs,
    eval_interval=eval_interval,
    eval_iters=eval_iters,
)

runner.train()
