from torch import nn
import torch 

class PositionalEmbedding(nn.Module):

    def __init__(self, context_length, d_model):
        
        super(PositionalEmbedding, self).__init__()

        # for each token position, give an embedding
        self.encoding = torch.zeros(context_length, d_model) # (T, C) 
        self.encoding.requires_grad = False 

        pos = torch.arange(0, context_length) # (T)
        pos = pos.float().unsqueeze(dim=1) # (T, 1)

        _2i = torch.arange(0, d_model, step=2).float() # (C)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        

    def forward(self, x):

        batch_size, context_length = x.size() # (B, T)

        return self.encoding[:context_length, :]