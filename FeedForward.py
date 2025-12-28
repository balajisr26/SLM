import torch
import torch.nn as nn
from modules.GELU import GELU


class MLP(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(emb_dim,4 * emb_dim)
                                              ,GELU()
                                              ,nn.Linear(4 * emb_dim,emb_dim)
                                    )
        
    def forward(self,x):
        return self.layers(x)

    
