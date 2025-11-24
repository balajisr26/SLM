import torch
import torch.nn as nn
from modules.HelperClasses import MLP,GELU,LayerNorm,MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = MultiHeadAttention(config.n_embd,config.n_embd,config.block_size
                                       ,config.dropout,config.n_head,qkv_bias=False
                                       )
      #  self.attn = CausalSelfAttention(config)
       # print("Attn Device:",self.attn.get_device())
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config.n_embd)
    def forward(self, x):
      #  print("x",x)
    #    print("Attn",self.attn(x))
    #    print("Layer Norm",self.ln1(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
