import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert d_out % num_heads==0,"d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.dropout = dropout
        self.w_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length)
                                               ,diagonal=1)
                             
        )

    
    def forward(self,x):
        b,num_tokens,d_in = x.shape
     #   print("MHA Shape",x.shape)
       # print(self.w_query(x).shape)
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

      #  print("Queries shape:", queries.shape)
      #  print("Keys shape:", keys.shape)
      #  print("Values shape:", values.shape)

        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)

      #  print("Queries shape:", queries.shape)
      #  print("Keys shape:", keys.shape)
      #  print("Values shape:", values.shape)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

      #  print("Queries shape:", queries.shape)
      #  print("Keys shape:", keys.shape)
      #  print("Values shape:", values.shape)

        attn_scores = queries @ keys.transpose(2,3)
       # print(attn_scores.shape)

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
       # print(attn_scores)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5,dim=-1)
       # print(attn_weights)
        attn_weights = self.dropout(attn_weights)

      #  print("Values shape",values.shape)
        context_vec = (attn_weights @ values).transpose(1,2)
      #  print("Context vec shape:",context_vec.shape)

        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
        #return context_vec.to(config.device)
       # print("Context Vec Shape",context_vec.shape)
      #  print(context_vec)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))

class LayerNorm(nn.Module):
    def __init__(self,emb_dim,bias):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MLP(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(emb_dim,4 * emb_dim)
                                              ,GELU()
                                              ,nn.Linear(4 * emb_dim,emb_dim)
                                    )
        
    def forward(self,x):
        return self.layers(x)
    
