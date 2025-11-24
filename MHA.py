import torch
import torch.nn as nn
import config

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




if __name__ == "__main__":
    import torch
    torch.manual_seed(123)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )
    
    batch = torch.stack((inputs, inputs), dim=0)
   # print(batch.shape)
    mha = MultiHeadAttention(3,2,6,0.0,2)
    mha(batch)
