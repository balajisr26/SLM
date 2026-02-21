import torch
import torch.nn as nn
from modules.Transformers import TransformerBlock, LayerNorm,RMSNorm
import torch.nn.functional as F
import math


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size,config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size,config.n_embd)
        self.drop_emb = nn.Dropout(config.dropout)
        self.trf_blocks = nn.Sequential(
             *[TransformerBlock(config) for _ in range(config.n_layers)])
        
       # self.final_norm = LayerNorm(config.n_embd,config.bias)
        self.final_norm = RMSNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.out_head.weight = self.tok_emb.weight # Weight Tying
      

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
      #  print("b",b,"t",t)
        tok_emb = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(0, t, dtype=torch.long, device=device))
   
        x  = tok_emb + pos
     
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
      
        
        if targets is not None:
            logits = self.out_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.out_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens given a conditioning sequence.
        idx: Tensor of shape (B, T)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

        