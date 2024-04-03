import json
with open('config.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

# required parameters
block_size = params['block_size']
d_model = params['d_model']
n_head = params['n_heads']
n_layers = params['n_layers']
learning_rate = params['learning_rate']
dropout = params['dropout']
norm_eps = params['norm_eps']

import torch
import torch.nn as nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
    
  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class AttentionHead(nn.Module):
  def __init__(self,
      head_size: int,
      d_model: int,
      block_size: int,
      dropout: float):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.rel_pos_embd = nn.Parameter(torch.randn(block_size, block_size, head_size))
  
  def forward(self, x: torch.Tensor, mask: bool = False):
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)
    scores = torch.matmul(query ,key.transpose(-2, -1)) / (key.shape[-1]**-0.5)
    
    if mask is True:
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    
    else:
      rel_pos_scores = torch.einsum('btc,tvc->btv', query, self.rel_pos_embd[:T, :T])
      scores = scores + rel_pos_scores

    att_mat = F.softmax(att_mat)
    value = self.value(x)
    output = torch.matmul(att_mat, value)
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self,
      d_model: int,
      block_size: int,
      n_head : int,
      dropout: float):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(d_model=d_model, dropout=dropout, block_size=block_size, head_size=head_size) for _ in range(n_head)])
    self.projection = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x: torch.Tensor, mask: bool = False):
    out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
    out = self.dropout(self.projection(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 5 * n_embd),
      nn.GELU(),
      nn.Linear(5 * n_embd, n_embd),
      nn.Dropout(dropout),
      )

  def forward(self, x: torch.Tensor):
    return self.net(x)
class DecoderBlock(nn.Module):
  def __init__(self, d_model: int,
        block_size: int,
        n_head: int,
        norm_eps: float,
        dropout: float):
    super().__init__()
    self.attention = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)
  
  def forward(self, x: torch.Tensor):
    x_out = self.attention(self.norm(x), mask=True)
    x_out = x + self.dropout(x_out)
    del x

    x = self.attention(self.norm(x_out), mask=False)
    x = x_out + self.dropout(x)
    del x_out

    x_out = self.ffwd(self.norm(x))
    x_out = x + self.dropout(x_out)
    del x
    
    return x_out
  
class Transformer(nn.Module):
  def __init__(self, vocab_size: int, block_size):
    super().__init__()
    self.block_size = block_size
    self.token_embeddings = nn.Embedding(vocab_size, d_model)
    self.pos_encodings = nn.Embedding(block_size, d_model)
    self.decoder = nn.ModuleList([DecoderBlock(n_head=n_head, d_model=d_model, dropout=dropout, norm_eps=norm_eps, block_size=block_size) for _ in range(n_layers)])
    self.norm_final = RMSNorm(d_model, eps=norm_eps)
    self.linear_final = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
  def forward(self, idx):
    B, T = idx.shape
    toked_model = self.token_embeddings(idx)
    pos_encod = self.pos_encodings(torch.arange(T, device=device))
    x = toked_model + pos_encod
    logits = self.linear_final(self.norm_final(self.decoder(x)))

    return logits
  
  def train_model(self, idx: torch.Tensor, targets: torch.Tensor):
    logits = self.forward(idx)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)

    return loss

  def generate(self, idx: torch.Tensor, max_token: int=10):
    for _ in range(max_token):
      idx_cond = idx[:, -self.block_size:]
      logits = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.argmax(probs, dim=-1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
