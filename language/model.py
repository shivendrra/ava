import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModelArgs:
  d_model:int = 1024
  n_layers:int = 12
  n_heads:int = 18
  norm_eps:float = 1e-5
  max_batch_size:int = 32
  block_size:int = 2048
  multiple_of:int = 256
  ffn_multiplier:Optional[float] = None
  dropout:float = 0.2

class RMSNorm(nn.Module):
  def __init__(self, dim:int, eps:float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.wei = nn.Parameter(torch.ones(dim))
  
  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  
  def forward(self, x):
    out = self._norm(x.float()).type_as(x)
    return out * self.wei

class SwiGLU(nn.Module):
  def __init__(self, w1, w2, w3) -> None:
    super().__init__()
    self.w1 = w1
    self.w2 = w2
    self.w3 = w3
    
  def forward(self, x):
    x1 = F.linear(x, self.w1.weight)
    x2 = F.linear(x, self.w2.weight)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, self.w3.weight)

class Head(nn.Module):
  def __init__(self, head_size, d_model, dropout, block_size, mask=False) -> None:
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.mask = mask
    if mask:
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    key, query, value = self.key(x), self.query(x), self.value(x)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** -0.5)
    if self.mask:
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    attention = self.dropout(F.softmax(scores))
    output = torch.matmul(attention, value)
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, dropout, n_head, block_size, mask) -> None:
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, d_model, dropout, block_size, mask) for _ in range(head_size)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, d_model, hidden_dim, multiple_of, ffn_multiplier, dropout) -> None:
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_multiplier is not None:
      hidden_dim = int(ffn_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of -1) // multiple_of)
    self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.swiglu = SwiGLU(self.w1, self.w2, self.w3)
  
  def forward(self, x):
    out = self.w2(self.w1(x))
    out = self.w3(self.swiglu(out))
    out = self.dropout(out)
    return out

class Decoder(nn.Module):
  def __init__(self, d_model, norm_eps, dropout, n_head, hidden_dim, multiple_of, ffn_multiplier, block_size) -> None:
    super().__init__()
    self.att = MultiHeadAttention(d_model, dropout, n_head,block_size, mask=True)
    self.f_att = MultiHeadAttention(d_model, dropout, n_head, block_size, mask=False)
    self.ffn = FeedForward(d_model, hidden_dim, multiple_of, ffn_multiplier, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, norm_eps)
  
  def forward(self, x):
    out = self.att(self.norm(x))
    x = x + self.dropout(self.ffn(out))
    
    out = self.f_att(self.norm(x))
    x = x + self.dropout(self.ffn(out))
    return x

class PositionalEmbeddings(nn.Module):
  def __init__(self, ) -> None:
    super().__init__()

class Transformer(nn.Module):
  def __init__(self, vocab_size, args: ModelArgs) -> None:
    super().__init__()
    self.block_size = args.block_size
    self.token_model = nn.Embedding(vocab_size, args.d_model)