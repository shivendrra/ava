import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelArgs:
  d_model:int = 1024
  n_layers:int = 12
  n_heads:int = 18
  n_ff:int = 10 * d_model

class RMSNorm(nn.Module):
  def __init__(self, dim:int, eps:float=1e-5):
    super().__init__()
    self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  def forward(self, x):
    out = self._norm(x.float()).type_as(x)
    return out * self.weight

class SwiGLU(nn.Module):
  """
    swiglu activation function
      SwiGLU(x,W,V,b,c,b) = Swish b(xW + b) * (xV + c)
    paper: https://paperswithcode.com/method/swiglu
  """
  def __init__(self, w1:torch.tensor, w2:torch.tensor, w3:torch.tensor) -> None:
    super().__init__()
    self.w1, self.w2 = w1, w2
  def forward(self, x):
    x1 = F.linear(x, self.w1.weight)
    x2 = F.linear(x, self.w2.weight)
    return F.silu(x1) * x2

class Attention(nn.Module):
  def __init__(self, head_size:int, d_model:int, block_size:int, dropout:float, bias:bool, masking:bool):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=bias)
    self.query = nn.Linear(d_model, head_size, bias=bias)
    self.value = nn.Linear(d_model, head_size, bias=bias)
    self.dropout = dropout
    self.relpos_embed = nn.Parameter(torch.randn(block_size, block_size, head_size))
    if masking:
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
      self.masking = True
    else: self.masking = False
  def forward(self, x):
    B, T, C = x.shape
    key, query, value = self.key(x), self.query(x), self.value(x)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** -0.5)
    if self.masking:
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    relpos_scores = torch.einsum('btc,tvc->btv', query, self.relpos_embed[:T, :T])
    scores = scores + relpos_scores
    output = F.softmax(scores, dim=-1)
    output = self.dropout(output)
    return output @ value

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model:int, block_size:int, dropout:float, n_head:int, bias:bool, masking:bool):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([Attention(head_size, d_model, block_size, dropout, bias, masking)])
    self.projection = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.dropout(self.projection(out))

class FeedForward(nn.Module):
  def __init__(self, d_model, hidden_dim, multiple_of, ffn_multiplier, dropout) -> None:
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_multiplier is not None:
      hidden_dim = int(ffn_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of -1) // multiple_of)
    self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.swiglu = SwiGLU(self.w1, self.w2)
  def forward(self, x):
    x = self.swiglu(self.w1(x))
    return self.dropout(self.w2(x))

class Decoder(nn.Module):
  def __init__(self, d_model:int, n_head:int, norm_eps:float, dropout:float, block_size:int, hidden_dim:int, multiple_of:int, ffn_multiplier:int):
    super().__init__()
    self.masked_attention = MultiHeadAttention(d_model, block_size, dropout, n_head, True, True)
    self.casual_attention = MultiHeadAttention(d_model, block_size, dropout, n_head, True, False)
    self.ffwd = FeedForward(d_model, hidden_dim, multiple_of, ffn_multiplier, dropout)
    self.norm = RMSNorm(d_model, norm_eps)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    x_out = self.norm(x)
    x = x + self.dropout(self.masked_attention(x_out))

    x_out = self.casual_attention(self.norm(x))
    x = x + self.dropout(x_out)

    x_out = self.norm(x)
    x = x + self.dropout(self.ffwd(x_out))
    return x

class PositionalEncodings(nn.Module):
  def __init__(self,):
    super().__init__()

class Transformer(nn.Module):
  def __init__(self, params:ModelArgs, vocab_size):
    super().__init__()
    self.token_embeddings = nn.Embedding()