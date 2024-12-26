import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelArgs:
  d_model:int = 1024
  n_layers:int = 12
  n_heads:int = 18
  n_ff_multiple:int = 10
  fnn_multiplier:int = None
  n_ff:int = n_ff_multiple * d_model
  dropout:float = 0.2
  norm_eps:float = 1e-5
  block_size:int = 1024

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

class RoPE(nn.Module):
  def __init__(self, head_size, block_size):
    super().__init__()
    self.head_size = head_size
    self.block_size = block_size
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
    position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)  # (block_size, 1)
    sinusoidal = torch.einsum("i,j->ij", position, inv_freq)  # Shape: (block_size, head_size // 2)
    self.register_buffer("cos_emb", sinusoidal.cos(), persistent=False)  # (block_size, head_size // 2)
    self.register_buffer("sin_emb", sinusoidal.sin(), persistent=False)  # (block_size, head_size // 2)

  def forward(self, q, k):
    # spliting tensors into even and odd components
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    # retrieving embeddings for current sequence length
    cos = self.cos_emb[:q.shape[1], :].unsqueeze(0).to(q.device)
    sin = self.sin_emb[:q.shape[1], :].unsqueeze(0).to(q.device)

    # applying rotations
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

class Head(nn.Module):
  def __init__(self, head_size, d_model, dropout, block_size, mask=False):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.mask = mask
    if mask:
      self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.pos_emb = RoPE(head_size, block_size)
  def forward(self, x):
    B, T, C = x.shape
    key, query, value = self.key(x), self.query(x), self.value(x)
    query, key = self.pos_emb(query, key)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)
    if self.mask:
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    attention = self.dropout(F.softmax(scores, dim=-1))
    output = torch.matmul(attention, value)
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, dropout, n_head, block_size, mask):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList(
      [Head(head_size, d_model, dropout, block_size, mask) for _ in range(n_head)]
    )
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

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs, vocab_size: int):
    super().__init__()
    self.block_size = params.block_size
    self.d_model = params.d_model
    self.n_layers = params.n_layers
    self.token_embeddings = nn.Embedding(vocab_size, self.d_model)
    self.positional_embeddings = nn.Embedding(self.block_size, self.d_model)
    self.decoder_layers = nn.ModuleList([Decoder(d_model=params.d_model, n_head=params.n_heads, norm_eps=params.norm_eps, dropout=params.dropout, block_size=params.block_size, hidden_dim=params.n_ff, multiple_of=params.n_ff_multiple, ffn_multiplier=params.ffn_multiplier) for _ in range(self.n_layers)])
    self.norm_final = RMSNorm(self.d_model, params.norm_eps)
    self.linear_final = nn.Linear(self.d_model, vocab_size, bias=False)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.size()
    token_embeddings = self.token_embeddings(idx)  # Shape: (B, T, d_model)
    pos_indices = torch.arange(T, device=idx.device).unsqueeze(0)  # Shape: (1, T)
    positional_embeddings = self.positional_embeddings(pos_indices)  # Shape: (1, T, d_model)
    x = token_embeddings + positional_embeddings  # Shape: (B, T, d_model)

    # through decoder layers
    for layer in self.decoder_layers:
      x = layer(x)

    # final normalization and projection
    x = self.norm_final(x)
    logits = self.linear_final(x)  # Shape: (B, T, vocab_size)

    # compute loss if targets are there
    loss = None
    if targets is not None:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss