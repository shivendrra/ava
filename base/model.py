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
    """
      Initialize the RMSNorm normalization layer.
      Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
      Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.
    """
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    """
      Apply the RMSNorm normalization to the input tensor.
        Args:
        x (torch.Tensor): The input tensor.
      Returns:
        torch.Tensor: The normalized tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    """
      Forward pass through the RMSNorm layer.
      Args:
          x (torch.Tensor): The input tensor.
      Returns:
          torch.Tensor: The output tensor after applying RMSNorm.
    """
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class UnMaskedHead(nn.Module):
  def __init__(self, head_size, d_model, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.rel_pos_embd = nn.Parameter(torch.randn(block_size, block_size, head_size))

  def forward(self, x):
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** -0.5)
    rel_pos_scores = torch.einsum('btc,tvc->btv', query, self.rel_pos_embd[:T, :T])
    scores = scores + rel_pos_scores

    att_mat = F.softmax(scores, dim=-1)
    att_mat = self.dropout(att_mat)
    value = self.value(x)
    output = torch.matmul(att_mat, value)
    return output

class UnMaskedAttention(nn.Module):
  def __init__(self, d_model, block_size, dropout, n_head):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([UnMaskedHead(d_model=d_model, dropout=dropout, block_size=block_size, head_size=head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class MaskedHead(nn.Module):
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=False)
    self.query = nn.Linear(d_model, head_size, bias=False)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** -0.5)
    scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

    att_mat = F.softmax(scores, dim=-1)
    att_mat = self.dropout(att_mat)
    value = self.value(x)
    output = torch.matmul(att_mat, value)
    return output

class CasualMaskedAttention(nn.Module):
  def __init__(self, d_model, block_size, dropout, n_head):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([MaskedHead(d_model=d_model, dropout=dropout, block_size=block_size, head_size=head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FinalHead(nn.Module):
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, att):
    B, T, C = x.shape
    key = self.key(att)
    query = self.query(att)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** -0.5)

    att_mat = F.softmax(scores, dim=-1)
    att_mat = self.dropout(att_mat)
    value = self.value(x)
    output = torch.matmul(att_mat, value)
    return output

class FinalAttention(nn.Module):
  def __init__(self, d_model, block_size, dropout, n_head):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([FinalHead(d_model=d_model, dropout=dropout, block_size=block_size, head_size=head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, att):
    out = torch.cat([h(x, att) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 4*d_model),
      nn.GELU(),
      nn.Linear(4*d_model, d_model),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class EncoderNetwork(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.s_att = UnMaskedAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)

  def forward(self, src):
    src = self.norm(src)
    src_out = src + self.dropout(self.s_att(src))

    src = self.norm(src_out)
    src_f = src + self.dropout(self.ffwd(src))

    del src_out, src
    return src_f

class DecoderNetwork(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.m_att = CasualMaskedAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.f_att = FinalAttention(d_model=d_model, n_head=n_head, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)

  def forward(self, src, att):
    m_att_out = self.norm(src)
    m_out = src + self.dropout(self.m_att(m_att_out))

    f_out = self.f_att(m_out, self.norm(att))
    f_out = m_out + self.dropout(f_out)

    src_f = self.norm(f_out)
    src_f = f_out + self.dropout(self.ffwd(src_f))

    del f_out, m_out, m_att_out, src, att
    return src_f

class Transformer(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.block_size = block_size
    self.toked_model = nn.Embedding(vocab_size, d_model)
    self.pos_encod = nn.Embedding(block_size, d_model)
    self.enc_layer = nn.ModuleList([EncoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])
    self.dec_layer = nn.ModuleList([DecoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])
    self.norm_final = RMSNorm(d_model, eps=norm_eps)
    self.linear_final = nn.Linear(d_model, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """
      initialize weights of linear and embedding layers

      Args:
        - module (nn.Module): the module to initialize weights for
    """
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    """
      forward pass of the transformer model

    Args:
      - idx (Tensor): input tensor representing token indices
      - targets (Tensor): target tensor for computing loss during training

    Returns:
      - logits (Tensor): output logits from the final linear layer
      - loss (Tensor): optional. computed cross-entropy loss if targets are provided, else None
    """
    B, T = idx.shape

    toked_model = self.toked_model(idx)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod

    for layer in self.enc_layer:
      x_out = layer(x)

    for layer in self.dec_layer:
      x_final = layer(x, x_out)

    x_final = self.norm_final(x_final)
    logits = self.linear_final(x_final)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss