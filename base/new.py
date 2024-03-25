import json
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

import torch
import torch.nn as nn
from torch.nn import functional as F

with open('config_enigma.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

batch_size = params['batch_size']
block_size = params['block_size']
n_head = params['n_head']
d_model = params['d_model']
n_layers = params['n_layer']
dropout = params['dropout']
norm_eps = params['norm_eps']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MaskedHead(nn.Module):
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    key = self.key(x)
    query = self.query(x)

    weights = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1]**-0.5)
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)

    value = self.value(x)
    out = torch.matmul(weights, value)
    return out

class MultiMaskedAttention(nn.Module):
  def __init__(self, d_model, n_head, dropout, block_size):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([MaskedHead(d_model=d_model, dropout=dropout, head_size=head_size, block_size=block_size) for _ in range(n_head)])
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)
    return out

class UnMaskedHead(nn.Module):
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    key = self.key(x)
    query = self.query(x)

    weights = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1]**-0.5)
    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)

    value = self.value(x)
    out = torch.matmul(weights, value)

    return out, weights

class EncoderAttention(nn.Module):
  def __init__(self, d_model, n_head, dropout, block_size):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([UnMaskedHead(d_model=d_model, dropout=dropout, head_size=head_size, block_size=block_size) for _ in range(n_head)])
    self.dropout = nn.Dropout(dropout)

  def forward(self, src):
    out_list = []
    weights_list = []
    for h in self.heads:
      out, weights = h(src)
      out_list.append(out)
      weights_list.append(weights)

    out = torch.cat(out_list, dim=-1)
    weights = torch.cat(weights_list, dim=-1)
    out = self.dropout(out)
    return out, weights

class FinalHead(nn.Module):
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=False)
    self.proj = nn.Linear(head_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, wei, val):
    wei = self.proj(wei)
    return torch.matmul(wei, val.transpose(-2, -1))

class DecoderAttention(nn.Module):
  def __init__(self, d_model, n_head, dropout, block_size):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([FinalHead(d_model=d_model, dropout=dropout, head_size=head_size, block_size=block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, wei, val):
    out= torch.cat([h(wei, val) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.ln1 = nn.Linear(d_model, 5*d_model)
    self.gelu = nn.GELU()
    self.ln2 = nn.Linear(5*d_model, d_model)
    self.drop = nn.Dropout(dropout)

  def forward(self, x):
    return self.drop(self.ln2(self.gelu(self.ln1(x))))

class EncoderNetwork(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.s_att = EncoderAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, src):
    src2, att = self.s_att(src)
    src = src + self.dropout(src2)
    src = self.norm1(src)

    src2 = self.ffwd(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)

    return src, att

class DecoderNetwork(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.m_att = MultiMaskedAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.d_att = DecoderAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, src, att):
    src2 = self.m_att(src)
    src = src + self.dropout(src2)
    src = src + self.norm1(src)

    trg2 = self.d_att(att, src)
    trg = trg + self.dropout(trg2)
    trg_f = trg + self.norm1(trg)

    src_f2 = self.ffwd(self.norm2(trg_f))
    src_f = src_f + self.dropout(src_f2)
    src_f = self.norm2(src_f)

    return src_f

class Transformer(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.toked_model = nn.Embedding(vocab_size, d_model)
    self.pos_encod = nn.Embedding(block_size, d_model)
    self.enc_layer = nn.ModuleList([EncoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])
    self.dec_layer = nn.ModuleList([DecoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])

    self.norm_final = nn.LayerNorm(d_model)
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

  def forward(self, idx, targets=None):
    B, T = idx.shape

    toked_model = self.toked_model(idx)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod

    for layer in self.enc_layer:
      wei, att = layer(x)

    for layer in self.dec_layer:
      x = layer(x, att)

    x = self.norm_final(x)
    logits = self.linear_final(x)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
    generated_tokens = []

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]

      scaled_logits = logits / temperature
      if top_k > 0:
        scaled_logits = self._top_k_filtering(scaled_logits, top_k)

      probs = F.softmax(scaled_logits, dim=-1)
      sampled_idx = torch.multinomial(probs, num_samples=1)
      generated_tokens.append(sampled_idx.item())
      idx = torch.cat((idx, sampled_idx), dim=1)

    return generated_tokens

  def _top_k_filtering(self, logits, top_k):
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1).expand_as(logits)
    filtered_logits = torch.where(logits < min_value, torch.ones_like(logits) * -float('inf'), logits)

    return filtered_logits