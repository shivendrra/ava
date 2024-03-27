import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class EncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead)
    self.norm1 = nn.LayerNorm(d_model)
    self.ffwd = FeedForward(d_model, d_ff, dropout)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, src, src_mask=None):
    src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
    src = src + self.dropout(src2)
    src = self.norm1(src)
    src2 = self.ffwd(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)
    
    del src2
    return src

class Encoder(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
    super().__init__()
    self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)])

  def forward(self, src, src_mask=None):
    for layer in self.layers:
      src = layer(src, src_mask=src_mask)
    return src

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(FeedForward, self).__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.dropout(nn.GELU(self.linear1(x)))
    x = self.linear2(x)
    return x

class TransformerASR(nn.Module):
  def __init__(self, input_dim, output_dim, n_heads=8, n_layers=6, d_model=512, d_ff=2048, dropout=0.1):
    super(TransformerASR, self).__init__()
    self.encoder = nn.Linear(input_dim, d_model)
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    self.encoder = Encoder(d_model, n_heads, d_ff, dropout, n_layers)
    self.decoder = nn.Linear(d_model, output_dim)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, src, src_mask=None):
    src = self.encoder(src)
    src = self.pos_encoder(src)
    output = self.encoder(src, src_mask=src_mask)
    output = self.decoder(output)
    return self.softmax(output)

input_dim = 80
output_dim = 30
model = TransformerASR(input_dim, output_dim)

batch_size = 32
seq_len = 100
dummy_input = torch.randn(batch_size, seq_len, input_dim)

src_mask = (dummy_input.sum(dim=2) != 0).transpose(0, 1)

output = model(dummy_input, src_mask=src_mask)
print("Output shape:", output.shape)