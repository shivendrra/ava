class ModelConfig:
  d_model:int = 256
  in_dim:int = 8
  n_embed:int = 512
  beta:float = 0.25
  n_heads:int = 12
  n_layers:int = 8
  dropout:float = 0.2

import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
  def __init__(self, dim:int, eps:float=1e-5):
    super().__init__()
    self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  def forward(self, x):
    out = self._norm(x.float()).type_as(x)
    return out * self.weight

class Encoder(nn.Module):
  def __init__(self, _in, d_model, n_layers, n_heads, dropout):
    super().__init__()
    self.embed = nn.Linear(_in, d_model)
    self.encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, bias=False, activation=F.gelu),
      num_layers=n_layers
    )
  def forward(self, x):
    x = self.embed(x)
    x = x.premute(1, 0, 2)
    z_e = self.encoder(x)
    return z_e.premute(1, 0, 2)

class Decoder(nn.Module):
  def __init__(self, d_model, _out, n_layers, n_heads, dropout):
    super().__init__()
    self.decoder = nn.TransformerDecoder(
      nn.TransformerDecoderLayer(d_model, n_heads, dropout=dropout, bias=False, activation=F.gelu),
      num_layers=n_layers
    )
    self.fc_out = nn.Linear(d_model, _out)
  
  def forward(self, z_q):
    z_q = z_q.permute(1, 0, 2)
    x_recon = self.decoder(z_q, z_q)
    x_recon = self.fc_out(x_recon.permute(1, 0, 2))
    return x_recon

class Quantizer(nn.Module):
  def __init__(self, n_embed, d_model, beta):
    super().__init__()
    self.n_embed, self.d_model, self.beta = n_embed, d_model, beta
    self.embeddings = nn.Embedding(n_embed, d_model)
    self.embeddings.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
  
  def forward(self, z_e):
    z_e_flat = z_e.view(-1, self.d_model)
    distance = torch.cdist(z_e_flat, self.embeddings.weight)
    encoding_indices = torch.argmin(distance, dim=1)
    z_q = self.embeddings(encoding_indices).view(z_e.shape)
    loss = self.beta * torch.mean((z_q.detach() - z_e) ** 2) + torch.mean((z_e.detach() - z_q) ** 2)

    z_q = z_e + (z_q - z_e).detach()
    return z_q, loss, encoding_indices.view(z_e.shape[:-1])

class AudioVQVAE(nn.Module):
  def __init__(self, args: ModelConfig):
    super().__init__()
    self.encoder = Encoder(args.in_dim, args.d_model, args.n_layers, args.n_heads, args.dropout)
    self.vq_layer = Quantizer(args.n_embed, args.d_model, args.beta)
    self.decoder = Decoder(args.d_model, args.in_dim, args.n_layers, args.n_heads, args.dropout)
  
  def forward(self, x):
    z_e = self.encoder(x)
    z_q, vq_loss, indices = self.vq_layer(z_e)
    x_recon = self.decoder(z_q)
    return x_recon, vq_loss, indices