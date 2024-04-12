import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConfigModel():
  d_model = 512
  block_size = 256
  n_head = 18
  n_layers = 12
  dropout = 0.2
  norm_eps = 1e-5
  n_ff = 5 * d_model

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

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, block_size, dropout):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    pe = torch.zeros(block_size, d_model)
    position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)

class NewGELU(nn.Module):
  def forward(self, input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
class SelfAttention(nn.Module):
  def __init__(self, head_size, d_model, block_size, dropout):
    super().__init__()
    self.w_k = nn.Linear(d_model, head_size, bias=True)
    self.w_q = nn.Linear(d_model, head_size, bias=True)
    self.w_v = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.rel_pos_embd = nn.Parameter(torch.randn(block_size, block_size, head_size))

  def forward(self, x, dec_x=None):
    batch_s, seq_len, _ = x.shape

    k = self.w_k(x)
    q = self.w_q(dec_x) if dec_x is not None else self.w_q(x)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (1.0 / math.sqrt(k.shape[-1]))
    rel_pos_scores = torch.einsum('btc,tvc->btv', q, self.rel_pos_embd[:seq_len, :seq_len])
    scores = scores + rel_pos_scores

    att_mat = F.softmax(scores, dim=-1)
    att_mat = self.dropout(att_mat)
    v = self.w_v(x)
    output = torch.matmul(att_mat, v)
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, block_size, dropout, n_head):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList([SelfAttention(d_model=d_model, dropout=dropout, block_size=block_size, head_size=head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, dec_x=None):
    out = torch.cat([h(x, dec_x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
  
class MaskedHead(nn.Module):
  def __init__(self, d_model, head_size, dropout, block_size):
    super().__init__()
    self.w_k = nn.Linear(d_model, head_size, bias=True)
    self.w_q = nn.Linear(d_model, head_size, bias=True)
    self.w_v = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    batch_s, seq_len, _ = x.shape
    k, q = self.w_k(x), self.w_q(x)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (1.0 / math.sqrt(k.shape[-1]))
    scores = scores.masked_fill(self.tril[:seq_len, :seq_len] == 0, float('-inf'))
    
    att_mat = F.softmax(scores, dim=-1)
    att_mat = self.dropout(att_mat)
    value = self.value(x)
    output = torch.matmul(att_mat, value)  
    return output

class MultiMaskedAttention(nn.Module):
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

class MLP(nn.Module):
  def __init__(self, n_ff, d_model, dropout):
    super().__init__()
    self.layer = nn.Linear(d_model, n_ff)
    self.n_gelu = NewGELU
    self.outputs = nn.Linear(n_ff, d_model)
    self.dropout = nn.Dropout(dropout)
  
  def forwward(self, x):
    x = self.n_gelu(self.layer(x))
    x = self.dropout(self.outputs(x))
    return x

class Encoder(nn.Module):
  def __init__(self, d_model, block_size, n_head, dropout, n_ff, norm_eps):
    super().__init__()
    self.s_att = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.mlp = MLP(n_ff, d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)
  
  def forward(self, aud_src):
    aud_src = aud_src + self.dropout(self.s_att(self.norm(aud_src)))
    aud_src = aud_src + self.dropout(self.mlp(self.norm(aud_src)))
    return aud_src

class Decoder(nn.Module):
  def __init__(self, d_model, block_size, n_head, dropout, n_ff, norm_eps):
    super().__init__()
    self.m_att = MultiMaskedAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.s_att = MultiHeadAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.mlp = MLP(n_ff, d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm = RMSNorm(d_model, eps=norm_eps)
  
  def forward(self, src, enc_trg):
    src = src + self.dropout(self.m_att(self.norm(src)))
    src = src + self.dropout(self.s_att(self.norm(enc_trg), self.norm(src)))
    src = src + self.dropout(self.mlp(self.norm(src)))
    return src

class Transformer(nn.Module):
  def __init__(self, vocab_size, config: ConfigModel):
    super().__init__()
    self.block_size = config.block_size
    self.toked_model = nn.Embedding(vocab_size, config.d_model)
    self.pos_encod = nn.Embedding(self.block_size, config.d_model)
    self.aud_pos = PositionalEncoding(config.d_model, self.block_size, config.dropout)
    self.encoder = nn.ModuleList([Encoder(config.d_model, self.block_size, config.n_head, config.dropout, config.n_ff, config.norm_eps) for _ in range(config.n_layers)])
    self.decoder = nn.ModuleList([Decoder(config.d_model, self.block_size, config.n_head, config.dropout, config.n_ff, config.norm_eps) for _ in range(config.n_layers)])
    self.norm_final = RMSNorm(config.d_model, eps=config.norm_eps)
    self.linear_final = nn.Linear(config.d_model, vocab_size)
    self.dropout = nn.Dropout(config.dropout)
  
  def forward(self, aud_in, txt_in):
    B, T = txt_in.shape
    
    toked_model = self.toked_model(txt_in)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod

    aud_tok = self.aud_pos(aud_in)

    for layer in self.encoder:
      aud_tok = layer(aud_tok)
    
    for layer in self.decoder:
      x_final = layer(x, aud_tok)
    x_final = self.norm_final(x_final)
    logits = self.linear_final(x_final)

    return logits

  def model_train(self, aud_in, txt_in, trg):
    assert trg is not None
    logits = self(aud_in, txt_in)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)

    return loss