import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnMaskedHead(nn.Module):
  def __init__(self, head_size, d_model, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=False)
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

class EncoderAttention(nn.Module):
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
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
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

class FeedForward(nn.Module):
  def __init__(self, d_model, dropout):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(d_model, 10*d_model),
      nn.GELU(),
      nn.Linear(10*d_model, d_model),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class EncoderNetwork(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.s_att = EncoderAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, src):
    src = self.norm1(src)
    src2 = self.s_att(src)
    src = src + self.dropout(src2)

    src = self.norm2(src)
    src2 = self.ffwd(src)
    src = src + self.dropout(src2)

    return src

class DecoderNetwork(nn.Module):
  def __init__(self, d_model, n_head, norm_eps, dropout, block_size):
    super().__init__()
    self.s_att = CasualMaskedAttention(n_head=n_head, d_model=d_model, dropout=dropout, block_size=block_size)
    self.ffwd = FeedForward(d_model, dropout)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
    self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

  def forward(self, src, att):
    src2 = self.s_att(src)
    src = src + self.dropout(src2)
    src = src + self.norm1(src)

    att2 = self.s_att(att)
    att2 = att + self.dropout(att2)
    trg = att2 + self.norm1(att2)

    src_f2 = self.ffwd(self.norm2(trg))
    src_f = src_f + self.dropout(src_f2)
    src_f = self.norm2(src_f)

    return src_f

class Transformer(nn.Module):
  def __init__(self, vocab_size, n_head, block_size, d_model, n_layers, norm_eps, dropout):
    super().__init__()
    self.block_size = block_size
    self.toked_model = nn.Embedding(vocab_size, d_model)
    self.pos_encod = nn.Embedding(block_size, d_model)
    self.enc_layer = nn.ModuleList([EncoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])
    self.dec_layer = nn.ModuleList([DecoderNetwork(n_head=n_head, norm_eps=norm_eps, block_size=block_size, dropout=dropout, d_model=d_model) for _ in range(n_layers)])

    self.norm_final = nn.LayerNorm(d_model)
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

  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
    """
      generate new tokens using the trained model

    Args:
      - idx (Tensor): input tensor representing initial token indices
      - max_new_tokens (int): max no of new tokens to generate
      - temperature (float): softmax temperature for sampling
      - top_k (int): no of top tokens to consider in sampling

    Returns:
      - generated_tokens (list): list of generated token indices
    """
    generated_tokens = []

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
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
  
  def generate_masked_tokens(self, idx, masked_indices, temperature=1.0, top_k=0):
    """
      Generate predictions for masked tokens using the trained model.

      Args:
        - idx (Tensor): input tensor representing token indices
        - masked_indices (Tensor): tensor of indices indicating masked positions
        - temperature (float): softmax temperature for sampling
        - top_k (int): no of top tokens to consider in sampling

      Returns:
        - predicted_tokens (Tensor): tensor of predicted token indices
    """
    B, T = idx.shape

    toked_model = self.toked_model(idx)
    pos_encod = self.pos_encod(torch.arange(T, device=device))
    x = toked_model + pos_encod

    for layer in self.enc_layer:
      x_out = layer(x)

    for layer in self.dec_layer:
      x_final = layer(x, x_out)

    x_masked = x_final.clone()
    x_masked[masked_indices] = self.toked_model(torch.tensor([6], device=device))

    x_masked = self.norm_final(x_masked)
    logits = self.linear_final(x_masked)

    masked_logits = logits[masked_indices].view(-1, logits.size(-1))
    scaled_logits = masked_logits / temperature
    if top_k > 0:
      scaled_logits = self._top_k_filtering(scaled_logits, top_k)

    probs = F.softmax(scaled_logits, dim=-1)
    predicted_indices = torch.argmax(probs, dim=-1)

    return predicted_indices
  
  def _top_k_filtering(self, logits, top_k):
    """
      filter logits to keep only the top-k tokens

    Args:
      - logits (Tensor): input tensor representing unscaled logits
      - top_k (int): no of top tokens to keep

    Returns:
      - filtered_logits (Tensor): filtered logits with only top-k tokens remaining
    """
    values, indices = torch.topk(logits, top_k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1).expand_as(logits)
    filtered_logits = torch.where(logits < min_value, torch.ones_like(logits) * -float('inf'), logits)

    return filtered_logits