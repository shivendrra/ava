import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelArgs:
  d_model:int = 1024
  n_layers:int = 12
  n_heads:int = 18
  n_ff_multiple:int = 10
  ffn_multiplier:int = 4
  n_ff:int = n_ff_multiple * d_model
  n_latent:int = 64
  dropout:float = 0.2
  norm_eps:float = 1e-5
  block_size:int = 1024
  n_experts:int = 4
  top_k:int = 2
  capacity_factor:int = 2
  device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
  def __init__(self, in_dim, hidden_dim):
    super().__init__()
    # project from in_dim to 2 * hidden_dim
    self.proj = nn.Linear(in_dim, 2 * hidden_dim, bias=False)
  
  def forward(self, x):
    # x: [batch, ..., in_dim]
    x_proj = self.proj(x)            # [batch, ..., 2 * hidden_dim]
    x1, x2 = x_proj.chunk(2, dim=-1)  # each [batch, ..., hidden_dim]
    return F.silu(x1) * x2

class RoPE(nn.Module):
  def __init__(self, head_size, block_size, device):
    super().__init__()
    self.head_size = head_size
    self.block_size = block_size
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
    position = torch.arange(0, self.block_size, dtype=torch.float, device=device)  # (block_size,)
    sinusoidal = torch.einsum("i,j->ij", position, inv_freq)  # Shape: (block_size, head_size // 2)
    self.register_buffer("cos_emb", sinusoidal.cos(), persistent=False)  # (block_size, head_size // 2)
    self.register_buffer("sin_emb", sinusoidal.sin(), persistent=False)  # (block_size, head_size // 2)

  def forward(self, q, k):
    # spliting tensors into even and odd components
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    assert q.size(-1) == self.head_size, f"Query size mismatch: {q.size(-1)} != {self.head_size}"
    assert k.size(-1) == self.head_size, f"Key size mismatch: {k.size(-1)} != {self.head_size}"
    # retrieving embeddings for current sequence length
    cos = self.cos_emb[:q.shape[1], :].unsqueeze(0).to(q.device)
    sin = self.sin_emb[:q.shape[1], :].unsqueeze(0).to(q.device)

    # applying rotations
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

class LatentHead(nn.Module):
  """
    Multi-Query Latent Attention
    discussed in paper by deepseek: https://arxiv.org/pdf/2502.07864
  """
  def __init__(self, head_size, d_model, dropout, block_size, device, latent_dim=None, mask=False):
    super().__init__()
    # Default latent dimension: compress to half of head_size if not provided
    if latent_dim is None:
      latent_dim = max(1, head_size // 2)
    self.query = nn.Linear(d_model, head_size, bias=True)

    # For keys: decompose the projection into two parts: Wa_K and Wb_K
    self.wa_key = nn.Linear(d_model, latent_dim, bias=False)
    self.wb_key = nn.Linear(latent_dim, head_size, bias=False)

    # For values: similar decomposition into Wa_V and Wb_V
    self.wa_value = nn.Linear(d_model, latent_dim, bias=False)
    self.wb_value = nn.Linear(latent_dim, head_size, bias=False)

    self.dropout = nn.Dropout(dropout)
    self.mask = mask
    if mask:
      self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.pos_emb = RoPE(head_size, block_size, device)    # Rotary Positional Embedding layer remains the same

  def forward(self, x):
    B, T, C = x.shape

    # Compute query, latent key and latent value
    query = self.query(x)                      # (B, T, head_size)
    latent_key = self.wa_key(x)                # (B, T, latent_dim)
    latent_value = self.wa_value(x)            # (B, T, latent_dim)

    # Expand the latent representations to full key and value
    key = self.wb_key(latent_key)              # (B, T, head_size)
    value = self.wb_value(latent_value)          # (B, T, head_size)
    query, key = self.pos_emb(query, key)     # cpply Rotary Positional Embedding to query and key
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)    # compute scaled dot-product attention scores

    # apply masking if needed (decoder part)
    if self.mask:
      if T > self.tril.size(0):
        self.tril = torch.tril(torch.ones(T, T, device=scores.device))
      scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

    attention = self.dropout(F.softmax(scores, dim=-1))   # apply softmax and dropout
    output = torch.matmul(attention, value)    # compute the output as the weighted sum of the value vectors
    return output

class MultiHeadLatentAttention(nn.Module):
  def __init__(self, d_model, dropout, n_head, block_size, device, mask, latent_dim=None):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList(
      [LatentHead(head_size, d_model, dropout, block_size, device, latent_dim, mask) for _ in range(n_head)]
    )
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class Head(nn.Module):
  def __init__(self, head_size, d_model, dropout, block_size, device, mask=False):
    super().__init__()
    self.key = nn.Linear(d_model, head_size, bias=True)
    self.query = nn.Linear(d_model, head_size, bias=True)
    self.value = nn.Linear(d_model, head_size, bias=True)
    self.dropout = nn.Dropout(dropout)
    self.mask = mask
    if mask:
      self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.pos_emb = RoPE(head_size, block_size, device)
  def forward(self, x):
    B, T, C = x.shape
    key, query, value = self.key(x), self.query(x), self.value(x)
    query, key = self.pos_emb(query, key)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)
    if T > self.tril.size(0):
      self.tril = torch.tril(torch.ones(T, T, device=scores.device))
    scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    attention = self.dropout(F.softmax(scores, dim=-1))
    output = torch.matmul(attention, value)
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, dropout, n_head, block_size, device, mask):
    head_size = d_model // n_head
    super().__init__()
    self.heads = nn.ModuleList(
      [Head(head_size, d_model, dropout, block_size, device, mask) for _ in range(n_head)]
    )
    self.proj = nn.Linear(n_head * head_size, d_model)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class Expert(nn.Module):
  def __init__(self, d_model, hidden_dim, dropout, ffn_multiplier=None) -> None:
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_multiplier is not None:
      hidden_dim = int(ffn_multiplier * hidden_dim)
    self.swiglu = SwiGLU(d_model, hidden_dim)   # SwiGLU will project from d_model to 2 * hidden_dim
    self.fc = nn.Linear(hidden_dim, d_model, bias=False)   # then project back to d_model
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    x = self.swiglu(x)  # apply SwiGLU activation on the input x
    return self.dropout(self.fc(x))   # then project back and apply dropout

class NoisyTopkRouter(nn.Module):
  def __init__(self, n_embed, n_experts, top_k) -> None:
    super().__init__()
    self.top_k = top_k
    # layer for router logits
    self.topkroute_linear = nn.Linear(n_embed, n_experts)
    self.noise_linear = nn.Linear(n_embed, n_experts)
  def forward(self, mh_output):
    # mh_ouput is the output tensor from multihead self attention block
    logits = self.topkroute_linear(mh_output)
    # noise logits
    noise_logits = self.noise_linear(mh_output)
    # adding scaled unit gaussian noise to the logits
    noise = torch.randn_like(logits)*F.softplus(noise_logits)
    noisy_logits = logits + noise
    top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
    zeros = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    return router_output, indices

class SparseMoE(nn.Module):
  def __init__(self, d_model, n_experts, top_k, n_ff, dropout, ffn_multiplier, capacity_factor=1.0) -> None:
    super().__init__()
    self.router = NoisyTopkRouter(d_model, n_experts, top_k)
    self.experts = nn.ModuleList([Expert(d_model=d_model, hidden_dim=n_ff, dropout=dropout, ffn_multiplier=ffn_multiplier) for _ in range(n_experts)])
    self.top_k, self.capacity_factor, self.n_experts = top_k, capacity_factor, n_experts
  def forward(self, x):
    batch_size, seq_len, _ = x.shape
    gating_output, indices = self.router(x)
    final_outputs = torch.zeros_like(x)

    flat_x = x.view(-1, x.size(-1))
    flat_gating_output = gating_output.view(-1, gating_output.size(-1))
    tokens_per_batch = batch_size * seq_len * self.top_k
    expert_capacity = int((tokens_per_batch / self.n_experts) * self.capacity_factor)
    updates = torch.zeros_like(flat_x)

    for i, expert in enumerate(self.experts):
      expert_mask = (indices == i).any(dim=-1)
      flat_mask = expert_mask.view(-1)
      selected_indices = torch.nonzero(flat_mask).squeeze(-1)
      limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
      if limited_indices.numel() > 0:
        expert_inputs = flat_x[limited_indices]
        expert_output = expert(expert_inputs)
        gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
        weighted_outputs = expert_output * gating_scores
        updates.index_add_(0, limited_indices, weighted_outputs)

    final_outputs += updates.view(batch_size, seq_len, -1)
    return final_outputs

class Block(nn.Module):
  def __init__(self, d_model, n_head, n_experts, top_k, n_ff, dropout, ffn_multiplier, block_size, device, n_latent, capacity_factor) -> None:
    super().__init__()
    self.sa = MultiHeadLatentAttention(d_model, dropout, n_head, block_size, device, False, n_latent)
    self.ca = MultiHeadLatentAttention(d_model, dropout, n_head, block_size, device, True, n_latent)
    self.smoe = SparseMoE(d_model, n_experts, top_k, n_ff, dropout, ffn_multiplier, capacity_factor)
    self.ln1 = RMSNorm(d_model)
    self.ln2 = RMSNorm(d_model)
  def forward(self, x):
    x = x + self.ca((self.ln1(x)))
    x = x + self.smoe(self.sa(self.ln2(x)))
    return x

def kaiming_init_weights(m):
  if isinstance (m, (nn.Linear)): nn.init.kaiming_normal_(m.weight)

class TransformerMoE(nn.Module):
  def __init__(self, params: ModelArgs, vocab_size: int):
    super().__init__()
    self.block_size = params.block_size
    self.d_model = params.d_model
    self.n_layers = params.n_layers
    self.token_embeddings = nn.Embedding(vocab_size, self.d_model)
    self.blocks = nn.ModuleList([Block(d_model=params.d_model, n_head=params.n_heads, n_experts=params.n_experts, top_k=params.top_k, n_ff=params.n_ff, ffn_multiplier=params.ffn_multiplier, dropout=params.dropout, block_size=params.block_size, device=params.device, n_latent=params.n_latent, capacity_factor=params.capacity_factor) for _ in range(self.n_layers)])
    self.norm_final = RMSNorm(self.d_model, params.norm_eps)
    self.linear_final = nn.Linear(self.d_model, vocab_size, bias=False)
    self.apply(kaiming_init_weights)

  def forward(self, idx, targets=None):
    B, T = idx.size()
    x = self.token_embeddings(idx)  # Shape: (B, T, d_model)
    # through decoder layers
    for layer in self.blocks:
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