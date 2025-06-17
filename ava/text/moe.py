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

def rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
  q = (q * cos) + (rotate_half(q) * sin)
  k = (k * cos) + (rotate_half(k) * sin)
  return q, k

def apply_rope_x(x, cos, sin):
  return (x * cos) + (rotate_half(x) * sin)

class MLA(torch.nn.Module):
  def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
    super().__init__()
    self.d_model, self.n_heads = d_model, n_heads
    self.dh = d_model // n_heads
    self.q_proj_dim, self.kv_proj_dim = d_model // 2, (2*d_model) // 3
    self.qk_nope_dim, self.qk_rope_dim = self.dh // 2, self.dh // 2

    ## Q projections
    # Lora
    self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
    self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, self.d_model)))
    self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        
    ## KV projections
    # Lora
    self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
    self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim, self.d_model + (self.n_heads * self.qk_nope_dim))))
    self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

    # output projection
    self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

    # RoPE
    self.max_seq_len = max_len
    self.rope_theta = rope_theta

    # https://github.com/lucidrains/rotary-embedding-torch/tree/main
    # visualize emb later to make sure it looks ok
    # we do self.dh here instead of self.qk_rope_dim because its better
    freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
    emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
    cos_cached = emb.cos()[None, None, :, :]
    sin_cached = emb.sin()[None, None, :, :]

    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
    # This is like a parameter but its a constant so we can use register_buffer
    self.register_buffer("cos_cached", cos_cached)
    self.register_buffer("sin_cached", sin_cached)

  def forward(self, x, kv_cache=None, past_length=0):
    B, S, D = x.size()

    # Q Projections
    compressed_q = x @ self.W_dq
    compressed_q = self.q_layernorm(compressed_q)
    Q = compressed_q @ self.W_uq
    Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
    Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

    # Q Decoupled RoPE
    cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
    sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
    Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

    # KV Projections
    if kv_cache is None:
      compressed_kv = x @ self.W_dkv
      KV_for_lora, K_for_rope = torch.split(compressed_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
      KV_for_lora = self.kv_layernorm(KV_for_lora)
    else:
      new_kv = x @ self.W_dkv
      compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
      new_kv, new_K_for_rope = torch.split(new_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
      old_kv, old_K_for_rope = torch.split(kv_cache, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
      new_kv = self.kv_layernorm(new_kv)
      old_kv = self.kv_layernorm(old_kv)
      KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
      K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)
            

      KV = KV_for_lora @ self.W_ukv
      KV = KV.view(B, -1, self.n_heads, self.dh+self.qk_nope_dim).transpose(1,2)
      K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
      S_full = K.size(2)        

      # K Rope
      K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1,2)
      cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
      sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
      K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

      # apply position encoding to each head
      K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

      # split into multiple heads
      q_heads = torch.cat([Q, Q_for_rope], dim=-1)
      k_heads = torch.cat([K, K_for_rope], dim=-1)
      v_heads = V # already reshaped before the split

      # make attention mask
      mask = torch.ones((S,S_full), device=x.device)
      mask = torch.tril(mask, diagonal=past_length)
      mask = mask[None, None, :, :]

      sq_mask = mask == 1
      # attention
      x = torch.nn.functional.scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=sq_mask)
      x = x.transpose(1, 2).reshape(B, S, D)

      # apply projection
      x = x @ self.W_o.T
      return x, compressed_kv

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
  def __init__(self, d_model, n_heads, n_experts, top_k, n_ff, dropout, ffn_multiplier, block_size, device, n_latent, capacity_factor) -> None:
    super().__init__()
    self.sa = MLA(d_model, n_heads, block_size) # replaced with newer accurate RoPE integrated MLA
    self.ca = MLA(d_model, n_heads, block_size) # replaced with newer accurate RoPE integrated MLA
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
    self.blocks = nn.ModuleList([Block(d_model=params.d_model, n_heads=params.n_heads, n_experts=params.n_experts, top_k=params.top_k, n_ff=params.n_ff, ffn_multiplier=params.ffn_multiplier, dropout=params.dropout, block_size=params.block_size, device=params.device, n_latent=params.n_latent, capacity_factor=params.capacity_factor) for _ in range(self.n_layers)])
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