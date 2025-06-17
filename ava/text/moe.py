import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
  def forward(self, x):
    out = self._norm(x.float()).type_as(x)
    return out * self.weight

class SwiGLU(nn.Module):
  """
  SwiGLU activation function
  SwiGLU(x,W,V,b,c,b) = Swish_b(xW + b) * (xV + c)
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

def apply_rope_x(x, cos, sin):
  return (x * cos) + (rotate_half(x) * sin)

class MLA(nn.Module):
  def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.dh = d_model // n_heads
    self.qk_nope_dim = self.dh // 2
    self.qk_rope_dim = self.dh - self.qk_nope_dim  # Ensure dimensions add up correctly

    # Standard linear projections (no LoRA)
    self.q_proj = nn.Linear(d_model, d_model, bias=False)
    self.k_proj = nn.Linear(d_model, d_model, bias=False)
    self.v_proj = nn.Linear(d_model, d_model, bias=False)
    self.o_proj = nn.Linear(d_model, d_model, bias=False)

    # RoPE parameters
    self.max_seq_len = max_len
    self.rope_theta = rope_theta

    # Precompute RoPE embeddings
    rope_dim = self.qk_rope_dim
    freqs = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
    cos_cached = emb.cos()[None, None, :, :]
    sin_cached = emb.sin()[None, None, :, :]

    self.register_buffer("cos_cached", cos_cached)
    self.register_buffer("sin_cached", sin_cached)

  def forward(self, x, kv_cache=None, past_length=0, is_causal=True):
    B, S, D = x.size()

    # Standard Q, K, V projections
    Q = self.q_proj(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)
    K = self.k_proj(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)
    V = self.v_proj(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)

    # Split Q and K into nope and rope parts
    Q_nope, Q_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
    K_nope, K_rope = torch.split(K, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

    # handling KV caching
    if kv_cache is not None:
      past_K_nope, past_K_rope, past_V = kv_cache
      K_nope = torch.cat([past_K_nope, K_nope], dim=2)
      K_rope = torch.cat([past_K_rope, K_rope], dim=2)
      V = torch.cat([past_V, V], dim=2)

    S_full = K_nope.size(2)

    # applying RoPE to Q
    rope_dim_half = self.qk_rope_dim // 2
    cos_q = self.cos_cached[:, :, past_length:past_length+S, :rope_dim_half]
    sin_q = self.sin_cached[:, :, past_length:past_length+S, :rope_dim_half]
    if self.qk_rope_dim % 2 != 0:
      cos_q = torch.cat([cos_q, cos_q[:, :, :, -1:]], dim=-1)
      sin_q = torch.cat([sin_q, sin_q[:, :, :, -1:]], dim=-1)
    else:
      cos_q = cos_q.repeat(1, 1, 1, 2)
      sin_q = sin_q.repeat(1, 1, 1, 2)
    
    Q_rope = apply_rope_x(Q_rope, cos_q, sin_q)

    # applying RoPE to K
    cos_k = self.cos_cached[:, :, :S_full, :rope_dim_half]
    sin_k = self.sin_cached[:, :, :S_full, :rope_dim_half]
    if self.qk_rope_dim % 2 != 0:
      cos_k = torch.cat([cos_k, cos_k[:, :, :, -1:]], dim=-1)
      sin_k = torch.cat([sin_k, sin_k[:, :, :, -1:]], dim=-1)
    else:
      cos_k = cos_k.repeat(1, 1, 1, 2)
      sin_k = sin_k.repeat(1, 1, 1, 2)
    
    K_rope = apply_rope_x(K_rope, cos_k, sin_k)

    # Combine nope and rope parts
    Q_final = torch.cat([Q_nope, Q_rope], dim=-1)
    K_final = torch.cat([K_nope, K_rope], dim=-1)

    # creating attention mask
    if is_causal:
      # Causal mask for decoder self-attention
      mask = torch.tril(torch.ones(S, S_full, device=x.device, dtype=torch.bool))
      if past_length > 0:
        # Adjust mask for cached keys/values
        causal_mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        prefix_mask = torch.ones(S, past_length, device=x.device, dtype=torch.bool)
        mask = torch.cat([prefix_mask, causal_mask], dim=1)
    else:
      mask = torch.ones(S, S_full, device=x.device, dtype=torch.bool) # Full attention mask for cross-attention

    # applying scaled dot-product attention
    x = F.scaled_dot_product_attention(Q_final, K_final, V, attn_mask=mask, dropout_p=0.0 if not self.training else 0.1)

    x = x.transpose(1, 2).reshape(B, S, D)
    x = self.o_proj(x)  # applying output projection
    new_kv_cache = (K_nope, K_rope, V) if kv_cache is not None or past_length > 0 else None    # Prepare new cache
    return x, new_kv_cache

class Expert(nn.Module):
  def __init__(self, d_model, hidden_dim, dropout, ffn_multiplier=None):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_multiplier is not None:
      hidden_dim = int(ffn_multiplier * hidden_dim)
    self.swiglu = SwiGLU(d_model, hidden_dim)
    self.fc = nn.Linear(hidden_dim, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.swiglu(x)
    return self.dropout(self.fc(x))

class NoisyTopkRouter(nn.Module):
  def __init__(self, n_embed, n_experts, top_k):
    super().__init__()
    self.top_k = top_k
    self.n_experts = n_experts
    
    # Router layers
    self.topkroute_linear = nn.Linear(n_embed, n_experts, bias=False)
    self.noise_linear = nn.Linear(n_embed, n_experts, bias=False)
  
  def forward(self, x):
    # x is the input tensor (batch_size, seq_len, d_model)
    logits = self.topkroute_linear(x)
    
    if self.training:
      # adding noise during training
      noise_logits = self.noise_linear(x)
      noise = torch.randn_like(logits) * F.softplus(noise_logits)
      noisy_logits = logits + noise
    else:
      noisy_logits = logits
    top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)     # Get top-k experts

    # creating sparse logits
    zeros = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    return router_output, indices

class SparseMoE(nn.Module):
  def __init__(self, d_model, n_experts, top_k, n_ff, dropout, ffn_multiplier, capacity_factor=1.25):
    super().__init__()
    self.router = NoisyTopkRouter(d_model, n_experts, top_k)
    self.experts = nn.ModuleList([Expert(d_model=d_model, hidden_dim=n_ff, dropout=dropout, ffn_multiplier=ffn_multiplier) for _ in range(n_experts)])
    self.top_k = top_k
    self.capacity_factor = capacity_factor
    self.n_experts = n_experts
  
  def forward(self, x):
    batch_size, seq_len, d_model = x.shape
    gating_output, indices = self.router(x)     # Get routing decisions

    # Flatten for easier processing
    flat_x = x.view(-1, d_model)
    flat_gating_output = gating_output.view(-1, self.n_experts)
    flat_indices = indices.view(-1, self.top_k)

    # Calculate expert capacity
    total_tokens = batch_size * seq_len
    tokens_per_expert = (total_tokens * self.top_k) / self.n_experts
    expert_capacity = int(tokens_per_expert * self.capacity_factor)
    final_output = torch.zeros_like(flat_x)     # Initialize output

    # processing each expert
    for expert_idx, expert in enumerate(self.experts):
      # Find tokens assigned to this expert
      expert_mask = (flat_indices == expert_idx).any(dim=-1)
      expert_tokens = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)

      # applyinging capacity limit
      if len(expert_tokens) > expert_capacity:
        expert_tokens = expert_tokens[:expert_capacity]

      if len(expert_tokens) > 0:
        # Get inputs for this expert
        expert_input = flat_x[expert_tokens]

        # processing through expert
        expert_output = expert(expert_input)
        gating_weights = flat_gating_output[expert_tokens, expert_idx:expert_idx+1]         # applyinginging gating weights
        weighted_output = expert_output * gating_weights
        final_output.index_adding_(0, expert_tokens, weighted_output)  # adding to final output
    return final_output.view(batch_size, seq_len, d_model)

class Block(nn.Module):
  def __init__(self, d_model, n_heads, n_experts, top_k, n_ff, dropout, ffn_multiplier, block_size, capacity_factor, rope_theta=10000.0):
    super().__init__()
    self.self_attn = MLA(d_model, n_heads, block_size, rope_theta)  # Self-attention (causal)
    self.cross_attn = MLA(d_model, n_heads, block_size, rope_theta) # Cross-attention (non-causal) - optional, can be used for encoder-decoder
    self.moe = SparseMoE(d_model, n_experts, top_k, n_ff, dropout, ffn_multiplier, capacity_factor)     # Sparse MoE
    # Layer norms ----------
    self.ln1 = RMSNorm(d_model)
    self.ln2 = RMSNorm(d_model)
    self.ln3 = RMSNorm(d_model)
  
  def forward(self, x, encoder_output=None, self_attn_cache=None, cross_attn_cache=None, past_length=0):
    # Self-attention with residual connection
    attn_out, new_self_cache = self.self_attn(
      self.ln1(x), 
      kv_cache=self_attn_cache, 
      past_length=past_length, 
      is_causal=True
    )
    x = x + attn_out

    # Cross-attention (if encoder output is provided)
    if encoder_output is not None:
      cross_out, new_cross_cache = self.cross_attn(
        self.ln2(x), 
        kv_cache=cross_attn_cache, 
        past_length=0, 
        is_causal=False
      )
      x = x + cross_out
    else:
      new_cross_cache = None
    
    # MoE with residual connection
    norm_input = self.ln3(x) if encoder_output is not None else self.ln2(x)
    moe_out = self.moe(norm_input)
    x = x + moe_out

    return x, new_self_cache, new_cross_cache

def kaiming_init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, nn.Parameter):
    if m.dim() >= 2:
      nn.init.kaiming_normal_(m)

class TransformerMoE(nn.Module):
  def __init__(self, params, vocab_size: int):
    super().__init__()
    self.block_size = params.block_size
    self.d_model = params.d_model
    self.n_layers = params.n_layers

    # Token embeddings
    self.token_embeddings = nn.Embedding(vocab_size, self.d_model)
    
    # Transformer blocks
    self.blocks = nn.ModuleList([
      Block(
        d_model=params.d_model,
        n_heads=params.n_heads,
        n_experts=params.n_experts,
        top_k=params.top_k,
        n_ff=params.n_ff,
        dropout=params.dropout,
        ffn_multiplier=params.ffn_multiplier,
        block_size=params.block_size,
        capacity_factor=params.capacity_factor,
        rope_theta=params.rope_theta
      ) for _ in range(self.n_layers)
    ])

    # Final layer norm and output projection
    self.norm_final = RMSNorm(self.d_model, params.norm_eps)
    self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

    # Weight tying (optional)
    self.lm_head.weight = self.token_embeddings.weight

    # Initialize weights
    self.apply(kaiming_init_weights)

    # Scale down output layer initialization
    with torch.no_grad():
      self.lm_head.weight *= 0.1

  def forward(self, idx, targets=None, encoder_output=None, kv_caches=None, past_length=0):
    B, T = idx.size()

    # Token embeddings
    x = self.token_embeddings(idx)

    # Initialize caches if not provided
    if kv_caches is None:
      kv_caches = [(None, None) for _ in range(self.n_layers)]

    new_kv_caches = []

    # Pass through transformer blocks
    for i, block in enumerate(self.blocks):
      self_cache, cross_cache = kv_caches[i]
      x, new_self_cache, new_cross_cache = block(
        x, 
        encoder_output=encoder_output,
        self_attn_cache=self_cache,
        cross_attn_cache=cross_cache,
        past_length=past_length
      )
      new_kv_caches.append((new_self_cache, new_cross_cache))

    # Final normalization and projection
    x = self.norm_final(x)
    logits = self.lm_head(x)

    # Compute loss if targets provided
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    return logits, loss, new_kv_caches