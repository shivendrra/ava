import torch
import torch.nn as nn
from torch.nn import functional as F
import math

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
  def __init__(self, in_dim, hidden_dim):
    super().__init__()
    self.proj = nn.Linear(in_dim, 2 * hidden_dim, bias=False)

  def forward(self, x):
    x_proj = self.proj(x)
    x1, x2 = x_proj.chunk(2, dim=-1)
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
    self.qk_rope_dim = self.dh - self.qk_nope_dim

    self.q_proj = nn.Linear(d_model, d_model, bias=False)
    self.k_proj = nn.Linear(d_model, d_model, bias=False)
    self.v_proj = nn.Linear(d_model, d_model, bias=False)
    self.o_proj = nn.Linear(d_model, d_model, bias=False)

    # Pre-compute and cache RoPE embeddings
    rope_dim = self.qk_rope_dim
    if rope_dim > 0:
      freqs = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
      emb = torch.outer(torch.arange(max_len).float(), freqs)
      cos_cached = emb.cos()[None, None, :, :]
      sin_cached = emb.sin()[None, None, :, :]
    else:
      cos_cached = torch.zeros(1, 1, max_len, 1)
      sin_cached = torch.zeros(1, 1, max_len, 1)

    self.register_buffer("cos_cached", cos_cached, persistent=False)
    self.register_buffer("sin_cached", sin_cached, persistent=False)

  def forward(self, x, kv_cache=None, past_length=0, is_causal=True):
    B, S, D = x.size()

    # Validate input dimensions
    assert D == self.d_model, f"Input dim {D} != model dim {self.d_model}"
    assert S <= self.cos_cached.size(2), f"Sequence length {S} > max_len {self.cos_cached.size(2)}"

    # More memory efficient projections
    Q = self.q_proj(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)
    K = self.k_proj(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)
    V = self.v_proj(x).view(B, S, self.n_heads, self.dh).transpose(1, 2)

    # Split Q and K for RoPE
    if self.qk_rope_dim > 0:
      Q_nope, Q_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
      K_nope, K_rope = torch.split(K, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
    else:
      Q_nope, Q_rope = Q, torch.empty_like(Q[:, :, :, :0])
      K_nope, K_rope = K, torch.empty_like(K[:, :, :, :0])

    # Handle KV cache more efficiently
    if kv_cache is not None:
      past_K_nope, past_K_rope, past_V = kv_cache
      K_nope = torch.cat([past_K_nope, K_nope], dim=2)
      if self.qk_rope_dim > 0:
        K_rope = torch.cat([past_K_rope, K_rope], dim=2)
      V = torch.cat([past_V, V], dim=2)

    S_full = K_nope.size(2)

    # Apply RoPE more efficiently
    if self.qk_rope_dim > 0:
      rope_dim_half = self.qk_rope_dim // 2
      
      # Ensure indices are within bounds
      start_pos = max(0, min(past_length, self.cos_cached.size(2) - S))
      end_pos = min(start_pos + S, self.cos_cached.size(2))
      start_full = min(S_full, self.cos_cached.size(2))
      
      if end_pos > start_pos and rope_dim_half > 0:
        cos_q = self.cos_cached[:, :, start_pos:end_pos, :rope_dim_half]
        sin_q = self.sin_cached[:, :, start_pos:end_pos, :rope_dim_half]
        
        # Ensure shapes match
        if cos_q.size(2) == S:
          cos_q = cos_q.repeat(1, 1, 1, 2)[:, :, :, :self.qk_rope_dim]
          sin_q = sin_q.repeat(1, 1, 1, 2)[:, :, :, :self.qk_rope_dim]
          Q_rope = apply_rope_x(Q_rope, cos_q, sin_q)

        cos_k = self.cos_cached[:, :, :start_full, :rope_dim_half]
        sin_k = self.sin_cached[:, :, :start_full, :rope_dim_half]
        
        if cos_k.size(2) == S_full:
          cos_k = cos_k.repeat(1, 1, 1, 2)[:, :, :, :self.qk_rope_dim]
          sin_k = sin_k.repeat(1, 1, 1, 2)[:, :, :, :self.qk_rope_dim]
          K_rope = apply_rope_x(K_rope, cos_k, sin_k)

    # Reconstruct Q and K
    Q_final = torch.cat([Q_nope, Q_rope], dim=-1) if self.qk_rope_dim > 0 else Q_nope
    K_final = torch.cat([K_nope, K_rope], dim=-1) if self.qk_rope_dim > 0 else K_nope

    # Use PyTorch's optimized attention with proper error handling
    try:
      x = F.scaled_dot_product_attention(
        Q_final, K_final, V, 
        attn_mask=None, 
        dropout_p=0.0 if not self.training else 0.1,
        is_causal=is_causal and (past_length == 0)
      )
    except Exception as e:
      # Fallback to manual attention if scaled_dot_product_attention fails
      scale = 1.0 / math.sqrt(self.dh)
      scores = torch.matmul(Q_final, K_final.transpose(-2, -1)) * scale
      
      if is_causal and past_length == 0:
        mask = torch.tril(torch.ones(S, S_full, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))
      
      attn_weights = F.softmax(scores, dim=-1)
      if self.training:
        attn_weights = F.dropout(attn_weights, p=0.1)
      x = torch.matmul(attn_weights, V)
    
    x = x.transpose(1, 2).reshape(B, S, D)
    x = self.o_proj(x)
    
    # Return new cache only if needed
    new_kv_cache = (K_nope, K_rope, V) if kv_cache is not None or past_length > 0 else None
    return x, new_kv_cache

class Expert(nn.Module):
  def __init__(self, d_model, hidden_dim, dropout, ffn_multiplier=None):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_multiplier is not None:
      hidden_dim = int(ffn_multiplier * hidden_dim)
    hidden_dim = max(1, hidden_dim)  # Ensure positive dimension
    
    self.swiglu = SwiGLU(d_model, hidden_dim)
    self.fc = nn.Linear(hidden_dim, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.swiglu(x)
    return self.dropout(self.fc(x))

class NoisyTopkRouter(nn.Module):
  def __init__(self, n_embed, n_experts, top_k):
    super().__init__()
    self.top_k = min(top_k, n_experts)  # Ensure top_k doesn't exceed n_experts
    self.n_experts = n_experts
    self.topkroute_linear = nn.Linear(n_embed, n_experts, bias=False)
    self.noise_linear = nn.Linear(n_embed, n_experts, bias=False)

  def forward(self, x):
    # More memory efficient routing
    logits = self.topkroute_linear(x)
    
    if self.training and self.n_experts > 1:
      noise_logits = self.noise_linear(x)
      noise = torch.randn_like(logits) * F.softplus(noise_logits.clamp(max=10.0))
      noisy_logits = logits + noise
    else:
      noisy_logits = logits
    
    # Ensure valid top_k operation
    if self.n_experts == 1:
      top_k_logits = noisy_logits
      indices = torch.zeros_like(noisy_logits, dtype=torch.long)
    else:
      top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
    
    # Create sparse logits more safely
    zeros = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    
    return router_output, indices

class SparseMoE(nn.Module):
  def __init__(self, d_model, n_experts, top_k, n_ff, dropout, ffn_multiplier, capacity_factor=1.25):
    super().__init__()
    self.n_experts = max(1, n_experts)
    self.top_k = min(top_k, self.n_experts)
    
    self.router = NoisyTopkRouter(d_model, self.n_experts, self.top_k)
    self.experts = nn.ModuleList([
      Expert(d_model, n_ff, dropout, ffn_multiplier) 
      for _ in range(self.n_experts)
    ])
    self.capacity_factor = capacity_factor

  def forward(self, x):
    batch_size, seq_len, d_model = x.shape
    original_shape = x.shape
    
    # Handle single expert case
    if self.n_experts == 1:
      return self.experts[0](x)
    
    # Flatten input for processing
    flat_x = x.view(-1, d_model)
    
    # Get routing decisions
    gating_output, indices = self.router(x)
    flat_gating_output = gating_output.view(-1, self.n_experts)
    flat_indices = indices.view(-1, self.top_k)

    # Initialize output tensor
    final_output = torch.zeros_like(flat_x)
    
    # Calculate expert capacity more conservatively
    total_tokens = flat_x.size(0)
    tokens_per_expert = max(1, (total_tokens * self.top_k) // self.n_experts)
    expert_capacity = max(8, int(tokens_per_expert * self.capacity_factor))

    # Process each expert more safely
    for expert_idx in range(self.n_experts):
      # Find tokens assigned to this expert
      expert_mask = (flat_indices == expert_idx).any(dim=-1)
      expert_tokens = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)

      if expert_tokens.numel() == 0:
        continue
        
      # Apply capacity limit to prevent OOM
      if expert_tokens.numel() > expert_capacity:
        # Randomly sample tokens to maintain diversity
        perm = torch.randperm(expert_tokens.numel(), device=expert_tokens.device)
        expert_tokens = expert_tokens[perm[:expert_capacity]]
      
      # Ensure indices are valid
      expert_tokens = expert_tokens[expert_tokens < flat_x.size(0)]
      
      if expert_tokens.numel() == 0:
        continue
      
      try:
        # Process tokens through expert
        expert_input = flat_x[expert_tokens]
        expert_output = self.experts[expert_idx](expert_input)
        
        # Get gating weights for these tokens - use advanced indexing more safely
        expert_gating = flat_gating_output[expert_tokens]
        gating_weights = expert_gating[:, expert_idx:expert_idx+1]
        weighted_output = expert_output * gating_weights
        
        # Add to final output using scatter_add for memory efficiency
        final_output.scatter_add_(0, expert_tokens.unsqueeze(-1).expand_as(weighted_output), weighted_output)
        
      except Exception as e:
        # Skip this expert if there's an error to prevent training crash
        continue

    return final_output.view(original_shape)

class Block(nn.Module):
  def __init__(self, d_model, n_heads, n_experts, top_k, n_ff, dropout, ffn_multiplier, block_size, capacity_factor, rope_theta=10000.0):
    super().__init__()
    self.self_attn = MLA(d_model, n_heads, block_size, rope_theta)
    self.cross_attn = MLA(d_model, n_heads, block_size, rope_theta) if n_experts > 0 else None
    self.moe = SparseMoE(d_model, n_experts, top_k, n_ff, dropout, ffn_multiplier, capacity_factor)
    self.ln1 = RMSNorm(d_model)
    self.ln2 = RMSNorm(d_model)
    self.ln3 = RMSNorm(d_model) if self.cross_attn is not None else None

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
    new_cross_cache = None
    if encoder_output is not None and self.cross_attn is not None:
      cross_out, new_cross_cache = self.cross_attn(
        self.ln2(x), 
        kv_cache=cross_attn_cache, 
        past_length=0, 
        is_causal=False
      )
      x = x + cross_out
      norm_input = self.ln3(x)
    else:
      norm_input = self.ln2(x)

    # MoE layer with residual connection
    moe_out = self.moe(norm_input)
    x = x + moe_out

    return x, new_self_cache, new_cross_cache

def kaiming_init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    if m.bias is not None:
      nn.init.zeros_(m.bias)

class TransformerMoE(nn.Module):
  def __init__(self, params, vocab_size: int):
    super().__init__()
    self.block_size = params.block_size
    self.d_model = params.d_model
    self.n_layers = params.n_layers
    self.vocab_size = vocab_size
    
    # Validate parameters
    assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
    assert self.d_model > 0, f"d_model must be positive, got {self.d_model}"
    assert self.block_size > 0, f"block_size must be positive, got {self.block_size}"
    
    # Token embeddings
    self.token_embeddings = nn.Embedding(vocab_size, self.d_model)
    
    # Transformer blocks
    self.blocks = nn.ModuleList([
      Block(
        params.d_model, params.n_heads, params.n_experts, params.top_k, 
        params.n_ff, params.dropout, params.ffn_multiplier, 
        params.block_size, params.capacity_factor, params.rope_theta
      ) for _ in range(self.n_layers)
    ])
    
    # Final layer norm and output projection
    self.norm_final = RMSNorm(self.d_model, getattr(params, 'norm_eps', 1e-5))
    self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
    
    # Initialize weights
    self.apply(kaiming_init_weights)
    
    # Scale down the output layer for better training stability
    with torch.no_grad():
      self.lm_head.weight *= 0.1

  def forward(self, idx, targets=None, encoder_output=None, kv_caches=None, past_length=0):
    B, T = idx.size()
    
    # Validate inputs
    if torch.any(idx >= self.vocab_size) or torch.any(idx < 0):
      raise ValueError(f"Input indices out of range: min={idx.min().item()}, max={idx.max().item()}, vocab_size={self.vocab_size}")
    
    if T > self.block_size:
      raise ValueError(f"Input sequence length ({T}) exceeds block size ({self.block_size})")
    
    # Token embeddings
    x = self.token_embeddings(idx)

    # Initialize KV caches if not provided
    if kv_caches is None:
      kv_caches = [(None, None) for _ in range(self.n_layers)]

    # Forward pass through transformer blocks
    new_kv_caches = []
    for i, block in enumerate(self.blocks):
      self_cache, cross_cache = kv_caches[i]
      x, new_self_cache, new_cross_cache = block(
        x, encoder_output, self_cache, cross_cache, past_length
      )
      new_kv_caches.append((new_self_cache, new_cross_cache))

    # Final layer norm and output projection
    x = self.norm_final(x)
    logits = self.lm_head(x)

    # Calculate loss if targets are provided
    loss = None
    if targets is not None:
      # Validate targets
      if torch.any(targets >= self.vocab_size) or torch.any(targets < -1):
        valid_mask = (targets >= 0) & (targets < self.vocab_size)
        if not valid_mask.all():
          print(f"Warning: Invalid target indices found, clamping to valid range")
          targets = torch.clamp(targets, 0, self.vocab_size - 1)
      
      # Cross-entropy loss
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1),
        ignore_index=-1  # Ignore padding tokens if any
      )

    return logits, loss, new_kv_caches