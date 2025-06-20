# MoE Transformer Documentation

## Overview

This codebase implements a **Mixture of Experts (MoE) Transformer** with several advanced architectural features including Multi-Head Latent Attention (MLA), RoPE positional embeddings, and sparse expert routing. The implementation is designed for efficient training and inference of large language models.

## Key Features

- **Sparse Mixture of Experts (MoE)**: Conditional computation using expert routing
- **Multi-Head Latent Attention (MLA)**: Advanced attention mechanism with RoPE
- **RMSNorm**: Root Mean Square Layer Normalization for improved stability
- **SwiGLU Activation**: Swish-Gated Linear Units for better performance
- **Rotary Position Embeddings (RoPE)**: Relative position encoding
- **KV Caching**: Efficient inference with key-value caching
- **Memory Optimizations**: Various techniques to reduce memory usage

## Architecture Components

### 1. RMSNorm
```python
class RMSNorm(nn.Module)
```
Root Mean Square Layer Normalization - a more efficient alternative to LayerNorm.

**Parameters:**
- `dim`: Input dimension
- `eps`: Small constant for numerical stability (default: 1e-5)

**Key Features:**
- No bias term (unlike LayerNorm)
- Better numerical stability
- Faster computation

### 2. SwiGLU
```python
class SwiGLU(nn.Module)
```
Swish-Gated Linear Unit activation function used in the expert networks.

**Parameters:**
- `in_dim`: Input dimension
- `hidden_dim`: Hidden dimension

**Formula:** `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)`

### 3. Multi-Head Latent Attention (MLA)
```python
class MLA(nn.Module)
```
Advanced attention mechanism with RoPE positional embeddings and efficient KV caching.

**Parameters:**
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `max_len`: Maximum sequence length (default: 1024)
- `rope_theta`: RoPE frequency base (default: 10000.0)

**Key Features:**
- **RoPE Integration**: Rotary positional embeddings for better position encoding
- **Efficient KV Caching**: Supports incremental decoding
- **Memory Optimization**: Uses PyTorch's `scaled_dot_product_attention` when available
- **Fallback Implementation**: Manual attention computation if optimized version fails

### 4. Expert Networks
```python
class Expert(nn.Module)
```
Individual expert in the MoE layer.

**Parameters:**
- `d_model`: Model dimension
- `hidden_dim`: Hidden dimension of the expert
- `dropout`: Dropout rate
- `ffn_multiplier`: Optional multiplier for hidden dimension

**Architecture:**
- SwiGLU activation
- Linear projection back to model dimension
- Dropout for regularization

### 5. Noisy Top-k Router
```python
class NoisyTopkRouter(nn.Module)
```
Routes tokens to the top-k experts with added noise during training.

**Parameters:**
- `n_embed`: Input embedding dimension
- `n_experts`: Total number of experts
- `top_k`: Number of experts to route each token to

**Features:**
- **Training Noise**: Adds learnable noise during training for better load balancing
- **Top-k Selection**: Routes each token to the k best experts
- **Sparse Routing**: Creates sparse expert assignments

### 6. Sparse MoE Layer
```python
class SparseMoE(nn.Module)
```
The main MoE layer that combines routing and expert computation.

**Parameters:**
- `d_model`: Model dimension
- `n_experts`: Number of experts
- `top_k`: Top-k routing
- `n_ff`: Expert hidden dimension
- `dropout`: Dropout rate
- `ffn_multiplier`: FFN dimension multiplier
- `capacity_factor`: Expert capacity multiplier (default: 1.25)

**Key Features:**
- **Capacity Management**: Prevents expert overload with capacity limits
- **Load Balancing**: Random sampling when experts exceed capacity
- **Error Handling**: Graceful fallback for expert computation errors
- **Memory Efficiency**: Optimized token processing and scatter operations

### 7. Transformer Block
```python
class Block(nn.Module)
```
A single transformer layer combining self-attention, optional cross-attention, and MoE.

**Parameters:**
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `n_experts`: Number of MoE experts
- `top_k`: Top-k expert routing
- `n_ff`: Expert hidden dimension
- `dropout`: Dropout rate
- `ffn_multiplier`: FFN multiplier
- `block_size`: Maximum sequence length
- `capacity_factor`: Expert capacity factor
- `rope_theta`: RoPE theta parameter

**Architecture:**
1. **Self-Attention** with RMSNorm
2. **Cross-Attention** (optional, for encoder-decoder)
3. **MoE Layer** with RMSNorm
4. **Residual Connections** throughout

### 8. Main Model
```python
class TransformerMoE(nn.Module)
```
The complete MoE Transformer model.

**Parameters via config:**
- Model architecture parameters
- Training hyperparameters
- Expert configuration

**Features:**
- **Token Embeddings**: Learnable token representations
- **Stacked Blocks**: Multiple transformer layers
- **KV Caching**: Efficient inference support
- **Loss Computation**: Built-in cross-entropy loss
- **Input Validation**: Comprehensive error checking

## Configuration

The model supports three pre-configured sizes via `config.json`:

### AVA_500M
- **Parameters**: ~500M
- **d_model**: 1024
- **n_layers**: 24
- **n_experts**: 4
- **top_k**: 2

### AVA_750M
- **Parameters**: ~750M
- **d_model**: 1280
- **n_layers**: 24
- **n_experts**: 6
- **top_k**: 2

### AVA_1B
- **Parameters**: ~1B
- **d_model**: 1536
- **n_layers**: 24
- **n_experts**: 8
- **top_k**: 2

## Usage Example

```python
import torch
from moe import TransformerMoE
import json

# Load configuration
with open('config.json', 'r') as f:
    configs = json.load(f)[0]  # Load first config dict

# Select model size
model_config = configs['AVA_500M']['ModelConfig']

# Create model parameter object
class ModelParams:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

params = ModelParams(model_config)

# Initialize model
model = TransformerMoE(params, vocab_size=params.vocab_size)

# Example forward pass
batch_size, seq_len = 2, 512
input_ids = torch.randint(0, params.vocab_size, (batch_size, seq_len))

# Training mode
model.train()
logits, loss, kv_caches = model(input_ids, targets=input_ids)

# Inference mode with KV caching
model.eval()
with torch.no_grad():
    logits, _, new_caches = model(input_ids, kv_caches=None)
    
    # Continue generation with caching
    next_token = torch.randint(0, params.vocab_size, (batch_size, 1))
    next_logits, _, updated_caches = model(
        next_token, 
        kv_caches=new_caches, 
        past_length=seq_len
    )
```

## Key Optimizations

### Memory Efficiency
1. **Sparse Expert Routing**: Only active experts process tokens
2. **Capacity Management**: Prevents memory spikes from expert overload
3. **KV Caching**: Reduces recomputation during inference
4. **Efficient Attention**: Uses PyTorch's optimized attention when available

### Training Stability
1. **RMSNorm**: More stable than LayerNorm
2. **Gradient Clipping**: Prevents gradient explosion
3. **Weight Initialization**: Kaiming initialization with output scaling
4. **Error Handling**: Graceful fallback mechanisms

### Performance Features
1. **Torch Compilation**: Support for `torch.compile`
2. **Mixed Precision**: bfloat16 training support
3. **Gradient Accumulation**: Effective larger batch sizes
4. **Optimized Routing**: Efficient sparse operations

## Training Configuration

Each model size includes optimized training hyperparameters:

- **Learning Rate Scheduling**: Warmup + cosine decay
- **Optimizer**: AdamW with weight decay
- **Gradient Accumulation**: For effective larger batches
- **Mixed Precision**: bfloat16 for memory efficiency
- **Evaluation**: Regular validation during training

## Advanced Features

### RoPE (Rotary Position Embeddings)
- Relative position encoding that works well with longer sequences
- Cached precomputed values for efficiency
- Supports both training and inference modes

### Expert Load Balancing
- Noisy routing during training for better load distribution
- Capacity limits prevent any single expert from being overwhelmed
- Random sampling maintains training diversity

### KV Caching System
- Efficient incremental decoding
- Supports both self-attention and cross-attention caching
- Memory-efficient cache management

## Error Handling & Robustness

The implementation includes extensive error handling:
- Input validation for tensor shapes and ranges
- Graceful fallback for attention computation
- Expert computation error recovery
- Comprehensive parameter validation

## Dependencies

- PyTorch 2.0+
- torch.nn.functional
- Standard library (math, json)

## Notes

- The model uses `bias=False` throughout for efficiency
- RoPE embeddings are precomputed and cached
- Expert capacity is dynamically managed to prevent OOM errors
- The implementation prioritizes memory efficiency and training stability