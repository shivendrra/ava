import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelArgs:
  d_model:int = 1024
  n_layers:int = 12
  n_heads:int = 18
  n_ff_multiple:int = 10
  fnn_multiplier:int = None
  n_ff:int = n_ff_multiple * d_model
  dropout:float = 0.2
  norm_eps:float = 1e-5
  block_size:int = 1024
  n_experts:int = 4
  top_k:int = 2
  device: str = "cuda" if torch.cuda.is_available() else "cpu"

