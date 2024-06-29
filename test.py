import torch
import tiktoken
import torch.nn as nn
import math
import numpy as np
tokenizer = tiktoken.get_encoding("p50k_base")
tokenizer = tiktoken.encoding_for_model("text-davinci-003")

input_text = "Hello, my name is shivendra and I'm from kanpur"
encoded = tokenizer.encode(input_text)
print(encoded)

encoded = torch.tensor(encoded, dtype=torch.long)
x = torch.stack([encoded for i in range(5)])
print(x)

B, T  = x.shape
z = x.view(B*T)
print(z)

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

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

length = 100
channels = 100
positional_embeddings = sinusoids(length, channels).numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.imshow(positional_embeddings, aspect='auto', cmap='viridis')
plt.colorbar()
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Positional Embeddings')
plt.show()