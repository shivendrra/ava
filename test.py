import torch
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

with open('captions.txt', 'r', encoding='utf-8') as file:
  captions = file.read()

print(len(captions)/1e6, 'million words')

chars = sorted(list(set(captions)))
vocab_size = len(chars)

# map of characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(captions), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 8
block_size = 16
max_iters = 100
eval_interval = 10
learning_rate = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
d_embd = 64
n_head = 8
n_layers = 8
dropout = 0.2
norm_eps = 1e-05
# ------------

torch.manual_seed(1400)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self, d_embd, n_head, dropout, block_size):
    head_size = d_embd // n_head
    super().__init__()
    self.key = nn.Linear(d_embd, head_size, bias=True)
    self.query = nn.Linear(d_embd, head_size, bias=True)
    self.value = nn.Linear(d_embd, head_size, bias=True)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    key = self.key(x)   # (B,T,hs)
    query = self.query(x) # (B,T,hs)

    # compute attention scores ("affinities")
    weights = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    weights = F.softmax(weights, dim=-1) # (B, T, T)
    weights = self.dropout(weights)

    # perform the weighted aggregation of the values
    value = self.value(x) # (B,T,hs)
    out = weights @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, d_embd, n_head, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(d_embd=d_embd, n_head=n_head, dropout=dropout, block_size=block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * (d_embd // n_head), d_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(out)

    return out

class FeedForward(nn.Module):
  """ dual linear layer with GELU function """
  def __init__(self, d_embd):
    super().__init__()
    self.fc1 = nn.Linear(d_embd, 4*d_embd) # n_ff = 4*d_embd
    self.fc2 = nn.Linear(4*d_embd, d_embd) # n_ff = 4*d_embd

  def forward(self, x):
    x = F.gelu(self.fc1(x)) # GELU insted of ReLU
    x = self.fc2(x)
    return x

class EncoderDecoderAttention(nn.Module):
  """ separate attention layer for decoder layer """

  def __init__(self, d_embd, n_head, dropout, block_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(d_embd, n_head, dropout, block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_head * (d_embd // n_head), d_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, query, key, value, mask=None):
    x = torch.cat((key, query, value), dim=-1)
    energies = []
    for head in self.heads:
        energy = head(x)
        energies.append(energy.unsqueeze(1))
    energy = torch.cat(energies, dim=1)
    energy = self.proj(energy)
    energy = self.dropout(energy)

    if mask is not None:
      energy = energy.masked_fill(mask == 0, float('-inf'))

    attention = F.softmax(energy, dim=-1)
    output = torch.matmul(attention, value)

    return output

class EncoderLayer(nn.Module):
  """ Encoder Layer """

  def __init__(self, d_embd, n_head, dropout, block_size):
    super().__init__()
    self.s_att = MultiHeadAttention(d_embd=d_embd, n_head=n_head, block_size=block_size, dropout=dropout)
    self.ffwd = FeedForward(d_embd=d_embd)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_embd)
    self.norm2 = nn.LayerNorm(d_embd)

  def forward(self, src, src_mask=None):
    src2 = self.s_att(src)
    src = src + self.dropout(src2)
    src = self.norm1(src)

    src2 = self.ffwd(src)
    src = src + self.dropout(src2)
    src = self.norm2(src)

    return src

class DecoderLayer(nn.Module):
  """ Decoder Layer """

  def __init__(self, d_embd, n_head, dropout, block_size) -> None:
    super().__init__()
    self.s_att = MultiHeadAttention(d_embd=d_embd, n_head=n_head, block_size=block_size, dropout=dropout)
    self.enc_att = EncoderDecoderAttention(d_embd=d_embd, n_head=n_head, block_size=block_size, dropout=dropout)
    self.ffwd = FeedForward(d_embd=d_embd)
    self.dropout = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_embd)
    self.norm2 = nn.LayerNorm(d_embd)
    self.norm3 = nn.LayerNorm(d_embd)

  def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
    trg2 = self.s_att(trg)
    trg = trg2 + self.dropout(trg2)
    trg = self.norm1(trg)

    trg2 = self.enc_att(trg, enc_src, enc_src)
    trg = trg + self.dropout(trg2)
    trg = self.norm2(trg)

    trg2 = self.ffwd(trg)
    trg = trg + self.dropout(trg2)
    trg = self.norm3(trg)

    return trg

class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.d_embd = d_embd
    self.block_size = block_size

    self.token_embd = nn.Embedding(vocab_size, d_embd)
    self.pos_embd = nn.Embedding(block_size, d_embd)
    self.enc_layer = nn.ModuleList([EncoderLayer(n_head=n_head, block_size=block_size, dropout=dropout, d_embd=d_embd) for _ in range(n_layers)])
    self.dec_layer = nn.ModuleList([DecoderLayer(n_head=n_head, block_size=block_size, dropout=dropout, d_embd=d_embd) for _ in range(n_layers)])

    self.norm_final = nn.LayerNorm(d_embd)
    self.lm_head = nn.Linear(d_embd, vocab_size)
    self.fc_out = nn.Linear(d_embd, vocab_size)
    self.dropout = nn.Dropout(dropout)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding) and module.weight.numel() > 0:
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def make_src_mask(self, src):
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

  def make_trg_mask(self, trg):
    trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_embd = self.token_embd(idx)
    pos_embd = self.pos_embd(torch.arange(T, device=device))
    x = tok_embd + pos_embd

    for layer in self.enc_layer:
      x = layer(x, None)

    for layer in self.dec_layer:
      x = layer(x, x)

    x = self.norm_final(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets, ignore_index=-52, reduction='mean')

    return logits, loss

  def generate(self, idx, max_tokens=50):
    for _ in range(max_tokens):
      idx_cond = idx[:, -self.block_size: ]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx, loss


model = Transformer()
# checkpoint_path = '/content/drive/MyDrive/52.9_transformer_model.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint)
m = model.to(device)

# no of parameters
n_param = sum(p.numel() for p in m.parameters())/1e6
print(n_param, 'million')

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
steps = []
train_losses = []
val_losses = []

for iter in range(max_iters):

  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    steps.append(iter)
    train_losses.append(losses['train'])
    val_losses.append(losses['val'])

  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

target_text = "Would you like to tell me your name because "
context = torch.tensor([encode(target_text)], dtype=torch.long, device=device)
generated_output = decode(m.generate(context, max_new_tokens=10)[0].tolist())
print(generated_output)