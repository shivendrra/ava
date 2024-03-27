"""
  use this file to train the model

  working:
    - imports vatious dependencies first, and then loads the training data
    - tokenizes it, per-character basis
    - loads the required hyper-parameters and the model file
    - trains it till 'max_iters' and saves the model state, and generates outputs
  
  with the current set configuration, model can reach upto ~60million parameters
  and can become ~99% accurate with next token prediction
"""

import torch
import json
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('../datasets/wiki_176m.txt', 'r', encoding='utf-8') as file:
  data = file.read()

print(f"{(len(data)/1e6):.2f} million letters")

from tokenizer import Tokenizer

tokenizer = Tokenizer()
vocab_size = tokenizer.get_vocab()

# Train and test splits
data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

with open('config.json', 'r', encoding='utf-8') as file:
  params = json.load(file)

# required parameters
batch_size = params['batch_size']
block_size = params['block_size']
max_iters = 1000
eval_interval = 100
eval_iters = 200
learning_rate = params['learning_rate']

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

from model import Transformer
model = Transformer(vocab_size)
m = model.to(device)

# no of parameters
n_param = sum(p.numel() for p in m.parameters())/1e6
print(f"{n_param:.2f} million")
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

torch.save(model.state_dict(), f'enigma_{n_param:.0f}m.pth')