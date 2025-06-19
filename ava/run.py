import json, time, math
import pickle
from contextlib import nullcontext
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import tiktoken

from .model import TransformerMoE

def load_config(config_path: str, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  with open(config_path, 'r') as f:
    configs = json.load(f)
  
  if isinstance(configs, list):
    configs = configs[0]  # taking first config if it's a list
    # [0] => 500M
    # [1] => 750M
    # [2] => 1B

  if model_name not in configs:
    raise ValueError(f"Model '{model_name}' not found in config. Available: {list(configs.keys())}")
  
  model_config = configs[model_name]['ModelConfig']
  train_config = configs[model_name]['TrainConfig']
  
  return model_config, train_config

def prepare_dataset(data_path: str, train_split: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
  """Load and prepare dataset for training"""
  print("Loading dataset...")

  with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()  
  print(f"Dataset loaded: {len(text):,} characters")

  enc = tiktoken.get_encoding("cl100k_base")  # AVA-4 tokenizer  
  print("Tokenizing dataset...")
  tokens = enc.encode(text)
  tokens = torch.tensor(tokens, dtype=torch.long)
  print(f"Tokenized: {len(tokens):,} tokens")
  
  # train/validation split
  n = int(train_split * len(tokens))
  train_data = tokens[:n]
  val_data = tokens[n:]

  print(f"Train tokens: {len(train_data):,}")
  print(f"Validation tokens: {len(val_data):,}")

  return train_data, val_data

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
  """Generate a batch of data for training"""
  # Ensure we don't go out of bounds
  max_start_idx = len(data) - block_size - 1
  if max_start_idx <= 0:
    raise ValueError(f"Dataset too small. Need at least {block_size + 1} tokens, got {len(data)}")
  
  ix = torch.randint(0, max_start_idx, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, eval_iters: int, batch_size: int, block_size: int, device: str) -> Dict[str, float]:
  """Estimate loss on train and validation sets"""
  model.eval()
  losses = {}
  
  for split, data in [('train', train_data), ('val', val_data)]:
    if len(data) <= block_size:
      print(f"Warning: {split} data too small, skipping evaluation")
      losses[split] = float('inf')
      losses[f'{split}_acc'] = 0.0
      continue
      
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    for _ in range(eval_iters):
      try:
        X, Y = get_batch(data, batch_size, block_size, device)
        logits, loss, _ = model(X, Y)
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == Y).sum().item()
        total_correct += correct
        total_tokens += Y.numel()
      except Exception as e:
        print(f"Error in evaluation: {e}")
        continue
    
    losses[split] = total_loss / eval_iters if total_loss > 0 else float('inf')
    losses[f'{split}_acc'] = total_correct / total_tokens if total_tokens > 0 else 0.0
  
  model.train()
  return losses

def get_lr(step: int, warmup_steps: int, lr_decay_steps: int, max_lr: float, min_lr: float) -> float:
  """Get learning rate with warmup and cosine decay"""
  # Linear warmup
  if step < warmup_steps:
    return max_lr * step / warmup_steps

  # Cosine decay
  if step > lr_decay_steps:
    return min_lr
  
  decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, betas: Tuple[float, float]) -> torch.optim.Optimizer:
  """Configure optimizer with weight decay for specific parameters"""
  # Separate parameters that should and shouldn't be weight decayed
  decay = set()
  no_decay = set()

  whitelist_weight_modules = (torch.nn.Linear, )
  blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
  
  for mn, m in model.named_modules():
    for pn, p in m.named_parameters():
      fpn = '%s.%s' % (mn, pn) if mn else pn
      
      if pn.endswith('bias'):
        no_decay.add(fpn)
      elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        decay.add(fpn)
      elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        no_decay.add(fpn)
  
  # creating parameter groups
  param_dict = {pn: p for pn, p in model.named_parameters()}
  optim_groups = [
    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
  ]

  optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
  return optimizer

def count_parameters(model: nn.Module) -> int:
  """Count total number of parameters in the model"""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model: nn.Module, model_config: Dict[str, Any]) -> None:
  """Print detailed model information"""
  print("\n" + "="*60)
  print("MODEL INFORMATION")
  print("="*60)

  total_params = count_parameters(model)
  print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
  
  print(f"Architecture:")
  print(f"  - Layers: {model_config['n_layers']}")
  print(f"  - Heads: {model_config['n_heads']}")
  print(f"  - Model Dimension: {model_config['d_model']}")
  print(f"  - Head Dimension: {model_config['d_model'] // model_config['n_heads']}")
  print(f"  - FFN Multiplier: {model_config['ffn_multiplier']}")
  print(f"  - Max Sequence Length: {model_config['max_seq_len']}")
  print(f"  - Vocabulary Size: {model_config['vocab_size']}")
  print(f"  - Dropout: {model_config['dropout']}")
  
  # memory estimation
  param_size = total_params * 4 / (1024**3)  # 4 bytes per parameter
  print(f"  - Estimated Model Size: {param_size:.2f} GB")
  print("="*60)

def main():
  """Main training function"""
  # configuration
  config_path = "config.json"
  model_name = "AVA_500M"  # change this to "AVA_750M" or "AVA_1B" as needed
  
  # google-drive dataset URL (replace with your actual URL)
  dataset_url = "https://drive.google.com/"
  dataset_path = "/teamspace/uploads/dataset.txt"

  # load configurations
  print("Loading configurations...")
  model_config, train_config = load_config(config_path, model_name)

  # set device
  device = train_config['device']
  if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
    print("CUDA not available, using CPU")
  
  print(f"Using device: {device}")

  # setting random seeds for reproducibility
  torch.manual_seed(1337)
  if device == 'cuda':
    torch.cuda.manual_seed(1337)

  # downloading and preparing dataset
  train_data, val_data = prepare_dataset(dataset_path)
  print("Initializing model...")

  # create a params object with the configuration
  class ModelConfig:
    def __init__(self, config_dict):
      for key, value in config_dict.items():
        setattr(self, key, value)
  
  params = ModelConfig(model_config)
  model = TransformerMoE(params=params, vocab_size=model_config['vocab_size'])
  model.to(device)
  
  # print model information
  print_model_info(model, model_config)

  # configure optimizer
  optimizer = configure_optimizers(
    model, 
    weight_decay=train_config['weight_decay'],
    learning_rate=train_config['learning_rate'],
    betas=(train_config['beta1'], train_config['beta2'])
  )

  # compiling model for faster training (PyTorch 2.0+)
  # Disable compilation initially to debug the error
  compile_model = train_config.get('compile', False)
  if compile_model:
    print("Model compilation disabled for debugging. Enable after fixing issues.")
    # print("Compiling model...")
    # try:
    #   model = torch.compile(model)
    #   print("Model compiled successfully!")
    # except:
    #   print("Model compilation failed, continuing without compilation")
  
  # Training setup
  scaler = torch.amp.GradScaler(enabled=(device == 'cuda' and train_config.get('dtype') == 'bfloat16'))
  
  # Fix autocast context
  if device == 'cuda':
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
  else:
    ctx = nullcontext()
  
  # Training parameters
  max_iters = train_config.get('max_iters', 600000)
  batch_size = train_config['batch_size']
  block_size = train_config['block_size']
  gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
  grad_clip = train_config.get('grad_clip', 1.0)
  eval_interval = train_config['eval_interval']
  eval_iters = train_config['eval_iters']
  log_interval = train_config.get('log_interval', 1)
  
  # Learning rate schedule parameters
  warmup_steps = train_config['warmup_steps']
  lr_decay_steps = train_config['lr_decay_steps']
  learning_rate = train_config['learning_rate']
  min_lr = train_config['min_lr']
  
  print(f"\nStarting training for {max_iters:,} iterations...")
  print(f"Batch size: {batch_size}, Block size: {block_size}")
  print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
  print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
  
  # Training loop
  model.train()
  step = 0
  start_time = time.time()
  
  # Initial evaluation
  try:
    losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
    print(f"\nStep {step:6d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | " f"Train Acc: {losses['train_acc']:.3f} | Val Acc: {losses['val_acc']:.3f}")
  except Exception as e:
    print(f"Initial evaluation failed: {e}")
    print("Continuing with training...")
  
  while step < max_iters:
    try:
      lr = get_lr(step, warmup_steps, lr_decay_steps, learning_rate, min_lr)
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      
      # Evaluate and log
      if step % eval_interval == 0 and step > 0:
        try:
          losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
          elapsed_time = time.time() - start_time
          avg_time_per_step = elapsed_time / step if step > 0 else 0
          
          print(f"Step {step:6d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | " f"Train Acc: {losses['train_acc']:.3f} | Val Acc: {losses['val_acc']:.3f} | " f"LR: {lr:.2e} | Time/Step: {avg_time_per_step:.2f}s")
        except Exception as e:
          print(f"Evaluation failed at step {step}: {e}")

      # Training step
      optimizer.zero_grad(set_to_none=True)
      loss_accum = 0.0
      
      for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch(train_data, batch_size, block_size, device)
        
        with ctx:
          logits, loss, _ = model(X, Y)
          loss = loss / gradient_accumulation_steps
          loss_accum += loss.detach()
        
        scaler.scale(loss).backward()
      
      # Gradient clipping
      if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      
      # Optimizer step
      scaler.step(optimizer)
      scaler.update()
      
      step += 1
      
      # Log training progress
      if step % log_interval == 0:
        elapsed_time = time.time() - start_time
        avg_time_per_step = elapsed_time / step
        print(f"Step {step:6d} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Time/Step: {avg_time_per_step:.2f}s")
        
    except Exception as e:
      print(f"Training error at step {step}: {e}")
      print("Stopping training...")
      break
  
  print(f"\nTraining completed! Total time: {(time.time() - start_time) / 3600:.2f} hours")
  
  # Save final model
  checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_config': model_config,
    'train_config': train_config,
    'step': step,
  }
  
  torch.save(checkpoint, f'{model_name}_final.pt')
  print(f"Model saved as {model_name}_final.pt")
  
  # Final evaluation
  try:
    losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
    print(f"\nFinal Results:")
    print(f"Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
    print(f"Train Accuracy: {losses['train_acc']:.3f} | Val Accuracy: {losses['val_acc']:.3f}")
  except Exception as e:
    print(f"Final evaluation failed: {e}")

if __name__ == "__main__":
  main()