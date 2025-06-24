from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import torch, math
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json, time, logging, gc
from pathlib import Path
import numpy as np
import tiktoken

from .text.moe import TransformerMoE
from .dataset import Dataset

class EarlyStopping:
  """Early stopping utility"""
  def __init__(self, patience: int = 7, min_delta: float = 0.001):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_loss = float('inf')

  def __call__(self, val_loss: float) -> bool:
    if val_loss < self.best_loss - self.min_delta:
      self.best_loss = val_loss
      self.counter = 0
      return False
    else:
      self.counter += 1
      return self.counter >= self.patience

class Config:
  """Configuration class for model parameters"""
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

class MultiTrainer:
  def __init__(self, config_path: str, model_name: str, dataset_path: List[str], dataset_names: List[str], save_dir: str, log_dir: str, encoding: str = "gpt2", device: str = 'cuda'):
    self.config_path = Path(config_path)
    self.dataset_path = dataset_path
    self.save_dir = Path(save_dir)
    self.log_dir = Path(log_dir)
    self.encoding = encoding
    self.dataset_names = dataset_names
    self.device = device

    # Create directories
    self.save_dir.mkdir(parents=True, exist_ok=True)
    self.log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    self.setup_logging()
    
    # Load datasets
    self.datasets = {}
    self.current_dataset_idx = 0
    self.load_datasets()
    
    # Load configurations
    self.model_config, self.train_config = self.load_config(model_name)
    
    # Initialize model
    vocab_size = tiktoken.get_encoding(encoding).n_vocab
    self.config = Config(**self.model_config)
    self.model = TransformerMoE(self.config, vocab_size).to(device)
    self.n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
    self.logger.info(f"Model initialized with {self.n_params:.2f}M parameters, vocab size: {vocab_size}")

    # Training state
    self.optimizer = None
    self.scheduler = None
    self.global_step = 0
    self.epoch = 0
    self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    self.training_history = []
    self.writer = SummaryWriter(self.log_dir)

  def setup_logging(self):
    log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    self.logger = logging.getLogger(__name__)

  def load_config(self, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(self.config_path, 'r') as f:
      configs = json.load(f)
    
    if isinstance(configs, list):
      configs = configs[0]

    if model_name not in configs:
      raise ValueError(f"Model '{model_name}' not found in config. Available: {list(configs.keys())}")
    
    model_config = configs[model_name]['ModelConfig']
    train_config = configs[model_name]['TrainConfig']
    
    return model_config, train_config

  def load_datasets(self):
    self.logger.info("Loading datasets...")
    for path, name in zip(self.dataset_path, self.dataset_names):
      try:
        dataset = Dataset(path, self.encoding, 0.25, 1600)
        self.datasets[name] = dataset
        stats = dataset.get_data_stats()
        self.logger.info(f"Dataset '{name}' loaded: {stats}")
      except Exception as e:
        self.logger.error(f"Failed to load dataset '{name}': {e}")
        continue

    if not self.datasets:
      raise ValueError("No datasets loaded!")

  def get_current_dataset(self) -> Dataset:
    return self.datasets[self.dataset_names[self.current_dataset_idx]]

  def switch_dataset(self):
    self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.dataset_names)
    current_name = self.dataset_names[self.current_dataset_idx]
    self.logger.info(f"Switched to dataset: {current_name}")

  def setup_training(self):
    learning_rate = self.train_config.get('learning_rate', 6e-4)
    weight_decay = self.train_config.get('weight_decay', 0.1)
    beta1 = self.train_config.get('beta1', 0.9)
    beta2 = self.train_config.get('beta2', 0.95)
    
    self.optimizer = optim.AdamW(
      self.model.parameters(), 
      lr=learning_rate,
      weight_decay=weight_decay, 
      betas=(beta1, beta2)
    )

    # Learning rate scheduler with warmup
    warmup_steps = self.train_config.get('warmup_steps', 2000)
    lr_decay_steps = self.train_config.get('lr_decay_steps', 600000)
    min_lr = self.train_config.get('min_lr', 6e-5)
    
    def lr_lambda(step):
      if step < warmup_steps:
        return step / warmup_steps
      # Cosine decay after warmup
      progress = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
      return max(min_lr / learning_rate, 0.5 * (1 + np.cos(np.pi * progress)))

    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    self.logger.info("Training setup completed")

  def train_step(self, batch_size: int, block_size: int) -> Dict[str, float]:
    self.model.train()
    
    # Get batch from current dataset
    dataset = self.get_current_dataset()
    
    try:
      x, y = dataset.get_batch("train", batch_size, block_size, self.device)
    except Exception as e:
      self.logger.error(f"Failed to get training batch: {e}")
      return {'loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf'), 'grad_norm': 0.0, 'lr': 0.0}

    # Validate batch shapes
    if x.dim() != 2 or y.dim() != 2 or x.shape != y.shape:
      self.logger.error(f"Invalid batch shapes: x={x.shape}, y={y.shape}")
      return {'loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf'), 'grad_norm': 0.0, 'lr': 0.0}

    # Forward pass
    self.optimizer.zero_grad()
    
    try:
      logits, loss, _ = self.model(x, targets=y)
      
      if torch.isnan(loss) or torch.isinf(loss):
        self.logger.warning("NaN or Inf loss detected")
        return {'loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf'), 'grad_norm': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}
        
    except Exception as e:
      self.logger.error(f"Forward pass failed: {e}")
      return {'loss': float('inf'), 'accuracy': 0.0, 'perplexity': float('inf'), 'grad_norm': 0.0, 'lr': 0.0}

    # Backward pass
    try:
      loss.backward()
      grad_clip = self.train_config.get('grad_clip', 1.0)
      grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
      self.optimizer.step()
      self.scheduler.step()
    except Exception as e:
      self.logger.error(f"Backward pass failed: {e}")
      return {'loss': loss.item(), 'accuracy': 0.0, 'perplexity': float('inf'), 'grad_norm': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}

    # Compute metrics
    try:
      with torch.no_grad():
        pred_tokens = torch.argmax(logits, dim=-1)
        accuracy = (pred_tokens == y).float().mean()
        perplexity = torch.exp(torch.clamp(loss, max=10.0))

      return {
        'loss': loss.item(),
        'accuracy': accuracy.item(), 
        'perplexity': perplexity.item(),
        'grad_norm': grad_norm.item(),
        'lr': self.optimizer.param_groups[0]['lr']
      }
    except Exception as e:
      self.logger.error(f"Metrics computation failed: {e}")
      return {'loss': loss.item(), 'accuracy': 0.0, 'perplexity': float('inf'), 'grad_norm': 0.0, 'lr': self.optimizer.param_groups[0]['lr']}

  # Simplified validate method in run.py  
  def validate(self, batch_size: int, block_size: int, num_batches: int = 5) -> Dict[str, float]:
    self.model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    
    with torch.no_grad():
      for dataset_name, dataset in self.datasets.items():
        for _ in range(2):  # Just 2 batches per dataset
          try:
            x, y = dataset.get_batch("val", min(batch_size//2, 4), block_size, self.device)
            
            if x.dim() != 2 or y.dim() != 2 or x.shape != y.shape:
              continue
              
            logits, loss, _ = self.model(x, targets=y)
            
            if torch.isnan(loss) or torch.isinf(loss):
              continue

            pred_tokens = torch.argmax(logits, dim=-1)
            accuracy = (pred_tokens == y).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_batches += 1
            
          except Exception as e:
            self.logger.warning(f"Validation batch failed: {e}")
            continue

    if total_batches > 0:
      avg_loss = total_loss / total_batches
      avg_accuracy = total_accuracy / total_batches
      perplexity = math.exp(min(avg_loss, 10.0))
    else:
      avg_loss = float('inf')
      avg_accuracy = 0.0
      perplexity = float('inf')

    return {
      'loss': avg_loss,
      'accuracy': avg_accuracy, 
      'perplexity': perplexity
    }

  def save_checkpoint(self, is_best: bool = False):
    checkpoint = {
      'epoch': self.epoch,
      'global_step': self.global_step,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scheduler_state_dict': self.scheduler.state_dict(),
      'model_config': self.model_config,
      'train_config': self.train_config,
      'training_history': self.training_history,
      'current_dataset_idx': self.current_dataset_idx
    }

    filename = "best_model.pth" if is_best else f"checkpoint_epoch_{self.epoch:03d}.pth"
    filepath = self.save_dir / filename

    try:
      torch.save(checkpoint, filepath)
      self.logger.info(f"Checkpoint saved: {filepath}")
      return str(filepath)
    except Exception as e:
      self.logger.error(f"Failed to save checkpoint: {e}")
      return None

  def load_checkpoint(self, checkpoint_path: str) -> bool:
    try:
      checkpoint = torch.load(checkpoint_path, map_location=self.device)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      
      if self.optimizer:
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if self.scheduler:
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
      self.epoch = checkpoint.get('epoch', 0)
      self.global_step = checkpoint.get('global_step', 0)
      self.training_history = checkpoint.get('training_history', [])
      self.current_dataset_idx = checkpoint.get('current_dataset_idx', 0)
      
      self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
      return True
    except Exception as e:
      self.logger.error(f"Failed to load checkpoint: {e}")
      return False

  def train(self, num_epochs: int = None, resume_from: Optional[str] = None):
    # Use config values if not provided
    if num_epochs is None:
      num_epochs = self.train_config.get('epochs', 1)
    
    batch_size = self.train_config.get('batch_size', 12)
    block_size = self.train_config.get('block_size', 768)
    eval_interval = self.train_config.get('eval_interval', 200)  # Reduced for more frequent validation
    log_interval = self.train_config.get('log_interval', 1)
    
    # Setup training
    self.setup_training()
    
    # Resume from checkpoint if provided
    if resume_from:
      self.load_checkpoint(resume_from)

    self.logger.info("Starting training...")
    self.logger.info(f"Parameters: epochs={num_epochs}, batch_size={batch_size}, block_size={block_size}")

    try:
      start_time = time.time()
      best_val_loss = float('inf')

      for epoch in range(self.epoch, num_epochs):
        self.epoch = epoch
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Calculate steps per epoch
        steps_per_epoch = 1000
        
        for step in range(steps_per_epoch):
          # Single training step
          metrics = self.train_step(batch_size, block_size)
          epoch_losses.append(metrics)
          self.global_step += 1
          
          # Log to tensorboard
          for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)
          
          # Print progress
          if self.global_step % log_interval == 0:
            current_dataset = self.dataset_names[self.current_dataset_idx]
            self.logger.info(f"Step {self.global_step}: Dataset={current_dataset}, "
                            f"Loss={metrics['loss']:.4f}, "
                            f"Acc={metrics['accuracy']:.3f}, "
                            f"PPL={metrics['perplexity']:.2f}, "
                            f"LR={metrics['lr']:.2e}")

          # Validation
          if self.global_step % eval_interval == 0:
            val_metrics = self.validate(batch_size // 2, block_size)  # Smaller batch for validation

            # Log validation metrics
            for key, value in val_metrics.items():
              self.writer.add_scalar(f'val/{key}', value, self.global_step)

            self.logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                            f"Acc: {val_metrics['accuracy']:.3f}, "
                            f"PPL: {val_metrics['perplexity']:.2f}")
            
            # Check for improvement
            if val_metrics['loss'] < best_val_loss:
              best_val_loss = val_metrics['loss']
              self.save_checkpoint(is_best=True)
              self.logger.info("New best model saved!")

            # Early stopping check
            if self.early_stopping(val_metrics['loss']):
              self.logger.info("Early stopping triggered")
              return
          
          # Dataset rotation every 500 steps
          if self.global_step % 500 == 0 and len(self.datasets) > 1:
            self.switch_dataset()

          # Memory cleanup
          if self.global_step % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
              torch.cuda.empty_cache()

        # End of epoch
        avg_epoch_loss = np.mean([m['loss'] for m in epoch_losses])
        avg_accuracy = np.mean([m['accuracy'] for m in epoch_losses])
        avg_perplexity = np.mean([m['perplexity'] for m in epoch_losses])
        epoch_time = time.time() - epoch_start_time

        self.training_history.append({
          'epoch': epoch,
          'avg_loss': avg_epoch_loss,
          'avg_accuracy': avg_accuracy,
          'avg_perplexity': avg_perplexity,
          'epoch_time': epoch_time,
          'global_step': self.global_step
        })

        self.logger.info(
          f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s, "
          f"Avg Loss: {avg_epoch_loss:.4f}, "
          f"Avg Acc: {avg_accuracy:.3f}, "
          f"Avg PPL: {avg_perplexity:.2f}"
        )

        # Save checkpoint every epoch
        self.save_checkpoint()

    except KeyboardInterrupt:
      self.logger.info("Training interrupted by user")
    except Exception as e:
      self.logger.error(f"Training failed: {e}")
      raise
    finally:
      # Final save and cleanup
      self.save_checkpoint()
      total_time = time.time() - start_time
      
      self.logger.info(f"Training completed in {total_time:.2f}s")
      self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
      
      self.writer.close()
      
      # Save training summary
      summary = {
        'total_time': total_time,
        'best_val_loss': best_val_loss,
        'total_steps': self.global_step,
        'final_epoch': self.epoch,
        'model_params': f"{self.n_params:.2f}M",
        'training_history': self.training_history
      }
      
      with open(self.save_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def main():
  """Example training setup"""
  
  dataset_paths = [
    "/teamspace/uploads/consolidated-2.5b.txt",
    "/teamspace/uploads/consolidated_10.txt"
  ]
  dataset_names = ["consolidate_2.5b", "consolidate_10"]
  
  # Initialize trainer
  trainer = MultiTrainer(
    config_path="config.json",
    model_name="AVA_500M",  # or "AVA_750M", "AVA_1B"
    dataset_path=dataset_paths,
    dataset_names=dataset_names,
    save_dir="checkpoints",
    log_dir="logs",
    encoding="p50k_base",
    device="cuda"
  )

  # Start training
  trainer.train()

if __name__ == "__main__":
  main()