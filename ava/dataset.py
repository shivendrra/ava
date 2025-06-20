import torch
from typing import *
import os, random
import tiktoken

class Dataset:
  """
  Initialize the Dataset
  Args:
    path (str): Path to the text data file
    encoding (str): encoding for tiktoken tokenizer
    ratio (float): Fraction of data to use for validation (default 0.25)
    random_seed (int): random seeding for batching
    max_data_size (int): Maximum number of characters to load (default 100000)
  """
  def __init__(self, path: str, encoding: str = "p50k_base", ratio: float = 0.25, random_seed: int = 1600, max_data_size: int = int(1e8)):
    self.path = path
    self.random_seed, self.ratio = random_seed, max(0.0, min(1.0, ratio))
    self.max_data_size, self.encoding = max_data_size, encoding
    try:
      self._tokenizer = tiktoken.get_encoding(encoding)
      self._vocab_size = self._tokenizer.n_vocab
    except Exception as e:
      raise ValueError(f"Failed to initialize the tokenizer: {e}")

    # initializing stats
    self.data_len, self.tokenized_data_len, self.train_data_len, self.val_data_len = 0, 0, 0, 0
    self._stats_computed = False

  def _load_and_tokenize_data(self):
    """Load and tokenize data without storing it permanently"""
    if not os.path.isfile(self.path):
      raise FileNotFoundError(f"{self.path} does not exist.")

    try:
      with open(self.path, "r", encoding="utf-8") as file:
        raw_data = file.read()
        if len(raw_data) > self.max_data_size:
          raw_data = raw_data[:self.max_data_size]
        self.data_len = len(raw_data)
    except Exception as e:
      raise IOError(f"Failed to read file {self.path}: {e}")

    try:
      tokens = self._tokenizer.encode(raw_data)
      tokenized_data = torch.tensor(tokens, dtype=torch.long)
      self.tokenized_data_len = len(tokenized_data)
      return tokenized_data
    except Exception as e:
      raise ValueError(f"Failed to tokenize data: {e}")

  def _compute_split_lengths(self):
    """Compute train/val split lengths without storing the data"""
    if not self._stats_computed:
      tokenized_data = self._load_and_tokenize_data()
      n = int(len(tokenized_data) * (1 - self.ratio))
      self.train_data_len = n
      self.val_data_len = len(tokenized_data) - n
      self._stats_computed = True

  def split_data(self):
    """Split the tokenized data into train and validation sets"""
    tokenized_data = self._load_and_tokenize_data()
    n = int(len(tokenized_data) * (1 - self.ratio))
    train_data = tokenized_data[:n]
    val_data = tokenized_data[n:]
    self.train_data_len, self.val_data_len = len(train_data), len(val_data)
    return train_data, val_data

  def get_batch(self, split_name: str, batch_size: int, block_size: int, device: str = "cuda"):
    """
    Samples a random batch of subsequences from the train or validation data
    Args:
      split_name (str): "train" or "val"
      batch_size (int): Number of samples in the batch
      block_size (int): Length of each subsequence
      device (str): Device to move the tensors to (e.g. "cpu" or "cuda")
    Returns:
      Tuple of tensors (x, y) where x is the input batch and y is the target batch
    """
    if split_name not in ["train", "val"]:
      raise ValueError("Split must be either 'train' or 'val'")
    if batch_size <= 0 or block_size <= 0:
      raise ValueError("batch_size & block_size must be positive")

    # Load and split data fresh each time to avoid storing
    train_data, val_data = self.split_data()  
    data = train_data if split_name == "train" else val_data
    if len(data) == 0:
      raise ValueError(f"No data available for split: '{split_name}'")
    if len(data) < block_size + 1:
      raise ValueError(f"Data length ({len(data)}) is less than block_size + 1 ({block_size + 1})")

    torch.manual_seed(self.random_seed + hash(split_name) % 10000)  # Set random seed for reproducibility
    random.seed(self.random_seed + hash(split_name) % 10000)  # Set random seed for reproducibility
    max_start = len(data) - block_size  # Generate random starting indices
    if max_start <= 0:
      raise ValueError("Block size is too large for available data")
    idx = torch.randint(0, max_start, (batch_size,))
    try:
      # Create input and target sequences
      x = torch.stack([data[i:i + block_size] for i in idx])
      y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])
      return x.to(device), y.to(device)
    except Exception as e:
      raise RuntimeError(f"Failed to create batch: {e}")

  def get_data_stats(self) -> Dict[str, Any]:
    """Get statistics about the dataset"""
    # Ensure split lengths are computed
    self._compute_split_lengths()

    return {
      'total_chars': self.data_len,
      'total_tokens': self.tokenized_data_len,
      'train_tokens': self.train_data_len,
      'val_tokens': self.val_data_len,
      'encoding': self.encoding,
      'vocab_size': self._vocab_size,
      'random_seed': self.random_seed,
      'split_ratio': self.ratio
    }

  def __len__(self) -> int:
    self._compute_split_lengths()
    return self.tokenized_data_len