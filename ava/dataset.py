import torch
import torch.nn.functional as F
from typing import *
import os, random
import tiktoken

class Dataset:
  """
  Initialize the Dataset
  Args:
    path (str): Path to the DNA data file
    encoding (str): encoding for tiktoken tokenizer
    ratio (float): Fraction of data to use for validation (default 0.25)
    random_seed (int): random seeding for batching
    max_data_size (int): Maximum number of characters to load (default 100000)
  """
  def __init__(self, path: str, encoding: str="p50k_base", ratio: float=0.25, random_seed: int=1600, max_data_size: int=1e8):
    self.path = self.ratio = path, max(0.0, min(1.0, ratio))
    self.random_seed, self.max_data_size = random_seed, max_data_size
    self.encoding, self.data_len = encoding, 0
    try:
      self._tokenizer = tiktoken.get_encoding(encoding)
      self._vocab_size = self._tokenizer.n_vocab
    except Exception as e:
      raise ValueError(f"Failed to initialize the tokenizer: {e}")

    self.data, self.train_data, self.val_data = "", None, None
    self._data_split = False

  def load_data(self):
    if not os.path.isfile(self.path):
      raise FileNotFoundError(f"{self.path} does not exist.")

    try:
      with open(self.path, "r", encoding="utf-8") as file:
        data = file.readlines()
        self.data_len = len(data)
        file.close()
    except Exception as e:
      raise IOError(f"Failed to read file {self.path}: {e}")

    return data

  def split(self):
    data = self.load_data()
    n = int(self.data_len * (1 - self.ratio))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

  def get_batch(self, split_name: str, batch_size: int, block_size: int, device: str="cuda"):
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
      raise ValueError("batch_size & block_size must be poitive")
    train_data, val_data = self.split()
    train_data, val_data = self._tokenizer.encode(train_data), self._tokenizer.decode(val_data)
    data = train_data if split_name == "train" else val_data
    if len(data) == 0:
      raise ValueError(f"No more data available for split: '{split_name}")
    if len(data) < block_size:
      raise ValueError(f"Data length ({len(data)}) is less then block_size ({block_size})")
    
    torch.manual_seed(self.random_seed)
    random.seed(self.random_seed)
    max_start = len(data) - block_size
    if max_start <= 0:
      raise ValueError("Block size is too large for available data")
    idx = torch.randint(0, max_start, (batch_size,))
    try:
      x = torch.stack([data[i:i+block_size] for i in idx])
      y = x.clone()
      return x.to(device), y.to(device)
    except Exception as e:
      raise RuntimeError(f"Failed to create batch: {e}")

  def get_data_stats(self) -> Dict[str, Any]:
    """Get statistics about the dataset"""
    train_data, val_data = self.split()
    return {
      'total_length': self.data_len,
      'train_tokens': len(train_data),
      'val_tokens': len(val_data),
      'encoding': self.encoding,
      'vocab_size': self._vocab_size,
      'random_seed': self.random_seed,
      'split_ratio': self.ratio
    }


  def __len__(self) -> int:
    return len(self.data_len)