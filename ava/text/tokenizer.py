import os
import sentencepiece as spm
import tiktoken

class TokenTrainer:
  def __init__(self, model_prefix="tokenizer", vocab_size=8000, model_type="bpe"):
    """
    Initializes the SentencePiece trainer.
    
    Args:
      model_prefix (str): Prefix for the trained model files.
      vocab_size (int): Target vocabulary size.
      model_type (str): Type of SentencePiece model (e.g. "bpe", "unigram").
    """
    self.model_prefix = model_prefix
    self.vocab_size = vocab_size
    self.model_type = model_type
    self.sp = spm.SentencePieceProcessor()
    self.model_path = f"{self.model_prefix}.model"
    
    # If model exists, load it.
    if os.path.exists(self.model_path):
      self.sp.load(self.model_path)

  def train(self, input_file, continue_training=False):
    """
    Trains a SentencePiece tokenizer from scratch or continues training an existing model.
    
    Args:
      input_file (str): Path to the text file containing training data.
      continue_training (bool): Whether to continue training from an existing model.
    """
    if continue_training and os.path.exists(self.model_path):
      existing_vocab_size = self.sp.get_piece_size()
      assert self.vocab_size > existing_vocab_size, "New vocab size must be larger than the existing one."
      print(f"Continuing training from {existing_vocab_size} to {self.vocab_size} vocabulary size...")
      spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=self.model_prefix,
        vocab_size=self.vocab_size,
        model_type=self.model_type,
        user_defined_symbols=[self.sp.id_to_piece(i) for i in range(existing_vocab_size)],
        input_sentence_size=1000000,
        shuffle_input_sentence=True
      )
    else:
      print(f"Training tokenizer from scratch with vocab size {self.vocab_size}...")
      spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=self.model_prefix,
        vocab_size=self.vocab_size,
        model_type=self.model_type,
        input_sentence_size=1000000,
        shuffle_input_sentence=True
      )
    # Load the trained model.
    self.sp.load(self.model_path)

  def save_model(self, directory="."):
    """
    Saves the trained model files to the specified directory.
    
    Args:
      directory (str): Directory to save the model.
    """
    os.makedirs(directory, exist_ok=True)
    os.rename(f"{self.model_prefix}.model", os.path.join(directory, f"{self.model_prefix}.model"))
    os.rename(f"{self.model_prefix}.vocab", os.path.join(directory, f"{self.model_prefix}.vocab"))
    print(f"Tokenizer saved to {directory}")

  def load_model(self, model_path):
    """
    Loads a trained SentencePiece model.
    
    Args:
      model_path (str): Path to the .model file.
    """
    if os.path.exists(model_path):
      self.sp.load(model_path)
      print(f"Loaded tokenizer model from {model_path}")
    else:
      raise FileNotFoundError(f"Model file {model_path} not found.")

  def get_vocab(self):
    """Returns the vocabulary as a list of tokens."""
    return [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]

  def get_vocab_size(self):
    """Returns the size of the vocabulary."""
    return self.sp.get_piece_size()

class Tokenizer:
  def __init__(self, mode="spm", encoding=None, model=None, spm_model_path=None):
    """
    Initializes the tokenizer.
    
    Args:
      mode (str): "spm" to use SentencePiece, "ttk" to use tiktoken.
      encoding (str): tiktoken encoding name (used in "ttk" mode).
      model (str): Model name for tiktoken (used in "ttk" mode).
      spm_model_path (str): Path to a pre-trained SentencePiece model (used in "spm" mode).
    """
    self.mode = mode.lower()
    if self.mode == "spm":
      if spm_model_path is None or not os.path.exists(spm_model_path):
        raise ValueError("For SentencePiece mode, a valid spm_model_path must be provided.")
      self.sp = spm.SentencePieceProcessor()
      self.sp.load(spm_model_path)
    elif self.mode == "ttk":
      self.encodings = encoding if encoding is not None else "p50k_base"
      self.model = model if model is not None else "text-davinci-003"
      self.tokenizer = tiktoken.get_encoding(self.encodings)
      self.tokenizer = tiktoken.encoding_for_model(self.model)
    else:
      raise ValueError("Mode must be either 'spm' or 'ttk'.")

  def encode(self, data):
    """
    Encodes input text into token IDs.
    
    Args:
      data (str): Input text.
      
    Returns:
      List[int]: Token IDs.
    """
    if self.mode == "spm":
      return self.sp.encode(data, out_type=int)
    elif self.mode == "ttk":
      return self.tokenizer.encode(data)

  def decode(self, tokens):
    """
    Decodes token IDs back into text.
    
    Args:
      tokens (List[int]): List of token IDs.
      
    Returns:
      str: Decoded text.
    """
    if self.mode == "spm":
      return self.sp.decode(tokens)
    elif self.mode == "ttk":
      return self.tokenizer.decode(tokens)

  def get_vocab_size(self):
    """
    Returns the vocabulary size.
    
    Returns:
      int: Vocabulary size.
    """
    if self.mode == "spm":
      return self.sp.get_piece_size()
    elif self.mode == "ttk":
      return self.tokenizer.n_vocab

  def get_vocab(self):
    """
    Returns the vocabulary as a list of token strings.
    
    Returns:
      List[str]: List of tokens.
    """
    if self.mode == "spm":
      return [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]
    elif self.mode == "ttk":
      # tiktoken may not provide a direct method; use an appropriate method based on your library version
      return [self.tokenizer.id_to_token(i) for i in range(self.tokenizer.n_vocab)]