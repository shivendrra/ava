import sentencepiece as spm
import tiktoken
import os

pre_encodings = 'p50k_base'
pre_model = 'text-davinci-003'

class Tokenizer:
  def __init__(self, encoding=None, model=None):
    self.encodings = encoding if encoding is not None else pre_encodings
    self.model = model if model is not None else pre_model
    self.tokenizer = tiktoken.get_encoding(self.encodings)
    self.tokenizer = tiktoken.encoding_for_model(self.model)
  def encode(self, data): return self.tokenizer.encode(data)
  def decode(self, tokens): return self.tokenizer.decode(tokens)
  def get_vocab(self): return self.tokenizer.n_vocab

class TokenTrainer:
  def __init__(self, model_prefix="tokenizer", vocab_size=8000, model_type="bpe"):
    """
    Initializes the tokenizer.
    
    Args:
      model_prefix (str): Prefix for the trained model files.
      vocab_size (int): Target vocabulary size.
      model_type (str): Type of SentencePiece model (bpe/unigram/word/char).
    """
    self.model_prefix = model_prefix
    self.vocab_size = vocab_size
    self.model_type = model_type
    self.sp = spm.SentencePieceProcessor()
    self.model_path = f"{self.model_prefix}.model"

    # Load the model if it already exists
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
      # Load existing model to continue training
      existing_vocab_size = self.sp.get_piece_size()
      assert self.vocab_size > existing_vocab_size, "New vocab size must be larger than the existing one."
      
      print(f"Continuing training from {existing_vocab_size} to {self.vocab_size} vocabulary size...")
      spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=self.model_prefix,
        vocab_size=self.vocab_size,
        model_type=self.model_type,
        user_defined_symbols=[self.sp.id_to_piece(i) for i in range(existing_vocab_size)],
        input_sentence_size=1000000,  # Use a subset of sentences if dataset is large
        shuffle_input_sentence=True
      )
    else:
      # Train from scratch
      print(f"Training tokenizer from scratch with vocab size {self.vocab_size}...")
      spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=self.model_prefix,
        vocab_size=self.vocab_size,
        model_type=self.model_type,
        input_sentence_size=1000000,
        shuffle_input_sentence=True
      )
    
    # Load the newly trained model
    self.sp.load(self.model_path)

  def encode(self, text):
    """
    Encodes a given text into token IDs.
    
    Args:
      text (str): Input text.
      
    Returns:
      List[int]: Tokenized IDs.
    """
    return self.sp.encode(text, out_type=int)

  def decode(self, token_ids):
    """
    Decodes a list of token IDs back to text.
    
    Args:
      token_ids (List[int]): List of token IDs.
      
    Returns:
      str: Decoded text.
    """
    return self.sp.decode(token_ids)

  def save_model(self, directory="."):
    """
    Saves the trained tokenizer model to the specified directory.
    
    Args:
      directory (str): Path to save the model files.
    """
    os.makedirs(directory, exist_ok=True)
    os.rename(f"{self.model_prefix}.model", os.path.join(directory, f"{self.model_prefix}.model"))
    os.rename(f"{self.model_prefix}.vocab", os.path.join(directory, f"{self.model_prefix}.vocab"))
    print(f"Tokenizer saved to {directory}")

  def load_model(self, model_path):
    """
    Loads a trained SentencePiece tokenizer model.
    
    Args:
      model_path (str): Path to the .model file.
    """
    if os.path.exists(model_path):
      self.sp.load(model_path)
      print(f"Loaded tokenizer model from {model_path}")
    else:
      raise FileNotFoundError(f"Model file {model_path} not found.")

  def get_vocab_size(self):
    """
    Returns the current vocabulary size of the tokenizer.
    
    Returns:
      int: Vocabulary size.
    """
    return self.sp.get_piece_size()

  def get_vocab(self):
    """
    Returns the vocabulary as a list of token strings.
    
    Returns:
      List[str]: List of vocabulary tokens.
    """
    return [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]


# sample training code logic
input_file = "dataset/train.txt"

trainer = TokenTrainer("cl_6k", 6000)
trainer.train(input_file)

print("Vocab size:", trainer.get_vocab_size())
print("Vocabulary:", trainer.get_vocab()[:50])

encoded = trainer.encode("Hello, how are you?")
print("Encoded:", encoded)

decoded = trainer.decode(encoded)
print("Decoded:", decoded)