from text.tokenizer import TokenTrainer, Tokenizer

# trainer = TokenTrainer(model_prefix="ct_16k", vocab_size=16000, model_type="bpe")
# trainer.train("dataset/consolidated_360m.txt", continue_training=False)
# trainer.save_model(directory="trained_models")

# For SentencePiece mode, pass the path to your pretrained model.
sp_tokenizer = Tokenizer(mode="spm", spm_model_path="model/ct_16k.model")
encoded = sp_tokenizer.encode("Hello, world!")
decoded = sp_tokenizer.decode(encoded)
print("SPM Mode - Encoded:", encoded)
print("SPM Mode - Decoded:", decoded)

# For tiktoken mode (preloaded vocabs are used)
ttk_tokenizer = Tokenizer(mode="ttk", encoding="p50k_base", model="text-davinci-003")
encoded_ttk = ttk_tokenizer.encode("Hello, world!")
decoded_ttk = ttk_tokenizer.decode(encoded_ttk)
print("TIKTOK Mode - Encoded:", encoded_ttk)
print("TIKTOK Mode - Decoded:", decoded_ttk)