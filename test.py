import torch
import tiktoken
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