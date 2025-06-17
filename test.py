import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
print(current_directory)

with open("ava/model/ct_16k.model", "rb") as f:
  data = f.read()
  data = data.decode(encoding="utf-8", errors="ignore")

print(data[:1000])