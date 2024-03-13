import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
import pandas as pd

input_directory = 'parquet files'
output_directory = 'parquet_out'

if not os.path.exists(output_directory):
  os.makedirs(output_directory)

parquet_files = [file for file in os.listdir(input_directory) if file.endswith('.parquet')]

for file_name in parquet_files:
  input_path = os.path.join(input_directory, file_name)
  output_path = os.path.join(output_directory, os.path.splitext(file_name)[0] + '.txt')

  df = pd.read_parquet(input_path)
  df.to_csv(output_path, sep='\t', index=False)

  print(f"Conversion complete: {file_name} -> {output_path}")

print("All Parquet files converted to text files.")
