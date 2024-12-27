"""
  @process.py
  - contains code logics for processing the datasets
  - handles automation logics for unzipping, un-parqueting, spliting/appending, etc.
** works fine, more to be added
"""

import os, shutil, gzip
import pandas as pd

class Process:
  def __init__(self, input_directory: str, output_directory: str):
    self.input_directory = input_directory
    self.output_directory = output_directory
    os.makedirs(self.output_directory, exist_ok=True)

  def combine_text_files(self, output_file):
    files = os.listdir(self.input_directory)
    output_path = os.path.join(self.output_directory, output_file)

    with open(output_path, 'a', encoding='utf-8') as output_file:
      for file_name in files:
        input_path = os.path.join(self.input_directory, file_name)

        if file_name.endswith('.txt') and os.path.isfile(input_path):
          print(f"Reading: {file_name}")
          with open(input_path, 'r', encoding='utf-8') as input_file:
            output_file.write(input_file.read())
            output_file.write('\n')

          print(f"Reading complete: {file_name}")
        else:
          print(f"Skipping non-text file: {file_name}")

  def split_file(self, input_file: str, num_files: int):
    input_path = os.path.join(self.input_directory, input_file)
    with open(input_path, 'r', encoding='utf-8') as f:
      total_lines = sum(1 for _ in f)
      lines_per_file = total_lines // num_files
      f.seek(0)

      for i in range(num_files):
        output_file = os.path.join(self.output_directory, f'chunk_0{i+1}.txt')
        with open(output_file, 'w', encoding='utf-8') as fw:
          lines_written = 0
          while lines_written < lines_per_file:
            line = f.readline()
            if not line:
              break
            fw.write(line)
            lines_written += 1
    print(f"File split completed into {num_files} files in '{self.output_directory}' directory.")

  def parquet_to_csv(self):
    parquet_files = [file for file in os.listdir(self.input_directory) if file.endswith('.parquet')]

    for file_name in parquet_files:
      input_path = os.path.join(self.input_directory, file_name)
      base_name = os.path.splitext(file_name)[0]
      csv_output_path = os.path.join(self.output_directory, base_name + '.csv')
      df = pd.read_parquet(input_path)
      df.to_csv(csv_output_path, sep=',', index=False)
      print(f"CSV Conversion complete: {file_name} -> {csv_output_path}")

  def parquet_to_txt(self):
    parquet_files = [file for file in os.listdir(self.input_directory) if file.endswith('.parquet')]

    for file_name in parquet_files:
      input_path = os.path.join(self.input_directory, file_name)
      base_name = os.path.splitext(file_name)[0]
      txt_output_path = os.path.join(self.output_directory, base_name + '.txt')
      df = pd.read_parquet(input_path)
      df.to_csv(txt_output_path, sep='\t', index=False)
      print(f"TXT Conversion complete: {file_name} -> {txt_output_path}")

  def convert_csv_to_parquet(self):
    csv_files = [file for file in os.listdir(self.input_directory) if file.endswith('.csv')]

    for file_name in csv_files:
      input_path = os.path.join(self.input_directory, file_name)
      base_name = os.path.splitext(file_name)[0]
      parquet_output_path = os.path.join(self.output_directory, base_name + '.parquet')
      df = pd.read_csv(input_path)
      df.to_parquet(parquet_output_path, index=False)
      print(f"Parquet Conversion complete: {file_name} -> {parquet_output_path}")

  def unzip(self):
    files = os.listdir(self.input_directory)

    for file_name in files:
      input_path = os.path.join(self.input_directory, file_name)
      output_path = os.path.join(self.output_directory, os.path.splitext(file_name)[0])
      if file_name.endswith('.gz'):
        print(f"Unzipping: {file_name}")
        with gzip.open(input_path, 'rb') as f_in:
          with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Unzipping complete: {output_path}")
      else:
        print(f"Skipping non-GZip file: {file_name}")