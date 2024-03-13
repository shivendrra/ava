import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

def consolidate_files(input_directory, output_file):
  total_words = 0
    
  with open(output_file, 'w', encoding='utf-8') as output:
    for filename in os.listdir(input_directory):
      if filename.endswith('.txt'):
        input_path = os.path.join(input_directory, filename)
        with open(input_path, 'r', encoding='utf-8') as input_file:
          content = input_file.read().strip()
          output.write(content)
          words_count = len(content.split())
          total_words += words_count

  with open(output_file, 'a', encoding='utf-8') as output:
    output.write(f"\nTotal words: {total_words}")

  return total_words

if __name__ == "__main__":
  input_directory = "parquet_out"
  output_file = "datasets/consolidated.txt"
  total_words = consolidate_files(input_directory, output_file)
  print(f"Files consolidated successfully.\nTotal words in the consolidated file: {total_words}")