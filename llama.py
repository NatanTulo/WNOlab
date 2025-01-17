import transformers
import torch
from datasets import Dataset
import time
from typing import List
import matplotlib.pyplot as plt
from collections import Counter

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "pad_token_id": 128001},
    device_map="auto",
)

def generate_text(example):
    messages = [
        {"role": "system", "content": "You are given data in form <filename>: <description>, you have to answer in form <filename> - <category>. Pictures are from a few datasets and you have to provide the main theme that could be the name of the dataset. Don't give any slashes, stick to one category for each picture. Categories should be as general as possible."},
        {"role": "user", "content": "".join(example["lines"])},
    ]
    output = pipeline(messages, max_new_tokens=2048)[0]["generated_text"][2]["content"]
    print(output)
    return {"text": output}

def process_and_save(lines: List[str]) -> float:
    start_time = time.time()
    dataset = Dataset.from_dict({"lines": [lines]})
    
    processed = dataset.map(
        generate_text,
        batched=False,
        load_from_cache_file=False,
        keep_in_memory=True,
        new_fingerprint="benchmark_all"
    )
    
    with open("opisy_blip_output.txt", "w", encoding="utf-8") as out:
        for output_text in processed["text"]:
            out.write(output_text + "\n")
            out.flush()
    
    end_time = time.time()
    return end_time - start_time

with open("./opisy_blip.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

execution_time = process_and_save(lines)
print(f"Przetworzono {len(lines)} opisów w {execution_time:.2f} sekund")

############################################################################################################
# plot
############################################################################################################

# Read and parse the data
categories = []
file_data = []  # Lista do przechowywania nazw plików i kategorii
with open('opisy_blip_output.txt', 'r') as file:
    for line in file:
        parts = line.split('-')
        if len(parts) < 2:
            continue
        filename = parts[0].strip()
        category = parts[1].strip()
        categories.append(category)
        file_data.append((filename, category))

# Count occurrences of each category
category_counts = Counter(categories)
# Sort the counter items in descending order
sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))

# Print filenames for categories with only one image
for filename, category in file_data:
    if category_counts.get(category, 0) == 1:
        print(f"{filename} (only {category})")

plt.figure(figsize=(15, 8))  # Zwiększony rozmiar figury
plt.bar(sorted_categories.keys(), sorted_categories.values(), color='coral')
plt.xlabel('Klasy')
plt.ylabel('Liczba obrazów')
plt.title('Liczba obrazów dla każdej najczęstszej klasy')
plt.xticks(rotation=90, fontsize=10)  # Zmniejszony rozmiar czcionki etykiet
plt.tight_layout()
plt.savefig('top_class_counts.png', dpi=300)  # Zapisanie wykresu z wyższą rozdzielczością
plt.show()