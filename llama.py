import transformers
import torch
from tqdm import tqdm
from datasets import Dataset
import time
from typing import List
import matplotlib.pyplot as plt

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

def process_and_save(lines: List[str], chunk_size: int) -> tuple[float, str]:
    start_time = time.time()
    chunked_lines = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    dataset = Dataset.from_dict({"lines": chunked_lines})
    
    processed = dataset.map(
        generate_text,
        batched=False,
        load_from_cache_file=False,
        keep_in_memory=True,
        new_fingerprint=f"benchmark_chunk_{chunk_size}"
    )
    
    # Save results to a chunk-specific file
    output_filename = f"opisy_blip_output_chunk{chunk_size}.txt"
    with open(output_filename, "w", encoding="utf-8") as out:
        for output_text in processed["text"]:
            out.write(output_text + "\n")
            out.flush()
    
    end_time = time.time()
    return end_time - start_time, output_filename

# Read the file
with open("./opisy_blip.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Test different chunk sizes
chunk_sizes = [26,52,104]  # Divisors or near-divisors of 104
results = {}
output_files = {}

for size in chunk_sizes:
    print(f"Testing chunk size: {size}")
    execution_time, output_file = process_and_save(lines, size)
    results[size] = execution_time
    output_files[size] = output_file
    print(f"Time taken: {execution_time:.2f} seconds")
    print(f"Results saved to: {output_file}")

# Save benchmark summary
with open("benchmark_summary.txt", "w", encoding="utf-8") as f:
    f.write("Benchmark Results:\n")
    f.write("================\n")
    for size in chunk_sizes:
        f.write(f"Chunk size: {size}\n")
        f.write(f"Time: {results[size]:.2f} seconds\n")
        f.write(f"Output file: {output_files[size]}\n")
        f.write("----------------\n")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.title('Execution Time vs Chunk Size')
plt.xlabel('Chunk Size')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.savefig('benchmark_results.png')
plt.show()