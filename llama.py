import transformers
import torch
from tqdm import tqdm
from datasets import Dataset

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16, "pad_token_id": 128001},
    device_map="auto",
)

with open("./opisy_blip.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

chunk_size = 10
chunked_lines = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
dataset = Dataset.from_dict({"lines": chunked_lines})

def generate_text(example):
    messages = [
        {"role": "system", "content": "You are given data in form <filename>: <description>, you have to answer in form <filename> - <category>. Pictures are from a few datasets and you have to provide the main theme that could be the name of the dataset. Don't give any slashes, stick to one category for each picture."},
        {"role": "user", "content": "".join(example["lines"])},
    ]
    output = pipeline(messages, max_new_tokens=256)[0]["generated_text"][2]["content"]
    print(output)
    return {"text": output}

processed = dataset.map(
    generate_text,
    batched=False,
    load_from_cache_file=False,
    keep_in_memory=True,
    new_fingerprint="my_pipeline_run"
)

with open("./opisy_blip_output.txt", "w", encoding="utf-8") as out:
    for output_text in processed["text"]:
        out.write(output_text + "\n")
        out.flush()