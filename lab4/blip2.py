from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
import os  # Dodano import modułu os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.float16
)

folder_path = "./Final_images_dataset"  # Definicja ścieżki do folderu
with open("opisy_blip2.txt", "w", encoding="utf-8") as f:
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert("RGB")
            
            prompt = "Task: Provide a detailed and comprehensive description of the picture. Answer:"
            inputs = processor(images=image,  return_tensors="pt").to(device="cuda", dtype=torch.float16)
            
            generated_ids = model.generate(**inputs, max_length=128, num_beams=16)
            description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            f.write(f"{file_name}: {description}\n")
