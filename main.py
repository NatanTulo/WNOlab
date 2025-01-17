from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from tqdm import tqdm

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

folder_path = "./Final_images_dataset"
output_file = "./opisy_blip.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file_name)
            raw_image = Image.open(image_path).convert("RGB")
            inputs = processor(raw_image, return_tensors="pt").to("cuda")
            attention_mask = inputs.get("attention_mask")
            out = model.generate(**inputs, attention_mask=attention_mask, max_length=64, num_beams=8)
            description = processor.decode(out[0], skip_special_tokens=True)
            f.write(f"{file_name}: {description}\n")
