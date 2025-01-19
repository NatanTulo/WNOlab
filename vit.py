from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
from tqdm import tqdm

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.config.pad_token_id = tokenizer.eos_token_id

max_length = 128
num_beams = 16
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    attention_mask = torch.ones((1, max_length), dtype=torch.long).to(device)
    output_ids = model.generate(
        pixel_values,
        attention_mask=attention_mask,
        **gen_kwargs
    )

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

folder_path = "./Final_images_dataset"
output_file = "./opisy_vit.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file_name)
            preds = predict_step([image_path])
            description = preds[0]
            f.write(f"{file_name}: {description}\n")
