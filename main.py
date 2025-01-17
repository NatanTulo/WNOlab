# Importy z biblioteki standardowej
import datetime
import os
import time
from collections import Counter
from typing import List

# Importy bibliotek zewnętrznych
from PIL import Image
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Inicjalizacja procesora i modelu BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# Definicja ścieżek do folderu z obrazami i pliku wyjściowego
folder_path = "./Final_images_dataset"
output_file = "./opisy_blip.txt"

# Inicjalizacja logowania z nazwą pliku zawierającą datę rozpoczęcia
start_datetime = datetime.datetime.now()
log_file_name = f"log_{start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

with open(log_file_name, "w", encoding="utf-8") as log_f:
    log_f.write(f"Program rozpoczęty: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Generowanie opisów obrazów i zapis do pliku
    with open(output_file, "w", encoding="utf-8") as f:
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, file_name)
                raw_image = Image.open(image_path).convert("RGB")
                inputs = processor(raw_image, return_tensors="pt").to("cuda")
                attention_mask = inputs.get("attention_mask")
                out = model.generate(**inputs, attention_mask=attention_mask, max_length=128, num_beams=16)
                description = processor.decode(out[0], skip_special_tokens=True)
                f.write(f"{file_name}: {description}\n")
    log_f.write(f"Opisy wygenerowane: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    with open(output_file, "r", encoding="utf-8") as f:
        descriptions = f.read()
    log_f.write(descriptions + "\n\n")

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Inicjalizacja pipeline do generowania tekstu
    pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16, "pad_token_id": 128001},
        device_map="auto",
    )

    # Definicja funkcji generującej tekst na podstawie opisów
    def generate_text(example):
        messages = [
            {"role": "system", "content": "You are given data in form <filename>: <description>, you have to answer in form <filename> - <category>. Pictures are from a few datasets and you have to provide the main theme that could be the name of the dataset. Don't give any slashes, stick to one category for each picture. Categories should be as general as possible."},
            {"role": "user", "content": "".join(example["lines"])},
        ]
        output = pipeline(messages, max_new_tokens=2048)[0]["generated_text"][2]["content"]
        print(output)
        return {"text": output}

    # Definicja funkcji przetwarzającej linie opisów i zapisującej wyniki
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

    # Odczyt wygenerowanych opisów z pliku
    with open("./opisy_blip.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Przetwarzanie opisów i pomiar czasu wykonania
    execution_time = process_and_save(lines)
    print(f"Przetworzono {len(lines)} opisów w {execution_time:.2f} sekund")
    log_f.write(f"Opisy przetworzone: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    with open("opisy_blip_output.txt", "r", encoding="utf-8") as f:
        categories = f.read()
    log_f.write(categories + "\n\n")

    # Analiza kategorii i wykrywanie odstających
    categories = []
    file_data = []
    with open('opisy_blip_output.txt', 'r') as file:
        for line in file:
            parts = line.split('-')
            if len(parts) < 2:
                continue
            filename = parts[0].strip()
            category = parts[1].strip()
            categories.append(category)
            file_data.append((filename, category))

    # Liczenie wystąpień każdej kategorii i sortowanie
    category_counts = Counter(categories)
    sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))

    # Wykrywanie kategorii z jedynym obrazem i zapisywanie ich
    outliers = [f"{filename} (only {category})" for filename, category in file_data if category_counts.get(category, 0) == 1]
    if outliers:
        with open('outliers.txt', 'a', encoding='utf-8') as outliers_file:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            outliers_file.write(f"{now}\n")
            for outlier_info in outliers:
                print(outlier_info)
                outliers_file.write(f"{outlier_info}\n")
            
            # Dodanie systemu liczenia punktów
            special_images = {"wno_04.jpg", "wno_06.jpg", "wno_08.jpg", "wno_09.jpg", "wno_87.jpg", "wno_88.jpg", "wno_15.jpeg", "wno_89.jpg"}
            points = 0.0
            allowed_incorrect = 1
            for outlier in outliers:
                filename = outlier.split()[0]
                if filename in special_images:
                    points += 0.5
                else:
                    if allowed_incorrect > 0:
                        allowed_incorrect -= 1
                    else:
                        points -= 0.5
            total_points = points + 1
            outliers_file.write(f"{points} +1 = {total_points}\n\n\n")
        log_f.write(f"Outliers wykryte: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        with open('outliers.txt', 'r', encoding='utf-8') as f:
            outliers_content = f.read()
        log_f.write(outliers_content + "\n\n")

    # Tworzenie wykresu liczby obrazów w każdej kategorii
    plt.figure(figsize=(15, 8))
    plt.bar(sorted_categories.keys(), sorted_categories.values(), color='coral')
    plt.xlabel('Klasy')
    plt.ylabel('Liczba obrazów')
    plt.title('Liczba obrazów dla każdej najczęstszej klasy')
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    
    # Dodanie daty i czasu do nazwy pliku wykresu
    plot_file_name = f"top_class_counts_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(plot_file_name, dpi=300)  # Zapisanie wykresu z unikalną nazwą
    plt.show()

    # Zapis końcowego czasu i całkowitego czasu wykonania
    end_datetime = datetime.datetime.now()
    total_time = end_datetime - start_datetime
    log_f.write(f"Program zakończony: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Całkowity czas wykonania: {total_time}\n")
