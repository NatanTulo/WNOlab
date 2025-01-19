# Importy z biblioteki standardowej
import datetime
import os
import signal
import sys
import time
from collections import Counter
from typing import List

# Importy bibliotek zewnętrznych
import google.generativeai as genai
import matplotlib.pyplot as plt
import torch
from PIL import Image
from datasets import Dataset
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor, pipeline

# Dodanie zmiennej wyboru API
api_choice = "llama"  # Możliwe wartości: "llama", "google"

# Inicjalizacja modelu Google AI
genai.configure(api_key="TOKEN")  # Dodaj swój klucz API
google_model = genai.GenerativeModel("gemini-1.5-flash")

# Definicja ścieżek do folderu z obrazami i pliku wyjściowego
folder_path = "./Final_images_dataset"
desc_file = ["./opisy_blip.txt","./opisy_blip2.txt","./opisy_vit.txt"]
output_file = "./kategorie.txt"

# Inicjalizacja logowania z nazwą pliku zawierającą datę rozpoczęcia
start_datetime = datetime.datetime.now()
timestamp_for_files = start_datetime.strftime('%Y-%m-%d_%H-%M-%S')
log_file_name = f"log_{timestamp_for_files}.txt"

def handle_sigint(signum, frame):
    print("Program interrupted by user (Ctrl+C).")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

def is_grayscale(image_path, tolerance=10):
    img = Image.open(image_path).convert("RGB")
    for pixel in img.getdata():
        r, g, b = pixel
        if abs(r - g) > tolerance or abs(g - b) > tolerance or abs(b - r) > tolerance:
            return False
    return True

with open(f"wyniki/{log_file_name}", "w", encoding="utf-8") as log_f:
    log_f.write(f"Program rozpoczęty: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Dodanie zmiennej wyboru regeneracji opisów
    regen_descriptions = False  # Nie generuj ponownie opisów

    if regen_descriptions:
        # Inicjalizacja procesora i modelu BLIP
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
        
        # Generowanie opisów obrazów i zapis do pliku
        with open(desc_file, "w", encoding="utf-8") as f:
            for file_name in tqdm(os.listdir(folder_path)):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(folder_path, file_name)
                    raw_image = Image.open(image_path).convert("RGB")
                    inputs = processor(raw_image, return_tensors="pt").to("cuda")
                    attention_mask = inputs.get("attention_mask")
                    out = blip_model.generate(**inputs, attention_mask=attention_mask, max_length=128, num_beams=16)
                    description = processor.decode(out[0], skip_special_tokens=True)
                    f.write(f"{file_name}: {description}\n")
        log_f.write(f"Opisy wygenerowane: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        log_f.write(f"Użyto istniejące opisy: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    lines = []
    for single_file in desc_file:
        with open(single_file, "r", encoding="utf-8") as f:
            lines.extend(f.readlines())
        log_f.write("".join(lines) + "\n\n")

    # Inicjalizacja pipeline do generowania tekstu
    if api_choice == "llama":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16, "pad_token_id": 128001},
            device_map="auto",
        )

    # Definicja funkcji generującej tekst za pomocą Llama
    def generate_text_llama(example):
        if api_choice != "llama":
            raise RuntimeError("Pipeline nie został zainicjalizowany. Ustaw api_choice na 'llama'.")
        messages = [
            {"role": "system", "content": "You are given data in form <filename>: <description>, you have to answer in form <filename> - <category>. Pictures are from a few datasets and you have to provide the main theme that could be the name of the dataset. Don't give any slashes, stick to one category for each picture."},
            {"role": "user", "content": "".join(example["lines"])},
        ]
        output = llm_pipeline(messages, max_new_tokens=2048)[0]["generated_text"][2]["content"]
        # print(output)
        return {"text": output}

    # Definicja funkcji generującej tekst za pomocą Google API
    def generate_text_google(example):
        messages = [
            {"role": "system", "content": "You analyze image descriptions in the format filename: description. Identify outliers—singular images that do not match the main dataset themes. Return filename - category, inferring the category from the description. Keep responses concise."},
            {"role": "user", "content": "".join(example["lines"])},
        ]
        resp = google_model.generate_content([messages[0]["content"], messages[1]["content"]])
        output_text = resp.text
        # print(output_text)
        return {"text": output_text}

    # Aktualizacja funkcji przetwarzającej linie opisów i zapisującej wyniki
    def process_and_save(lines: List[str]) -> float:
        start_time = time.time()
        dataset = Dataset.from_dict({"lines": [lines]})
        
        if api_choice == "llama":
            processed = dataset.map(
                generate_text_llama,
                batched=False,
                load_from_cache_file=False,
                keep_in_memory=True,
                new_fingerprint="benchmark_llama"
            )
        elif api_choice == "google":
            processed = dataset.map(
                generate_text_google,
                batched=False,
                load_from_cache_file=False,
                keep_in_memory=True,
                new_fingerprint="benchmark_google"
            )
        else:
            raise ValueError("Nieprawidłowy wybór API. Użyj 'llama' lub 'google'.")
        
        with open(output_file, "w", encoding="utf-8") as out:
            for output_text in processed["text"]:
                out.write(output_text + "\n")
        
        end_time = time.time()
        return end_time - start_time

    # Odczyt wygenerowanych opisów z pliku
    lines = []
    for single_file in desc_file:
        with open(single_file, "r", encoding="utf-8") as f:
            lines.extend(f.readlines())

    benchmark_results = {
        desc: {
            "llama": {"total_points": [], "plus_points": [], "minus_points": []},
            "google": {"total_points": [], "plus_points": [], "minus_points": []}
        }
        for desc in desc_file
    }

    # Główna pętla po plikach opisów
    for current_desc_file in desc_file:
        print(f"\nTestowanie pliku: {current_desc_file}")
        
        # Wczytaj opisy z aktualnego pliku
        with open(current_desc_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Pętla po API
        for test_api in ["llama", "google"]:
            print(f"\nTestowanie API: {test_api}")
            for attempt in range(1, 6):
                print(f"\nPróba {attempt}/5")
                api_choice = test_api

                # Przetwarzanie opisów i pomiar czasu wykonania
                execution_time = process_and_save(lines)
                print(f"Przetworzono {len(lines)} opisów w {execution_time:.2f} sekund")
                log_f.write(f"Opisy przetworzone: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                with open(output_file, "r", encoding="utf-8") as f:
                    categories = f.read()
                log_f.write(categories + "\n\n")

                # Analiza kategorii i wykrywanie odstających
                categories = []
                file_data = []
                with open(output_file, 'r') as file:
                    for line in file:
                        parts = line.split('-')
                        if len(parts) < 2:
                            continue
                        filename = parts[0].strip()
                        category = parts[1].strip()
                        categories.append(category)
                        file_data.append((filename, category))

                color_status_map = {}
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        is_gray = is_grayscale(os.path.join(folder_path, file_name))
                        color_status_map[file_name] = "grayscale" if is_gray else "color"

                updated_file_data = []
                for (filename, category) in file_data:
                    color_status = color_status_map.get(filename, "color")
                    updated_file_data.append((filename, category, color_status))
                file_data = updated_file_data

                # Liczenie wystąpień każdej kategorii i sortowanie
                category_counts = Counter(categories)
                sorted_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
                
                print(f"Liczba różnych kategorii: {len(category_counts)}")  # Dodano print liczby kategorii

                # Wykrywanie kategorii z jedynym obrazem i zapisywanie ich
                outliers = [
                    f"{filename} (only {category})"
                    for filename, category, color in file_data  # Zmień rozpakowanie na trzy wartości
                    if category_counts.get(category, 0) == 1
                ]

                # Dodatkowa detekcja outlierów dla kolor/grayscale
                color_counts = Counter([cd[2] for cd in file_data])
                if color_counts["grayscale"] == 1:
                    for (filename, cat, colorstatus) in file_data:
                        if colorstatus == "grayscale":
                            outliers.append(f"{filename} (only grayscale)")
                            break

                if color_counts["color"] == 1:
                    for (filename, cat, colorstatus) in file_data:
                        if colorstatus == "color":
                            outliers.append(f"{filename} (only color)")
                            break

                if outliers:
                    with open('outliers.txt', 'a', encoding='utf-8') as outliers_file:
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        outliers_file.write(f"{now}\n")
                        
                        # Dodanie systemu liczenia punktów
                        special_images = {"wno_04.jpg", "wno_06.jpg", "wno_08.jpg", "wno_09.jpg", "wno_87.jpg", "wno_88.jpg", "wno_15.jpeg", "wno_89.jpg"}
                        plus_points = 0.0
                        minus_points = 0.0
                        allowed_incorrect = 1
                        
                        for outlier in outliers:
                            filename = outlier.split()[0]
                            if filename in special_images:
                                plus_points += 0.5
                                print(f"Dodano +0.5 punktu za specjalny obraz: {filename}. plus_points = {plus_points}")
                            else:
                                if allowed_incorrect > 0:
                                    allowed_incorrect -= 1
                                    print(f"Zmniejszono allowed_incorrect do {allowed_incorrect}")
                                else:
                                    minus_points += 0.5
                                    print(f"Dodano -0.5 punktu za nieakceptowany obraz: {filename}. minus_points = {minus_points}")
                            outliers_file.write(f"{outlier}\n")
                            print(outlier)
                        
                        total_points = plus_points - minus_points + 1
                        print(f"Suma punktów: {plus_points} - {minus_points} + 1 = {total_points}")
                        outliers_file.write(f"{plus_points} - {minus_points} + 1 = {total_points}\n\n\n")
                    
                    log_f.write(f"Outliers wykryte: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    for outlier_info in outliers:
                        log_f.write(f"{outlier_info}\n")
                    log_f.write(f"{plus_points} - {minus_points} + 1 = {total_points}\n\n\n")

                # Zapisz wyniki do odpowiedniej struktury
                benchmark_results[current_desc_file][test_api]["total_points"].append(total_points)
                benchmark_results[current_desc_file][test_api]["plus_points"].append(plus_points)
                benchmark_results[current_desc_file][test_api]["minus_points"].append(minus_points)

    # Zapis wyników do pliku
    with open("benchmark_results.txt", "w", encoding="utf-8") as bench_f:
        for desc in benchmark_results:
            bench_f.write(f"\nWyniki dla pliku {desc}:\n")
            for test_api in benchmark_results[desc]:
                bench_f.write(f"{test_api}:\n")
                for i in range(len(benchmark_results[desc][test_api]["total_points"])):
                    tp = benchmark_results[desc][test_api]["total_points"][i]
                    pp = benchmark_results[desc][test_api]["plus_points"][i]
                    mp = benchmark_results[desc][test_api]["minus_points"][i]
                    bench_f.write(f"  Próba {i+1}: total={tp}, plus={pp}, minus={mp}\n")

    # Generowanie wykresów dla każdego pliku opisów
    for desc in benchmark_results:
        desc_name = os.path.splitext(os.path.basename(desc))[0]
        
        # Wykres total points
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot(range(1, 6), benchmark_results[desc]["llama"]["total_points"], marker='o')
        axes[0].set_title(f"Total Points (llama) - {desc_name}")
        axes[1].plot(range(1, 6), benchmark_results[desc]["google"]["total_points"], marker='o', color='red')
        axes[1].set_title(f"Total Points (google) - {desc_name}")
        plt.savefig(f"benchmark_total_points_{desc_name}.png")
        plt.close()

        # Wykres plus/minus points
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
        axes2[0].plot(range(1, 6), benchmark_results[desc]["llama"]["plus_points"], label="plus")
        axes2[0].plot(range(1, 6), benchmark_results[desc]["llama"]["minus_points"], label="minus")
        axes2[0].legend()
        axes2[0].set_title(f"llama (plus/minus) - {desc_name}")
        axes2[1].plot(range(1, 6), benchmark_results[desc]["google"]["plus_points"], label="plus")
        axes2[1].plot(range(1, 6), benchmark_results[desc]["google"]["minus_points"], label="minus")
        axes2[1].legend()
        axes2[1].set_title(f"google (plus/minus) - {desc_name}")
        plt.savefig(f"benchmark_plus_minus_points_{desc_name}.png")
        plt.close()

    # Tworzenie wykresu liczby obrazów w każdej kategorii
    plt.figure(figsize=(15, 8))
    plt.bar(sorted_categories.keys(), sorted_categories.values(), color='coral')
    plt.xlabel('Klasy')
    plt.ylabel('Liczba obrazów')
    plt.title('Liczba obrazów dla każdej najczęstszej klasy')
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    
    # Użycie wspólnego znacznika czasowego dla nazwy pliku wykresu
    plot_file_name = f"wyniki/top_class_counts_{timestamp_for_files}.png"
    plt.savefig(plot_file_name, dpi=300)
    # plt.show()

    # Zapis końcowego czasu i całkowitego czasu wykonania
    end_datetime = datetime.datetime.now()
    total_time = end_datetime - start_datetime
    log_f.write(f"Program zakończony: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"Całkowity czas wykonania: {total_time}\n")
