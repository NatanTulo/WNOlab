# Importy z biblioteki standardowej
import datetime
import os
from collections import Counter

# Importy bibliotek zewnętrznych
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageStat

def is_grayscale(image_path):
    img = Image.open(image_path).convert("RGB")
    grayscale_img = img.convert("L").convert("RGB")
    return list(img.getdata()) == list(grayscale_img.getdata())

folder_path = "./Final_images_dataset"
start_datetime = datetime.datetime.now()
timestamp_for_files = start_datetime.strftime('%Y-%m-%d_%H-%M-%S')
# Analiza kategorii i wykrywanie odstających
categories = []
file_data = []
with open('opisy_output.txt', 'r', encoding='utf-8') as file:
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
        for outlier_info in outliers:
            print(outlier_info)
            outliers_file.write(f"{outlier_info}\n")
        
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
        total_points = plus_points - minus_points + 1
        outliers_file.write(f"{plus_points} - {minus_points} + 1 = {total_points}\n\n\n")
        print(f"{plus_points} - {minus_points} + 1 = {total_points}")

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
