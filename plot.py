import matplotlib.pyplot as plt
from collections import Counter
import datetime  # Dodany import

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
outliers = [f"{filename} (only {category})" for filename, category in file_data if category_counts.get(category, 0) == 1]
if outliers:
    with open('outliers.txt', 'a', encoding='utf-8') as outliers_file:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outliers_file.write(f"{now}\n")
        for outlier_info in outliers:
            print(outlier_info)
            outliers_file.write(f"{outlier_info}\n")
        outliers_file.write("\n\n\n")  # Oddzielenie dla następnego uruchomienia programu

plt.figure(figsize=(15, 8))  # Zwiększony rozmiar figury
plt.bar(sorted_categories.keys(), sorted_categories.values(), color='coral')
plt.xlabel('Klasy')
plt.ylabel('Liczba obrazów')
plt.title('Liczba obrazów dla każdej najczęstszej klasy')
plt.xticks(rotation=90, fontsize=10)  # Zmniejszony rozmiar czcionki etykiet
plt.tight_layout()
plt.savefig('top_class_counts.png', dpi=300)  # Zapisanie wykresu z wyższą rozdzielczością
plt.show()