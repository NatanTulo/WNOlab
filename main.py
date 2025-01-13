from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import collections

# Load a model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
model = YOLO("yolo11x-cls.pt")  # load a detection model

def verbose_dict(result):
    data = {}
    probs = result.probs
    data['filename'] = result.path  # Dodanie nazwy pliku obrazu

    if len(result) == 0:
        if probs is not None:
            data['status'] = 'no detections'
        return data

    if probs is not None:
        data['probs'] = {model.names[int(j)]: float(probs.data[j]) for j in probs.top5}

    return data

results = model.predict("Final_images_dataset/*")

# Przetwarzanie wyników do słownika
verbose_data = [verbose_dict(result) for result in results]

# Agregacja prawdopodobieństw dla każdej klasy
class_probs = collections.defaultdict(float)
for data in verbose_data:
    if 'probs' in data:
        for cls, prob in data['probs'].items():
            class_probs[cls] += prob

# Sortowanie klas od największego do najmniejszego
sorted_class_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))

# Tworzenie wykresu słupkowego z posortowanymi klasami
plt.figure(figsize=(15, 8))  # Zwiększony rozmiar figury
plt.bar(sorted_class_probs.keys(), sorted_class_probs.values(), color='skyblue')
plt.xlabel('Klasy')
plt.ylabel('Suma prawdopodobieństw')
plt.title('Sumy prawdopodobieństw dla klas')
plt.xticks(rotation=90, fontsize=5)  # Zmniejszony rozmiar czcionki etykiet
plt.tight_layout()
plt.savefig('class_probabilities.png', dpi=300)  # Zapisanie wykresu z wyższą rozdzielczością
plt.show()

# Agregacja liczby obrazów dla każdej najczęstszej kategorii
top_class_counts = collections.defaultdict(int)
class_to_image = {}  # Słownik mapujący klasę do nazwy pliku obrazu
for data in verbose_data:
    if 'probs' in data:
        top_class = max(data['probs'], key=data['probs'].get)
        top_class_counts[top_class] += 1
        if top_class_counts[top_class] == 1:
            class_to_image[top_class] = data.get('filename', 'unknown')  # Zakładając, że 'filename' jest dostępny w 'data'

# Sortowanie klas od największego do najmniejszego
sorted_top_class_counts = dict(sorted(top_class_counts.items(), key=lambda item: item[1], reverse=True))

# Tworzenie wykresu słupkowego dla liczby obrazów w każdej kategorii
plt.figure(figsize=(15, 8))  # Zwiększony rozmiar figury
plt.bar(sorted_top_class_counts.keys(), sorted_top_class_counts.values(), color='coral')
plt.xlabel('Klasy')
plt.ylabel('Liczba obrazów')
plt.title('Liczba obrazów dla każdej najczęstszej klasy')
plt.xticks(rotation=90, fontsize=10)  # Zmniejszony rozmiar czcionki etykiet
plt.tight_layout()
plt.savefig('top_class_counts.png', dpi=300)  # Zapisanie wykresu z wyższą rozdzielczością
plt.show()

# Wypisanie kategorii z liczbą obrazów równą 1
single_image_classes = [cls for cls, count in top_class_counts.items() if count == 1]
for cls in single_image_classes:
    image_filename = class_to_image.get(cls, 'unknown')
    print(f"{image_filename}; {cls}; {top_class_counts[cls]}")

# Filtrowanie plików z kategorią 'tiger_shark'
tiger_shark_files = [
    data.get('filename', 'unknown') 
    for data in verbose_data 
    if 'probs' in data and max(data['probs'], key=data['probs'].get) == 'tiger_shark'
]
print("Pliki z kategorią 'tiger_shark':", tiger_shark_files)
