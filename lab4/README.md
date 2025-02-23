# Sprawozdanie z zadania

## Polecenie:
- Na podstawie cech obrazu, bądź abstrakcyjnych cech wyciągniętych z sieci neuronowych należy:
- **1 pkt** – znaleźć liczbę kategorii obrazów w danym datasecie.
- **0.5 pkt/poprawny obraz** – Znaleźć obrazy nie pasujące (każdy jedyny w swoim rodzaju) do danego datasetu zgodne z listą poniżej:
- wno_04.jpg (jedyny stadion), wno_06.jpg (jedyny w skali szarości), wno_08.jpg (jedyny człowiek z bronią), wno_09.jpg (jedyny zachód słońca), wno_87.jpg (jedyne buty na zdjęciu), wno_88.jpg (jedyne anime), wno_15.jpeg (jedyny picasso w secie), wno_89.jpg (jedyny obraz olejny)
- **-0.5 pkt/niepoprawny obraz** – Jedno zdjęcie jest dopuszczalne jako niepoprawny outlayer, za każde następne punkty będą odejmowane.

## Instrukcje uruchomienia:
1. Umieść obrazy w folderze `./Final_images_dataset`.
2. Uruchom skrypt(y) znajdujące się w katalogu `lab4` (np. `main.py`, `plot.py`).
3. Wyniki zapisywane są w plikach: opisy_blip.txt, kategorie.txt, outliers.txt oraz w folderze `wyniki`.

## Status realizacji:
- Wyodrębnianie kategorii – wykonane.
- Detekcja outlierów oraz system punktacji – wykonane - max 6.

# Podsumowanie
- Analiza opisów obrazów i wyodrębnianie kategorii.
- Wykrywanie outlierów na podstawie unikalności cech/opisów.

