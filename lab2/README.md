# Otoczka wypukła - Tygrysy

## Polecenie:
- Zadanie polega na implementacji algorytmu znajdowania otoczki wypukłej (covex hull) wokół punktów nazywanych tygrysami. Nie wolno używać innych bibliotek niż standardowe oraz numpy i matplotlib do wizualizacji (nie wolno też użyć żadnego gotowego algorytmu liczenia otoczki).
- Wygenerować losowo 20 punktów (tygrysów) na płaszczyźnie (x, y) z losowym kierunkiem (alpha).
- **1 pkt** – Zaimplementować algorytm Jarvisa lub Grahama do znalezienia otoczki wypukłej.
- **2 pkt** – Użycie klasy do implementacji zmodyfikowanych tygrysów.
- **1 pkt** – Dodanie aspektu czasu i ruchu tygrysów.
- **1 pkt** – Krokowe rozwijanie otoczki (wizualizacja etapów).
- **1* pkt** – Narysowanie odpowiednio zwróconego tygrysa z SVG.

## Instrukcje uruchomienia:
1. Uruchom skrypt `main.py`:
   ```
   python main.py
   ```
2. Wynik (otoczka wypukła) zostanie wyświetlony na wykresie przy użyciu matplotlib.

## Status realizacji:
- Losowe generowanie 20 punktów – wykonane.
- Algorytm Grahama zaimplementowany – wykonany (wybrano Grahama, a nie Jarvisa).
- Rozszerzenie danych o cechy dodatkowe – wykonane.
- Aspekt dynamiczny (ruch tygrysów) oraz krokowe rozwijanie otoczki – nie zaimplementowano.
- Wizualizacja SVG – nie zaimplementowano.
