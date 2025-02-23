# Projekt analizy danych z D435i

## Polecenie:
- Pobranie paczki z plikiem `d435i_walking.bag` zawierającym dane z:
   - `/device_0/sensor_2/Accel_0/imu/data`
   - `/device_0/sensor_2/Gyro_0/imu/data`
   - `/device_0/sensor_0/Depth_0/image/data`
   - `/device_0/sensor_1/Color_0/image/data`
- **1 pkt** – Otwarcie '.bag' i wczytanie powyższych topic'ów.  
- **1 pkt** – Wygenerować pseudorzut z góry w RGB.
- **1 pkt** – Wyznaczyć zerowe położenie kamery i trajektorię ruchu na podstawie danych IMU.
- **1 pkt** – Zwizualizować trajektorię jako wykres 3D na przygotowanym pseudorzucie.
- **1* pkt** – Użycie Open3D do wizualizacji.
- **3* pkt** – Użycie algorytmu SLAM do ulepszenia trajektorii i pseudorzutu.

## Instrukcje uruchomienia:
1. Upewnij się, że masz zainstalowane biblioteki: bagpy, pandas, numpy, scipy, opencv-python, tqdm, open3d.
2. Uruchom:
   ```
   python main.py
   ```

## Status realizacji:
- Otwarcie danych z paczki BAG i wczytanie tematów – wykonane.
- Generacja pseudorzutu RGB – wykonana.
- Wyznaczenie pozycji kamery i trajektorii – wykonane.
- Wizualizacja 3D przy użyciu Open3D – wykonane.
- Algorytm SLAM – nie zaimplementowano.

## Wnioski:
Podstawowe zadania związane z analizą danych i wizualizacją trajektorii zostały zrealizowane, jednak algorytm SLAM pozostaje do implementacji.
