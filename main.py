import bagpy
from bagpy import bagreader
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import ast
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D

b = bagreader('data/d435i_walking.bag')

# get the list of topics
print(b.topic_table)

# Save topic paths to a text file
with open('topics.txt', 'w') as f:
    for topic in b.topic_table['Topics']:
        f.write(topic + "\n")

# Odczyt danych z przyspieszeń
if not os.path.isfile('data/d435i_walking/device_0-sensor_2-Accel_0-imu-data.csv'):
    print("Plik z akcelerometru nie istnieje.")
    accel_msgs = b.message_by_topic('/device_0/sensor_2/Accel_0/imu/data')
    if accel_msgs:
        try:
            accel_df = pd.read_csv(accel_msgs)
        except FileNotFoundError:
            print("Brak pliku akcelerometru.")
    else:
        print("Brak danych w temacie Accelerometer.")
else:
    print("Dane z akcelerometru już istnieją.")
    accel_df = pd.read_csv('data/d435i_walking/device_0-sensor_2-Accel_0-imu-data.csv')

# Odczyt danych z żyroskopu
if not os.path.isfile('data/d435i_walking/device_0-sensor_2-Gyro_0-imu-data.csv'):
    print("Plik z żyroskopu nie istnieje.")
    gyro_msgs = b.message_by_topic('/device_0/sensor_2/Gyro_0/imu/data')
    if gyro_msgs:
        try:
            gyro_df = pd.read_csv(gyro_msgs)
        except FileNotFoundError:
            print("Brak pliku żyroskopu.")
    else:
        print("Brak danych w temacie Gyroscope.")
else:
    print("Dane z żyroskopu już istnieją.")
    gyro_df = pd.read_csv('data/d435i_walking/device_0-sensor_2-Gyro_0-imu-data.csv')

# Odczyt danych z kamery głębi
if not os.path.isfile('data/d435i_walking/device_0-sensor_0-Depth_0-image-data.csv'):
    print("Plik z głębi nie istnieje.")
    depth_msgs = b.message_by_topic('/device_0/sensor_0/Depth_0/image/data')
    if depth_msgs:
        try:
            depth_df = pd.read_csv(depth_msgs)
        except FileNotFoundError:
            print("Brak pliku głębi.")
    else:
        print("Brak danych w temacie Depth Camera.")
else:
    print("Dane z kamery głębi już istnieją.")
    depth_df = pd.read_csv('data/d435i_walking/device_0-sensor_0-Depth_0-image-data.csv')

# Odczyt danych z kamery kolorowej
if not os.path.isfile('data/d435i_walking/device_0-sensor_1-Color_0-image-data.csv'):
    print("Plik z koloru nie istnieje.")
    color_msgs = b.message_by_topic('/device_0/sensor_1/Color_0/image/data')
    if color_msgs:
        try:
            color_df = pd.read_csv(color_msgs)
        except FileNotFoundError:
            print("Brak pliku koloru.")
    else:
        print("Brak danych w temacie Color Camera.")
else:
    print("Dane z kamery kolorowej już istnieją.")
    color_df = pd.read_csv('data/d435i_walking/device_0-sensor_1-Color_0-image-data.csv')

first = color_df.head(1)
width = first['width'][0]
height = first['height'][0]
data = first['data'][0]

# Convert string representation of bytes to actual bytes
try:
    # Try to evaluate the string as a literal
    data_bytes = ast.literal_eval(data)
    # Convert to bytes if it's a list of integers
    if isinstance(data_bytes, list):
        data_bytes = bytes(data_bytes)
except:
    # If the data is already in the correct format, use it directly
    data_bytes = data.encode('latin-1')

# Create the image array from bytes
img_array = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height, width, 3))

fig, ax = plt.subplots()
ax.imshow(img_array)
ax.set_title("Pseudorzut z góry (RGB) z color_df")
plt.show()

class PoseTracker:
    def __init__(self, dt):
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)  # Macierz rotacji
        self.dt = dt
        
    def update(self, angular_velocity, acceleration):
        # Aktualizacja orientacji przez małe obroty
        rotation = Rotation.from_rotvec((angular_velocity * self.dt))
        self.orientation = rotation.apply(self.orientation)
        
        # Usunięcie grawitacji i przekształcenie przyspieszenia do układu globalnego
        acceleration_global = self.orientation @ acceleration
        acceleration_global[2] += 9.81  # Kompensacja grawitacji
        
        # Całkowanie przyspieszenia
        self.velocity += acceleration_global * self.dt
        self.position += self.velocity * self.dt
        
        return self.position

# Przygotowanie danych
gyro_data = gyro_df[['angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']].values
accel_data = accel_df[['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].values

# Obliczenie dt (zakładając stałą częstotliwość próbkowania)
gyro_times = gyro_df['Time'].values
dt = np.mean(np.diff(gyro_times))

# Śledzenie pozycji
tracker = PoseTracker(dt)
positions = []

for i in range(len(gyro_data)):
    if i < len(accel_data):  # Upewniamy się, że mamy dane z obu czujników
        pos = tracker.update(gyro_data[i], accel_data[i])
        positions.append(pos.copy())

positions = np.array(positions)

# Wizualizacja trajektorii 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Trajektoria kamery')

# Dodanie punktu początkowego i końcowego
ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', s=100, label='Start')
ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', s=100, label='End')
ax.legend()

plt.show()

# Wyświetlenie podstawowych statystyk
print(f"Całkowita przebyta odległość: {np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))):.2f} m")
print(f"Przemieszczenie (w linii prostej): {np.sqrt(np.sum((positions[-1] - positions[0])**2)):.2f} m")