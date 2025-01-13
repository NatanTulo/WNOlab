import bagpy
from bagpy import bagreader
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

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
