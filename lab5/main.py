# Importy bibliotek
from bagpy import bagreader
import pandas as pd
import os
import numpy as np
import ast
from scipy.spatial.transform import Rotation
import cv2
from tqdm import tqdm
import open3d as o3d

"""
Program do analizy danych z kamery Intel RealSense D435i
Przetwarza dane z akcelerometru, żyroskopu oraz kamery RGB-D
Tworzy mapę otoczenia wraz z wizualizacją trajektorii ruchu
"""

# 1. WCZYTYWANIE DANYCH
b = bagreader('data/d435i_walking.bag')

print(b.topic_table)

with open('topics.txt', 'w') as f:
    for topic in b.topic_table['Topics']:
        f.write(topic + "\n")

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

try:
    data_bytes = ast.literal_eval(data)
    if isinstance(data_bytes, list):
        data_bytes = bytes(data_bytes)
except:
    data_bytes = data.encode('latin-1')

img_array = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height, width, 3))

# 2. DEFINICJE FUNKCJI I KLAS
def calibrate_imu(accel_data, gyro_data, static_duration=100):
    """Kalibracja danych IMU poprzez usunięcie biasu"""
    accel_bias = np.mean(accel_data[:static_duration], axis=0)
    gyro_bias = np.mean(gyro_data[:static_duration], axis=0)
    
    accel_calibrated = accel_data - accel_bias
    gyro_calibrated = gyro_data - gyro_bias
    
    return accel_calibrated, gyro_calibrated

def lowpass_filter(data, alpha=0.1):
    """Filtr dolnoprzepustowy dla danych sensorycznych"""
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    return filtered_data

# 3. KLASY PRZETWARZAJĄCE DANE
class PoseTracker:
    """Śledzenie pozycji na podstawie danych IMU"""
    def __init__(self, dt):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)
        self.dt = dt
        self.velocity_threshold = 0.01
        self.zero_velocity_count = 0
        self.zero_velocity_threshold = 5
        
    def update(self, angular_velocity, acceleration):
        rotation = Rotation.from_rotvec((angular_velocity * self.dt))
        self.orientation = rotation.apply(self.orientation)
        
        acceleration_global = self.orientation @ acceleration
        acceleration_global[2] += 9.81
        
        if np.linalg.norm(acceleration_global) < 0.1:
            self.zero_velocity_count += 1
        else:
            self.zero_velocity_count = 0
            
        if self.zero_velocity_count > self.zero_velocity_threshold:
            self.velocity *= 0.5
            
        self.velocity += acceleration_global * self.dt
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 2.0:
            self.velocity = self.velocity * 2.0 / velocity_magnitude
            
        self.position += self.velocity * self.dt
        
        return self.position

class TopDownViewProcessor:
    """Przetwarzanie widoku z góry na podstawie danych z kamery"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fx = 386.952
        self.fy = 386.952
        self.cx = width / 2
        self.cy = height / 2
        
        self.top_down_size = 1000
        self.scale = 200
        self.top_down_view = np.zeros((self.top_down_size, self.top_down_size, 3), dtype=np.uint8)
        
        self.center_x = None
        self.center_y = None
        
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_z = float('inf')
        self.max_z = float('-inf')
        
        self.min_depth = 0.3
        self.max_depth = 5.0
        self.accumulation_count = np.zeros((self.top_down_size, self.top_down_size), dtype=np.float32)

    def process_frame(self, color_frame, depth_frame, orientation):
        depth_meters = depth_frame.astype(float) / 1000.0
        valid_depth = (depth_meters > self.min_depth) & (depth_meters < self.max_depth)
        
        rows, cols = depth_frame.shape
        pixel_coords = np.mgrid[0:rows, 0:cols].reshape(2, -1)
        z = depth_meters.reshape(-1)
        
        x = (pixel_coords[1] - self.cx) * z / self.fx
        y = (pixel_coords[0] - self.cy) * z / self.fy
        
        points = np.vstack((x, y, z))
        points = orientation.apply(points.T).T
        
        valid_points = valid_depth.reshape(-1) & ~np.isnan(points[2])
        x_proj = points[0][valid_points]
        z_proj = points[2][valid_points]
        y_proj = points[1][valid_points]
        
        if len(x_proj) > 0:
            self.min_x = min(self.min_x, x_proj.min())
            self.max_x = max(self.max_x, x_proj.max())
            self.min_z = min(self.min_z, z_proj.min())
            self.max_z = max(self.max_z, z_proj.max())
            
            if self.center_x is None:
                self.center_x = self.top_down_size // 2
                self.center_y = self.top_down_size // 2
            
            target_x = (self.min_z + self.max_z) / 2
            target_y = -(self.min_x + self.max_x) / 2
            
            self.center_x = int(self.top_down_size // 2 - target_x * self.scale)
            self.center_y = int(self.top_down_size // 2 - target_y * self.scale)
        
        height_valid = (y_proj > -1.0) & (y_proj < 1.0)
        x_proj = x_proj[height_valid]
        z_proj = z_proj[height_valid]
        
        colors = color_frame.reshape(-1, 3)[valid_points][height_valid]
        
        pixel_x = (z_proj * self.scale + self.center_x).astype(int)
        pixel_z = (-x_proj * self.scale + self.center_y).astype(int)
        
        valid = (pixel_x >= 0) & (pixel_x < self.top_down_size) & \
                (pixel_z >= 0) & (pixel_z < self.top_down_size)
        
        frame_view = np.zeros_like(self.top_down_view, dtype=np.float32)
        accumulation = np.zeros((self.top_down_size, self.top_down_size), dtype=np.float32)
        
        valid_pixels_z = pixel_z[valid]
        valid_pixels_x = pixel_x[valid]
        valid_colors = colors[valid]
        
        np.add.at(frame_view, (valid_pixels_z, valid_pixels_x), valid_colors)
        np.add.at(accumulation, (valid_pixels_z, valid_pixels_x), 1)
        
        mask = accumulation > 0
        frame_view[mask] = frame_view[mask] / accumulation[mask, np.newaxis]
        
        self.accumulation_count += accumulation
        mask = self.accumulation_count > 0
        self.top_down_view[mask] = ((self.top_down_view[mask].astype(np.float32) * 
                                   (self.accumulation_count[mask, np.newaxis] - accumulation[mask, np.newaxis]) + 
                                   frame_view[mask] * accumulation[mask, np.newaxis]) / 
                                   self.accumulation_count[mask, np.newaxis]).astype(np.uint8)
        
        return self.top_down_view

# 4. PRZETWARZANIE DANYCH IMU
gyro_data = gyro_df[['angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']].values
accel_data = accel_df[['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].values

accel_calibrated, gyro_calibrated = calibrate_imu(accel_data, gyro_data)

accel_filtered = np.apply_along_axis(lowpass_filter, 0, accel_calibrated)
gyro_filtered = np.apply_along_axis(lowpass_filter, 0, gyro_calibrated)

gyro_times = gyro_df['Time'].values
dt = np.mean(np.diff(gyro_times))

tracker = PoseTracker(dt)
positions = []

for i in range(len(gyro_filtered)):
    if i < len(accel_filtered):
        pos = tracker.update(gyro_filtered[i], accel_filtered[i])
        positions.append(pos.copy())

positions = np.array(positions)

# 5. PRZETWARZANIE KLATEK OBRAZU
processor = TopDownViewProcessor(width=width, height=height)

cumulative_rotation = Rotation.from_euler('xyz', [0, 0, 0])
top_down_views = []

for i in tqdm(range(len(depth_df)), desc="Przetwarzanie klatek"):
    if i >= len(color_df) or i >= len(gyro_df):
        break
        
    depth_data = depth_df.iloc[i]['data']
    try:
        depth_bytes = ast.literal_eval(depth_data)
        if isinstance(depth_bytes, list):
            depth_bytes = bytes(depth_bytes)
    except:
        depth_bytes = depth_data.encode('latin-1')
    
    depth_height = depth_df.iloc[i]['height']
    depth_width = depth_df.iloc[i]['width']
    
    depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
    
    depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    
    color_data = color_df.iloc[i]['data']
    try:
        color_bytes = ast.literal_eval(color_data)
        if isinstance(color_bytes, list):
            color_bytes = bytes(color_bytes)
    except:
        color_bytes = color_data.encode('latin-1')
    color_frame = np.frombuffer(color_bytes, dtype=np.uint8).reshape((height, width, 3))
    
    angular_velocity = gyro_filtered[i]
    cumulative_rotation = cumulative_rotation * Rotation.from_rotvec(angular_velocity * dt)
    
    top_down = processor.process_frame(color_frame, depth_frame, cumulative_rotation)
    top_down_views.append(top_down.copy())

# 6. PRZYGOTOWANIE WIZUALIZACJI
final_view = top_down_views[-1]

nonzero_coords = np.nonzero(np.any(final_view > 0, axis=2))
if len(nonzero_coords[0]) > 0:
    min_row, max_row = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
    min_col, max_col = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])
    
    padding = 50
    min_row = max(0, min_row - padding)
    max_row = min(final_view.shape[0], max_row + padding)
    min_col = max(0, min_col - padding)
    max_col = min(final_view.shape[1], max_col + padding)
    
    final_view = final_view[min_row:max_row, min_col:max_col]

final_view = cv2.GaussianBlur(final_view, (3, 3), 0)
final_view = cv2.convertScaleAbs(final_view, alpha=1.5, beta=10)

positions[:, 2] = positions[:, 2] / 75

max_range = np.array([
    positions[:, 0].max() - positions[:, 0].min(),
    positions[:, 1].max() - positions[:, 1].min(),
    positions[:, 2].max() - positions[:, 2].min()
]).max() / 2.0

mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

# 7. TWORZENIE GEOMETRII 3D
trajectory = o3d.geometry.LineSet()
points = o3d.utility.Vector3dVector(positions)
lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(positions)-1)])
trajectory.points = points
trajectory.lines = lines
trajectory.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(lines))])

start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
start_sphere.translate(positions[0])
start_sphere.paint_uniform_color([0, 1, 0])

end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
end_sphere.translate(positions[-1])
end_sphere.paint_uniform_color([1, 0, 0])

vertices = []
triangles = []
colors = []

rows, cols = final_view.shape[:2]
for i in range(rows):
    for j in range(cols):
        x = mid_x - max_range + (2 * max_range * j / (cols-1))
        y = mid_y - max_range + (2 * max_range * i / (rows-1))
        z = max_range/2
        vertices.append([x, y, z])
        colors.append(final_view[i, j] / 255.0)

for i in range(rows-1):
    for j in range(cols-1):
        v0 = i * cols + j
        v1 = v0 + 1
        v2 = (i + 1) * cols + j
        v3 = v2 + 1
        triangles.extend([[v0, v2, v1], [v1, v2, v3]])

floor_mesh = o3d.geometry.TriangleMesh()
floor_mesh.vertices = o3d.utility.Vector3dVector(vertices)
floor_mesh.triangles = o3d.utility.Vector3iVector(triangles)
floor_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# 8. KONFIGURACJA I URUCHOMIENIE WIZUALIZACJI
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(trajectory)
vis.add_geometry(start_sphere)
vis.add_geometry(end_sphere)
vis.add_geometry(floor_mesh)
vis.add_geometry(coord_frame)

ctr = vis.get_view_control()
ctr.set_zoom(0.5)
ctr.set_front([0, 0, -1])
ctr.set_lookat([mid_x, mid_y, 0])
ctr.set_up([0, 1, 0])

vis.run()
vis.destroy_window()

# 9. WYŚWIETLENIE STATYSTYK
print(f"Całkowita przebyta odległość: {np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))):.2f} m")
print(f"Przemieszczenie (w linii prostej): {np.sqrt(np.sum((positions[-1] - positions[0])**2)):.2f} m")
