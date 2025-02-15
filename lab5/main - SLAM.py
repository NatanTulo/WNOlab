import bagpy
from bagpy import bagreader
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import ast
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import g2o
from tqdm import tqdm  # Dodano import tqdm

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

# Przed przetwarzaniem danych, dodajemy kalibrację i filtrację
def calibrate_imu(accel_data, gyro_data, static_duration=100):
    # Używamy pierwszych n próbek do kalibracji
    accel_bias = np.mean(accel_data[:static_duration], axis=0)
    gyro_bias = np.mean(gyro_data[:static_duration], axis=0)
    
    # Odejmujemy bias
    accel_calibrated = accel_data - accel_bias
    gyro_calibrated = gyro_data - gyro_bias
    
    return accel_calibrated, gyro_calibrated

def lowpass_filter(data, alpha=0.1):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    return filtered_data

class PoseTracker:
    def __init__(self, dt):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)
        self.dt = dt
        self.velocity_threshold = 0.01  # m/s
        self.zero_velocity_count = 0
        self.zero_velocity_threshold = 5  # liczba próbek
        
    def update(self, angular_velocity, acceleration):
        # Aktualizacja orientacji
        rotation = Rotation.from_rotvec((angular_velocity * self.dt))
        self.orientation = rotation.apply(self.orientation)
        
        # Przekształcenie przyspieszenia do układu globalnego
        acceleration_global = self.orientation @ acceleration
        acceleration_global[2] += 9.81
        
        # Wykrywanie stanu spoczynku
        if np.linalg.norm(acceleration_global) < 0.1:  # Prawie brak przyspieszenia
            self.zero_velocity_count += 1
        else:
            self.zero_velocity_count = 0
            
        if self.zero_velocity_count > self.zero_velocity_threshold:
            self.velocity *= 0.5  # Stopniowe zmniejszanie prędkości
            
        # Ograniczenie maksymalnej prędkości
        self.velocity += acceleration_global * self.dt
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > 2.0:  # Maksymalna prędkość 2 m/s
            self.velocity = self.velocity * 2.0 / velocity_magnitude
            
        self.position += self.velocity * self.dt
        
        return self.position

class SLAMProcessor:
    def __init__(self):
        self.global_map = np.zeros((0, 3))  # 3D points in global map
        self.global_colors = np.zeros((0, 3))  # Colors for points
        self.poses = []  # Camera poses
        self.keyframes = []  # Selected keyframes
        self.loop_candidates = []
        
    def icp_registration(self, source, target, max_iterations=50, tolerance=1e-8):
        if len(source) < 3 or len(target) < 3:
            return np.eye(4), False
            
        # Initialize nearest neighbor search
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(target)
        
        # Initial transformation
        T = np.eye(4)
        
        for iteration in range(max_iterations):
            # Find nearest neighbors
            distances, indices = neigh.kneighbors(source)
            
            # Compute centroids
            source_centroid = np.mean(source, axis=0)
            target_centroid = np.mean(target[indices[:, 0]], axis=0)
            
            # Center point clouds
            source_centered = source - source_centroid
            target_centered = target[indices[:, 0]] - target_centroid
            
            # Compute optimal rotation
            H = source_centered.T @ target_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure right-handed coordinate system
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = target_centroid - R @ source_centroid
            
            # Update transformation
            T_current = np.eye(4)
            T_current[:3, :3] = R
            T_current[:3, 3] = t
            
            # Update source points
            source = (R @ source.T).T + t
            
            # Check convergence
            if np.mean(distances) < tolerance:
                break
                
        return T_current, True

class TopDownViewProcessor:
    def __init__(self, width, height):
        # D435i typical intrinsics
        self.width = width
        self.height = height
        self.fx = 386.952
        self.fy = 386.952
        self.cx = width / 2
        self.cy = height / 2
        
        # Modified parameters for better visibility
        self.top_down_size = 1200  # Increased size
        self.scale = 200  # Reduced scale to see more area
        self.top_down_view = np.zeros((self.top_down_size, self.top_down_size, 3), dtype=np.uint8)
        self.confidence_map = np.zeros((self.top_down_size, self.top_down_size), dtype=np.float32)
        
        # Adjusted parameters
        self.min_depth = 0.3
        self.max_depth = 5.0  # Increased max depth
        
        # Center position more to the right to show all data
        self.center_x = int(self.top_down_size * 0.5)  # Center horizontally
        self.center_y = int(self.top_down_size * 0.25)   # Move center up to show more of the scene
        self.slam = SLAMProcessor()
        self.previous_points = None
        self.previous_colors = None
        self.cumulative_transform = np.eye(4)
        
    def process_frame(self, color_frame, depth_frame, orientation):
        # Don't decay previous view - remove that line
        
        # Convert depth to meters
        depth_meters = depth_frame.astype(float) / 1000.0
        valid_depth = (depth_meters > self.min_depth) & (depth_meters < self.max_depth)
        
        # Generate point cloud
        rows, cols = depth_frame.shape
        pixel_coords = np.mgrid[0:rows, 0:cols].reshape(2, -1)
        z = depth_meters.reshape(-1)
        
        x = (pixel_coords[1] - self.cx) * z / self.fx
        y = (pixel_coords[0] - self.cy) * z / self.fy
        
        # Transform points using IMU orientation
        points = np.vstack((x, y, z))
        points = orientation.apply(points.T).T
        
        valid_points = valid_depth.reshape(-1) & ~np.isnan(points[2])
        x_proj = points[0][valid_points]
        z_proj = points[2][valid_points]
        y_proj = points[1][valid_points]
        
        # Tighter height filtering
        height_valid = (y_proj > -0.3) & (y_proj < 0.3)
        x_proj = x_proj[height_valid]
        z_proj = z_proj[height_valid]
        y_proj = y_proj[height_valid]  # Dodano filtrowanie y_proj
        colors = color_frame.reshape(-1, 3)[valid_points][height_valid]  # Dodano filtrowanie colors
        
        # Convert to image coordinates with proper orientation
        pixel_x = (-z_proj * self.scale + self.center_x).astype(int)  # Forward/backward
        pixel_z = (-x_proj * self.scale + self.center_y).astype(int)   # Left/right
        
        valid = (pixel_x >= 0) & (pixel_x < self.top_down_size) & \
                (pixel_z >= 0) & (pixel_z < self.top_down_size)
        
        # Create frame view with confidence-based weighting
        frame_view = np.zeros_like(self.top_down_view, dtype=np.float32)
        new_confidence = np.zeros_like(self.confidence_map)
        
        valid_pixels_z = pixel_z[valid]
        valid_pixels_x = pixel_x[valid]
        valid_colors = colors[valid]
        
        # Calculate confidence based on depth
        confidences = 1.0 / (z_proj[valid] + 0.1)
        confidences = confidences / np.max(confidences)  # Normalize confidences
        
        # Update pixels with confidence-based weights
        np.add.at(frame_view, (valid_pixels_z, valid_pixels_x), 
                 valid_colors * confidences[:, np.newaxis])
        np.add.at(new_confidence, (valid_pixels_z, valid_pixels_x), confidences)
        
        # Blend with existing view based on confidence
        mask = new_confidence > 0
        combined_confidence = np.maximum(self.confidence_map, new_confidence)
        
        # Where we have new data, blend based on relative confidence
        update_mask = mask & (self.confidence_map > 0)
        if np.any(update_mask):
            alpha = new_confidence[update_mask] / combined_confidence[update_mask]
            self.top_down_view[update_mask] = (
                (1 - alpha[:, np.newaxis]) * self.top_down_view[update_mask] +
                alpha[:, np.newaxis] * frame_view[update_mask] / 
                new_confidence[update_mask, np.newaxis]
            ).astype(np.uint8)
        
        # Where we only have new data, just add it
        new_only_mask = mask & (self.confidence_map == 0)
        if np.any(new_only_mask):
            self.top_down_view[new_only_mask] = (
                frame_view[new_only_mask] / new_confidence[new_only_mask, np.newaxis]
            ).astype(np.uint8)
        
        # Update confidence map
        self.confidence_map = combined_confidence
        
        # Create current frame point cloud
        valid_points = np.vstack((x_proj, y_proj, z_proj)).T
        current_colors = colors
        
        # Perform ICP if we have previous points
        if self.previous_points is not None:
            T, success = self.slam.icp_registration(valid_points, self.previous_points)
            if success:
                self.cumulative_transform = T @ self.cumulative_transform
                
                # Transform current points to global frame
                valid_points = (self.cumulative_transform[:3, :3] @ valid_points.T).T + self.cumulative_transform[:3, 3]
                
                # Add to global map with downsampling
                if len(self.slam.global_map) == 0:
                    self.slam.global_map = valid_points
                    self.slam.global_colors = current_colors
                else:
                    # Downsample and merge
                    merged_points = np.vstack((self.slam.global_map, valid_points))
                    merged_colors = np.vstack((self.slam.global_colors, current_colors))
                    
                    # Simple voxel grid downsampling
                    voxel_size = 0.05  # 5cm voxels
                    voxel_indices = np.floor(merged_points / voxel_size)
                    unique_indices, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
                    
                    self.slam.global_map = merged_points[unique_idx]
                    self.slam.global_colors = merged_colors[unique_idx]
        
        # Update previous points
        self.previous_points = valid_points
        self.previous_colors = current_colors
        
        # Project global map to top-down view
        if len(self.slam.global_map) > 0:
            global_x = self.slam.global_map[:, 0]
            global_z = self.slam.global_map[:, 2]
            
            pixel_x = (-global_z * self.scale + self.center_x).astype(int)
            pixel_z = (-global_x * self.scale + self.center_y).astype(int)
            
            valid = (pixel_x >= 0) & (pixel_x < self.top_down_size) & \
                    (pixel_z >= 0) & (pixel_z < self.top_down_size)
            
            # Update top-down view with global map
            self.top_down_view[pixel_z[valid], pixel_x[valid]] = self.slam.global_colors[valid]
        
        return self.top_down_view

# Przygotowanie danych
gyro_data = gyro_df[['angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']].values
accel_data = accel_df[['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']].values

# Kalibracja i filtracja
accel_calibrated, gyro_calibrated = calibrate_imu(accel_data, gyro_data)

# Aplikacja filtru dolnoprzepustowego
accel_filtered = np.apply_along_axis(lowpass_filter, 0, accel_calibrated)
gyro_filtered = np.apply_along_axis(lowpass_filter, 0, gyro_calibrated)

# Obliczenie dt (zakładając stałą częstotliwość próbkowania)
gyro_times = gyro_df['Time'].values
dt = np.mean(np.diff(gyro_times))

# Śledzenie pozycji
tracker = PoseTracker(dt)
positions = []

for i in range(len(gyro_filtered)):
    if i < len(accel_filtered):  # Upewniamy się, że mamy dane z obu czujników
        pos = tracker.update(gyro_filtered[i], accel_filtered[i])
        positions.append(pos.copy())

positions = np.array(positions)

# After loading the data, initialize the processor
# These are approximate values for D435i, adjust if needed
processor = TopDownViewProcessor(width=width, height=height)

# Process each frame
cumulative_rotation = Rotation.from_euler('xyz', [0, 0, 0])
top_down_views = []

# Procesowanie każdej klatki z paskiem postępu
for i in tqdm(range(len(depth_df)), desc="Przetwarzanie klatek"):
    if i >= len(color_df) or i >= len(gyro_df):
        break
        
    # Get depth frame
    depth_data = depth_df.iloc[i]['data']
    try:
        depth_bytes = ast.literal_eval(depth_data)
        if isinstance(depth_bytes, list):
            depth_bytes = bytes(depth_bytes)
    except:
        depth_bytes = depth_data.encode('latin-1')
    
    # Get depth frame dimensions
    depth_height = depth_df.iloc[i]['height']
    depth_width = depth_df.iloc[i]['width']
    
    # Reshape depth frame
    depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
    
    # Resize depth frame to match color frame dimensions
    depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Get color frame
    color_data = color_df.iloc[i]['data']
    try:
        color_bytes = ast.literal_eval(color_data)
        if isinstance(color_bytes, list):
            color_bytes = bytes(color_bytes)
    except:
        color_bytes = color_data.encode('latin-1')
    color_frame = np.frombuffer(color_bytes, dtype=np.uint8).reshape((height, width, 3))
    
    # Update orientation using gyroscope data
    angular_velocity = gyro_filtered[i]
    cumulative_rotation = cumulative_rotation * Rotation.from_rotvec(angular_velocity * dt)
    
    # Process frame
    top_down = processor.process_frame(color_frame, depth_frame, cumulative_rotation)
    top_down_views.append(top_down.copy())

# After processing frames, modify the final view processing
final_view = top_down_views[-1]

# Find non-zero regions with larger padding
nonzero_coords = np.nonzero(np.any(final_view > 0, axis=2))
if len(nonzero_coords[0]) > 0:
    min_row, max_row = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
    min_col, max_col = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])
    
    # Add padding proportional to the view size
    padding = int(final_view.shape[0] * 0.1)  # 10% padding
    min_row = max(0, min_row - padding)
    max_row = min(final_view.shape[0], max_row + padding * 3)  # More padding below
    min_col = max(0, min_col - padding)
    max_col = min(final_view.shape[1], max_col + padding)
    
    final_view = final_view[min_row:max_row, min_col:max_col]

# Adjusted image enhancement
final_view = cv2.GaussianBlur(final_view, (5, 5), 0)  # Increased blur radius
final_view = cv2.convertScaleAbs(final_view, alpha=1.5, beta=0)  # Adjusted contrast, removed brightness boost
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better detail visibility
lab = cv2.cvtColor(final_view, cv2.COLOR_RGB2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab[...,0] = clahe.apply(lab[...,0])
final_view = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

final_view = cv2.rotate(final_view, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees right

# Display with adjusted figure size and labels
plt.figure(figsize=(15, 15))
plt.imshow(final_view)
plt.title('Top-Down View of Environment')
plt.xlabel('← Left    Right →')
plt.ylabel('Forward →')
plt.axis('equal')

# Add grid for better spatial reference
plt.grid(True, alpha=0.3)

# Update video writer with rotation
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('top_down_view.mp4', fourcc, 30.0, (processor.top_down_size, processor.top_down_size))

for view in top_down_views:
    rotated_view = cv2.rotate(view, cv2.ROTATE_90_CLOCKWISE)
    out.write(cv2.cvtColor(rotated_view, cv2.COLOR_RGB2BGR))
out.release()

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
