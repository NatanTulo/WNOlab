from bagpy import bagreader
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import ast
from scipy.spatial.transform import Rotation
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import g2o
from tqdm import tqdm  # Add this import for progress bars
from scipy.spatial import cKDTree  # Add this import
import time
from multiprocessing import Pool, cpu_count
import signal
import sys

# Add this function at the start of the file
def load_data(bag_file):
    b = bagreader(bag_file)
    data = {}
    
    # Load accelerometer data
    if os.path.isfile('data/d435i_walking/device_0-sensor_2-Accel_0-imu-data.csv'):
        print("Dane z akcelerometru już istnieją.")
        data['accel_df'] = pd.read_csv('data/d435i_walking/device_0-sensor_2-Accel_0-imu-data.csv')
    else:
        print("Wczytuję dane z akcelerometru...")
        accel_msgs = b.message_by_topic('/device_0/sensor_2/Accel_0/imu/data')
        data['accel_df'] = pd.read_csv(accel_msgs)

    # Similar for other sensors...
    # Odczyt danych z żyroskopu
    if not os.path.isfile('data/d435i_walking/device_0-sensor_2-Gyro_0-imu-data.csv'):
        print("Plik z żyroskopu nie istnieje.")
        gyro_msgs = b.message_by_topic('/device_0/sensor_2/Gyro_0/imu/data')
        if gyro_msgs:
            try:
                data['gyro_df'] = pd.read_csv(gyro_msgs)
            except FileNotFoundError:
                print("Brak pliku żyroskopu.")
        else:
            print("Brak danych w temacie Gyroscope.")
    else:
        print("Dane z żyroskopu już istnieją.")
        data['gyro_df'] = pd.read_csv('data/d435i_walking/device_0-sensor_2-Gyro_0-imu-data.csv')

    # Odczyt danych z kamery głębi
    if not os.path.isfile('data/d435i_walking/device_0-sensor_0-Depth_0-image-data.csv'):
        print("Plik z głębi nie istnieje.")
        depth_msgs = b.message_by_topic('/device_0/sensor_0/Depth_0/image/data')
        if depth_msgs:
            try:
                data['depth_df'] = pd.read_csv(depth_msgs)
            except FileNotFoundError:
                print("Brak pliku głębi.")
        else:
            print("Brak danych w temacie Depth Camera.")
    else:
        print("Dane z kamery głębi już istnieją.")
        data['depth_df'] = pd.read_csv('data/d435i_walking/device_0-sensor_0-Depth_0-image-data.csv')

    # Odczyt danych z kamery kolorowej
    if not os.path.isfile('data/d435i_walking/device_0-sensor_1-Color_0-image-data.csv'):
        print("Plik z koloru nie istnieje.")
        color_msgs = b.message_by_topic('/device_0/sensor_1/Color_0/image/data')
        if color_msgs:
            try:
                data['color_df'] = pd.read_csv(color_msgs)
            except FileNotFoundError:
                print("Brak pliku koloru.")
        else:
            print("Brak danych w temacie Color Camera.")
    else:
        print("Dane z kamery kolorowej już istnieją.")
        data['color_df'] = pd.read_csv('data/d435i_walking/device_0-sensor_1-Color_0-image-data.csv')

    return data

def process_frame_parallel(args):
    global width, height, dt, processor  # This won't work in multiprocessing
    # Instead, unpack needed values from args
    i, (depth_data, color_data, angular_velocity, frame_width, frame_height, time_dt) = args
    # Get depth frame
    try:
        depth_bytes = ast.literal_eval(depth_data['data'])
        if isinstance(depth_bytes, list):
            depth_bytes = bytes(depth_bytes)
    except:
        depth_bytes = depth_data['data'].encode('latin-1')
    
    # Get depth frame dimensions and reshape
    depth_height = depth_data['height']
    depth_width = depth_data['width']
    depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((depth_height, depth_width))
    
    # Resize depth frame to match color frame dimensions
    depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Process color frame
    try:
        color_bytes = ast.literal_eval(color_data['data'])
        if isinstance(color_bytes, list):
            color_bytes = bytes(color_bytes)
    except:
        color_bytes = color_data['data'].encode('latin-1')
    color_frame = np.frombuffer(color_bytes, dtype=np.uint8).reshape((height, width, 3))
    
    # Update orientation and process frame
    cumulative_rotation = Rotation.from_rotvec(angular_velocity * dt)
    
    top_down = processor.process_frame(color_frame, depth_frame, cumulative_rotation)
    return top_down

class ProcessWrapper:
    def __init__(self, width, height):
        self.processor = TopDownViewProcessor(width, height)
        self.width = width
        self.height = height
        self.cumulative_rotation = Rotation.from_euler('xyz', [0, 0, 0])
        
    def get_color_frame(self, color_data, i):
        """Helper method to handle color frame extraction"""
        try:
            # Get frame dimensions from data
            frame_width = int(color_data['width'])
            frame_height = int(color_data['height'])
            expected_size = frame_width * frame_height * 3
            
            # Get the raw data
            try:
                if isinstance(color_data['data'], str):
                    color_bytes = ast.literal_eval(color_data['data'])
                    if isinstance(color_bytes, list):
                        color_bytes = bytes(color_bytes)
                    else:
                        color_bytes = color_data['data'].encode('latin-1')
                else:
                    color_bytes = color_data['data']
                    
                # Get actual size and pad/truncate if needed
                actual_size = len(color_bytes)
                if actual_size % 3 != 0:
                    # Ensure size is multiple of 3 (RGB)
                    color_bytes = color_bytes[:(actual_size - (actual_size % 3))]
                    actual_size = len(color_bytes)
                
                # Check if we have enough data for a full frame
                required_pixels = frame_width * frame_height
                actual_pixels = actual_size // 3
                
                if actual_pixels < required_pixels:
                    # Pad with black pixels
                    padding_size = (required_pixels - actual_pixels) * 3
                    color_bytes = color_bytes + b'\0' * padding_size
                elif actual_pixels > required_pixels:
                    # Truncate to exact frame size
                    color_bytes = color_bytes[:required_pixels * 3]
                
                # Reshape to RGB image
                color_array = np.frombuffer(color_bytes, dtype=np.uint8)
                color_frame = color_array.reshape((frame_height, frame_width, 3))
                
                # Resize if needed
                if (frame_width, frame_height) != (self.width, self.height):
                    color_frame = cv2.resize(color_frame, (self.width, self.height), 
                                          interpolation=cv2.INTER_LINEAR)
                
                return color_frame
                
            except Exception as e:
                print(f"Error parsing color data for frame {i}: {str(e)}")
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error in get_color_frame for frame {i}: {str(e)}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def get_depth_frame(self, depth_data, i):
        """Helper method to handle depth frame extraction"""
        try:
            depth_height = int(depth_data['height'])
            depth_width = int(depth_data['width'])
            
            try:
                depth_bytes = ast.literal_eval(depth_data['data'])
                if isinstance(depth_bytes, list):
                    depth_bytes = bytes(depth_bytes)
                else:
                    depth_bytes = depth_data['data'].encode('latin-1')
                
                depth_frame = np.frombuffer(depth_bytes, dtype=np.uint16)
                expected_size = depth_width * depth_height
                if depth_frame.size < expected_size:
                    print(f"Depth frame {i} ma niepoprawny rozmiar ({depth_frame.size} zamiast {expected_size}). Pomijam.")
                    return None
                depth_frame = depth_frame[:expected_size].reshape((depth_height, depth_width))
                
                # Resize to match required dimensions
                if (depth_width, depth_height) != (self.width, self.height):
                    depth_frame = cv2.resize(depth_frame, (self.width, self.height), 
                                          interpolation=cv2.INTER_NEAREST)
                
                return depth_frame
                
            except Exception as e:
                print(f"Error parsing depth data for frame {i}: {str(e)}")
                return np.zeros((self.height, self.width), dtype=np.uint16)
                
        except Exception as e:
            print(f"Error in get_depth_frame for frame {i}: {str(e)}")
            return np.zeros((self.height, self.width), dtype=np.uint16)

    def process_single_frame(self, args):
        i, (depth_data, color_data, angular_velocity, dt) = args
        try:
            # Get frames using helper methods
            depth_frame = self.get_depth_frame(depth_data, i)
            if depth_frame is None:
                return None
            color_frame = self.get_color_frame(color_data, i)
            
            if depth_frame is None or color_frame is None:
                return None
                
            # Update orientation and process frame
            self.cumulative_rotation = self.cumulative_rotation * Rotation.from_rotvec(angular_velocity * dt)
            
            return self.processor.process_frame(color_frame, depth_frame, self.cumulative_rotation)
            
        except Exception as e:
            print(f"Error in process_single_frame {i}: {str(e)}")
            return None

class ProcessManager:
    def __init__(self):
        self.pool = None
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        print("\nReceived Ctrl+C. Cleaning up...")
        if self.pool:
            self.pool.terminate()
            self.pool.join()
        sys.exit(0)
        
    def process_frames(self, frame_data, total_frames):
        try:
            self.pool = Pool(processes=max(1, cpu_count()-1))
            results = list(tqdm(
                self.pool.imap(process_frame_parallel, frame_data),
                total=total_frames,
                desc="Processing frames"
            ))
            self.pool.close()
            self.pool.join()
            return results
        except KeyboardInterrupt:
            print("\nInterrupted by user. Cleaning up...")
            if self.pool:
                self.pool.terminate()
                self.pool.join()
            sys.exit(0)
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            if self.pool:
                self.pool.terminate()
                self.pool.join()
            raise

if __name__ == '__main__':
    # Load data once at the start
    data = load_data('data/d435i_walking.bag')
    
    # Extract dataframes from loaded data
    accel_df = data['accel_df']
    gyro_df = data['gyro_df']
    depth_df = data['depth_df']
    color_df = data['color_df']

    # Get frame dimensions first
    first = color_df.head(1)
    width = first['width'][0]
    height = first['height'][0]
    
    # Now print dimensions with all information available
    print("\nLoaded data dimensions:")
    print(f"Depth frames: {len(depth_df)}")
    print(f"Color frames: {len(color_df)}")
    print(f"IMU data points: {len(gyro_df)}")
    print(f"Frame dimensions: {width}x{height}")

    # Get first frame data
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

    # fig, ax = plt.subplots()
    # ax.imshow(img_array)
    # ax.set_title("Widok pierwszej klatki")
    # plt.show()

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
            self.tree = None
            self.max_points = 1000  # Limit points for ICP
            self.voxel_size = 0.1  # Increased voxel size for faster processing
            
        def downsample_points(self, points, colors):
            # Voxel grid downsampling
            voxel_indices = np.floor(points / self.voxel_size)
            unique_indices, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
            
            # Average points and colors in each voxel
            downsampled_points = np.zeros((len(unique_indices), 3))
            downsampled_colors = np.zeros((len(unique_indices), 3))
            np.add.at(downsampled_points, inverse, points)
            np.add.at(downsampled_colors, inverse, colors)
            counts = np.bincount(inverse)
            downsampled_points /= counts[:, np.newaxis]
            downsampled_colors /= counts[:, np.newaxis]
            
            return downsampled_points.astype(np.float32), downsampled_colors.astype(np.uint8)
            
        def icp_registration(self, source, target, max_iterations=20, tolerance=1e-4):
            if len(source) < 3 or len(target) < 3:
                return np.eye(4), False
                
            # Downsample points for faster ICP
            source = source[np.random.choice(len(source), min(self.max_points, len(source)), replace=False)]
            target = target[np.random.choice(len(target), min(self.max_points, len(target)), replace=False)]
            
            # Use KD-tree for faster neighbor search
            tree = cKDTree(target)
            T = np.eye(4)
            
            for iteration in range(max_iterations):
                # Find nearest neighbors using KD-tree
                distances, indices = tree.query(source, k=1)
                
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
            self.last_update_time = time.time()
            self.update_interval = 0.1  # Update display every 0.1 seconds
            print(f"Initialized processor with size {self.top_down_size}x{self.top_down_size}")
            
        def process_frame(self, color_frame, depth_frame, orientation):
            if depth_frame is None or depth_frame.size == 0:
                print("Niepoprawna ramka głębi. Pomijam przetwarzanie.")
                return self.top_down_view
            try:
                # Downsample input images for faster processing
                scale_factor = 2
                small_depth = cv2.resize(depth_frame, (depth_frame.shape[1]//scale_factor, 
                                                    depth_frame.shape[0]//scale_factor))
                small_color = cv2.resize(color_frame, (color_frame.shape[1]//scale_factor, 
                                                    color_frame.shape[0]//scale_factor))
                
                # Convert depth to meters
                depth_meters = small_depth.astype(float) / 1000.0
                valid_depth = (depth_meters > self.min_depth) & (depth_meters < self.max_depth)
                
                # Generate point cloud
                rows, cols = small_depth.shape
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
                y_proj = y_proj[height_valid]  # Make sure y_proj is filtered too
                colors = small_color.reshape(-1, 3)[valid_points][height_valid]
                
                # Jeśli nie ma żadnych punktów, pomijamy
                if len(x_proj) == 0:
                    print("Brak poprawnych punktów w tej ramce. Pomijam.")
                    return self.top_down_view

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
                # Jeśli confidences puste, pomijamy
                if len(confidences) == 0:
                    print("Brak zaufania do punktów w tej ramce. Pomijam.")
                    return self.top_down_view
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
                valid_points = np.vstack((x_proj, y_proj, z_proj)).T  # Now all arrays should have same first dimension
                current_colors = colors
                
                # Perform ICP if we have previous points
                if self.previous_points is not None and len(valid_points) > 0:
                    print(f"Processing points: current={len(valid_points)}, previous={len(self.previous_points)}")
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
                
                # Update visualization less frequently
                current_time = time.time()
                if current_time - self.last_update_time > self.update_interval:
                    self.last_update_time = current_time
                    # Update visualization code here
                
                return self.top_down_view
                
            except Exception as e:
                print(f"Error in process_frame: {str(e)}")
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
    print(f"Loaded data dimensions:")
    print(f"Depth frames: {len(depth_df)}")
    print(f"Color frames: {len(color_df)}")
    print(f"IMU data points: {len(gyro_df)}")
    print(f"Frame dimensions: {width}x{height}")

    # Initialize processor and variables
    processor = TopDownViewProcessor(width=width, height=height)
    total_frames = min(len(depth_df), len(color_df), len(gyro_df))
    cumulative_rotation = Rotation.from_euler('xyz', [0, 0, 0])
    top_down_views = []

    print("Starting frame processing...")

    # Replace the parallel processing section with:
    process_manager = ProcessManager()
    
    # Prepare data for parallel processing
    frame_data = [(i, (depth_df.iloc[i], color_df.iloc[i], gyro_filtered[i])) 
                  for i in range(total_frames)]
    
    # Initialize processor wrapper
    process_wrapper = ProcessWrapper(width, height)
    
    # Prepare data for parallel processing with additional parameters
    frame_data = [(i, (depth_df.iloc[i], color_df.iloc[i], gyro_filtered[i], dt)) 
                  for i in range(total_frames)]
    
    try:
        print("Starting frame processing...")
        # Process frames sequentially instead of in parallel
        top_down_views = []
        for args in tqdm(frame_data, desc="Processing frames"):
            view = process_wrapper.process_single_frame(args)
            if view is not None:  # Only append valid frames
                top_down_views.append(view.copy())
        print("Frame processing complete")
        
        if not top_down_views:
            print("No frames were processed successfully")
            sys.exit(1)
            
        # Continue with the rest of the processing...
        # After processing frames, modify the final view processing
        if top_down_views:
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
            plt.show()
        else:
            print("No frames were processed successfully")

        # Update video writer with rotation
        if top_down_views:
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
        
    except KeyboardInterrupt:
        print("\nExiting program...")
        sys.exit(0)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup code
        cv2.destroyAllWindows()
        plt.close('all')
