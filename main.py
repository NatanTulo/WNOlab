from PIL import Image
from tqdm import tqdm
import cv2
from math import sqrt, pi, exp, atan2
import time
from concurrent.futures import ThreadPoolExecutor

def gaussian_kernel(size=5, sigma=1.4):
    n = size // 2
    return [[1/(2*pi*sigma**2) * exp(-(x*x + y*y)/(2*sigma**2)) 
            for x in range(-n, n+1)] 
            for y in range(-n, n+1)]

def convolve_chunk(args):
    chunk, kernel, start_y, width, k_size, offset = args
    return [[sum(kernel[i][j] * chunk[i][x-offset+j]
                for i in range(k_size)
                for j in range(k_size))
             for x in range(offset, width-offset)]
            for y in range(len(chunk))]

def convolve(image, kernel):
    height, width = len(image), len(image[0])
    k_size = len(kernel)
    offset = k_size // 2
    
    num_threads = 8
    chunk_size = (height - 2*offset) // num_threads
    
    chunks = [(image[i*chunk_size+offset-offset : (i*chunk_size+chunk_size+offset if i < num_threads-1 else height-offset)+offset],
               kernel, i*chunk_size+offset, width, k_size, offset)
              for i in range(num_threads)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(convolve_chunk, chunks))
    
    return [row for chunk_result in results for row in chunk_result]

def sobel_filters():
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    return Kx, Ky

def parallel_gradient_calculation(image, Kx, Ky):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_x = executor.submit(convolve, image, Kx)
        future_y = executor.submit(convolve, image, Ky)
        Gx, Gy = future_x.result(), future_y.result()
    return Gx, Gy

def non_maximum_suppression(mag, angle):
    height, width = len(mag), len(mag[0])
    angle_mod = [[a % 180 for a in row] for row in angle]
    
    def get_neighbors(y, x):
        curr_angle = angle_mod[y][x]
        return (
            [mag[y][x-1], mag[y][x+1]] if (curr_angle < 22.5 or curr_angle >= 157.5) else
            [mag[y-1][x+1], mag[y+1][x-1]] if (22.5 <= curr_angle < 67.5) else
            [mag[y-1][x], mag[y+1][x]] if (67.5 <= curr_angle < 112.5) else
            [mag[y-1][x-1], mag[y+1][x+1]]
        )
    
    return [[mag[y][x] if (0 < y < height-1 and 0 < x < width-1 and 
                          mag[y][x] >= max(get_neighbors(y, x))) else 0
             for x in range(width)]
            for y in range(height)]

def convert_to_bytes(matrix):
    return bytes(bytearray([min(255, max(0, int(val))) 
                           for row in matrix 
                           for val in row]))

def canny_edge_detection(image_path, output_path, low_threshold=50, high_threshold=150):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = img.tolist()
    
    with tqdm(total=5, desc="Canny Edge Detection") as pbar:
        # 1. Gaussian Blur
        gauss_kernel = gaussian_kernel()
        blurred = convolve(image, gauss_kernel)
        pbar.update(1)
        
        # 2. Gradient calculation
        Kx, Ky = sobel_filters()
        Gx, Gy = parallel_gradient_calculation(blurred, Kx, Ky)
        pbar.update(1)
        
        # Oblicz magnitude i kąty używając list comprehension
        height, width = len(Gx), len(Gx[0])
        magnitude = [[sqrt(Gx[y][x]**2 + Gy[y][x]**2) 
                     for x in range(width)] 
                     for y in range(height)]
        angle = [[atan2(Gy[y][x], Gx[y][x]) * 180/pi 
                 for x in range(width)] 
                 for y in range(height)]
        pbar.update(1)
        
        # 3. Non-maximum suppression
        suppressed = non_maximum_suppression(magnitude, angle)
        pbar.update(1)
        
        # 4. Double threshold
        result = [[255 if suppressed[y][x] >= high_threshold else
                  (128 if suppressed[y][x] >= low_threshold else 0)
                  for x in range(len(suppressed[0]))] for y in range(len(suppressed))]
        pbar.update(1)
    
    # Konwertuj wynik na obraz PIL i zapisz
    result_bytes = convert_to_bytes(result)
    result_img = Image.frombytes('L', (len(result[0]), len(result)), result_bytes)
    result_img.save(output_path)

def prewitt_edge_detection(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = img.tolist()
    
    kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    kernel_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    
    with tqdm(total=3, desc="Prewitt Edge Detection") as pbar:
        Gx, Gy = parallel_gradient_calculation(image, kernel_x, kernel_y)
        pbar.update(2)
        
        # Oblicz magnitude używając list comprehension
        height, width = len(Gx), len(Gx[0])
        result = [[min(255, int(sqrt(Gx[y][x]**2 + Gy[y][x]**2)))
                  for x in range(width)]
                  for y in range(height)]
        pbar.update(1)
        
        result_bytes = convert_to_bytes(result)
        result_img = Image.frombytes('L', (width, height), result_bytes)
        result_img.save(output_path)

if __name__ == "__main__":
    start_time = time.time()
    
    prewitt_edge_detection("pg.jpg", "pg_edges_prewitt.jpg")
    canny_edge_detection("pg.jpg", "pg_edges_canny.jpg")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nCałkowity czas wykonania: {execution_time:.2f} sekund")
