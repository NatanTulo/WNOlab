from PIL import Image
from tqdm import tqdm
import cv2
from math import sqrt, pi, atan2
import time

def gaussian_kernel(size=5, sigma=1.4):
    # Generowanie kernela Gaussa używając list comprehension
    n = size // 2
    return [[1/(2*pi*sigma**2) * pow(2.718, -(x*x + y*y)/(2*sigma**2)) 
            for x in range(-n, n+1)] for y in range(-n, n+1)]

def convolve(image, kernel):
    height, width = len(image), len(image[0])
    k_size = len(kernel)
    offset = k_size // 2
    
    with tqdm(total=height-2*offset, desc="Applying convolution") as pbar:
        result = [[sum(kernel[i][j] * image[y-offset+i][x-offset+j] 
                  for i in range(k_size) for j in range(k_size))
                  for x in range(offset, width-offset)]
                  for y in range(offset, height-offset)]
        [pbar.update(1) for _ in range(height-2*offset)]
    return result

def sobel_filters():
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    return Kx, Ky

def non_maximum_suppression(mag, angle):
    height, width = len(mag), len(mag[0])
    # Utwórz najpierw pełną tablicę zer o wymiarach obrazu wejściowego
    result = [[0 for _ in range(width)] for _ in range(height)]
    
    # Wypełnij środkową część tablicy używając list comprehension dla wartości niezerowych
    inner_result = [[mag[y][x] if all([
            (mag[y][x] >= (mag[y][x+1] if (0 <= angle[y][x] % 180 < 22.5) or (157.5 <= angle[y][x] % 180 <= 180) else
                          mag[y+1][x-1] if 22.5 <= angle[y][x] % 180 < 67.5 else
                          mag[y+1][x] if 67.5 <= angle[y][x] % 180 < 112.5 else
                          mag[y-1][x-1])),
            (mag[y][x] >= (mag[y][x-1] if (0 <= angle[y][x] % 180 < 22.5) or (157.5 <= angle[y][x] % 180 <= 180) else
                          mag[y-1][x+1] if 22.5 <= angle[y][x] % 180 < 67.5 else
                          mag[y-1][x] if 67.5 <= angle[y][x] % 180 < 112.5 else
                          mag[y+1][x+1]))
            ]) else 0
            for x in range(1, width-1)]
            for y in range(1, height-1)]
    
    result = [[inner_result[y-1][x-1] if 0 < y < height-1 and 0 < x < width-1 else 0
              for x in range(width)]
             for y in range(height)]
    
    return result

def convert_to_bytes(matrix):
    height, width = len(matrix), len(matrix[0])
    with tqdm(total=height, desc="Converting to image") as pbar:
        byte_array = bytearray([min(255, max(0, int(matrix[y][x])))
                              for y in range(height)
                              for x in range(width)])
        [pbar.update(1) for _ in range(height)]
    return bytes(byte_array)

def canny_edge_detection(image_path, output_path, low_threshold=50, high_threshold=150):
    # Wczytaj obraz jako listę list
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = img.tolist()
    
    with tqdm(total=5, desc="Canny Edge Detection") as pbar:
        # 1. Gaussian Blur
        gauss_kernel = gaussian_kernel()
        blurred = convolve(image, gauss_kernel)
        pbar.update(1)
        
        # 2. Gradient calculation
        Kx, Ky = sobel_filters()
        Gx = convolve(blurred, Kx)
        Gy = convolve(blurred, Ky)
        pbar.update(1)
        
        # Oblicz magnitude i kąty
        height, width = len(Gx), len(Gx[0])
        magnitude = [[sqrt(Gx[y][x]**2 + Gy[y][x]**2) for x in range(width)] for y in range(height)]
        angle = [[atan2(Gy[y][x], Gx[y][x]) * 180/pi for x in range(width)] for y in range(height)]
        pbar.update(1)
        
        # 3. Non-maximum suppression
        suppressed = non_maximum_suppression(magnitude, angle)
        pbar.update(1)
        
        # 4. Double threshold
        result = [[255 if suppressed[y][x] >= high_threshold else
                  (128 if suppressed[y][x] >= low_threshold else 0)
                  for x in range(width)] for y in range(height)]
        pbar.update(1)
    
    # Konwertuj wynik na obraz PIL i zapisz
    result_bytes = convert_to_bytes(result)
    result_img = Image.frombytes('L', (width, height), result_bytes)
    result_img.save(output_path)

def prewitt_edge_detection(image_path, output_path):
    # Wczytaj obraz jako listę list
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = img.tolist()
    
    # Kernele Prewitta
    kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    kernel_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    
    with tqdm(total=3, desc="Prewitt Edge Detection") as pbar:
        # Aplikuj kernele
        Gx = convolve(image, kernel_x)
        pbar.update(1)
        
        Gy = convolve(image, kernel_y)
        pbar.update(1)
        
        # Oblicz magnitude używając list comprehension
        height, width = len(Gx), len(Gx[0])
        result = [[min(255, int(sqrt(Gx[y][x]**2 + Gy[y][x]**2)))
                  for x in range(width)] for y in range(height)]
        pbar.update(1)
    
    # Konwertuj wynik na obraz PIL i zapisz
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
