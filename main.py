from PIL import Image
from tqdm import tqdm
import cv2

def prewitt_edge_detection(image_path, output_path):
    # Wczytaj obraz i przekonwertuj na skalę szarości
    img = Image.open(image_path).convert('L')
    width, height = img.size
    
    # Przygotuj nowy obraz na wynik
    edge_img = Image.new('L', (width, height))
    
    # Kernele Prewitta
    kernel_x = [[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]]
    
    kernel_y = [[-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]]
    
    # Iteruj przez piksele z paskiem postępu
    pixels = img.load()
    edge_pixels = edge_img.load()
    
    for y in tqdm(range(1, height-1), desc="Wykrywanie krawędzi"):
        for x in range(1, width-1):
            px = py = 0
            
            # Aplikuj kernel
            for i in range(3):
                for j in range(3):
                    val = pixels[x + i - 1, y + j - 1]
                    px += kernel_x[i][j] * val
                    py += kernel_y[i][j] * val
            
            # Oblicz magnitude
            magnitude = min(255, int((px * px + py * py) ** 0.5))
            edge_pixels[x, y] = magnitude
    
    # Zapisz wynik
    edge_img.save(output_path)

def canny_edge_detection(image_path, output_path, threshold1=100, threshold2=200):
    # Wczytaj obraz w skali szarości przy pomocy OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Zastosuj rozmycie Gaussa aby zredukować szum
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Zastosuj detekcję krawędzi Canny
    with tqdm(total=1, desc="Wykrywanie krawędzi (Canny)") as pbar:
        edges = cv2.Canny(blurred, threshold1, threshold2)
        pbar.update(1)
    
    # Zapisz wynik
    cv2.imwrite(output_path, edges)

if __name__ == "__main__":
    prewitt_edge_detection("pg.jpg", "pg_edges_prewitt.jpg")
    canny_edge_detection("pg.jpg", "pg_edges_canny.jpg")
