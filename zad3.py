import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry
a = 1  # Długość boku trójkąta równobocznego
r = a*np.sqrt(3)/3  # Promień okręgu opisanego na trójkącie
steps = 500  # Liczba kroków animacji
interval = 20  # Interwał między klatkami w ms
total_time = steps * interval / 1000  # Całkowity czas animacji w sekundach
first_point_x = -0.7

# Funkcja bazowa
def f(x):
    return 5 * x**3 - 10 * x**2 + np.sin(x) - 4

# Dane funkcji
x_vals = np.linspace(-1, 2.4, 5000)  # Gęsta siatka dla precyzyjnych obliczeń
y_vals = f(x_vals)

# Czas dla każdej klatki
t_vals = np.linspace(0, total_time, steps)

# Rysowanie
fig, ax = plt.subplots()
ax.set_xlim(-2, 3)
ax.set_ylim(-15, 5)
ax.set_aspect('equal')

# Elementy graficzne
ax.plot(x_vals, y_vals, 'g-', lw=0.5)  # Funkcja bazowa
point1, = ax.plot([], [], 'ro')  # wierzchołek trójkąta
point2, = ax.plot([], [], 'go')  # wierzchołek trójkąta
point3, = ax.plot([], [], 'bo')  # wierzchołek trójkąta
side1, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
side2, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
side3, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
radius_line, = ax.plot([], [], 'b-', lw=1)  # Promień wodzący
curve, = ax.plot([], [], 'k-', lw=1)  # Epicykloida

# Ścieżka epicykloidy
x_curve, y_curve = [], []

def init():
    x1, y1 = first_point_x, f(first_point_x)
    x1, y1, x2, y2, x3, y3 = find_triangle(first_point_x, a, x_vals, y_vals)
    
    point1.set_data([x1], [y1])
    point2.set_data([x2], [y2])
    # point3.set_data([x3], [y3])
    side1.set_data([x1, x2], [y1, y2])
    side2.set_data([x2, x3], [y2, y3])
    side3.set_data([x3, x1], [y3, y1])
    radius_line.set_data([], [])
    curve.set_data([], [])
    return point1, side1, side2, side3, radius_line, curve, point2#, point3

def find_next_x(x1, y1, a, x_vals, y_vals):
    # Znajdujemy tylko punkty na prawo od x1
    mask = x_vals > x1
    x_vals_right = x_vals[mask]
    y_vals_right = y_vals[mask]
    
    # Obliczamy odległości tylko dla punktów po prawej stronie
    distances = np.sqrt((x_vals_right - x1)**2 + (y_vals_right - y1)**2)
    closest_idx = np.abs(distances - a).argmin()
    
    return x_vals_right[closest_idx]

def find_triangle(x1, a, x_vals, y_vals):
    # Aktualizacja pierwszego punktu
    y1 = f(x1)
    
    # Znalezienie drugiego punktu w odległości 'a' w linii prostej
    x2 = find_next_x(x1, y1, a, x_vals, y_vals)
    y2 = f(x2)

    # Obliczenie wektora kierunkowego prostej point1-point2
    dx = x2 - x1
    dy = y2 - y1
    
    # Obrót wektora o 90 stopni (prostopadły)
    perpendicular_dx = dy  
    perpendicular_dy = -dx 
    
    # Normalizacja wektora prostopadłego
    length = np.sqrt(perpendicular_dx**2 + perpendicular_dy**2)
    perpendicular_dx /= length
    perpendicular_dy /= length
    
    # Wyznaczenie punktu na prostej prostopadłej
    x3 = (x1+x2)/2 + perpendicular_dx * a*np.sqrt(3)/2 
    y3 = (y1+y2)/2 + perpendicular_dy * a*np.sqrt(3)/2
    
    # Sprawdzenie czy punkt jest poniżej funkcji i odbicie jeśli tak
    y3_function = f(x3)
    if y3 < y3_function:  # jeśli punkt jest poniżej funkcji
        # Odbijamy punkt względem prostej point1-point2
        x3 = (x1+x2)/2 - perpendicular_dx * a*np.sqrt(3)/2
        y3 = (y1+y2)/2 - perpendicular_dy * a*np.sqrt(3)/2
    
    return x1, y1, x2, y2, x3, y3

def rotate_point(x, y, cx, cy, angle):
    # Przesunięcie punktu względem środka obrotu
    translated_x = x - cx
    translated_y = y - cy
    
    # Obrót
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated_x = translated_x * cos_a - translated_y * sin_a
    rotated_y = translated_x * sin_a + translated_y * cos_a
    
    return rotated_x + cx, rotated_y + cy

def update(frame):
    global first_point_x
    if frame == 0:
        x_curve.clear()
        y_curve.clear()
    
    # Znajdujemy pierwotne położenie trójkąta
    x1, y1, x2, y2, x3, y3 = find_triangle(first_point_x, a, x_vals, y_vals)
    
    # Obliczamy kąt obrotu (frame określa postęp animacji)
    angle = -(2 * np.pi * frame) / steps
    
    # Obracamy punkty 1 i 3 wokół punktu 2
    x1, y1 = rotate_point(x1, y1, x2, y2, angle)
    x3, y3 = rotate_point(x3, y3, x2, y2, angle)
    
    # Rysowanie linii między punktami
    side1.set_data([x1, x2], [y1, y2])
    side2.set_data([x2, x3], [y2, y3])
    side3.set_data([x3, x1], [y3, y1])
    
    point1.set_data([x1], [y1])
    point2.set_data([x2], [y2])
    # point3.set_data([x3], [y3])
    
    x_curve.append(x1)
    y_curve.append(y1)
    curve.set_data(x_curve, y_curve)
    print(x1,x2)
    
    return point1, side1, side2, side3, radius_line, curve , point2#, point3

ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval, blit=True)
plt.show()