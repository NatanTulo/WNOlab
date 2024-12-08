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
point2, = ax.plot([], [], 'ro')  # wierzchołek trójkąta
point3, = ax.plot([], [], 'ro')  # wierzchołek trójkąta
side1, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
side2, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
side3, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
radius_line, = ax.plot([], [], 'b-', lw=1)  # Promień wodzący
curve, = ax.plot([], [], 'k-', lw=1)  # Epicykloida

# Ścieżka epicykloidy
x_curve, y_curve = [], []

def init():
    x1, y1 = first_point_x, f(first_point_x)
    point1.set_data([x1], [y1])
    point2.set_data([], [])
    point3.set_data([], [])
    side1.set_data([], [])
    side2.set_data([], [])
    side3.set_data([], [])
    radius_line.set_data([], [])
    curve.set_data([], [])
    return point1, point2, point3, radius_line, curve

def find_next_x(x1, y1, a, x_vals, y_vals):
    distances = np.sqrt((x_vals - x1)**2 + (y_vals - y1)**2)
    closest_idx = np.abs(distances - a).argmin()
    return x_vals[closest_idx]

def update(frame):
    global first_point_x
    if frame == 0:
        x_curve.clear()
        y_curve.clear()
    
    # Aktualizacja pierwszego punktu
    x1, y1 = first_point_x, f(first_point_x)
    point1.set_data([x1], [y1])
    
    # Znalezienie drugiego punktu w odległości 'a' w linii prostej
    x2 = find_next_x(x1, y1, a, x_vals, y_vals)
    y2 = f(x2)
    # point2.set_data([x2], [y2])

    # Obliczenie wektora kierunkowego prostej point1-point2
    dx = x2 - x1
    dy = y2 - y1
    
    # Obrót wektora o 90 stopni (prostopadły)
    perpendicular_dx = -dy
    perpendicular_dy = dx
    
    # Normalizacja wektora prostopadłego
    length = np.sqrt(perpendicular_dx**2 + perpendicular_dy**2)
    perpendicular_dx /= length
    perpendicular_dy /= length
    
    # Wyznaczenie punktu na prostej prostopadłej
    x3 = (x1+x2)/2 - perpendicular_dx * a*np.sqrt(3)/2
    y3 = (y1+y2)/2 - perpendicular_dy * a*np.sqrt(3)/2
    # point3.set_data([x3], [y3])

    # Rysowanie linii prostopadłej
    # radius_line.set_data([x3, x4], [y3, y4])
    
    # Rysowanie linii między punktami
    side1.set_data([x1, x2], [y1, y2])
    side2.set_data([x2, x3], [y2, y3])
    side3.set_data([x3, x1], [y3, y1])
    # print(np.sqrt((x1-x2)**2 + (y1-y2)**2), np.sqrt((x2-x3)**2 + (y2-y3)**2), np.sqrt((x3-x1)**2 + (y3-y1)**2))
    # first_point_x = x2
    
    return point1, point2, point3, radius_line, curve

ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval, blit=True)
plt.show()