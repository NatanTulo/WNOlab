import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def funkcja(x):
    return 5*x*x*x - 10*x*x + np.sin(x) - 4

def funkcja_pochodna(x, h=1e-5):
    return (funkcja(x + h) - funkcja(x)) / h

t0 = np.arange(-5.0, 5.0, 0.02)
t1 = np.arange(0.0, 1.2, 0.02)

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

# Rysowanie podstawowej funkcji
line0 = ax.plot(t0, funkcja(t0), 'b')

# Okrąg
line1, = ax.plot([], [], 'k')

# Inicjalizacja pustych linii
line2, = ax.plot([], [], 'k')
line3, = ax.plot([], [], 'r--')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

def update(frame):
    mapped_frame = map_value(frame, 0, 1, -0.6, 2.6) # skrajne wartości X dla widoku funkcji
    
    # Obliczenie punktu na funkcji
    y_func = funkcja(mapped_frame)
    
    # Obliczenie pochodnej funkcji w danym punkcie
    derivative = funkcja_pochodna(mapped_frame)
    
    # Kąt styczny do funkcji
    angle = np.arctan(derivative)
    
    # Promień okręgu
    r = 0.5
    
    # Obliczenie współrzędnych okręgu
    x1 = np.sin(2*np.pi*t1) * r
    y1 = np.cos(2*np.pi*t1) * r
    
    # Obrót i translacja okręgu
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    rotated_points = np.dot(np.column_stack([x1, y1]), rotation_matrix)
    
    # Przesunięcie okręgu do punktu na funkcji
    perpendicular_angle = angle + np.pi/2
    x_offset = r * np.cos(perpendicular_angle)
    y_offset = r * np.sin(perpendicular_angle)
    
    x1 = rotated_points[:, 0] + mapped_frame + x_offset
    y1 = rotated_points[:, 1] + y_func + y_offset
    
    line1.set_data(x1, y1)
    
    return line1, line2, line3

ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 100), init_func=init, blit=True)
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-10, 10)
plt.show()