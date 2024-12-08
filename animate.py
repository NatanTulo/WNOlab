import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def funkcja(x):
    return 5*x**3 - 10*x**2 + np.sin(x) - 4

def funkcja_pochodna(x, h=1e-5):
    return (funkcja(x + h) - funkcja(x)) / h

# Przedział dla animacji
t1 = np.linspace(0, 2 * np.pi, 100)

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

# Rysowanie podstawowej funkcji
x_vals = np.linspace(-5, 5, 500)
y_vals = funkcja(x_vals)
ax.plot(x_vals, y_vals, 'b', label="Funkcja")

# Okrąg (będzie aktualizowany)
circle_line, = ax.plot([], [], 'r', label="Okrąg")

def init():
    circle_line.set_data([], [])
    return circle_line,

def update(frame):
    # Przekształcenie klatki w przedział (-0.6, 2.6)
    mapped_frame = map_value(frame, 0, 1, -5, 5)
    
    # Obliczenie punktu na funkcji
    y_func = funkcja(mapped_frame)
    
    # Obliczenie pochodnej i kąta
    derivative = funkcja_pochodna(mapped_frame)
    angle = np.arctan(derivative)
    
    # Parametry okręgu
    r = 1  # Promień okręgu
    x_circle = np.cos(t1) * r
    y_circle = np.sin(t1) * r
    
    # Obrót okręgu i przesunięcie do punktu na funkcji
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    rotated_circle = np.dot(rotation_matrix, np.array([x_circle, y_circle]))
    x_rotated, y_rotated = rotated_circle[0, :] + mapped_frame, rotated_circle[1, :] + y_func
    
    # Ustawienie danych okręgu
    circle_line.set_data(x_rotated, y_rotated)
    return circle_line,

# Animacja
ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 200), init_func=init, blit=True)
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(-20, 10)
plt.legend()
plt.show()
