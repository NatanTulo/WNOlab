import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# Parametry
r = 0.5  # Promień okręgu
steps = 500  # Liczba kroków animacji
interval = 20  # Interwał między klatkami w ms
total_time = steps * interval / 1000  # Całkowity czas animacji w sekundach

# Funkcja bazowa
def f(x):
    return 5 * x**3 - 10 * x**2 + np.sin(x) - 4

# Pochodna funkcji
def df(x):
    return 15 * x**2 - 20 * x + np.cos(x)

# Dane funkcji
x_vals = np.linspace(-1, 2.4, 5000)  # Gęsta siatka dla precyzyjnych obliczeń
y_vals = f(x_vals)

# Obliczenie długości łuku
dx = np.diff(x_vals)
dy = np.diff(y_vals)
ds = np.sqrt(dx**2 + dy**2)
s_vals = np.concatenate(([0], np.cumsum(ds)))  # Długość łuku jako funkcja x

# Funkcje interpolacyjne
x_of_s = interp1d(s_vals, x_vals)
y_of_s = interp1d(s_vals, y_vals)
df_of_s = interp1d(s_vals, df(x_vals))

# Całkowita długość łuku
total_length = s_vals[-1]

# Prędkość kątowa
omega = (total_length / r) / total_time

# Czas dla każdej klatki
t_vals = np.linspace(0, total_time, steps)

# Kąt toczenia i długość łuku w czasie
theta_vals = -omega * t_vals
s_t = r * (-theta_vals)

# Pozycja na krzywej
x_outer = x_of_s(s_t)
y_outer = y_of_s(s_t)

# Kąt normalnej do krzywej
normals = np.arctan(df_of_s(s_t))

# Pozycja środka okręgu
x_circle = x_outer - r * np.sin(normals)
y_circle = y_outer + r * np.cos(normals)

# Pozycja punktu na obracającym się okręgu
x_inner_circle = x_circle + r * np.cos(theta_vals)
y_inner_circle = y_circle + r * np.sin(theta_vals)

# Rysowanie
fig, ax = plt.subplots()
ax.set_xlim(-2, 3)
ax.set_ylim(-15, 5)
ax.set_aspect('equal')

# Elementy graficzne
ax.plot(x_vals, y_vals, 'g-', lw=0.5)  # Funkcja bazowa
small_circle, = ax.plot([], [], 'b--', lw=0.5)  # Okrąg toczący się
point_inner, = ax.plot([], [], 'ro')  # Punkt na obrzeżu okręgu
radius_line, = ax.plot([], [], 'b-', lw=1)  # Promień wodzący
curve, = ax.plot([], [], 'k-', lw=1)  # Epicykloida

# Ścieżka epicykloidy
x_curve, y_curve = [], []

def init():
    small_circle.set_data([], [])
    point_inner.set_data([], [])
    radius_line.set_data([], [])
    curve.set_data([], [])
    return small_circle, point_inner, radius_line, curve

def update(frame):
    if frame == 0:
        x_curve.clear()
        y_curve.clear()
    # Aktualny kąt i pozycje
    theta = theta_vals[frame]
    x_c = x_circle[frame]
    y_c = y_circle[frame]
    
    # Zaktualizuj mały okrąg
    theta_full = np.linspace(0, 2 * np.pi, 100)
    small_circle.set_data(
        x_c + r * np.cos(theta_full),
        y_c + r * np.sin(theta_full)
    )
    # Punkt na obrzeżu okręgu
    x_p = x_inner_circle[frame]
    y_p = y_inner_circle[frame]
    point_inner.set_data([x_p], [y_p])
    # Promień wodzący
    radius_line.set_data([x_c, x_p], [y_c, y_p])
    # Zaktualizuj epicykloidę
    x_curve.append(x_p)
    y_curve.append(y_p)
    curve.set_data(x_curve, y_curve)
    return small_circle, point_inner, radius_line, curve

ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval, blit=True)
plt.show()
