import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry
R = 10  # Promień dużego okręgu
r = 1  # Promień mniejszego okręgu
steps = 500  # Liczba kroków animacji
rotations = 2  # Liczba obrotów krzywej cyklometrycznej

# Generowanie danych
theta_mega = np.linspace(0, 2 * np.pi * rotations, steps)
theta_mini = np.linspace(0, 2 * np.pi * rotations, steps)
x_mega_outer = (R + r) * np.cos(theta_mega) - r * np.cos((R + r) / r * theta_mega)
y_mega_outer = (R + r) * np.sin(theta_mega) - r * np.sin((R + r) / r * theta_mega)

x_mini_outer = (R - r) * np.cos(theta_mini) + r * np.cos((R - r) / r * theta_mini)
y_mini_outer = (R - r) * np.sin(theta_mini) - r * np.sin((R - r) / r * theta_mini)

# Dane dla mniejszego koła
x_mega_circle = (R + r) * np.cos(theta_mega)
y_mega_circle = (R + r) * np.sin(theta_mega)

x_mini_circle = (R - r) * np.cos(theta_mini)
y_mini_circle = (R - r) * np.sin(theta_mini)

# Dane dla punktu na mniejszym okręgu
x_inner_mega_circle = x_mega_circle + r * np.cos(theta_mega)
y_inner_mega_circle = y_mega_circle + r * np.sin(theta_mega)

x_inner_mini_circle = x_mini_circle + r * np.cos(theta_mini)
y_inner_mini_circle = y_mini_circle + r * np.sin(theta_mini)

# Rysowanie
fig, ax = plt.subplots()
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')

# Elementy graficzne
main_circle, = ax.plot([], [], 'g--', lw=0.5)  # Duży okrąg
small_mega_circle, = ax.plot([], [], 'b--', lw=0.5)  # Mały okrąg
small_mini_circle, = ax.plot([], [], 'b--', lw=0.5)  # Mały okrąg
mega_radius_line, = ax.plot([], [], 'b-', lw=1)  # Promień wodzący
mini_radius_line, = ax.plot([], [], 'b-', lw=1)  # Promień wodzący
mega_curve, = ax.plot([], [], 'k-', lw=1)  # Ścieżka krzywej cyklometrycznej
mini_curve, = ax.plot([], [], 'k-', lw=1)

# Ścieżka krzywej
x_mega_curve, y_mega_curve = [], []
x_mini_curve, y_mini_curve = [], []


def init():
    main_circle.set_data([], [])
    small_mega_circle.set_data([], [])
    mega_radius_line.set_data([], [])
    mega_curve.set_data([], [])

    small_mini_circle.set_data([], [])
    mini_radius_line.set_data([], [])
    mini_curve.set_data([], [])

    return main_circle, small_mega_circle, mega_radius_line, mega_curve, small_mini_circle, mini_radius_line, mini_curve


def update(frame):
    if(frame == 1):
        x_mega_curve.clear()
        y_mega_curve.clear()
        x_mini_curve.clear()
        y_mini_curve.clear()
        
    # Zaktualizuj duży okrąg
    main_circle.set_data(R * np.cos(theta_mega), R * np.sin(theta_mega))

    # Zaktualizuj mały okrąg
    small_mega_circle.set_data(
        x_mega_circle[frame] + r * np.cos(theta_mega),
        y_mega_circle[frame] + r * np.sin(theta_mega)
    )
    small_mini_circle.set_data(
        x_mini_circle[frame] + r * np.cos(theta_mini),
        y_mini_circle[frame] - r * np.sin(theta_mini)
    )

    # Promień wodzący (od mniejszego okręgu do punktu stycznego)
    mega_radius_line.set_data(
        [x_mega_circle[frame], x_mega_outer[frame]], 
        [y_mega_circle[frame], y_mega_outer[frame]]
    )
    mini_radius_line.set_data(
        [x_mini_circle[frame], x_mini_outer[frame]], 
        [y_mini_circle[frame], y_mini_outer[frame]]
    )

    # Zaktualizuj krzywą
    x_mega_curve.append(x_mega_outer[frame])
    y_mega_curve.append(y_mega_outer[frame])
    mega_curve.set_data(x_mega_curve, y_mega_curve)

    x_mini_curve.append(x_mini_outer[frame])
    y_mini_curve.append(y_mini_outer[frame])
    mini_curve.set_data(x_mini_curve, y_mini_curve)

    return main_circle, small_mega_circle, mega_radius_line, mega_curve, small_mini_circle, mini_radius_line, mini_curve

ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=20, blit=True)
plt.show()