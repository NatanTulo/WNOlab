import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math  # dodajemy import math

# Parametry
a = 1  # Długość boku trójkąta równobocznego
steps = 1000  # Liczba kroków animacji
interval = 10  # Interwał między klatkami w ms
first_point_x = -1  # Będzie aktualizowany w zależności od wybranej funkcji
rotation_point = 2  # Domyślnie zaczynamy od obrotu wokół punktu 2

triangle_state = None  # Przechowuje stan trójkąta w momencie kolizji (x1,y1,x2,y2,x3,y3)

# Dodaj po innych zmiennych globalnych
current_triangle_state = None  # Przechowuje aktualny stan trójkąta po każdym obrocie

def create_safe_function(expression):
    """Tworzy bezpieczną funkcję z wyrażenia wprowadzonego przez użytkownika"""
    global first_point_x
    
    # Dozwolone funkcje matematyczne
    safe_dict = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
        'abs': abs, 'pi': np.pi, 'e': np.e,
        'pow': np.power  # dodajemy funkcję potęgowania
    }
    
    # Zamiana wyrażenia e^x na exp(x)
    if 'e^' in expression:
        expression = expression.replace('e^', 'exp(') + ')'
    
    # Funkcja domyślna i jej parametry
    if not expression.strip():
        print("Użyję funkcji domyślnej.")
        first_point_x = -1
        return lambda x: 5 * x**3 - 10 * x**2 + np.sin(x) - 4
    
    try:
        # Tworzymy funkcję lambda z wyrażenia
        return lambda x: eval(expression, {"__builtins__": {}}, {**safe_dict, 'x': x, 'np': np})
    except:
        print("Błąd w wyrażeniu funkcji. Użyję funkcji domyślnej.")
        first_point_x = -1
        return lambda x: 5 * x**3 - 10 * x**2 + np.sin(x) - 4

# Pobieramy funkcję od użytkownika
print("Wprowadź funkcję matematyczną używając zmiennej 'x' (np. 'exp(x)' lub '5*x**3 - 10*x**2 + sin(x) - 4'):")
user_function = input()
f = create_safe_function(user_function)

# Pobieranie punktu startowego
print("\nCzy chcesz ustawić własny punkt startowy? (t/n):")
custom_start = input().lower().startswith('t')

if custom_start:
    try:
        print("Podaj współrzędną x punktu startowego:")
        first_point_x = float(input())
    except:
        print("Błędny format. Używam domyślnej wartości x = -1")
        first_point_x = -1

# Ustawiamy domyślne zakresy
if not user_function.strip():  # dla funkcji domyślnej
    x_min, x_max = -2, 3
    y_min, y_max = -15, 5
else:  # dla własnej funkcji
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10

# Pobieranie zakresów osi dla obu przypadków
print("\nCzy chcesz użyć własnych zakresów osi? (t/n):")
custom_range = input().lower().startswith('t')

if custom_range:
    try:
        print("Podaj zakres x (xmin xmax):")
        x_min, x_max = map(float, input().split())
        print("Podaj zakres y (ymin ymax):")
        y_min, y_max = map(float, input().split())
    except:
        print("Błędny format. Używam domyślnych zakresów.")
        # Zachowujemy już ustawione zakresy

# Dane funkcji
x_vals = np.linspace(x_min, x_max, 5000)  # Gęsta siatka dla precyzyjnych obliczeń
y_vals = f(x_vals)

# Rysowanie
fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')

# Elementy graficzne
ax.plot(x_vals, y_vals, 'g-', lw=0.5)  # Funkcja bazowa
point1, = ax.plot([], [], 'ro')  # wierzchołek trójkąta
point2, = ax.plot([], [], 'g.')  # wierzchołek trójkąta
point3, = ax.plot([], [], 'b.')  # wierzchołek trójkąta
side1, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
side2, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
side3, = ax.plot([], [], 'b-', lw=1)  # Bok trójkąta
radius_line, = ax.plot([], [], 'k-', lw=1)  # Promień wodzący
curve, = ax.plot([], [], 'k-', lw=1)  # Epicykloida

# Ścieżka epicykloidy
x_curve, y_curve = [], []

def init():
    x1, y1 = first_point_x, f(first_point_x)
    x1, y1, x2, y2, x3, y3 = find_triangle(first_point_x, a, x_vals, y_vals)
    
    point1.set_data([x1], [y1])
    point2.set_data([x2], [y2])
    point3.set_data([x3], [y3])
    side1.set_data([x1, x2], [y1, y2])
    side2.set_data([x2, x3], [y2, y3])
    side3.set_data([x3, x1], [y3, y1])
    radius_line.set_data([(x3+x2)/2,x1], [(y3+y2)/2,y1])
    curve.set_data([], [])
    return point1, point2, point3, side1, side2, side3, radius_line, curve

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

def find_closest_point_on_function(x, y, window=0.1):
    # Szukamy najbliższego punktu w oknie wokół x
    x_window = np.linspace(x - window, x + window, 1000)
    y_window = f(x_window)
    
    # Obliczamy odległości do wszystkich punktów w oknie
    distances = np.sqrt((x_window - x)**2 + (y_window - y)**2)
    
    # Znajdujemy indeks najbliższego punktu
    closest_idx = np.argmin(distances)
    return  distances[closest_idx]

def is_point_on_function(x, y, tolerance=0.1):
    # Znajdujemy najbliższy punkt na funkcji i odległość do niego
    min_distance = find_closest_point_on_function(x, y)

    is_close = min_distance < tolerance
    
    print(f"Odległość od funkcji: {min_distance}, isSkok: {min_distance<tolerance}")
    return is_close

def check_collision_with_function(x1, y1, x2, y2, x3, y3):
    global rotation_point
    
    # Sprawdzamy tylko jeden konkretny punkt w zależności od aktualnego punktu obrotu
    if rotation_point == 2 and is_point_on_function(x3, y3):
        print("Kolizja z punktem 3")
        return 3
    elif rotation_point == 3 and is_point_on_function(x1, y1):
        print("Kolizja z punktem 1")
        return 1
    elif rotation_point == 1 and is_point_on_function(x2, y2):
        print("Kolizja z punktem 2")
        return 2
    
    return None

def update(frame):
    global first_point_x, rotation_point, triangle_state, current_triangle_state
    if frame == 0:
        x_curve.clear()
        y_curve.clear()
        rotation_point = 2
        triangle_state = None
        current_triangle_state = None
    
    if current_triangle_state is None:
        x1, y1, x2, y2, x3, y3 = find_triangle(first_point_x, a, x_vals, y_vals)
    else:
        x1, y1, x2, y2, x3, y3 = current_triangle_state
        
    # Stały przyrost kąta w każdej klatce
    angle_increment = -2 * np.pi / 100  # Pełny obrót co 100 klatek
    
    # Wykonujemy obrót wokół odpowiedniego punktu używając przyrostu kąta
    if rotation_point == 1:
        x2, y2 = rotate_point(x2, y2, x1, y1, angle_increment)
        x3, y3 = rotate_point(x3, y3, x1, y1, angle_increment)
    elif rotation_point == 2:
        x1, y1 = rotate_point(x1, y1, x2, y2, angle_increment)
        x3, y3 = rotate_point(x3, y3, x2, y2, angle_increment)
    elif rotation_point == 3:
        x1, y1 = rotate_point(x1, y1, x3, y3, angle_increment)
        x2, y2 = rotate_point(x2, y2, x3, y3, angle_increment)
    
    # Aktualizujemy aktualny stan trójkąta po każdym obrocie
    current_triangle_state = (x1, y1, x2, y2, x3, y3)
    
    # Sprawdzamy kolizję i ustawiamy nowy punkt obrotu
    collision_point = check_collision_with_function(x1, y1, x2, y2, x3, y3)
    if collision_point is not None:
        rotation_point = collision_point
        triangle_state = current_triangle_state
    
    side1.set_data([x1, x2], [y1, y2])
    side2.set_data([x2, x3], [y2, y3])
    side3.set_data([x3, x1], [y3, y1])
    
    point1.set_data([x1], [y1])
    point2.set_data([x2], [y2])
    point3.set_data([x3], [y3])

    radius_line.set_data([(x3+x2)/2,x1], [(y3+y2)/2,y1])
    
    x_curve.append(x1)
    y_curve.append(y1)
    curve.set_data(x_curve, y_curve)
    print(f'frame: {frame}')
    return point1, point2, point3, side1, side2, side3, radius_line, curve

ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval, blit=True)
plt.show()