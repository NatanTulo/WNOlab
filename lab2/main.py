import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alpha = np.random.uniform(0, 360)
        self.beta = np.random.uniform(0, 90)
        self.gamma = np.random.uniform(0, 45)
        self.a = np.random.uniform(0, 10)
        self.b = 2

        # Konwersja kątów na radiany
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)

        # Obliczanie punktów końcowych ramion kąta beta
        angle1 = alpha_rad - beta_rad/2
        angle2 = alpha_rad + beta_rad/2
        self.arm1_end = (
            self.x + np.cos(angle1) * self.a,
            self.y + np.sin(angle1) * self.a
        )
        self.arm2_end = (
            self.x + np.cos(angle2) * self.a,
            self.y + np.sin(angle2) * self.a
        )

        # Punkt końcowy wektora alpha
        self.alpha_end = (
            self.x + np.cos(alpha_rad) * self.a,
            self.y + np.sin(alpha_rad) * self.a
        )

        # Punkt końcowy przeciwnego wektora (b)
        self.tail_end = (
            self.x - np.cos(alpha_rad) * self.b,
            self.y - np.sin(alpha_rad) * self.b
        )

        # Obliczanie punktów końcowych prostopadłej
        perpendicular_angle = alpha_rad + np.pi/2
        perpendicular_length = self.a * np.tan(beta_rad/2)
        
        self.left_end = (
            self.tail_end[0] + np.cos(perpendicular_angle) * perpendicular_length,
            self.tail_end[1] + np.sin(perpendicular_angle) * perpendicular_length
        )
        
        self.right_end = (
            self.tail_end[0] - np.cos(perpendicular_angle) * perpendicular_length,
            self.tail_end[1] - np.sin(perpendicular_angle) * perpendicular_length
        )

        # Obliczanie punktów końcowych pod kątem gamma
        angle_to_left = np.arctan2(self.left_end[1] - self.y, self.left_end[0] - self.x)
        angle_to_right = np.arctan2(self.right_end[1] - self.y, self.right_end[0] - self.x)
        
        self.left_gamma_end = (
            self.x + np.cos(angle_to_left - gamma_rad) * self.a,
            self.y + np.sin(angle_to_left - gamma_rad) * self.a
        )
        
        self.right_gamma_end = (
            self.x + np.cos(angle_to_right + gamma_rad) * self.a,
            self.y + np.sin(angle_to_right + gamma_rad) * self.a
        )

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Point index out of range")

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counterclockwise

def dist(p, q):
    return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def angle_to_point(origin, point):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    if dx == 0 and dy == 0:
        return float('inf')
    angle = np.arctan2(dy, dx)
    return angle

def sort_by_angle(points, start):
    angles = [angle_to_point(start, point) for point in points]
    indices = np.argsort(angles)
    sorted_points = [points[i] for i in indices]
    return sorted_points

def graham_scan(points):
    start = min(points, key=lambda p: (p[1], p[0]))

    # Sort points by polar angle with the start point
    sorted_points = sort_by_angle(points, start)

    hull = [sorted_points[0], sorted_points[1]]

    for i in range(2, len(sorted_points)):
        while len(hull) > 1 and orientation(hull[-2], hull[-1], sorted_points[i]) != 2:
            hull.pop()
        hull.append(sorted_points[i])

    return hull

def get_all_points_from_point(point):
    """Zwraca wszystkie punkty końcowe dla danego punktu jako listę krotek (x,y)"""
    return [
        (point.x, point.y),
        point.arm1_end,
        point.arm2_end,
        point.alpha_end,
        point.tail_end,
        point.left_end,
        point.right_end,
        point.left_gamma_end,
        point.right_gamma_end
    ]

class ExtendedPoint:
    """Klasa opakowująca punkty do użycia w algorytmie Grahama"""
    def __init__(self, x, y, original_point=None):
        self.x = x
        self.y = y
        self.original_point = original_point

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Point index out of range")

def plot_points_and_hull(points, hull):
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    plt.scatter(x_coords, y_coords, color='blue', label='Points')
    plt.axis('equal')
    
    # Rysowanie kątów dla każdego punktu
    for point in points:
        # Rysowanie pierwszego ramienia
        plt.plot([point.x, point.arm1_end[0]], [point.y, point.arm1_end[1]], color='green')
        
        # Rysowanie drugiego ramienia
        plt.plot([point.x, point.arm2_end[0]], [point.y, point.arm2_end[1]], color='green')
        
        # Rysowanie dwusiecznej (alpha)
        plt.plot([point.x, point.alpha_end[0]], [point.y, point.alpha_end[1]], color='red')
        
        # Rysowanie przeciwnego wektora o długości b
        tail_end_x, tail_end_y = point.tail_end
        plt.plot([point.x, tail_end_x], [point.y, tail_end_y], color='blue')
        
        # Rysowanie trójkąta prostokątnego
        triangle_x = [point.x, point.right_end[0], tail_end_x, point.left_end[0]]
        triangle_y = [point.y, point.right_end[1], tail_end_y, point.left_end[1]]
        plt.fill(triangle_x, triangle_y, alpha=0.2, fc='yellow', ec='orange')
        
        # Dodanie odcinków z punktu początkowego do końców pod kątem gamma
        plt.plot([point.x, point.left_gamma_end[0]], 
                [point.y, point.left_gamma_end[1]], 
                color='purple', linewidth=1)
        
        plt.plot([point.x, point.right_gamma_end[0]], 
                [point.y, point.right_gamma_end[1]], 
                color='purple', linewidth=1)

    # Rysowanie otoczki wypukłej
    hull_with_closure = hull + [hull[0]]
    hull_x = [p.x for p in hull_with_closure]
    hull_y = [p.y for p in hull_with_closure]
    plt.plot(hull_x, hull_y, marker='.', linestyle='-', color='red', label='Convex Hull')
    
    plt.legend()
    plt.title('Random Points and Convex Hull with Directions')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def main():
    num_points = 20
    random_coords = np.random.rand(num_points, 2) * 100
    original_points = [Point(x, y) for x, y in random_coords]
    
    # Zbieramy wszystkie punkty końcowe
    all_points = []
    for point in original_points:
        for x, y in get_all_points_from_point(point):
            all_points.append(ExtendedPoint(x, y, point))
    
    # Znajdujemy otoczkę wypukłą dla wszystkich punktów
    hull = graham_scan(all_points)
    plot_points_and_hull(original_points, hull)

if __name__ == '__main__':
    main()