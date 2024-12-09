import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alpha = 90 #np.random.uniform(0, 360)  # kierunek w stopniach
        self.beta = 90 #np.random.uniform(0, 90)    # kąt rozwarcia w stopniach
        self.a = np.random.uniform(0, 10)       # długość wektora
        self.b = 2                              # długość przeciwnego wektora

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

def plot_points_and_hull(points, hull):
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    plt.scatter(x_coords, y_coords, color='blue', label='Points')
    plt.axis('equal')
    
    # Rysowanie kątów dla każdego punktu
    for point in points:
        alpha_rad = np.radians(point.alpha)
        beta_rad = np.radians(point.beta)
        
        # Obliczanie kątów dla obu ramion
        angle1 = alpha_rad - beta_rad/2
        angle2 = alpha_rad + beta_rad/2
        
        # Rysowanie pierwszego ramienia
        dx1 = np.cos(angle1) * point.a
        dy1 = np.sin(angle1) * point.a
        plt.plot([point.x, point.x + dx1], [point.y, point.y + dy1], color='green')
        
        # Rysowanie drugiego ramienia
        dx2 = np.cos(angle2) * point.a
        dy2 = np.sin(angle2) * point.a
        plt.plot([point.x, point.x + dx2], [point.y, point.y + dy2], color='green')
        
        # Rysowanie dwusiecznej (alpha)
        dx_alpha = np.cos(alpha_rad) * point.a
        dy_alpha = np.sin(alpha_rad) * point.a
        plt.plot([point.x, point.x + dx_alpha], [point.y, point.y + dy_alpha], color='red')
        
        # Rysowanie przeciwnego wektora o długości b
        dx_opposite = -np.cos(alpha_rad) * point.b
        dy_opposite = -np.sin(alpha_rad) * point.b
        tail_end_x = point.x + dx_opposite
        tail_end_y = point.y + dy_opposite
        plt.plot([point.x, tail_end_x], [point.y, tail_end_y], color='blue')
        
        # Rysowanie poprawnego trójkąta prostokątnego w miejscu przeciwnym
        left_angle = alpha_rad + np.pi + beta_rad/2
        right_angle = alpha_rad + np.pi - beta_rad/2

        # Obliczanie punktów końcowych ogona
        tail_end_x = point.x + dx_opposite
        tail_end_y = point.y + dy_opposite

        # Punkty lewego i prawego ramienia
        left_end_x = point.x + np.cos(left_angle) * point.b
        left_end_y = point.y + np.sin(left_angle) * point.b

        right_end_x = point.x + np.cos(right_angle) * point.b
        right_end_y = point.y + np.sin(right_angle) * point.b

        # Tworzenie poprawnego trójkąta prostokątnego
        triangle_x = [point.x, left_end_x, right_end_x]
        triangle_y = [point.y, left_end_y, right_end_y]

        plt.fill(triangle_x, triangle_y, alpha=0.5, fc='blue', ec='black')


    # Rysowanie trójkąta prostokątnego
    triangle_x = [point.x, tail_end_x, right_end_x, left_end_x]
    triangle_y = [point.y, tail_end_y, right_end_y, left_end_y]
    plt.fill(triangle_x, triangle_y, alpha=0.2, fc='yellow', ec='orange')

    # Rysowanie otoczki wypukłej
    hull_with_closure = hull + [hull[0]]
    hull_x = [p.x for p in hull_with_closure]
    hull_y = [p.y for p in hull_with_closure]
    plt.plot(hull_x, hull_y, marker='o', linestyle='-', color='red', label='Convex Hull')
    
    plt.legend()
    plt.title('Random Points and Convex Hull with Directions')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def main():
    num_points = 20
    random_coords = np.random.rand(num_points, 2) * 100
    points = [Point(x, y) for x, y in random_coords]
    hull = graham_scan(points)
    plot_points_and_hull(points, hull)

if __name__ == '__main__':
    main()