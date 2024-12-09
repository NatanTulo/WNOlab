import numpy as np
import matplotlib.pyplot as plt


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
    plt.scatter(*zip(*points), color='blue', label='Points')
    # Append the first point to the hull to connect it with the last point
    hull_with_closure = hull + [hull[0]]
    plt.plot(*zip(*hull_with_closure), marker='o', linestyle='-', color='red', label='Convex Hull')
    plt.legend()
    plt.title('Random Points and Convex Hull')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def main():
    num_points = 90
    points = np.random.rand(num_points, 2) * 100  # Zmiana zakresu na <0;100>
    hull = graham_scan(points)
    plot_points_and_hull(points, hull)

if __name__ == '__main__':
    main()