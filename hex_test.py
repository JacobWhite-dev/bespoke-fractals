import math
import numpy as np
import farey
import matplotlib.pyplot as plt

def hex(p, q):
    # Get hextant
    angle = np.arctan2(p, q)

    # Get bounding vectors
    n = 6
    interior_angle = 2 * math.pi / n
    tant = (angle // interior_angle) % n

    angle_up = (tant + 1) * interior_angle
    angle_low = tant * interior_angle

    x1 = np.cos(angle_up)
    x2 = np.cos(angle_low)
    x3 = q

    y1 = np.sin(angle_up)
    y2 = np.sin(angle_low)
    y3 = p

    a = (x1 * y3 - x3 * y1 + x3 * y2 - x2 * y3) / (x1 * y2 - x2 * y1)
    return a

def poly(p, q, points):

    angles = np.array([np.arctan2(point[1], point[0]) for point in points])
    angle = np.arctan2(p, q)

    angles = angles % (2 * math.pi)
    angles = np.concatenate(([0], angles))
    angle = angle % (2 * math.pi)
    #print(angles, angle)

    num_points = np.size(points, axis = 0)

    tent = np.max(np.where(angle >= angles)) % num_points
    lower = (tent - 1) % num_points
    upper = (tent) % num_points
    #upper = np.min(np.where(angle < angles)) % num_points
    #print(lower, upper, np.size(points, axis = 0))

    x1 = points[upper][0]
    x2 = points[lower][0]
    x3 = q

    y1 = points[upper][1]
    y2 = points[lower][1]
    y3 = p

    a = (x1 * y3 - x3 * y1 + x3 * y2 - x2 * y3) / (x1 * y2 - x2 * y1)
    #return a
    #print("Lower: [{},{}]; Upper: [{},{}]; Point: [{},{}]; a = {}".format(x1, y1, x2, y2, x3, y3, a))
    return round(a, -1)

xx, yy = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100))
grid = np.dstack((xx, yy))

points = np.array([[5,5], [1,2], [-3,3], [-5, -5], [-1, -2], [3, -3]])

result = np.array([[poly(item[1], item[0], points) for item in row] for row in grid])

plt.imshow(result)
plt.show()