import math
import numpy as np

def lp_norm(p, q, l):
    return math.pow(math.pow(abs(p), l) + math.pow(abs(q),l), 1/l)

def rotated_diamond(p, q):
    [P, Q] = rotate(25, np.array([[p], [q]]))

    # Normalise with l2 - eliminates nice fractal shape
    #norm = math.pow(math.pow(P, 2) + math.pow(Q, 2), 0.5)
    #P = P / norm
    #Q = Q / norm

    #P = P / 40

    #print("{},{} -> {},{}".format(p, q, P, Q))
    return math.pow(math.pow(abs(P), 0.33) + math.pow(abs(Q), 0.33), 1/0.33)

def wacko(p,q):
    return math.pow(math.pow(abs(p * q), 0.33) + math.pow(abs(q / p if p != 0 else 0), 0.33), 1/0.33)

def new(p,q):
    #return p * q
    return rotated_diamond(p, q)

def spiral(p, q):
    return abs(1 - math.exp(math.atan2(q, p)))

def conc(p, q):
    return math.sqrt(math.pow(p, 2) + math.pow(q, 2)) % 6

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
    return a
    #print("Lower: [{},{}]; Upper: [{},{}]; Point: [{},{}]; a = {}".format(x1, y1, x2, y2, x3, y3, a))

#points = np.array([[5,5], [1,2], [-3,3], [-5, -5], [-1, -2], [3, -3]])

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

def rando(p, q):
    return random.randint(0,1)

def ellipse(p, q):
    return math.pow((p - 2 * q), 2) + p * q
