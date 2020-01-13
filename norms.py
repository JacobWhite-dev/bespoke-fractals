'''

Created on Thurs Jan 9 2020

@author: uqjwhi35
'''

import numpy as np
import scipy.optimize as opt 
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

def lp(x, p):
    return np.power(np.sum(np.power(np.absolute(x), p), axis = -1), 1. / p)

def lp_cmplx(x, p):
    return lp(np.array([np.real(x), np.imag(x)]), p)

def lp_sum(x, p):
    return np.sum(lp(x, p))

def lp_mean(x, p):
    return np.mean(lp(x, p))

def lp_variance(x, p):
    norms = lp(x, p)
    return np.sum(np.power(np.max(norms) - np.mean(norms), 2))

def fit_norm(x, p0):
    hull = ConvexHull((x))
    points = x
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
    plt.show()
    return opt.differential_evolution(lambda p : lp_variance(hull.vertices, p), bounds = ((0, 1000),))

# Generate some data
N = 500 # Points to generate
dim = 2
count = 0

p = 0.5

# Initialise data
data = np.zeros([N, dim])

while count < N:
    [x, y] = np.random.random(size = [2]) * 2 - 1
    if lp([x, y], p) <= 1:
        data[count, :] = np.array([x, y])
        count += 1

plt.plot(data[:, 0], data[:, 1], 'ro')
plt.show()

result = fit_norm(data, 2)

print("Expected: {}".format(lp_sum(data, p)))
print("Found: {}".format(lp_sum(data, result.x)))
print(result.x, result.success)
