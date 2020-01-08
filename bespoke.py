'''

Created on Wed Jan 8 2020

@author: uqjwhi35
'''

'''
How this is gonna work:
1. Get Shape of interest
2. Find norm corresponding to that shape (HARD)
3. Create fractal
'''

import farey
import finite
import radon
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fftpack as fftpack
import pyfftw

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

#fareyVectors = farey.Farey()
#fareyVectors.compactOn()
#fareyVectors.generateFiniteWithCoverage(320)

#finiteAnglesSorted, anglesSorted = fareyVectors.sort('Euclidean')

#print(finiteAnglesSorted)
#print(anglesSorted)

#kspace = np.zeros((N, N))

def myFiniteFractal(N, K, sortBy = lambda p,q : abs(p) + abs(q), twoQuads=True, centered=False):
    '''
    Create the finite fractal for image size N given the Katz criterion K
    
    sortBy can be 'Euclidean' for L2 norm or 'length' for L1 norm
    twoQuads can be used to cover the half plane
    
    Returns lines, angles, mValues, fractal formed (as an image), oversampling filter if applicable
    '''
    fareyVectors = farey.Farey()        
    fareyVectors.compactOn()
    fareyVectors.generateFiniteWithCoverage(N)
    
    #sort to reorder result for prettier printing
    finiteAnglesSorted, anglesSorted = fareyVectors.sortCustom(sortBy)
    
    kSpace = np.zeros((N,N))
    lines, angles, mValues = finite.computeKatzLines(kSpace, anglesSorted, finiteAnglesSorted, K, centered, twoQuads)
    mu = len(lines)
    print("Number of finite lines in fractal:", mu)
    
    samplesImage1 = np.zeros((N,N), np.float32)
    for line in lines:
        u, v = line
        for x, y in zip(u, v):
            samplesImage1[x, y] += 1
    #determine oversampling because of power of two size
    #this is fixed for choice of M and m values
    oversamplingFilter = np.zeros((N,N), np.uint32)
    onesSlice = np.ones(N, np.uint32)
    for m in mValues:
        radon.setSlice(m, oversamplingFilter, onesSlice, 2)
    oversamplingFilter[oversamplingFilter==0] = 1
    samplesImage1 /= oversamplingFilter
#    samplesImage1 = fftpack.fftshift(samplesImage1)
    
    return lines, angles, mValues, samplesImage1, oversamplingFilter

def rotate(angle, point):
    rotMat = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return np.matmul(rotMat, point)

def rotated_diamond(p, q):
    [P, Q] = rotate(30, np.array([[p], [q]]))
    #print("{},{} -> {},{}".format(p, q, P, Q))
    return math.pow(abs(P), 0.33) + math.pow(abs(Q), 0.33)

N = 10000
K = 1

l3 = lambda p,q: (math.pow((p), 0.5) + math.pow((q), 0.5))
#l3 = lambda p,q: math.pow(p, 2) / 8 + math.pow(q, 2)
lines, angles, mValues, fractal, overSamplingFilter = myFiniteFractal(N, K, sortBy = rotated_diamond, twoQuads = True)

print(fractal)

# Tile center region further
radius = N/250
centerX = N/2
centerY = N/2
count = 0
fractal = fftpack.fftshift(fractal)
for i, row in enumerate(fractal):
    for j, col in enumerate(row):
        distance = rotated_diamond(i - float(centerX), j - float(centerY))
        #distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
        if distance < radius:
            if not fractal[i, j] > 0: #already selected
                count += 1
                fractal[i, j] = 1
#fractal = fftpack.fftshift(fractal)

print(fractal)
#totalSamples = mu*(N-1)+count+1
#actualR = float(totalSamples/N**2)
#print("Number of total sampled points:", totalSamples)
#print("Actual Reduction factor:", actualR)

plt.imshow(fractal)
plt.show()


