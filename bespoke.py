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

np.seterr(divide='ignore', invalid='ignore') # Skip dividing by zero

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

def P2R(radii, angles):
    angles = angles * 2 * math.pi / 360
    return radii * np.exp(1j*angles)

def rotate(angle, point):
    rotMat = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return np.matmul(rotMat, point)

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
    #fareyVectors.generateFinite(N)
    
    #sort to reorder result for prettier printing
    finiteAnglesSorted, anglesSorted = fareyVectors.sortCustom(sortBy)
    
    ## Rotate the whole fractal
    #anglesSorted = [angle * P2R(1, 63) for angle in anglesSorted]
    #for index in range(len(anglesSorted)):
    #    angle = anglesSorted[index]
    #    p = np.imag(angle)
    #    q = np.real(angle)
    #    p = p + N if p < 0 else p
    #    q = q + N if q < 0 else q

    #    anglesSorted[index] = q + 1j * p

    #print(anglesSorted)
 
    #finiteAnglesSorted = [farey.toFinite(angle, N) for angle in anglesSorted]

    kSpace = np.zeros((N,N))
    lines, angles, mValues = finite.computeKatzLines(kSpace, anglesSorted, finiteAnglesSorted, K, centered, twoQuads)
    mu = len(lines)
    print("Number of finite lines in fractal:", mu)
    
    samplesImage1 = np.zeros((N,N), np.float32)
    for line in lines:
        u, v = line

        for x, y in zip(u, v):
            
            # Rotate
            #[x, y] = rotate(30, np.array([[x], [y]]))
            #x = int(x) % N
            #y = int(y) % N

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

def totient(n):
    amount = 0        
    for k in range(1, n + 1):
        if math.gcd(n, k) == 1:
            amount += 1
    return amount

def summatory_totient(n):
    result = 0
    for i in range(1, n+1):
        result += totient(i)
    return result


def get_farey_index(p, q, n):

    if p == 1 and q == 1:
        return summatory_totient(n)

    sum = 0
    A = []
    for i in range(0, n + 1):
        A.append(math.floor((p * i) / q))

    for i in range(2, n + 1):
        for j in range(i + i, n + 1, i):
            A[j] -= A[i]

    for i in range(1, n + 1):
        sum += A[i]

    return sum

def rem_dist(p, q, N):
    return np.power(p / q - get_farey_index(p, q, N) / N, 2) if q != 0 else 0

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

#print(get_farey_index(1, 1, 4))
#exit()

N = 1024
K = 1

def novel_frac():
    _, _, _, q1, _ = myFiniteFractal(int(N / 2), K, sortBy = new, twoQuads = True)
    _, _, _, q2, _ = myFiniteFractal(int(N / 2), K, sortBy = lambda p,q: new(-p, q), twoQuads = True)
    _, _, _, q3, _ = myFiniteFractal(int(N / 2), K, sortBy = lambda p,q: new(-p, -q), twoQuads = True)
    _, _, _, q4, _ = myFiniteFractal(int(N / 2), K, sortBy = lambda p,q: new(p, -q), twoQuads = True)

    # Rotate

    q2 = np.fliplr(q2)
    q3 = np.fliplr(np.flipud(q3))
    q4 = np.flipud(q4)

    topHalf = np.concatenate((q2, q1), axis = 1)
    bottomHalf = np.concatenate((q3, q4), axis = 1)
    result = np.concatenate((topHalf, bottomHalf), axis = 0)

    return result

#plt.imshow(novel_frac())
#plt.show()

l3 = lambda p,q: (math.pow((p), 0.5) + math.pow((q), 0.5))
#l3 = lambda p,q: math.pow(p, 2) / 8 + math.pow(q, 2)
lines, angles, mValues, fractal, overSamplingFilter = myFiniteFractal(N, K, sortBy = rotated_diamond, twoQuads = True)

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
                #fractal[i, j] = 1
#fractal = fftpack.fftshift(fractal)

plt.imshow(fractal)
plt.show()
exit()

print(fractal)
#totalSamples = mu*(N-1)+count+1
#actualR = float(totalSamples/N**2)
#print("Number of total sampled points:", totalSamples)
#print("Actual Reduction factor:", actualR)




