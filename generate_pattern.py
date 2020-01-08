# -*- coding: utf-8 -*-
"""
Create a feasible MRI sampling pattern composed of
periodic lines in k-space given an image of desired
sampling points.

Created on Mon Dec 2 2019

@author: uqjwhi35
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import pyfftw
import math
import farey as fr

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def pseudomodinv(a, m):
    a = a % m; 
    for x in range(1, m) : 
        if ((a * x) % m == 1) : 
            return x 
    return 1

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        return pseudomodinv(a, m)
    else:
        return x % m

input = 'D:\\Winter Research\\Data\\NYU Stanford Knee\\knee_test\\slice_100_sum.nii.gz'
#input = 'D:\\Winter Research\\Data\\NYU Stanford Knee\\slices_comb\\case_000_slice_100.nii.gz'
output = '*'

# Open the input file and get the data
img = nib.load(input)
samplePoints = img.get_data()
l, w = samplePoints.shape

assert l == w

N = l

# ifft shift
samplePoints = fftpack.ifftshift(samplePoints)

# Create a new array to store the result in
pattern = np.zeros(samplePoints.shape)

# Find indices of values
x_samps, y_samps = np.nonzero(samplePoints)
mValues = np.zeros(N)

# For each index, find the radial line that goes through it and draw it in
#for index in range(0, len(x_samp)):

for i in range(0, len(x_samps)):
    p = x_samps[i]
    q = y_samps[i]
    #mValue, inv = fr.toFinite(fr.farey(p, q), N)
    #print("{} {}".format(p, q))
    try:
        mValue = (modinv(p, N) * q) % N
    except:
        #print("Bad p:{} q:{}".format(p, q))
        continue

    mValues[mValue] = mValues[mValue] + samplePoints[p, q]
    #print(mValues)

#print(mValues)
# Sort the gradients by how frequently they appear
grads = np.argsort(mValues)
grads = grads[-128:]
print(grads)

# Draw lines
for grad in grads:
    for x in range(0, N):
        pattern[x % N, (x * grad) % N] = 1

pattern = fftpack.fftshift(pattern)

# Print reduction factor
reductionFactor = np.size(pattern) / np.size(np.nonzero(pattern))
print(reductionFactor)

plt.imshow(pattern)
plt.show()



