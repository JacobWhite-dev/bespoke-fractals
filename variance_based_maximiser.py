"""


Created on Tues Dec 3 2019

@author: uqjwhi35
"""

import math
import numpy as np
import filenames
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import pyfftw

import finite

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

# Images will be 3D numpy array:
# Dims:
# 0 - images
# 1 - row of image
# 2 - column of image

def normalise_images(images):
    numImages = images.shape[0]
    mean = np.mean(images, 0)
    mean = mean.reshape(1, mean.shape[0], mean.shape[1])
    print(mean.shape)
    return images - mean

def normalise_images2(images):
    numImages = images.shape[0]
    for image in images:
        image = (image - np.mean(image)) / np.std(image)

    return images

def project_images(images, pattern):

    numImages = images.shape[0]
    norm = np.linalg.norm(pattern)

    # Calculate projection
    paddedProd = np.zeros(images.shape)
    for i in range(0, numImages):
        paddedProd[i, :, :] = np.where(pattern, images[i, :, :], float('nan'))

    # Flatten and remove nans
    projections = np.zeros([numImages, np.count_nonzero(pattern)])
    for i in range(0, numImages):
        flat = paddedProd[i, :, :].flatten()
        projections[i, :] = flat[~np.isnan(flat)]

    # Normalise
    projections = projections / norm
    
    return projections

def pattern_score(images, pattern):
    proj = project_images(images, pattern)
    cov = (np.cov(proj, rowvar = False))

    if len(cov.shape) < 2:
      det = float(cov)
      trace = float(cov)
    else:
      det = np.linalg.det(cov)
      trace = np.trace(cov)

    return det, trace

path = 'D:\\Winter Research\\Data\\NYU Stanford Knee\\slices_decomb\\'
slice =  100
caseIndex = 0

# Get the files containing the slices
imageList, caseList = filenames.getSortedFileListAndCases(path, caseIndex, 
                                                               '*_slice_' + 
                                                               str(slice) + 
                                                               '.nii.gz', True)

cases = len(np.unique(caseList))

images = np.zeros([cases, 320, 320])

count = 0

for image, case in zip(imageList, caseList):
    img = nib.load(image)
    print('Loaded', image)
    
    data = img.get_data()
    images[count, :, :] = data
    count = count + 1

# Perform the thing
# Here, we try to find the optimal pattern

startingPoint = np.zeros(102400)

dets = np.zeros(102400)
traces = np.zeros(102400)

#images = normalise_images(images)
images = normalise_images2(images)

## Try specific patterns:
#uniform = np.ones([320, 320])

#N = 320    # Dimension of fractal 
#K = 2.25   # Reduction factor

## Setup fractal
#lines, angles, \
#    mValues, fractal, \
#    oversampling = finite.finiteFractal(N, K, sortBy='Euclidean', 
#                                        twoQuads=True)
#mu = len(lines)
## Tile center region further
#radius = N/8
#centerX = N/2
#centerY = N/2
#count = 0
#fractal = fftpack.fftshift(fractal)
#for i, row in enumerate(fractal):
#    for j, col in enumerate(row):
#        distance = math.sqrt( (i-float(centerX))**2 + (j-float(centerY))**2)
#        if distance < radius:
#            if not fractal[i, j] > 0: #already selected
#                count += 1
#                fractal[i, j] = 1
#fractal = fftpack.fftshift(fractal)
#totalSamples = mu*(N-1)+count+1
#actualR = float(totalSamples/N**2)
#print("Number of total sampled points:", totalSamples)
#print("Actual Reduction factor:", actualR)

#random = (np.random.rand(320, 320)).round()

#det, trace = pattern_score(images, uniform)
#print("Uniform - Det: {} Trace: {}".format(det, trace))
#det, trace = pattern_score(images, random)
#print("Random - Det: {} Trace: {}".format(det, trace))
#det, trace = pattern_score(images, fractal)
#print("Fractal - Det: {} Trace: {}".format(det, trace))

#exit()

for i in range(102400):
    startingPoint[i] = 1
    dets[i], traces[i] = pattern_score(images, startingPoint.reshape(320, 320))
    startingPoint[i] = 0
    print("{} out of 102400 processed".format(i))

try:
    plt.imshow(dets.reshape(320, 320))
except:
    pass
plt.show()

file1 = open(r"D:\Winter Research\dets100.txt","w")
file1.write(str(dets))


plt.imshow(np.log(dets.reshape(320, 320)))
plt.show()






