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

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

input = 'D:\\Winter Research\\Data\\NYU Stanford Knee\\knee_test\\slice_100_sum.nii.gz'
output = '*'

# Open the input file and get the data
img = nib.load(input)
samplePoints = img.get_data()

# ifft shift
samplePoints = fftpack.ifftshift(samplePoints)

# Create a new array to store the result in
pattern = np.zeros(samplePoints.shape)

# Find indices of values
(x_samp, y_samp) = np.where(samplePoints > 0)
print(x_samp)
print(y_samp)

#ratios = []

# For each index, find the radial line that goes through it and draw it in
#for index in range(0, len(x_samp)):
#
#    if x_samp[index] == 0:
#        if y_samp[index] == 0:
#            continue
#        ratio = float('nan')
#    else:
#        ratio = float(y_samp[index]) / float(x_samp[index])
#
#    ratios.append(ratio)

# Fill in new image 




