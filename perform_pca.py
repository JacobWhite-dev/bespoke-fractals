# -*- coding: utf-8 -*-
'''
Perform a PCA on k-space data

Created on Thurs Nov 21 2019

@author: uqjwhi35
'''

# Load module for accessing files
import filenames

# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
import scipy.fftpack as fftpack
import pyfftw
import math
import matplotlib.pyplot as plt
import pandas as pd

# Import data
path = "slices_decomb/"               # Path
sliceNum = "100"
caseIndex = 0

# Get the slice files
imageList, caseList = filenames.getSortedFileListAndCases(path, 
                                                        caseIndex, 
                                                        '*_slice_' + sliceNum + '.nii.gz', True)

faces = pd.DataFrame([])

for image, case in zip(imageList, caseList):
    img = nib.load(image)

    face = pd.Series(np.ndarray.flatten(np.absolute(img.get_data())),name=image)
    faces = faces.append(face)

fig, axes = plt.subplots(4,5,figsize=(9,9),
subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.iloc[i].values.reshape(320,320),cmap='gray')
plt.show()

# Perform PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#n_components = 0.80 means it will return the Eigenvectors wthat have 80% of the variation in the dataset
faces_normalised = StandardScaler()
faces = faces_normalised.fit_transform(faces)
faces_pca = PCA(n_components=0.8)
faces_pca.fit(faces)

fig, axes = plt.subplots(1,len(faces_pca.components_),figsize=(9,3),
 subplot_kw={'xticks':[], 'yticks':[]},
 gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.components_[i].reshape(320,320),cmap='gray')
print(faces_pca.explained_variance_ratio_)
plt.show()

