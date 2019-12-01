# -*- coding: utf-8 -*-
'''
Perform a a UMAP dimensionality reduction on
knee MRI data.

Created on Mon Dec 2 2019

@author: uqjwhi35
'''

import numpy as np
import nibabel as nib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import filenames

path = 'D:\\Winter Research\\Data\\NYU Stanford Knee\\slices'
slices = '127'
caseIndex = 0

imageList, caseList = filenames.getSortedFileListAndCases(path, caseIndex, 
                                                               '*_slice_' + 
                                                               slices + 
                                                               '.nii.gz', True)

imageList, sliceList = filenames.getSortedFileListAndCases(path, caseIndex + 1, 
                                                               '*_slice_' + 
                                                               slices + 
                                                               '.nii.gz', True)

knees = []
channels = []

for image, case, sliceIndex in zip(imageList, caseList, sliceList):
    img = nib.load(image)
    print("Loaded", image)

    # Get the numpy array version of the image
    data = img.get_data()
    for channel in range(0, data.shape[0]):
        knees.append((np.absolute(data[channel, :, :])).flatten())
        channels.append(channel)

reducer = umap.UMAP(random_state = 42)
reducer.fit(knees)

embedding = reducer.embedding_
print(embedding.shape)

plt.scatter(embedding[:, 0], embedding[:, 1], c=channels, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the knee dataset', fontsize=24)
plt.show()




