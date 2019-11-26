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
import scipy.signal as sig
import pyfftw
import math
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import argparse

def main():
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Perform primary component analysis on MRI data')
    parser.add_argument("-p", "--path", default = "./", type = str, help = "Path containing files")
    parser.add_argument("--slices", default = "*", type = str, help = "Slices to process, given as a unix style expression")
    parser.add_argument("--size", default = "320,320", type = str, help = "Image size as a list (e.g. [256, 256])")
    parser.add_argument("--var", default = 0.8, type = float, help = "Variance to account for")
    parser.add_argument("-t", "--threshold", type = int, help = "Percentile threshold to apply to each component")
    parser.add_argument("-b", "--binary", action = "store_true", help = "Convert all elements to a 1 or 0")
    parser.add_argument("-s", "--save", type = str, help = "Path to save outputs to")
    parser.add_argument("-q", "--quiet", action = "store_true", help = "Suppress output labels")
    parser.add_argument("-c", "--channel", type = int, help = "Channel to use in multi-channel data")

    # Get args
    args = parser.parse_args()
    path = args.path
    slices = args.slices
    size = args.size
    var = args.var
    thresh = args.threshold
    binary = args.binary
    save = args.save
    quiet = args.quiet
    chan = args.channel

    # Check args
    if var and ((var > 1) or (var < 0)):
        print("Invalid variance. Variance must be between 0 and 1.")
        return

    if thresh and ((thresh > 100) or (thresh < 0)):
        print("Invalid threshold. Threshold must be a percentile between 0 and 100")
        return

    if not thresh and binary:
        print("Binary mode is only available when threshold is set.")
        return

    if size:
        try:
            size = [int(i) for i in size.split(',')]
        except:
            print("Sizes must be integers")
            return

    # Import data
    caseIndex = 0

    # Get the slice files
    imageList, sliceList = filenames.getSortedFileListAndCases(path, caseIndex + 1, '*_slice_' + slices + '.nii.gz', True)
    uniqueSlices = np.unique(sliceList)
    numSlices = len(uniqueSlices)

    # Initialise arrays
    outputs = []
    contribution =[]
    
    # Perform PCA for a given slice
    for slice in uniqueSlices:
        imageList, caseList = filenames.getSortedFileListAndCases(path, caseIndex, '*_slice_' + str(slice) + '.nii.gz', True)

        print("Processing slice", str(slice))
         
        vecs = pd.DataFrame([])

        for image, case in zip(imageList, caseList):

            img = nib.load(image)
            data = img.get_data()

            if chan is not None:
                data = data[chan, :, :]

            vec = pd.Series(np.ndarray.flatten(np.absolute(data)),name=image)
            vecs = vecs.append(vec)

        # Perform PCA
        faces_normalised = StandardScaler()
        vecs = faces_normalised.fit_transform(vecs)
        faces_pca = PCA(n_components=var)
        faces_pca.fit(vecs)
        result = np.abs(faces_pca.components_)

        if thresh != None:
            result[result < np.percentile(result, thresh)] = 0

        if binary:
            result[result != 0] = 1

        if save:
            for i in range(0, len(result)):
                saved = nib.Nifti1Image(result[i].reshape(size), np.eye(4))
                outname = (save + "/slice_" + str(slice).zfill(3) + "_comp_" + str(i).zfill(3) + ".nii.gz")
                saved.to_filename(outname)

        outputs.append(result)
        contribution.append(faces_pca.explained_variance_ratio_)


    maxEls = np.max([len(element) for element in outputs])

    for element in range(0, len(outputs)):
        for output in range(0, len(outputs[element])):
            ax = plt.subplot2grid((len(outputs), maxEls),(element, output))

            # Flipping, transposing is done so that orientation matches that in nifti file
            ax.imshow(np.flipud(np.fliplr(np.transpose(outputs[element][output].reshape(size)))), cmap='jet')
            plt.xticks([])
            plt.yticks([])

            # Add slice label
            if not output:
                ax.set_ylabel(str(uniqueSlices[element]))
            
            
            if not quiet:
                plt.title("{0:.2f}".format(contribution[element][output]))
        
    plt.show()

if __name__ == "__main__":
    main()
