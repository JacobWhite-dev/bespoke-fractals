# -*- coding: utf-8 -*-
"""
Deconstruct images into k-space data.

Created on Wed Nov 20 2019

@author: uqjwhi35
"""

# Load module for command-line arguments
import sys
import getopt

# Load module for accessing files
import filenames

# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
import scipy.fftpack as fftpack
import pyfftw
import math

def main(argv):

    # Default values
    complexOutput = 1 # Flag for if final image can have complex values
    caseIndex = 0     # Index of case number in filenames

    # Get arguments
    try:
        opts, args = getopt.getopt(argv, "i:o:vc")
    except getopt.GetoptError:
        print("reconstruct.py -i <inputpath> -o <outputpath>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            path = arg
        elif opt == '-o':
            outpath = arg
        elif opt == '-v':
            complexOutput = 0 if (arg == 0) else 1
        elif opt == '-c':
            caseIndex = arg
            
    #   break

    # Make sure that the paths end in a slash
    if not path.endswith("/"):
        path += "/"

    if not outpath.endswith("/"):
        outpath += "/"

    # Print the paths
    print("Input path is " + path)
    print("Output path is " + outpath)

    # Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
    fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
    fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
    fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
    fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    outputPrefix = "case_"

    # Get the files containing the slices
    imageList, \
        caseList = filenames.getSortedFileListAndCases(path, 
                                                        caseIndex, 
                                                        "*.nii.gz", True)
    imageList, \
        sliceList = filenames.getSortedFileListAndCases(path, 
                                                        caseIndex + 1, 
                                                        "*.nii.gz", True)

    count = 0 # Number of slices processed

    # Process each slice
    for image, case, sliceIndex in zip(imageList, caseList, sliceList):
        img = nib.load(image)
        print("Loaded", image)

        # Get the numpy array version of the image
        data = img.get_data() #numpy array without orientation
        lx, ly = data.shape
        print("Image shape:", data.shape)
    
        # 2D FFT
        newImage = np.zeros((lx, ly), dtype = complex)

        newImage[:, :] = fftpack.fftshift(fftpack.
                                            fft2(np.
                                                conjugate(data[:, :])))

        if not complexOutput:
            newImage = np.absolute(newImage)
#       endif

        # Save the output image
        slice = nib.Nifti1Image(newImage, np.eye(4))
        outname = (outpath + outputPrefix + str(case).zfill(3) + "_slice_" + str(sliceIndex) + ".nii.gz")
        slice.to_filename(outname)
        count += 1
    
#       break
 
    print("Total", count, "processed") 

#   break

if __name__ == "__main__":
    main(sys.argv[1:])
