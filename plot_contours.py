
# -*- coding: utf-8 -*-
"""
Plot k-space data using a contour plot

Created on Tue Nov 19 2019

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
    complexOutput = 0 # Flag for if final image can have complex values
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

if __name__ == "__main__":
    main(sys.argv[1:])