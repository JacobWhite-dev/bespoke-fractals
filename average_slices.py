# -*- coding: utf-8 -*-
"""
Average the k-space data for each slice from a collection of
3D volumes.

Created on Mon Nov 18 2019

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

def main(argv):

    # Default values
    caseIndex = 0     # Index of case number in filenames
    N = 320           # Number of pixels in each dimension
    channels = 8      # Number of channels

    # Get arguments
    try:
        opts, args = getopt.getopt(argv, "i:o:cNC")
    except getopt.GetoptError:
        print("average_slices.py -i <inputpath> -o <outputpath>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            path = arg
        elif opt == '-o':
            outpath = arg
        elif opt == '-c':
            caseIndex = int(arg)
        elif opt == '-N':
            N = int(arg)
        elif opt == '-C':
            channels = int(arg)
            
    #   break

    # Make sure that the paths end in a slash
    if not path.endswith("/"):
        path += "/"

    if not outpath.endswith("/"):
        outpath += "/"

    # Print the paths
    print("Input path is " + path)
    print("Output path is " + outpath)

    outputPrefix = "case_"        # Output file prefix
    sliceIndex = caseIndex + 1    # Position of slice number in filename

    # Images are assumed to be channels x N x N arrays
    N = 320        # Dimension of each channel's image
    channels = 8   # Number of channels in each image

    # Get the files containing the original slices
    _, sliceList = filenames.getSortedFileListAndCases(path, caseIndex + 1, "*.nii.gz", True)

    # For each slice, 
    for sliceIndex in np.unique(sliceList):

        # Get all the images of that slice
        imageList, caseList = filenames.getSortedFileListAndCases(path, caseIndex, "*_" + str(sliceIndex) + ".nii.gz", True)

        # Initialise new image which will be the average of all the cases
        averageImage = np.zeros((channels, N, N), dtype = complex)

        # Set the number of processed cases to 0
        count = 0

        # For each image of the slice, 
        for image, case in zip(imageList, caseList):

            # Load the image
            img = nib.load(image)
            print("Loaded", image)
            data = img.get_data()
        
            # For each channel, add the data to the average image
            for channel in range(0, channels):
                averageImage[channel, :, :] += data[channel, :, :]
            #   break

            # Increment the number of cases processed
            count += 1

        #   break

        # Divide the average image by the number of cases processed
        np.divide(averageImage, count)

        # Save the averaged image
        sliceAveraged = nib.Nifti1Image(averageImage, np.eye(4))
        outname = (outpath + outputPrefix + str(0).zfill(3) + 
                   "_slice_" + str(sliceIndex) + ".nii.gz")
        sliceAveraged.to_filename(outname)

    #   break

if __name__ == "__main__":
    main(sys.argv[1:])





        

