# -*- coding: utf-8 -*-
'''
Perform a PCA on nifti image files

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

# Load modules for data analysis and plotting
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load module for argument parsing
import argparse

def init_parser():
    """Initialise an argument parser object and return it."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description= "Perform primary component" +
                                     " analysis on MRI data")
    parser.add_argument("-p", "--path", default = "./", type = str, 
                        help = "Path containing files")
    parser.add_argument("--slices", default = "*", type = str, 
                        help = "Slices to process, given as a unix style " +
                        "expression")
    parser.add_argument("--size", default = "320,320", type = str, 
                        help = "Image size as a list (e.g. [256, 256])")
    parser.add_argument("--var", default = 0.8, type = float, 
                        help = "Variance to account for")
    parser.add_argument("-t", "--threshold", type = int, default = 0,                     
                        help = "Percentile threshold to apply to each " + 
                        "component")
    parser.add_argument("-T", "--superthreshold", type = int, 
                        default = 0, help = "Threshold additional results")
    parser.add_argument("-b", "--binary", action = "count", default = 0,
                        help = "Convert all elements to a 1 or 0")
    parser.add_argument("-s", "--save", type = str, 
                        help = "Path to save outputs to")
    parser.add_argument("-q", "--quiet", action = "store_true", 
                        help = "Suppress output labels")
    parser.add_argument("-c", "--channel", type = int, default = 0,
                        help = "Channel to use in multi-channel data")
    parser.add_argument("--sum", action = "store_true", default = False,
                        help = "Display the sum of the found components")
    parser.add_argument("--mean", action = "store_true", default = False,
                        help = "Show the mean for each slice")
    parser.add_argument("--caseindex", type = int, default = 0)

    return parser

def main():
    """Main function. Performs a PCA on images."""

    # Create the parser
    parser = init_parser()

    # Get arguments
    args = parser.parse_args()
    
    path = args.path                  # Input path
    slices = args.slices              # Slices to perform PCA on
    size = args.size                  # Size of the images
    var = args.var                    # Variance to account for in PCA
    thresh = args.threshold           # Threshold for data
    superThresh = args.superthreshold # Threshold for secondary data
    binary = args.binary              # Flag for outputting data as binary
    save = args.save                  # Path to save images to
    quiet = args.quiet                # Flag for quiet output
    chan = args.channel               # Channel to use for multi-channel data
    showSum = args.sum                # Flag for showing the sum of components
    showMean = args.mean              # Flag for showing the mean of the data
    caseIndex = args.caseindex        # Index of case number in filenames


    # Process arguments
    # Convert size to a list of integers
    try: 
        size = [int(i) for i in size.split(',')]
    except:
        print("Sizes must be integers")
        return

    # Ensure variance is between 0 and 1
    if var and ((var > 1) or (var < 0)):
        print("Invalid variance. Variance must be between 0 and 1.")
        return

    # Ensure threshold is between 0 and 100
    if thresh and ((thresh > 100) or (thresh < 0)):
        print("Invalid threshold. Threshold must be a percentile between " + 
              "0 and 100")
        return

    # Get the slice files
    _, sliceList = filenames.getSortedFileListAndCases(path, caseIndex + 1, 
                                                               '*_slice_' + 
                                                               slices + 
                                                               '.nii.gz', True)
   
    # Find the unique slice numbers and the number of these
    uniqueSlices = np.unique(sliceList)
    numSlices = len(uniqueSlices)

    # Initialise output lists
    outputs = []     # Output components [slice x components]
    contribution =[] # Contribution to variance of each component
    means = []       # Means for each slice
    
    # Perform PCA on the data for a given slice
    for sliceIndex in uniqueSlices:

        print("Processing slice", str(sliceIndex))

        # Find all the images of a given slice
        imageList, caseList = filenames.getSortedFileListAndCases(path, 
                                                                  caseIndex, 
                                                                  '*_slice_' + 
                                                                  str(sliceIndex) + 
                                                                  '.nii.gz', 
                                                                  True)
        
        # Create dataframe for cases of the slice
        sliceData = pd.DataFrame([])

        # For each image, 
        for image, case in zip(imageList, caseList):

            # Load the image 
            img = nib.load(image)
            data = img.get_data()
            print("Loaded case", str(case))

            # If multi-channel data has been specified, 
            # use the chosen channel
            if len(data.shape) == 3:
                data = data[chan, :, :]
            
            # If case data not 2D, skip it
            if len(data.shape) != 2:
                continue

            # Take the absolute value of the image and flatten it. Then
            # add it to the dataset
            newSliceData = pd.Series(np.ndarray.flatten(np.absolute(data)), 
                                     name = image)
            sliceData = sliceData.append(newSliceData)


        # Get the mean of the data for the slice
        mean = np.mean(np.array(sliceData), 0)
        means.append(mean)

        # Perform the PCA
        standardiser = StandardScaler()
        sliceData = standardiser.fit_transform(sliceData)
        pca = PCA(n_components = var)
        pca.fit(sliceData)
        components = np.abs(pca.components_)

        # Threshold the data
        components[components < np.percentile(components, thresh)] = 0

        # Convert the data to a binary if desired
        if binary:
            for component in components:
                component = [1 if x > 0 else 0 for x in component]
            mean = [1 if x > 0 else 0 for x in mean]
        
        # Append the results to the outputs vector
        outputs.append(components)
        contribution.append(pca.explained_variance_ratio_)


    # Calculate the sum of each set of components
    componentsSum = [np.sum(np.absolute(components), 0) for components in outputs]
    for component in componentsSum:
        component[component < np.percentile(component, superThresh)] = 0
        if binary > 1:
            component[component > 0] = 1

    # Save the components if desired
    if save:
        # Save the components
        for sliceIndex in range(0, numSlices):
            for i in range(0, len(outputs[sliceIndex])):
                saved = nib.Nifti1Image(outputs[sliceIndex][i].reshape(size), np.eye(4))
                outname = (save + "/slice_" + str(uniqueSlices[sliceIndex]).zfill(3) + 
                            "_comp_" + str(i).zfill(3) + ".nii.gz")
                saved.to_filename(outname)

            if showMean:
                saved = nib.Nifti1Image(means[sliceIndex].reshape(size), np.eye(4))
                outname = (save + "/slice_" + str(uniqueSlices[sliceIndex]).zfill(3) + 
                            "_mean.nii.gz")
                saved.to_filename(outname)

            if showSum:
                saved = nib.Nifti1Image(componentsSum[sliceIndex].reshape(size), np.eye(4))
                outname = (save + "/slice_" + str(uniqueSlices[sliceIndex]).zfill(3) + 
                            "_sum.nii.gz")
                saved.to_filename(outname)


    # Display results
    maxComponents = np.max([len(components) for components in outputs])

    # Show the PCA
    for sliceIndex in range(0, len(outputs)):
        for component in range(0, len(outputs[sliceIndex])):

            # Create the subplot
            ax = plt.subplot2grid((len(outputs), maxComponents + showMean + 
                                   showSum), (sliceIndex, component))

            # Show the image. The flipping and transposing performed is
            # done so that the image orientation matches that observed
            # when using a NIFTI file viewer like SMILX
            ax.imshow(np.flipud(np.fliplr(np.transpose(outputs[sliceIndex]
                                                       [component].
                                                       reshape(size)))), 
                      cmap ='jet')

            # Clear axes ticks
            plt.xticks([])
            plt.yticks([])

            # Add slice label on the y-axis of the first component
            if component == 0:
                ax.set_ylabel(str(uniqueSlices[sliceIndex]))
            
            # If not in quiet mode, show the variance contributions of
            # each component
            if not quiet:
                plt.title("{0:.2f}".format(contribution[sliceIndex]
                                           [component]))

        # Show the mean if desired
        if showMean:
            ax = plt.subplot2grid((len(outputs), maxComponents + showMean + 
                                   showSum), (sliceIndex, 
                                              maxComponents + showMean - 1))
            ax.imshow(np.flipud(np.fliplr(np.transpose(means[sliceIndex].reshape(size)))), 
                      cmap='jet')
            plt.xticks([])
            plt.yticks([])
            plt.title("Mean")

        # Show the sum if desired
        if showSum:
            ax = plt.subplot2grid((len(outputs), maxComponents + showMean + 
                                   showSum), (sliceIndex, maxComponents + 
                                              showMean + showSum - 1))
            ax.imshow(np.flipud(np.fliplr(np.
                                          transpose(componentsSum[sliceIndex].
                                                    reshape(size)))), 
                      cmap='jet')
            plt.xticks([])
            plt.yticks([])
            plt.title("Sum")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
