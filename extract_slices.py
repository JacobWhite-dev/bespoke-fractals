# -*- coding: utf-8 -*-
"""
Read Nifti 3D volumes via NiBabel and extract slices from 3D volume with 
multiple channels then save them as nifti files.

This is a modified version of the script batch_extract_slices.py created 
by uqscha22.

Created on Wed Jul 3 2019

@author: uqjwhi35
"""

# Load module for accessing files
import filenames

# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib

path = "images/"            # Path containing 3D volumes
imgPath = "kspace/"        # Subdirectory containing k-space data
outpath = "slices2/"         # Outpath for sliced images
outputPrefix = "case_"

caseIndex = 0    # Index of case number in filenames
sliceView = 0    # Which axis to take the slices along

sliceNum = 256   # Number of slices in each 3D volume

# Initiaise the list of offsets
offsets = []
for i in range(0, sliceNum):
    offsets.append(i)


# Get the directories containing the 3D volumes
dirList, caseList = filenames.getSortedFileListAndCases(path, 
                                                        caseIndex, 
                                                        '*', True)

# Process each 3D volume
for dir, case in zip(dirList, caseList):

    # Get list of filenames and case IDs from path where 3D volumes are
    imageList, caseList = filenames.getSortedFileListAndCases(dir+'/'+imgPath, 
                                                              caseIndex, 
                                                              "*.nii.gz")
    
    if not imageList:
        print("Image Not Found. Skipping", case)
        continue

    # Load nifti file
    image = imageList[0]
    img = nib.load(dir+'/'+imgPath+image)
    print("Loaded", image)

    # Get the numpy array version of the image
    data = img.get_data() #numpy array without orientation
    print("Image shape:", data.shape)

    # Initialise the slice count
    count = 0
    print("Completed {0}/{1} slices".format(count, sliceNum), 
          end='\r', flush=True)

    # Extract each slice
    for offset in offsets:

        # Extract a slice from 3D volume to save. Note that the data 
        # is multi-channel, and thus each slice is also multi-channel. 
        # These channels constitute the fourth dimension of the data array
        if sliceView == 1: 
            img_slice = data[:,offset,:,:]
        elif sliceView == 0:
            img_slice = data[offset,:,:,:] 
        else:
            img_slice = data[:,:,offset,:]
 
        # We flip the along the x and y axes when reading so that the orientation of
        # the resulting image is consistent with what SMILX shows. Note that
        # flipping the k-space data, which we are doing here, is equivalent
        # to flipping in the image space, due to the time-reversal property
        # of the Fourier transform, which states that F{x(-t)} = X(-w). This
        # may be easily generalised to two dimensions
        img_slice = np.flip(img_slice, 1)
        img_slice = np.flip(img_slice, 2)

        # Save slice
        slice = nib.Nifti1Image(img_slice, np.eye(4))
        outname = (outpath + outputPrefix + str(case).zfill(3) + 
                   "_slice_" + str(count) + ".nii.gz")
        slice.to_filename(outname)
        count += 1
    
        print("Completed {0}/{1} slices".format(count, sliceNum),
             end='\r', flush=True)

#       break

    print() # Flush output

#   break

print("Slicing Complete")