# -*- coding: utf-8 -*-
'''
Plot k-space data using a contour plot

Created on Tue Nov 19 2019

@author: uqjwhi35
'''

# Load module for command-line arguments
import sys
import getopt

# Load module for accessing files
import filenames

# Load modules for arrays and nifti file support
import numpy as np
import nibabel as nib
import scipy.signal as sig
import scipy.fftpack as fftpack
import pyfftw
import math
import matplotlib.pyplot as plt

def plot_contour(data, thresh, levels, filter, channel, ofile):

    # Create meshgrid
    lx, ly = data.shape
    x_vals = np.arange(0, lx, 1)
    y_vals = np.arange(0, ly, 1)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Create figure
    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
   
    # Calculate values
    data = np.absolute(data)

    if filter != None:
        data = sig.convolve2d(data, filter, mode = 'same', boundary = 'fill', fillvalue = 0)
    
    if thresh != None:
        data[data < (thresh * np.max(data))] = thresh * np.max(data)

    Z = np.log(data)

    # Plot the contour
    cp = ax.contourf(X, Y, Z, 9)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour Plot')
    plt.colorbar(cp)

    if ofile != None:
        if channel != None:
            ofile = ofile + '_chan_' + str(channel)
        plt.savefig(ofile + '.png')

    plt.show()

def main(argv):

    # Default values
    ofile = None
    thresh = None
    filter = None
    levels = 4

    try:
        opts, args = getopt.getopt(argv, 'i:otlf', 
                                   ['ifile=', 'ofile=', 'threshold=', 
                                    'levels=', 'filter='])
    except getopt.GetoptError:
        print('plot_contours.py -i <ifile>')
        sys.exit(2)
    print(opts)
    print(args)
    for opt, arg in opts:
        if opt in ('-i', '--ifile'):
            ifile = arg
        elif opt in ('-o', '--ofile'):
            ofile = arg
        elif opt in ('-t', '--threshold'):
            try:
                thresh = float(arg)
            except:
                print('Threshold must be a float')
                sys.exit(2)
        elif opt in ('-l', '--levels'):
            try:
                levels = int(arg)
                print(levels)
            except:
                try:
                   levels = list(arg)
                except:
                   print('Levels must be an integer or an array of integers')
                   sys.exit(2)

        elif opt in ('-f', '--filter'):
            try:
                filter = np.array(list(arg))
            except:
                print('Filter must be a list')
                sys.exit(2)
            
    img = nib.load(ifile)
    data = img.get_data()

    if len(data.shape) == 2:
        plot_contour(data, thresh, levels, filter, None, ofile)
    elif len(data.shape) == 3:
        channels, _, _ = data.shape
        for channel in range(0, channels):
            plot_contour(data[channel, :, :], thresh, levels, filter, channel, ofile)
    else:
        print('Unknown image dimensions')
        sys.exit(2)

if __name__ == '__main__':
    main(sys.argv[1:])