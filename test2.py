import finite
import farey
import numbertheory as nt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import bespoke
import scipy.fftpack as fftpack
#import pyfftw
import itertools
#import ffmpeg
import pylab as pl
from fracdim import fractal_dimension
import sys

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
#fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
#fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
#fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
#fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
#pyfftw.interfaces.cache.enable()

if __name__ == "__main__":

    #rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'

    ims = []

    #N = 2 * 3 * 5 * 7 * 11 - 1;
    #N = 3124

    N = 255

    #fig = plt.figure()

    #img = np.zeros((N, N))

    #fareyVectors = farey.Farey()
    #fareyVectors.compactOn()
    #fareyVectors.generate(N, octants = 5)8
    #vecs = fareyVectors.sort(type = 'Euclidean')
    #numVecs = len(vecs)
    #vecs = vecs[0: numVecs // 100]

    #K = 1
    #_, angles, _, fractal, _ = finite.finiteFractal(N, K, sortBy = 'Euclidean', twoQuads = True)
    #img = fractal
    #print(fractal_dimension(fractal))

    #plt.imshow(fractal)
    #plt.show()

    #sys.exit()

    #vecs[0:numVecs // 12] = vecs[0:numVecs // 12] * 2

    vecs = np.array([a + 1j * b for a,b in itertools.product(range(-N//10, N//10), range(-N // 10, N//10))])
    #vecs = vecs[np.abs(vecs) < N//6]
    #print(vecs)
    #print(np.abs(vecs) < N//6)
    #vecs = np.array(vecs)
    #vecs = vecs[np.abs(vecs) < N/6]
    #numVecs = len(vecs)
    #vecs = vecs[0: numVecs // 3]
    #vecs = np.array([1, 1j, 2 + 1j, -1, -1j, -2 - 1j])
    #vecs = np.random.choice(vecs, size = N * N // 8)

    # normalise
    #vecsNorm = [vec / np.abs(vec) for vec in vecs]

    #vecs2 = vecs

    # Set up formatting for the movie files
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    img = np.zeros((N, N))
    
    ###for j in [13]:
    for j in range(N):

        # UNCOMMENT THIS TO ONLY GET REPEATS
        if j != 0:
           if (N % j != -1 % j) and (N % j != -1 % j):
                continue
                #pass
        #print(j)

        #if j != 156:
        #    continue

        #vecs = fareyVectors.sortCustom(bespoke.ellipse)
        vecs2 = [j * vec for vec in vecs]

        #vecs2 = [j * (vec) for vec in vecs] 

        #img = np.zeros((N, N))

        #i = 1
        for vec in vecs2:

            img[int(np.imag(vec)) % N, int(np.real(vec)) % N] = 1

        #img = fftpack.fftshift(img)
        #im = plt.imshow(img, animated = True)
        #plt.show()
        #ims.append([im])

    ###ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    ###ani.save('wave.mp4', writer=writer)

    #plt.imshow(img)
    img = fftpack.fftshift(img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print(fractal_dimension(img))

    #sys.exit()

    # Do a box count
    image = 1 - img
    pixels=[]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]>0:
                pixels.append((i,j))
 
    Lx=image.shape[1]
    Ly=image.shape[0]
    print(Lx, Ly)
    pixels=pl.array(pixels)
    print(pixels.shape)

    # computing the fractal dimension
    #considering only scales in a logarithmic list
    scales=np.logspace(0.1, np.log(Lx / 8) / np.log(2), num=100, endpoint=False, base=2)
    Ns=[]
    # looping over several scales
    for scale in scales:
        #print ("======= Scale :",scale)
        # computing the histogram

        H, edges=np.histogramdd(pixels, bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))
        Ns.append(np.sum(H>0))
    
    print(Ns)
 
    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)

    plt.plot(np.log(scales), np.log(Ns))
    plt.show()

    dims = np.divide(np.diff(np.log(Ns)), np.diff(np.log(scales)))
    dims_av = np.convolve(dims, np.array([0.33, 0.33, 0.33]), mode = 'same')

    plt.plot(np.log(scales)[:-1], dims_av)
    plt.show()

    print ("The Hausdorff dimension is", -coeffs[0])
        


