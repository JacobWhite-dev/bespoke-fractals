import finite
import farey
import numbertheory as nt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import bespoke
import scipy.fftpack as fftpack
import pyfftw
import itertools

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

if __name__ == "__main__":

    ims = []

    N = 113

    fig = plt.figure()

    #img = np.zeros((N, N))

    fareyVectors = farey.Farey()
    fareyVectors.compactOn()
    fareyVectors.generate(N, octants = 1)
    vecs = fareyVectors.sort(type = 'Euclidean')
    numVecs = len(vecs)
    vecs = vecs[0:numVecs // 6]
    vecs[0:numVecs // 12] = vecs[0:numVecs // 12] * 2

    vecs = [a + 1j * b for a,b in itertools.product(range(N // 2), range(N // 2))]

    # normalise
    #vecsNorm = [vec / np.abs(vec) for vec in vecs]

    for j in range(N):

        #vecs = fareyVectors.sortCustom(bespoke.ellipse)
        #vecs2 = [j * vec for vec in vecsNorm]
        vecs2 = [j * vec for vec in vecs]

        img = np.zeros((N, N))

        #i = len(vecs2) // 2
        for vec in vecs2:

            if int(np.real(vec)) % 2 == 0:
                img[int(np.imag(vec)) % N, int(np.real(vec)) % N] += 5
            else:
                img[int(np.imag(vec)) % N, int(np.real(vec)) % N] += 1

            #i += 1

        #img = fftpack.fftshift(img)
        im = plt.imshow(img, animated = True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=1000)

    plt.show()
        


