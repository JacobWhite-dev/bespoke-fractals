import scipy.fftpack as fftpack
#import pyfftw
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams

# Monkey patch in fftn and ifftn from pyfftw.interfaces.scipy_fftpack
#fftpack.fft2 = pyfftw.interfaces.scipy_fftpack.fft2
#fftpack.ifft2 = pyfftw.interfaces.scipy_fftpack.ifft2
#fftpack.fft = pyfftw.interfaces.scipy_fftpack.fft
#fftpack.ifft = pyfftw.interfaces.scipy_fftpack.ifft

# Turn on the cache for optimum performance
#pyfftw.interfaces.cache.enable()

def scale_vec(frame, original, scaled, og, scal, scale, vecs):
    vec = vecs[frame]
    N = original.shape[0]

    a = np.real(vec)
    b = np.imag(vec)
    norm_z = int(np.abs(vec))

    original[int(b) % N, int(a) % N] = 0
    scaled[int(b * scale) % N, int(a * scale) % N] = 1

    og.set_data(fftpack.fftshift(original))
    scal.set_data(fftpack.fftshift(scaled))

    return og, scal

rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'
ims = []
N = 311
scale = 78

fig = plt.figure();
    
s = 50
    
vecs = []

# Make vectors representing square
for i in np.arange(-s//2, s//2):
    for j in np.arange(-s//2, s//2):
        vecs.append(i + 1j * j)

# Make vecs for any shape

original = np.zeros((N, N))

for vec in vecs:
    original[int(np.imag(vec)) % N, int(np.real(vec)) % N] = 1

#scaled = np.copy(original)
scaled = np.zeros_like(original)
scaled[0, 0] = 1

fig, axes = plt.subplots(1, 2)
og = axes[0].imshow(original)
scal = axes[1].imshow(scaled)
ani = animation.FuncAnimation(fig, scale_vec, len(vecs), fargs=(original, scaled, og, scal, scale, vecs),
                                   interval=1, blit=True)
plt.show()
