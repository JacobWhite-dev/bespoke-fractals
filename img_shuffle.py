from skimage import io
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import finite2

def kal_round(x, sigma):
    if sigma <= 0:
        return int(np.floor(x))
    else:
        return int(np.ceil(x))

def kaleidoscope3(img, kappa, sigma):
    imgNew = np.zeros_like(img, dtype = int)

    #img = img // (2 * np.abs(sigma))

    h, w = img.shape

    rows = np.arange(h)
    cols = np.arange(w)

    for r in rows:
        for c in cols:

            # Might need to be ceiling of h / kappa not floor
            #m1 = np.mod(((h // kappa)) * np.mod(r, kappa) + sigma * (r // kappa), h)
            #m2 = np.mod(((w // kappa)) * np.mod(c, kappa) + sigma * (c // kappa), w)

            m1 = np.mod(kal_round(h / kappa, sigma) * np.mod(r, kappa) + sigma * (r // kappa), h)
            m2 = np.mod(kal_round(w / kappa, sigma) * np.mod(c, kappa) + sigma * (c // kappa), w)

            #print(r, c, m1, m2)

            imgNew[m1, m2] += img[r, c]


    return imgNew

def kaleidoscope(img, kappa, sigma):
    imgNew = np.zeros_like(img, dtype = int);

    #img = img // (2 * np.abs(sigma))

    h, w = img.shape
    print(h)

    rows = np.arange(h)
    cols = np.arange(w)

    for r in rows:
        for c in cols:

            # Might need to be ceiling of h / kappa not floor
            #m1 = np.mod(((h // kappa)) * np.mod(r, kappa) + sigma * (r // kappa), h)
            #m2 = np.mod(((w // kappa)) * np.mod(c, kappa) + sigma * (c // kappa), w)

            m1 = np.mod(math.ceil(h / kappa) * np.mod(r, kappa) + sigma * (r // kappa), h)
            m2 = np.mod(math.ceil(w / kappa) * np.mod(c, kappa) + sigma * (c // kappa), w)

            #print(r, c, m1, m2)

            imgNew[m1, m2] += img[r, c]


    return imgNew

def kaleidoscope2(img, kappa, sigma):
    imgNew = np.zeros_like(img)

    img = img // np.abs(sigma) # Should this be in the equation?

    h, w = img.shape

    rows = np.arange(h)
    cols = np.arange(w)

    for k1 in rows:
        for k2 in cols:
            for n1 in rows:
                for n2 in cols:

                    k01 = np.mod((h // kappa) * np.mod(n1, kappa) + sigma * (n1 // kappa), h)
                    k02 = np.mod((w // kappa) * np.mod(n2, kappa) + sigma * (n2 // kappa), w)

                    if (k1 == k01) and (k2 == k02):
                        imgNew[k1, k2] = img[n1, n2]

    return imgNew


def shuffle(img, a):
    
    imgNew = np.zeros_like(img)

    h, w = img.shape

    rows = np.arange(h)
    cols = np.arange(w)

    for r in rows:
        for c in cols:
            if img[r, c] == 0:
                continue

            imgNew[(a * r) % h, (a * c) % w] += img[r, c]

    return imgNew

lena = io.imread("9.gif", as_gray= True)
#lena = lena[250:561, 250:561]
plt.imshow(lena)
plt.show()
print(lena.shape)
#new_lena = np.zeros((1024, 1024))
#new_lena[0:512, 0:512] = lena

#img = shuffle(new_lena, 205)
img3 = shuffle(lena, 205)
img2 = kaleidoscope3(lena, 3, 103)
#print(img.shape)

#img1r = kaleidoscope3(lena[:,:,0], 5, 3)
#img1g = kaleidoscope3(lena[:,:,1], 5, 3)
#img1b = kaleidoscope3(lena[:,:,2], 5, 3)

#img1 = np.stack([img1r, img1g, img1b], axis = -1)

##img = np.abs(img2)

#plt.imsave("lpc3,3.png", img1)
#plt.imsave("aaa.png", img2, cmap = 'gray')
#plt.imsave("aaacomp.png", np.abs(img3 - img2), cmap = 'gray')

print(np.max(img3 - img2))

plt.imshow(img2)
plt.axis("off")
plt.show()

#plt.imshow(img2)
#plt.axis("off")
#plt.show()

#circleImg = np.zeros((1069, 1069))
#cv2.circle(circleImg, (512, 512), 300, 20)

#ims = []
#fig = plt.figure()
### Make wave
#for a in np.arange(0, np.max(circleImg.shape), 1):
#    img = shuffle(circleImg, a)

#    im = plt.imshow(img, cmap = 'gray', animated = True)
#    #plt.title("a = {}".format(a))
#    ims.append([im])
#    print(a)

#ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)

#plt.show()