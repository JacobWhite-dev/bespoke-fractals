import finite
import farey
import numbertheory as nt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    N = 17
    a = 1
    b = 10

    lst_of_multiples = np.zeros((N, 2))
    lst_of_thingos = np.zeros((N, 2))

    # find grad
    m = nt.minverse(a, N) * b

    for i in range(0, N):
        lst_of_multiples[i, :] = np.array([[(a * i) % N, (b * i) % N]])
        lst_of_thingos[i, :] = np.array([[i % N, (m * i) % N]])

    img1 = np.zeros((N, N))
    img2 = np.zeros((N, N))

    for i in range(N):
        img1[int(lst_of_multiples[i, 0]), int(lst_of_multiples[i, 1])] = 1
        img2[int(lst_of_thingos[i, 0]), int(lst_of_thingos[i, 1])] = 1

    # Plot each
    fig, axes = plt.subplots(1, 2)

    print(img1)

    axes[0].imshow(img1)
    axes[1].imshow(img2)

    plt.show()

    

