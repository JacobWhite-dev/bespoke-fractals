import numpy as np

for N in range(0, 200):
    for L in range(0, N):
        works = False
        for m in range(0, N):
            if works == True:
                break
            for nu in range(1, N):
                if np.ceil(m * N / nu) == L or np.floor(m * N / nu) == L:
                    works = True

        if works == False:
            print("N = {}, L = {} does not work".format(N, L))
