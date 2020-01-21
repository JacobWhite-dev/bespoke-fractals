import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import numba
import scipy
import math
import metrics

data_df = pd.read_csv('result.csv', header = None, skiprows = lambda x: x % 2 == 0)
files_df = pd.read_csv('result.csv', header = None, skiprows = lambda x: x % 2 == 1)
files = list(files_df.values)
files = [str(file).split('_') for file in files]
files = [file[0][5:] for file in files]

subjects_df = pd.read_csv('subjects.csv', header = 0)

print(subjects_df)

data = data_df.values


sides = [subjects_df.loc('Side', files[0][i]) for i in range(len(data))]
print(sides)

print(data.shape)

points = data.shape[0]
dim = data.shape[1] / 2

means = data[:, 0:int(dim - 1)]

reducer = umap.UMAP(n_components = 2)
#reducer = umap.UMAP(n_components = 2)

embedding = reducer.fit_transform(data)

plt.plot(embedding[:, 0], embedding[:, 1], 'bo')
plt.show()

#print(data)
