from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import numba
import scipy
import math
import metrics
import sklearn.manifold as skman 
import sklearn.decomposition as skdec
import time
from visualiser import Visualiser

## Testing Metrics
#data = np.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [-1, -1, -1, 2, 2, 2], [-1, 1, -1, 2, 2, 2]]).astype(np.float32)

#reducer = umap.UMAP(n_components = 2, metric = metrics.bhattacharyya_mvn_vec, metric_kwds = {'dim': 3}, n_neighbors = 3)
#embedding = reducer.fit_transform(data)

#plt.scatter(embedding[:, 0], embedding[:, 1], c = [0, 1, 2, 3], cmap='Spectral', s=5)
#plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(5)-0.5).set_ticks(np.arange(4))
#plt.show()
#exit()


data_df = pd.read_csv('result.csv', header = None, skiprows = lambda x: x % 2 == 0)
files_df = pd.read_csv('result.csv', header = None, skiprows = lambda x: x % 2 == 1)
files_df = files_df.rename(columns = {0 :'ID'})

#for index, row in files_df.iterrows():
#    row[0] = str(row[0]).split('_')[0][3:]

subjects_df = pd.read_csv('subjects.csv', header = 0)

df = pd.concat([files_df, data_df], axis = 1)
df = pd.concat([df, subjects_df], axis = 1)

files = files_df.values
#print(files)
data = data_df.values
labels = ["left" in str(file) or "LEFT" in str(file) or "L_E_F_T" in str(file) for file in files]
labels = [int(label) for label in labels]
labels2 = [str(name[0]).split('_')[2] for name in files]
labels2 = [int(string) if string.isdigit() else np.nan for string in labels2]
labels2 = [5 if el > 5 else el for el in labels2]
slices = [int(str(name[0]).split('_')[-1][:-4]) for name in files]
print(slices)
#print(labels2)
#print(labels)

points = data.shape[0]
dim = 32

means = data[:, :dim]

#print(np.mean(data[:, :dim]))
#print(np.std(data[:, :dim]))

#print(np.mean(data[:, dim:]))
#print(np.std(data[:, dim:]))

var = np.square(data[:, dim:])
data = np.concatenate([means, var], axis = 1)
print(data.shape)

# Pre-apply PCA
#new_dim = 10
#pca = skdec.PCA(n_components = new_dim)
#pca.fit(means)
#new_means = pca.transform(means)

# Here, we have to go from a diagonal covariance matrix to a full one, due to the
# basis change applied by the PCA

#new_comp = pca.components_
#print(new_comp.shape)

#new_var = []
## Now, we apply this transformation to the covariance matrix
#for el in var:
#    cov = np.diag(el)
#    new_cov = new_I.T @ cov
#    new_var.append(np.reshape(new_cov, (new_dim * new_dim,)))

#new_var = np.array(new_var)
##new_var = pca.transform(var)
#print(new_means.shape, new_var.shape)

#new_data = np.concatenate([new_means, new_var], axis = 1)
#print(new_data.shape)

#data = data[::50, :]
#labels = labels[::50]
#new_data = new_data[:, :]
#new_means = new_means[:, :]

reducer5 = umap.UMAP(n_components = 3, verbose = True)
reducer = umap.UMAP(n_components = 2, metric = metrics.fast_bhattacharyya_mvn, metric_kwds = {'dim': dim}, verbose = True)
reducer2 = skman.Isomap(n_components = 3)
reducer3 = skman.LocallyLinearEmbedding(n_components = 2)
reducer4 = skman.SpectralEmbedding(n_components = 2)
#reducer = umap.UMAP(n_components = 3, metric = metrics.bhattacharyya_mvn_vec, metric_kwds = {'dim': dim, 'full_cov': False}, verbose = True)
#reducer = umap.UMAP(n_components = 2)

labels = {'Leg': labels[::20], 'Slice' : slices[::20]}
visualiser = Visualiser(data[::20], labels, reducer)
#visualiser = Visualiser(means[::50], [labels[::50], labels2[::50]], reducer4)
visualiser.fit_transform()
visualiser.plot_result()

##embedding = reducer.fit_transform(new_means)
##embedding = reducer.fit_transform(new_data)
#t = time.time()
#embedding = reducer.fit_transform(data[::50])
#elapsed = time.time() - t
#print("Fitting and transforming took {} seconds".format(elapsed))

## 2D Plot
##plt.scatter(embedding[:, 0], embedding[:, 1], c= labels[:], cmap='Spectral', s=5)
##plt.show()

## 3D Plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c = labels[::50], cmap = 'Spectral', s = 5)
#plt.show()

#print(data)
