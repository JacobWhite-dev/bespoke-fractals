"""
UMAP dimensionality reducer for use with VAE latent spaces.

Created on Mon Feb 10 2020 

@author: uqjwhi35
"""

import umap
from metrics import fast_bhattacharyya_mvn

class VAEUMAP(umap.UMAP):
    '''
    Wrapper class for initialising a UMAP reducer object using an appropriate
    metric for variational autoencoder (VAE) latent spaces (i.e. ones in which
    the points in the space represent distributions). The metric used is the
    Bhattacharyya distance: https://en.wikipedia.org/wiki/Bhattacharyya_distance

    This class is identical to the umap.UMAP class other than that:
        1. The number of dimensions of the latent space must be provided. This
           is done to reduce computation time.
        2. Each point in the latent space provided is assumed to be of the form
           [m1, m2, ..., mn, s1, s2, ..., sn] where mi is the mean of the 
           distribution along the ith dimension and si is the variance of the
           distribution along this same dimension.

    All of the keyword arguments supported by the umap.UMAP class are supported,
    other than metric and metric_kwds. These are listed in the UMAP 
    documentation at https://umap-learn.readthedocs.io/en/latest/api.html.
    '''

    def __init__(self, dim, **kwargs):
        super().__init__(metric = fast_bhattacharyya_mvn, 
                         metric_kwds = {'dim': dim}, 
                         **kwargs)