'''
Custom metrics for use with the UMAP. Particular focus is given to metrics
suitable for performing dimensionality reduction on VAE latent spaces, i.e.
metrics for distance between two distributions as opposed to two points. 

Created on Fri Jan 17 2020

@author: uqjwhi35
'''

import math
import numba
import numpy as np

###############################################
# Generic Functions for Distribution Measures #
###############################################

@numba.njit()
def unpack_vec(v, dim = 1, full_cov = False):
    '''
    Extract the mean and covariance matrix from a vector.

    Parameters:
        v - means and variances concatenated into a single vector, i.e.
                    v0 = [m0, m1, ..., mn, s0, s1, ..., sn]
                where mn and sn are the mean and variance of the distribution
                along a given dimension
        dim - dimension of the distribution. Defaults to 1

    Output:
        m, S - mean and covariance matrix respectively
    '''

    # Extract means and variances
    m = v[:dim].astype(np.float64) # Means for distribution
    s = v[dim:].astype(np.float64) # Variances for distribution

    # Construct covariance matrix from variances, assuming covariance
    # matrix is diagonal (full_cov = True) or giving all covariances
    # (full_cov = True)
    if full_cov:
        S = np.reshape(s, (dim, dim)) # Covariance matrix for distribution
    else:
        S = np.diag(s) # Covariance matrix for distribution

    return m, S

@numba.njit()
def vec_metric(v0, v1, measure, dim = 1, full_cov = False):
    '''
    Calculate a given measure between two normal distributions.

    Parameters:
        v0, v1 - means and variances for distribution 0 and distribution 1
                 respectively concatenated into a single vector, i.e.
                    v0 = [m0, m1, ..., mn, s0, s1, ..., sn]
                 where mn and sn are the mean and variance of the distribution
                 along a given dimension
        measure - function taking 
        dim - dimension of the distributions. Defaults to 1

    Output:
        Value of the measure between the two distributions

    '''

    m0, S0 = unpack_vec(v0, dim, full_cov)
    m1, S1 = unpack_vec(v1, dim, full_cov)

    return measure(m0, S0, m1, S1)

####################################
# Kullback–Leibler (KL) Divergence #
####################################

'''
The Kullback–Leibler (KL) divergence is a directional measure of 
how one probability distribution is different from another. It
may be thought of as a measure of surprise. A KL divergence of 0
indicates the two distributions are identical. Note that due to
its asymmetry (and the fact it does not obey the triangle
inequality), KL divergence is a measure and not a metric.
'''

@numba.njit()
def KL_mvn(m0, S0, m1, S1):
    '''
    Calculate KL divergence of one multivariate normal distribution 
    (distribution 0) to another (distribution 1), given the means
    and covariance matrices of each.
    
    Parameters:
        m0, m1 - vector of means for distribution 0 and 1 respectively
        S0, S1 - covariance matrices for distribution 0 and 1 respectively

    Output:
        KL divergence from distribution 0 to distribution 1
    '''

    # Type casting to avoid numba issues
    m0 = m0.astype(np.float64)
    m1 = m1.astype(np.float64)
    S0 = S0.astype(np.float64)
    S1 = S1.astype(np.float64)

    # Pre-calculations
    dm = m0 - m1             # Difference of means
    iS1 = np.linalg.inv(S1)  # Inverse of covariance matrix 1
    k = m0.shape[0]          # Dimension of distributions

    # Terms of divergence
    trace_term = np.trace(iS1 @ S0)
    means_term = dm.T @ iS1 @ dm - k
    log_term = math.log(np.linalg.det(S1) / np.linalg.det(S0))

    # Final result
    return 0.5 * (trace_term + means_term + log_term)

@numba.njit()
def KL_mvn_vec(v0, v1, dim = 1, full_cov = False):
    '''
    Calculate KL divergence of one multivariate normal distribution 
    (distribution 0) to another (distribution 1), given a single vector
    containing the means and variances for each.
    
    Parameters:
        v0, v1 - means and variances for distribution 0 and distribution 1
                 respectively concatenated into a single vector, i.e.
                    v0 = [m0, m1, ..., mn, s0, s1, ..., sn]
                 where mn and sn are the mean and variance of the distribution
                 along a given dimension
        dim - dimension of the distributions. Defaults to 1

    Output:
        KL divergence from distribution 0 to distribution 1
    '''

    return vec_metric(v0, v1, KL_mvn, dim, full_cov)

##########################
# Bhattacharyya Distance #
##########################

'''
The Bhattacharyya distance measures the similarity of two probability
distributions. It is heavily related to the Bhattacharyya coefficient,
which is a measure of the overlap between two samples or distributions.
'''

@numba.njit()
def bhattacharyya_mvn(m0, S0, m1, S1):
    '''
    Calculate Bhattacharyya distance between two multivariate normal 
    distributions (0 and 1), given the means and covariance matrices of each.
    
    Parameters:
        m0, m1 - vector of means for distribution 0 and 1 respectively
        S0, S1 - covariance matrices for distribution 0 and 1 respectively

    Output:
        Bhattacharyya distance between distributions 0 and 1
    '''

    # Type casting to avoid numba issues
    m0 = m0.astype(np.float64)
    m1 = m1.astype(np.float64)
    S0 = S0.astype(np.float64)
    S1 = S1.astype(np.float64)

    # Pre-calculations
    dm = m0 - m1             # Difference of means
    S = 0.5 * (S0 + S1)      # Mean of covariance matrices
    iS = np.linalg.inv(S)    # Inverse of mean of covariance matrices

    # Terms of distance
    means_term = 0.125 * dm.T @ iS @ dm
    log_term = 0.5 * math.log(np.linalg.det(S) / 
                              (math.sqrt(np.linalg.det(S0) * 
                                         np.linalg.det(S1))))

    # Final result
    return means_term + log_term

@numba.njit()
def bhattacharyya_mvn_vec(v0, v1, dim = 1, full_cov = False):
    '''
    Calculate Bhattacharyya distance between two multivariate normal 
    distributions (0 and 1), given a single vector containing the 
    means and variances for each.
    
    Parameters:
        v0, v1 - means and variances for distribution 0 and distribution 1
                 respectively concatenated into a single vector, i.e.
                    v0 = [m0, m1, ..., mn, s0, s1, ..., sn]
                 where mn and sn are the mean and variance of the distribution
                 along a given dimension
        dim - dimension of the distributions. Defaults to 1

    Output:
        Bhattacharyya distance between distribution 0 and distribution 1
    '''

    return vec_metric(v0, v1, bhattacharyya_mvn, dim, full_cov)

@numba.njit(parallel = True, fastmath = True)
def fast_bhat(v0, v1, dim = 1):
    # INCORRECT


    # Type casting to avoid numba issues
    m0 = v0[:dim].astype(np.float64)
    m1 = v1[:dim].astype(np.float64)
    s0 = v0[dim:].astype(np.float64)
    s1 = v1[dim:].astype(np.float64)

    # Pre-calculations
    dm = m0 - m1             # Difference of means
    s = s0 + s1              # Sum of covariance matrices
    
    return 0.25 * (np.sum(dm * dm / s) + (np.sum(s0) + np.sum(s1)))






