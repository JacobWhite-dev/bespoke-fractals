import numba
import numpy as np

@numba.njit()
def KL_mvn(m0, S0, m1, S1):
    '''
    m0, m1 are vectors of means
    S0, S1 are covariance matrices
    '''

    delta_m = m1 - m2
    iS1 = np.inv(S1)
    k = m0.shape[0]

    return 0.5 * (np.trace(iS1 @ S0) + delta_m.T @ iS1 @ delta_m - k + 
                  np.log(np.det(S1) / np.det(S0)))

@numba.njit()
def KL_mvn_vec(m0, s0, m1, s1):
    return KL_mvn(m0, np.diag(s0), m1, np.diag(s1))

@numba.njit()
def KL_mvn_single(a, b, n):
    return KL_mvn_vec(a[:n], a[n:], b[:n], b[n:])


    

