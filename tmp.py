import math
import numbertheory as nt # local modules

def farey(p, q):
    '''
    Convenience member for creating a Farey vector from a Farey fraction p/q
    '''
    return complex(int(q), int(p))

def getX(angle):
    '''
    Convenience function for extracting the consistent coordinate from a Farey vector
    This is based on matrix coordinate system, i.e. x is taken along rows and is normally q of p/q vector
    '''
    if not isinstance(angle, complex):
        print("Warning: Angle provided not of correct type. Use the farey() member.")

    return angle.real

def getY(angle):
    '''
    Convenience function for extracting the consistent coordinate from a Farey vector
    This is based on matrix coordinate system, i.e. y is taken along columns and is normally p of p/q vector
    '''
    if not isinstance(angle, complex):
        print("Warning: Angle provided not of correct type. Use the farey() member.")
    
    return angle.imag

def get_pq(angle):
    '''
    Return p, q tuple of the angle provided using module convention
    '''
    p = int(angle.imag)
    q = int(angle.real)
    
    return p, q

def projection_length(angle, M, N):
    '''
    Return the number of bins for projection at angle of a MxN image.
    '''
    p, q = get_pq(angle)
    return (N - 1) * abs(M) + (M - 1) * abs(N) + 1 #no. of bins

def total(n):
    '''
    Return the approx total Farey vectors/angles possible for given n
    '''
    return int(0.304 * n * n + 0.5)

def size(mu):
    '''
    Given number of projections mu, return the approx size n
    '''
    return int(math.sqrt( mu / 0.304))

def angle(angle, radians=True):
    '''
    Given p and q, return the corresponding angle (in Radians by default)
    '''
    p, q = get_pq(angle)

    theta = 0 if p == 0 else math.atan(q / float(p))

    return theta if radians else 180 / math.pi * theta

def toFinite(fareyVector, N):
    '''
    Return the finite vector corresponding to the Farey vector provided for a given modulus/length N
    and the multiplicative inverse of the relevant Farey angle
    '''
    p, q = get_pq(fareyVector)
    coprime = nt.is_coprime(abs(q), N)
