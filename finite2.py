# -*- coding: utf-8 -*-
'''
Created on Fri Oct 16 2020

@author: uqjwhi35
'''

import radon
import farey #local module
import measures #local module
import scipy.fftpack as fftpack
import numpy as np
import math
import matplotlib.pyplot as plt

def isMultiplying(N, x, s):
    return (N % x == -s % x) and -x < s and s < x
       

def isKatzCriterion(P, Q, angles, K = 1):
    '''
    Return true if angle set meets Katz criterion for exact reconstruction of
    discrete arrays
    '''
    sumOfP = 0
    sumOfQ = 0
    n = len(angles)
    for j in range(0, n):
        p, q = farey.get_pq(angles[j])
        sumOfP += abs(p)
        sumOfQ += abs(q)
        
    if sumOfP > K*P or sumOfQ > K*Q:
        return True
    else:
        return False

def computeKatzLines(kSpace, anglesSorted, finiteAnglesSorted, K, centered = True, twoQuads = False):
    '''
    compute finite lines coordinates given Katz criterion
    Returns a list or list of slice 2-tuples and corresponding list of angles and m values
    perp computes the perpendicular (s) lines as well
    '''
    N, M = kSpace.shape
    lines = []
    angles = []
    mValues = []
    for m, angle in zip(finiteAnglesSorted, anglesSorted):
        if isKatzCriterion(M, N, angles, K):
            print("Katz Criterion Met. Breaking")
            break
        m, inv = farey.toFinite(angle, np.minimum(M, N))
        u, v = radon.getSliceCoordinates2(m, kSpace, centered)
        lines.append((u,v))
        mValues.append(m)
        angles.append(angle)
        #second quadrant
        if twoQuads:
            if m != 0 and m != N: #dont repeat these

                # Rotational symmetry
                p, q = farey.get_pq(angle)
                newAngle = farey.farey(-q,p) #not confirmed - SWAPPED P AND Q
                angles.append(newAngle)
                m, inv = farey.toFinite(newAngle, N)
                u, v = radon.getSliceCoordinates2(m, kSpace, centered)
                lines.append((u,v))
                mValues.append(m)
                
    return lines, angles, mValues

def sortedFarey(M, N, K, sortBy = measures.hex, twoQuads=True, centered=False):

    fareyVectors = farey.Farey()
    fareyVectors.compactOn()
    fareyVectors.generateFiniteWithCoverage(np.minimum(M, N))

    vecs = fareyVectors.vectors
    vecs_2q = [farey.farey(-np.imag(vec), np.real(vec)) for vec in vecs]
    fareyVectors.vectors.extend(vecs_2q)
    fareyVectors.generateFinite(N)

    finiteAnglesSorted, anglesSorted = fareyVectors.sortCustom(sortBy)

    kSpace = np.zeros((N,N))
    lines, angles, mValues = computeKatzLines(kSpace, anglesSorted, finiteAnglesSorted, K, centered, twoQuads = False)
    mu = len(lines)
    print("Number of finite lines in fractal:", mu)

    return angles

def fractalise(M, N, angles, smearing = 'all', propagate = 0):

    kSpace = np.zeros((M, N))

    multiples = range(0, np.maximum(M, N) // 2)

    # Filter out unwanted multiples
    if smearing == 'all':
        multiples = range(0, np.maximum(M, N) // 2 + 1)
    else:

        multiples = [0]

        for i in range(1, np.maximum(M, N) // 2):
            for s in smearing:
                if isMultiplying(N, i, s) and isMultiplying(M, i, s):
                    multiples.append(i)

    fractal = []

    # Fractalise
    for i in multiples:
        for angle in angles:
            kSpace[int((np.imag(angle) * -i)) % M, (int(np.real(angle) * i)) % N] = 1
            kSpace[int((np.imag(angle) * i)) % M, (int(np.real(angle) * -i)) % N] = 1

        if propagate:
            fractal.append(np.copy(fftpack.fftshift(kSpace)))

    if not propagate:
        return fftpack.fftshift(kSpace)
    else:
        return fractal
   

def half_circle_points(N):
    angles = []

    for a in range(0, N):
        for b in range(0, N):
            if (a * a + b * b) < (N * N / 9):
                angles.append(a + 1j * b)

    return angles

def circle_points(N):
    angles = []

    for a in range(-N, N):
        for b in range(-N, N):
            if (a * a + b * b) < (N * N / 100):
                angles.append(a + 1j * b)

    return angles

def spiral_points(N):
    angles = []

    t = 0

    while 1:
        r = np.exp((t + 10)/100)
        theta = np.exp(t / 100)

        x = np.round(r * np.cos(theta))
        y = np.round(r * np.sin(theta))

        if x > N / 6 or y > M / 6:
            break

        a = int(x)
        b = int(y)

        #if math.gcd(a, b) == 1:

        angles.append(a + 1j * b)

        t += 0.01

    return angles

def spiral_points2(N):

    angles = []

    for a in range(0, N):
        for b in range(0, N):
            r0 = np.sqrt(a^2 + b^2)
            t = math.atan2(b,a)

            r = t * np.exp(0.1)

            if np.abs(r - r0) % (2 * math.pi) <= 1:
                angles.append(a + 1j * b)