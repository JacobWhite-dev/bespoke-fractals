import numpy as np
from matplotlib import pyplot as plt
import math

sequence = np.array([2, 2, 2, 1, 1, 1, 0, 0])

#Python program for Extended Euclidean algorithm
def egcd(a, b):
	if a == 0:
		return (b, 0, 1)
	else:
		gcd, x, y = egcd(b % a, a)
		return (gcd, y - (b//a) * x, x)

def kaleidoscope(sequence, t):
    result = np.zeros_like(sequence)

    N = np.size(sequence, axis = 0)

    for n in np.arange(N):
        m = (t * (n % t * (n % (N // t))) - (N % t) * (n // (N //t))) % N
        result[m] += sequence[n]

    return result

#result = np.zeros((np.size(sequence), np.size(sequence)))

#result[0, :] = sequence

#for t in range(1, np.size(sequence)):
#    result[t, :] = kaleidoscope(sequence, t)

#plt.imshow(result)
#plt.show()

def delta(n, N):
    if n % N == 0:
        return 1
    else:
        return 0

def k0(nu, sigma, N):
    return np.array([np.mod(int(np.ceil(N/nu)) * np.mod(n, nu) + sigma * np.floor(n / nu), N) for n in range(0, N)])

def check_solns2(s, sigma, nu, N):

    for x in range(-nu + 1, nu):
        for y in range(-s + 1, s):
            if x == 0 and y == 0:
                continue

            if (nu * y + x) >= N or (nu * y + x) <= -N:
                continue

            if (s * x + sigma * y) % N == 0:
                    return True

    return False

def check_solns(s, sigma, nu, N):

    #m = int(np.ceil(s * (nu + sigma) / N))

    d = math.gcd(s, sigma)

    gcd, x, y = egcd(s, sigma)

    #print(gcd, x, y)

    if N % gcd != 0:
        return False

    x = x * N / gcd;
    y = y * N / gcd;

    #if (m * N) % d == 0:
    #    return True
    #else:
    #    return False

    #d = math.gcd(s, sigma)

    u = int(s / d)
    v = int(sigma / d)

    if (x < nu) and (x > -nu) and (y < s) and (y > -s):
        return True
    else:
        return False

    #return False
    #if math.gcd(math.gcd(s, sigma), N) == 1:
    #    return False

    #for x in range(-nu + 1, nu):
    #    for y in range(-s + 1, s):
    #        if x == 0 and y == 0:
    #            continue

    #        if (s * x + sigma * y) % N == 0:
    #                return True

    #return False

def test3(N, nu, sigma):

    L = int(np.ceil(N / nu))

    if math.gcd(N, math.gcd(L, sigma)) == 1:
        return True, True
    else:
        return False, False

def test2(N, nu, sigma):

    L = int(np.ceil(N / nu))

    if math.gcd(L,N) <= N / nu and math.gcd(sigma, N) <= N / (N // nu):
        return True, False
    else:
        return False, False

def test(N, nu, sigma):

    #return (N % nu == 0 or N % nu == -sigma % nu)

    #return ((np.ceil(N / nu) * sigma) % N != 0)

    #if (N % nu == 0 or (N + sigma) % nu == 0):
    #    #if math.gcd(N // nu, sigma) == 1 and math.gcd((N + sigma) // nu, sigma) == 1:
    #    return True
    #return False

    sigma = sigma % N;

    #s = int(np.ceil(N / nu))

    #return not check_solns(s, sigma, nu, N), False

    #sigma = sigma % N

    if sigma == 0:
        return False, False

    if N % nu == 0:
        if math.gcd(N // nu, sigma) == 1:
            #print("top path")
            return True, False

    if (N + sigma) % nu == 0:

        spacing = int(np.round(N / nu))

        if N + sigma != spacing * nu:
            return False, True

        if math.gcd((N + sigma) // nu, N) == 1:
            return True, True

        #if math.gcd(spacing, N) == math.gcd(sigma, N) == 1: 
        #if math.gcd(spacing, sigma) == 1 and math.gcd(spacing, N) == 1:
        #z = math.gcd(spacing, sigma)

            # Bottom path is wrong sometimes (TOP PATH IS ALWAYS RIGHT)
            
            #print("bottom path")
            #return True, True

    return False, False

    #return (N % nu == 0 or N % nu == -sigma % nu)

    #if N % nu == 0:
    #    if (N / sigma) % nu != 0:
    #        return True
    #elif N % nu == -sigma % nu:
    #    if ((N + sigma) / sigma) % nu != 0:
    #        return True

    #else:
    #    return False

def kaleidoscope(x, nu, sigma):
    N = np.size(x)
    X = np.zeros_like(x)

    flag = 0
    k_zero = k0(nu, sigma, N)
    #res = not check_solns2(int(np.ceil(N / nu)), sigma % N, nu, N);
    #path = False
    res, path = test(N, nu, sigma)
    if np.size(np.unique(k_zero)) == np.size(k_zero):
        #print(N, nu)
        #print((N % nu == 0 or N % nu == -sigma % nu))
        
        # Seems to be NECESSARY, but not SUFFICIENT
        #assert((N % nu == 0 or N % nu == -sigma % nu))
        
        if not res:
            print("FN:", N, nu, sigma, path)
        #assert(test(N, nu, sigma))
    else:
        if res:
            print("FP:", N, nu, sigma, path)
        pass
        #print(k_zero)
        #print(N % nu != 0 and N % nu != -sigma % nu)
        #print(not (N % nu == 0 or N % nu == -sigma % nu))

    for k in range(0, N):
        for n in range(0, N):
            f = x[n] * delta(k - k_zero[n], N)

            #if (f != 0 and X[k] != 0):
            #    if not flag:
            #        print(N, nu)
            #        flag = 1

            X[k] += f 

    return X



print(k0(5, 2, 7))
print(k0(4, 3, 7))
sys.exit()

for N in range(2, 101):
    print(N)
    x = np.array([n for n in range(0, N)])
    for nu in range(1, N):
        for sigma in range (-N + 1, N):
            X = kaleidoscope(x, nu , sigma)
            #Y = kaleidoscope(X, N - nu, sigma)

            #if test(N, nu, sigma) and np.any(X - Y):
            #    print("Failure to invert:", N, nu, sigma)
            continue
        #print(x)
        #if np.any(k0(nu, 1, N) - kaleidoscope(x, nu, 1)):
        #    print(nu)
        #    #print(k0(nu, 1, N))
            #print(kaleidoscope(x, nu, 1))

        #print(nu)
        #print(kaleidoscope(x, nu, 1))
        #print(nu)
        #print(k0(nu, 1, N))
        #print("-----------------")




sys.exit()

N = 13
nu = 3
sigma = -1

print(k0(nu, sigma, N))

#sys.exit()

x = np.array([i for i in range(0, N)])
y = x
print(y)

i = 0

while True:
    y = kaleidoscope(y, nu, sigma)
    i +=1
    print(y)
    if not np.any(y - x):
        print(i)
        break

def lcm(a, b):
    return abs(b) // math.gcd(a, b)

print(np.power(int(np.ceil(N / nu)), i) % N == 1)

#x = kaleidoscope(np.array([i for i in range(0, 15)]), nu, sigma)
#y = kaleidoscope(x, 3, 2)
#z = kaleidoscope(y, 3, 2)
#a = kaleidoscope(z, 3, 2)
#b = kaleidoscope(a, 3, 2)
#c = kaleidoscope(b, 3, 2)
#d = kaleidoscope(c, 3, 2)
#e = kaleidoscope(d, 3, 2)


#print(x)
#print(y)
#print(z)
#print(a)
#print(b)
#print(c)
#print(d)
#print(e)

