import numpy as np
import torch
from itertools import chain, combinations, product
from scipy.stats import entropy
import numpy as np
from sympy.utilities.iterables import multiset_partitions
import math

def _primes(n, prime_numbers = []):
    """ Returns  a list of primes <= n """
    if len(prime_numbers) > 0:
        if prime_numbers[-1] >= n:
            return [p for p in prime_numbers if p <= n]
    else:
        return _generate_primes(n + 1, prime_numbers)

def _limit(n):
    return int(math.ceil(math.sqrt(n)))

def _generate_primes(n, prime_numbers = []):
    start = 3

    if len(prime_numbers) > 0:
        sieve = [False] * prime_numbers[-1]

        start = len(sieve)

        while len(sieve) < n:
            sieve.append(True)

        for p in prime_numbers:
            if p < n:
                sieve[p] = True
    else:
        sieve = [True] * n

    for i in range(start, _limit(n), 2):
        if sieve[i]:
            sieve[i * i :: 2 * i] = [False] * int((n - i * i - 1) / (2 * i) + 1)

    return [2] + [i for i in range(3, n, 2) if sieve[i]]

def _is_prime(n, prime_numbers = []):
    """ Returns true if n is prime; otherwise false """
    if n < 2:
        return False

    if len(prime_numbers) == 0 or prime_numbers[-1] < n:
        prime_numbers = _generate_primes(n + 1, prime_numbers)

    for p in prime_numbers:
        if n < p:
            return False
        if n == p:
            return True

    return False

def _factor(n, prime_numbers = []):
    """ Returns list of prime factors of a integer """
    factors = []

    if not isinstance(n, int):
        return factors

    n = abs(n)

    if n == 0 or n == 1:
        factors.append(n)
    else:
        limit = _limit(n)

        if len(prime_numbers) == 0 or prime_numbers[-1] < limit:
            prime_numbers = _generate_primes(limit + 1, prime_numbers)

        for p in prime_numbers:
            while True:
                if n % p == 0:
                    factors.append(p)

                    n = n // p
                else:
                    break

            if p > n:
                break

        if n != 1:
            factors.append(n)

        if len(factors) == 0:
            factors.append(n)

    return factors



def ind2sub(siz, idx):
    n = len(siz)
    b = len(idx)
    subs = []
    k = np.cumprod(siz[:-1])
    k = np.concatenate((np.ones(1), k))

    for i in range(n - 1, -1, -1):
        subs.append(torch.floor(idx.float() / k[i]))
        idx = torch.fmod(idx, k[i])

    return torch.stack(subs[::-1], dim=1)


def svd_fix(x):
    n = x.shape[0]
    m = x.shape[1]
    
    if n > m:
        u, s, v = torch.svd(x)
        
    else:
        u, s, v = torch.svd(x.t())
        v, u = u, v
    
    return u, s, v

def _get_all_factors(n, d=3):
    
    p = _factor(n)
    if len(p) < d:
        p = p + [1,] * (d - len(p))
    raw_factors = multiset_partitions(p, d)
    clean_factors = [tuple(sorted(np.prod(_) for _ in f)) for f in raw_factors]
    clean_factors = list(set(clean_factors))
    return(clean_factors)
    
    
def get_tt_shape(n, d=3):
    factors = _get_all_factors(n, d)
    if len(factors) == 0:
        raise ValueError('No possible factorizations, try lower d')
    entropies = [entropy(f) for f in factors]
    i = np.argmax(entropies)
    
    return list(factors[i])
    
    
def _roundup(n, k):
    return int(np.ceil(n / 10**k)) * 10**k

def suggest_tt(n, d=3):
    
    e = []
    
    for i in range(len(str(n))):
        e.append(entropy(get_tt_shape(_roundup(n, i), d=d)))

    i = np.argmax(np.array(e))
   
    factors = get_tt_shape(int(_roundup(n, i)), d=d)

    # print('n = %d | suggested voc size: %d with factors %a' % (n, roundup(n, i), factors))

    return factors
