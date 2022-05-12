import numba
import numpy as np


@numba.njit
def discrete_rvs(p):
    p = p / np.sum(p)
    return np.random.multinomial(1, p).argmax()


@numba.njit
def log_normalize(x):
    return x - log_sum_exp(x)


@numba.njit
def log_sum_exp(log_X):
    max_exp = np.max(log_X)

    if np.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += np.exp(x - max_exp)

    return np.log(total) + max_exp


@numba.jit(nopython=True)
def log_factorial(x):
    return log_gamma(x + 1)


@numba.vectorize([numba.float64(numba.float64)])
def log_gamma(x):
    return np.math.lgamma(x)


@numba.njit()
def log_poisson_pdf(x, rate):
    return -rate + x * np.log(rate) - log_factorial(x)  
