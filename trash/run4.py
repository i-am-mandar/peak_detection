#run4.py

import time
from numba import njit
import numpy as np

# compare number-accerletated for loop convolution to numpy's func

@njit
def convolve_1d(signal, kernal):
    """
    numba accerletion cuts computation time down by a factor of 30
    head to head with numpy's convolve(), this function comes in about 25% slower
    """
    n_sig = signal.size
    n_ker = kernal.size
    n_conv = n_sig - n_ker + 1
    # precalculating the reversed kernel cuts the computation time down
    # by a factor of 3
    rev_kernal = kernal[::-1].copy()
    result = np.zeros(n_conv)
    for i_conv in range(n_conv):
        # using np.dot() instead of np.sum over the products cuts
        # the computation time is down by a factor of 5
        result[i_conv] = np.dot(signal[i_conv: i_conv + n_ker], rev_kernal)
    return result
    
n_iter = 10000
n_arr = 1000
n_kernel = 100
t_numba = 0
t_numpy = 0

for i in range(n_iter + 1):
    signal = np.random.normal(size=n_arr)
    kernel = np.random.normal(size=n_kernel)
    start = time.time()
    result = np.convolve(signal, kernel, mode="valid")
    if i > 0:
        t_numpy += time.time() - start
    

for i in range(n_iter + 1):
    signal = np.random.normal(size=n_arr)
    kernel = np.random.normal(size=n_kernel)
    start = time.time()
    result = convolve_1d(signal, kernel)
    if i > 0:
        t_numba += time.time() - start 
        
print(f'Numpy: {1e6 * t_numpy / n_iter} us')
print(f'Numba: {1e6 * t_numba / n_iter} us')