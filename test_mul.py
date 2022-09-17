#!/usr/bin/env python3

import gpu_library
import numpy as np
import time

dtype = np.float32
size = 10000000
power = 4
factor = 3.0
arr1 = np.linspace(1.0,100.0, size, dtype=dtype)
arr2 = np.linspace(1.0,100.0, size, dtype=dtype)

factor = dtype(factor)

t0 = time.time()
for _ in range(power):
    gpu_library.multiply_with_scalar(arr1, factor)
print("gpu time: {}".format(time.time()-t0))
t0 = time.time()
arr2 = arr2 * (factor ** power)
print("cpu time: {}".format(time.time()-t0))
if not np.allclose(arr1,arr2):
    print("results mismatch: Max discrepancy: {}".format(
        np.max(np.abs(arr1-arr2))
    ))
else:
    print("results match")
