#!/usr/bin/env python3

import numpy as np
import time

# basic_module demo
import basic_module


dtype = np.float32
size = 10000000
power = 4
factor = 3.0
arr1 = np.linspace(1.0, 100.0, size, dtype=dtype)
arr2 = np.linspace(1.0, 100.0, size, dtype=dtype)

factor = dtype(factor)

t0 = time.time()
for _ in range(power):
    basic_module.multiply_with_scalar(arr1, factor)
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

# thrust_demo
import thrust_demo
dtype = np.float32
size = 100000
rndarr = np.random.normal(size=size).astype(dtype)
sum_diff = np.sum(rndarr)-thrust_demo.sum_array(rndarr)
if sum_diff == 0:
    print("thrust demo match")
else:
    print(f"thrust demo mismatch: Diff. : {sum_diff}")
