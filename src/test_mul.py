#!/usr/bin/env python3

import platform
if platform.system() == 'Windows':
    # https://bugs.python.org/issue43173
    import os
    from pathlib import Path
    for path in os.environ['Path'].split(';'):
        if len(list(Path(path).glob('cudart*.dll'))) > 0:
            os.add_dll_directory(path)
            break
else:
    # Things should be taken care of by $LD_LIBRARY_PATH
    pass

import basic_module.basic_module as basic_module
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
