# pybind11-cuda

>CMake+nvcc+msvc==pure_chaos. I learned it the hard way so you don't have to.

Starting point for GPU accelerated python libraries

Adapted from original work from https://github.com/PWhiddy/pybind11-cuda

Present work uses [modern CMake/Cuda](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9444-build-systems-exploring-modern-cmake-cuda-v2.pdf) approach

# Prerequisites

CUDA

Python 3.6 or greater

CMake >= 3.18 (for CUDA support and the new FindPython3 module)

# To build

You can use [variable CMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html) instead of CUDAFLAGS:

```bash
mkdir build; cd build
# provide a default cuda hardware architecture to build for
cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..
make
```

Test it with
```python3 ./src/test_cxx_module.py```


# Features demonstrated

- Compiles out of the box with cmake, even in Windows with `msvc`
- Easy-to-modify demos with _modern c++ experience_ by using libs 
such as `Thrust` and `cutlass`
- Numpy integration
- C++ Templating for composable kernels with generic data types

# Caveats

- The search order for `cuDNN` in `cutlass` is a bit surprising as of now (v2.10.0).
It is recommended to copy your desired version of `cuDNN` into your current CUDA directory.
And take notice on the detected path reported by `cutlass`'s `CMakeLists.txt`.

Originally based on https://github.com/torstem/demo-cuda-pybind11
