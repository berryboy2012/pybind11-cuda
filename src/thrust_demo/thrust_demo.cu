#pragma warning(push, 0)
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#pragma warning(pop)

namespace py=pybind11;


// wrap C++ function with NumPy array IO
template <typename Tv>
Tv sum_array(py::array_t<Tv, py::array::c_style | py::array::forcecast>& array)
{
    thrust::host_vector<Tv> h_vec((size_t) array.size());
    std::memcpy(h_vec.data(),array.data(),array.size()*sizeof(Tv));
    // Transfer to device and compute the sum.
    thrust::device_vector<Tv> d_vec = h_vec;
    auto x = thrust::reduce(d_vec.begin(), d_vec.end(), static_cast<Tv>(0), thrust::plus<Tv>());
    return x;

}

PYBIND11_MODULE(thrust_demo, m)
{
  m.def("sum_array", &sum_array<double>);
  m.def("sum_array", &sum_array<float>);
}
