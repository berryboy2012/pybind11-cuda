#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
template <typename T>
__global__ void kernel
(T *vec, T scalar, size_t num_elements)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] * scalar;
  }
}

template <typename T>
void run_kernel
(T *vec, T scalar, size_t num_elements)
{
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid((unsigned int)ceil((double)num_elements / dimBlock.x));
  
  kernel<T><<<dimGrid, dimBlock>>>
    (vec, scalar, num_elements);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}

template <typename Tv>
void map_array(py::array_t<Tv, py::array::c_style | py::array::forcecast> vec,
               const py::buffer& scalar)
{
  auto ha = vec.request(true);
  Tv sca;
  // Ahh, should find a better way to do this.
  auto scalar_info = scalar.request();
  if (scalar_info.format != py::format_descriptor<Tv>::format()){
      throw std::runtime_error("Unexpected scalar type: should be the same as vector's type.");
  }else {
      sca = *reinterpret_cast<Tv *>(scalar_info.ptr);
  }

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "vec.ndim != 1" << std::endl;
    strstr << "vec.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  auto size = ha.shape[0];
  auto size_bytes = size*sizeof(Tv);
  Tv *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  Tv* ptr = reinterpret_cast<Tv*>(ha.ptr);
  error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel<Tv>(gpu_ptr, sca, size);

  error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

PYBIND11_MODULE(basic_module, m)
{
  m.def("multiply_with_scalar", &map_array<double>);
  m.def("multiply_with_scalar", &map_array<float>);
}
