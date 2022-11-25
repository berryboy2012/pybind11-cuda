// Based on cutlass/examples/00_basic_gemm/00_basic_gemm.cu
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>
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
// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"
/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
        int M,
        int N,
        int K,
        float alpha,
        thrust::device_vector<float> const &A,
        int lda,
        thrust::device_vector<float> const &B,
        int ldb,
        float beta,
        thrust::device_vector<float> &C,
        int ldc) {

    // Define type definition for single-precision CUTLASS GEMM with row-major
    // input matrices and 128x128x8 threadblock tile size (chosen by default).
    //
    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as
    // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
    //
    // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

    using RowMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
            float,        // Data-type of A matrix
            RowMajor,  // Layout of A matrix
            float,        // Data-type of B matrix
            RowMajor,  // Layout of B matrix
            float,        // Data-type of C matrix
            RowMajor>; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                {thrust::raw_pointer_cast(A.data()), lda},    // Tensor-ref for source matrix A
                                {thrust::raw_pointer_cast(B.data()), ldb},    // Tensor-ref for source matrix B
                                {thrust::raw_pointer_cast(C.data()), ldc},    // Tensor-ref for source matrix C
                                {thrust::raw_pointer_cast(C.data()), ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //
    auto status = gemm_operator(args);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess) {
        std::cerr<<cutlass::cutlassGetStatusString(status)<<std::endl;
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}
thrust::device_vector<float> CutlassGemm(int M, int N, int K,
                                         float alpha, float beta,
                                         thrust::device_vector<float> const &matA,
                                         thrust::device_vector<float> const &matB) {

    // Compute leading dimensions for each matrix.
    int lda = K;
    int ldb = N;
    int ldc = N;

    thrust::device_vector<float> matC(M*N);


    //
    // Launch CUTLASS GEMM.
    //
    auto result = CutlassSgemmNN(M, N, K, alpha,
                                 matA, lda,
                                 matB, ldb,
                                 beta,
                                 matC, ldc);

    if (result != cudaSuccess) {
        std::stringstream strstr;
        strstr << "CUTLASS GEMM kernel failed: "
               << cudaGetErrorString(result) << std::endl;
        throw std::runtime_error(strstr.str());
    }

    return matC;
}
namespace py = pybind11;

// wrap C++ function with NumPy array IO
template<typename Tv>
py::array_t<Tv> matmul(py::array_t<Tv, py::array::c_style | py::array::forcecast> &array_A,
                       py::array_t<Tv, py::array::c_style | py::array::forcecast> &array_B) {
    auto arr_A_info = array_A.request(false);
    auto arr_B_info = array_B.request(false);
    if ((arr_A_info.ndim != 2) | (arr_B_info.ndim != 2)){
        std::stringstream strstr;
        strstr << "the number of dims of A or B is not 2" << std::endl;
        throw std::runtime_error(strstr.str());
    }
    if (arr_A_info.shape[1] != arr_B_info.shape[0]){
        std::stringstream strstr;
        strstr << "incompatible shape between A and B" << std::endl;
        throw std::runtime_error(strstr.str());
    }
    int M = arr_A_info.shape[0];
    int N = arr_B_info.shape[1];
    int K = arr_A_info.shape[1];
    thrust::device_vector<Tv> matA((size_t) array_A.size());
    thrust::copy(array_A.data(),array_A.data()+array_A.size(),matA.begin());
    thrust::device_vector<Tv> matB((size_t) array_B.size());
    thrust::copy(array_B.data(),array_B.data()+array_B.size(),matB.begin());
    auto matC = CutlassGemm(M,N,K,1.0,0.0,matA,matB);
    auto result = py::array_t<Tv>(py::buffer_info(
            nullptr,//ptr
            sizeof(Tv),//sizeof(element)
            py::format_descriptor<Tv>::format(),//python type string
            2,//ndim
            {M, N},//shape
            {sizeof(Tv)*N, sizeof(Tv)}//stride for each dim
            ));
    auto res_info = result.request(true);
    thrust::copy(matC.begin(),matC.end(),static_cast<Tv*>(res_info.ptr));
    return result;
}

PYBIND11_MODULE(cutlass_demo, m) {
    m.def("matmul", &matmul<float>);
}
