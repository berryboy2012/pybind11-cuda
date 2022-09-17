add_subdirectory(./../pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
pybind11_add_module(gpu_library gpu_library.cu)

set(cxx_flags
    # Don't split the next line, or things won't work for NVCC-MSVC!
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>$<$<CXX_COMPILER_ID:MSVC>:/Wall>")
if(DEFINED CMAKE_CUDA_COMPILER)
    target_compile_options(gpu_library PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cxx_flags}>)
else()
    target_compile_options(gpu_library PRIVATE ${cxx_flags})
endif()

set_target_properties(gpu_library
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "52;72;86"
        )

set(Python_FIND_ABI "OFF" "ANY" "ANY")
target_link_libraries(gpu_library PUBLIC cudart)

include(CMakePrintHelpers)
cmake_print_properties(TARGETS gpu_library PROPERTIES CUDA_ARCHITECTURES Python_FIND_ABI
        CMAKE_CXX_FLAGS CMAKE_CUDA_FLAGS)
