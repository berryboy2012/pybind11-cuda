pybind11_add_module(cutlass_demo cutlass_demo.cu helper.h)
target_include_directories(cutlass_demo BEFORE PRIVATE ./../../cutlass/include)
target_include_directories(cutlass_demo BEFORE PRIVATE ./../../cutlass/tools/util/include)
target_include_directories(cutlass_demo BEFORE PRIVATE ./../../thrust)
target_include_directories(cutlass_demo BEFORE PRIVATE ./../../thrust/dependencies/cub)

add_dependencies(cutlass_demo copy-test-py)

add_custom_target(copy-init-py-cutlass_demo ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)
add_dependencies(cutlass_demo copy-init-py-cutlass_demo)
set(cxx_flags
    # Don't split the next line, or things won't work for NVCC-MSVC!
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>$<$<CXX_COMPILER_ID:MSVC>:/Wall>")
if(DEFINED CMAKE_CUDA_COMPILER)
    target_compile_options(cutlass_demo PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cxx_flags}>)
else()
    target_compile_options(cutlass_demo PRIVATE ${cxx_flags})
endif()

set_target_properties(cutlass_demo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86"
        )


target_link_libraries(cutlass_demo PUBLIC cudart)

include(CMakePrintHelpers)
cmake_print_properties(TARGETS cutlass_demo PROPERTIES CUDA_ARCHITECTURES)
