pybind11_add_module(thrust_demo thrust_demo.cu)
target_include_directories(thrust_demo BEFORE PRIVATE ./../../thrust)
target_include_directories(thrust_demo BEFORE PRIVATE ./../../thrust/dependencies/cub)
add_dependencies(thrust_demo copy-test-py)

add_custom_target(copy-init-py-thrust_demo ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)
add_dependencies(thrust_demo copy-init-py-thrust_demo)
set(cxx_flags
    # Don't split the next line, or things won't work for NVCC-MSVC!
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>$<$<CXX_COMPILER_ID:MSVC>:/Wall>")
if(DEFINED CMAKE_CUDA_COMPILER)
    target_compile_options(thrust_demo PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cxx_flags}>)
else()
    target_compile_options(thrust_demo PRIVATE ${cxx_flags})
endif()

set_target_properties(thrust_demo
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "52;70;86"
        )


target_link_libraries(thrust_demo PUBLIC cudart)

include(CMakePrintHelpers)
cmake_print_properties(TARGETS thrust_demo PROPERTIES CUDA_ARCHITECTURES)
