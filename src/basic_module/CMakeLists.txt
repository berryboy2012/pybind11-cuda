pybind11_add_module(basic_module basic_module.cu)
add_custom_target(copy-cudart ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cudart64_110.dll)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cudart64_110.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../../cudart64_110.dll
        ${CMAKE_CURRENT_BINARY_DIR}/cudart64_110.dll
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../cudart64_110.dll)
add_dependencies(basic_module copy-test-py copy-cudart)
set(cxx_flags
    # Don't split the next line, or things won't work for NVCC-MSVC!
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>$<$<CXX_COMPILER_ID:MSVC>:/Wall>")
if(DEFINED CMAKE_CUDA_COMPILER)
    target_compile_options(basic_module PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${cxx_flags}>)
else()
    target_compile_options(basic_module PRIVATE ${cxx_flags})
endif()

set_target_properties(basic_module
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "52;70;86"
        )


target_link_libraries(basic_module PUBLIC cudart)

include(CMakePrintHelpers)
cmake_print_properties(TARGETS basic_module PROPERTIES CUDA_ARCHITECTURES)
