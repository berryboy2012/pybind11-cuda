add_subdirectory(./../pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
add_subdirectory(./../thrust ${CMAKE_CURRENT_BINARY_DIR}/thrust)
add_subdirectory(./../cutlass ${CMAKE_CURRENT_BINARY_DIR}/cutlass)

add_subdirectory(basic_module)
add_subdirectory(thrust_demo)
add_subdirectory(cutlass_demo)

add_custom_target(copy-test-py ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test_cxx_module.py)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_cxx_module.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test_cxx_module.py
        ${CMAKE_CURRENT_BINARY_DIR}/test_cxx_module.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_cxx_module.py)
