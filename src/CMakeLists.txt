add_subdirectory(./../pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
add_subdirectory(basic_module)

add_custom_target(copy-test-py ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test_mul.py)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_mul.py
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/test_mul.py
        ${CMAKE_CURRENT_BINARY_DIR}/test_mul.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_mul.py)
