enable_testing()

add_subdirectory(lattice)
add_subdirectory(solver)
add_subdirectory(sampler)
add_subdirectory(preconditioner)
add_subdirectory(smoother)
add_subdirectory(linear_operator)
add_subdirectory(intergrid)
add_subdirectory(auxilliary)
add_subdirectory(qoi)

# Main executable
add_executable(driver driver.cc ${SOURCES})
target_link_libraries(driver Eigen3::Eigen)
set_target_properties(driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# Tests
add_executable(test test.cc ${SOURCES})
target_link_libraries(test Eigen3::Eigen GTest::gtest_main)
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

include(GoogleTest)
gtest_discover_tests(test)