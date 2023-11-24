enable_testing()

add_subdirectory(lattice)
add_subdirectory(solver)
add_subdirectory(sampler)
add_subdirectory(preconditioner)
add_subdirectory(smoother)
add_subdirectory(linear_operator)
add_subdirectory(intergrid)
add_subdirectory(auxilliary)
add_subdirectory(measurements)

link_libraries(Eigen3::Eigen)

# Main executables
# sampling with multigrid Monte Carlo
add_executable(driver_mgmc driver_mgmc.cc ${SOURCES})
set_target_properties(driver_mgmc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# deterministic multigrid
add_executable(driver_mg driver_mg.cc ${SOURCES})
set_target_properties(driver_mg PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# compute spectrum
add_executable(driver_spectrum driver_spectrum.cc ${SOURCES})
set_target_properties(driver_spectrum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

if(USE_THOROUGH_TESTS)
    set(THOROUGH_TESTING "true")
endif()

configure_file(config.h.in config.h)

# Make sure we can include the generated config.h
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(test test.cc ${SOURCES})
target_include_directories(test PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(test GTest::gtest_main)

set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

include(GoogleTest)
gtest_discover_tests(test)