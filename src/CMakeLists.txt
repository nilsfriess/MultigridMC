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
target_link_libraries(driver Eigen3::Eigen libconfig::config ${CONFIG++_LIBRARY})
set_target_properties(driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# Tests
option(USE_THOROUGH_TESTS "Use thorough tests?" ON)

if(USE_THOROUGH_TESTS)
    set(THOROUGH_TESTING "true")
endif()

configure_file(config.h.in config.h)

add_executable(test test.cc ${SOURCES})
target_include_directories(test PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(test Eigen3::Eigen GTest::gtest_main)

set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

include(GoogleTest)
gtest_discover_tests(test)