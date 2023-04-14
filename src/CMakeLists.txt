enable_testing()

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/smoother.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/diffusion_operator_2d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/intergrid_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cholesky_solver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/multigrid_preconditioner.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/iterative_solver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.cc
)

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