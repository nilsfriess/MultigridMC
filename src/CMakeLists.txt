set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/diffusion_operator_2d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/intergrid_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_operator.cc)

add_executable(driver driver.cc ${SOURCES})
set_target_properties(driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

target_link_libraries(driver Eigen3::Eigen)