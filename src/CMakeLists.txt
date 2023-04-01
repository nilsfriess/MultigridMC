set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/diffusion_operator_2d.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/intergrid_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_operator.cc)

add_executable(driver driver.cc ${SOURCES})

target_link_libraries(driver Eigen3::Eigen)