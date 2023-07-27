set(SOURCES
${CMAKE_CURRENT_SOURCE_DIR}/common.cc
${CMAKE_CURRENT_SOURCE_DIR}/quadrature.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/parameters.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/statistics.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer2d.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/vtk_writer3d.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cholesky_wrapper.cc
  ${SOURCES} CACHE INTERNAL "sources"
)
