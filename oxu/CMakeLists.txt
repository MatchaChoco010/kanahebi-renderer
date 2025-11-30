set(TARGET_NAME oxu)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

add_library(${TARGET_NAME} STATIC ${SOURCES})

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  ${TARGET_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
  ${TARGET_NAME} PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  OptiX::OptiX
)
