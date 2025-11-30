set(TARGET_NAME kanahebi-cli)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

add_executable(${TARGET_NAME} ${SOURCES})

target_include_directories(
  ${TARGET_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WIN32)
  target_link_libraries(
    ${TARGET_NAME} PRIVATE
    usdImagingGL
    CUDA::cudart
    CUDA::cuda_driver
    OptiX::OptiX
    CLI11::CLI11
    OpenGL::GL
    oxu
  )
else()
  find_library(EGL_LIBRARY EGL REQUIRED)
  if(NOT EGL_LIBRARY)
      message(FATAL_ERROR "EGL library not found")
  endif()
  target_link_libraries(
    ${TARGET_NAME} PRIVATE
    usdImagingGL
    CUDA::cudart
    CUDA::cuda_driver
    OptiX::OptiX
    CLI11::CLI11
    OpenGL::GL
    ${EGL_LIBRARY}
    oxu
  )
endif()

if(WIN32)
  set(CLI_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/cli-windows/")
elseif(UNIX)
  set(CLI_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/cli-linux/")
endif()

if(WIN32)
  file(GLOB USD_LIBS "${PXR_ROOT}/lib/*.dll")
  file(GLOB USD_DEPS_LIBS "${PXR_ROOT}/bin/*.dll")
  list(APPEND USD_LIBS ${USD_DEPS_LIBS})
else()
  file(GLOB USD_LIBS "${PXR_ROOT}/lib/*.so*")
  file(GLOB USD_LIB46S "${PXR_ROOT}/lib64/*.so*")
  list(APPEND USD_LIBS ${USD_LIB46S})
endif()

add_custom_target(${TARGET_NAME}_post_build ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CLI_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:${TARGET_NAME}>"
    "${CLI_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TARGET_NAME}>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${USD_LIBS} "${CLI_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${PXR_ROOT}/lib/usd" "${CLI_OUTPUT_DIR}/usd"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/kanahebi-cli.sh" "${CLI_OUTPUT_DIR}"
  COMMENT "Deploying ${TARGET_NAME} plugin to ${CLI_OUTPUT_DIR}"
)
add_dependencies(${TARGET_NAME}_post_build ${TARGET_NAME})
