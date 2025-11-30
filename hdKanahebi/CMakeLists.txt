set(TARGET_NAME hdKanahebi)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE OPTIX_KERNELS CONFIGURE_DEPENDS "kernel/*.cu")

add_library(${TARGET_NAME} SHARED ${SOURCES})

if (MSVC)
  target_link_options(${TARGET_NAME} PRIVATE /OPT:NOREF /OPT:NOICF)
  set_property(TARGET ${TARGET_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS " /Zc:inline-")
endif()

target_include_directories(
  ${TARGET_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if (${BUILD_TYPE} STREQUAL "blender-4.5")
  add_compile_definitions("IS_BLENDER_4_5=1")
  target_include_directories(
    ${TARGET_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/dependencies/OpenUSD/include/
  )
  target_link_libraries(
    ${TARGET_NAME} PRIVATE
    usd_ms
    CUDA::cudart
    CUDA::cuda_driver
    OptiX::OptiX
    OpenGL::GL
    glm::glm
    OpenQMC::OpenQMC
    oxu
  )
else()
  target_link_libraries(
    ${TARGET_NAME} PRIVATE
    glf
    hd
    hgi
    usd
    CUDA::cudart
    CUDA::cuda_driver
    OptiX::OptiX
    OpenGL::GL
    glm::glm
    OpenQMC::OpenQMC
    oxu
  )
endif()

set(PTX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
compile_ptx(
  SOURCES ${OPTIX_KERNELS}
  INCLUDE_DIRS
    "include"
    "kernel"
    "${CMAKE_SOURCE_DIR}/external/openqmc/include"
  TARGET_PATH ${PTX_OUTPUT_DIR}
  TARGET_NAME ${TARGET_NAME}_ptx
  NVCC_OPTIONS
    -DOQMC_ARCH_GPU=1
    -D__DEVICE_BUILD__
    -U__AVX__ -U__AVX2__ -U__SSE__ -U__SSE2__ -U__SSE4_2__
)
add_dependencies(${TARGET_NAME} ${TARGET_NAME}_ptx)

if (${BUILD_TYPE} STREQUAL "houdini-21.0")
  if(WIN32)
    set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/houdini-21.0-windows/hdKanahebi")
  elseif(UNIX)
    set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/houdini-21.0-linux/hdKanahebi")
  endif()
elseif (${BUILD_TYPE} STREQUAL "blender-4.5")
  if(WIN32)
    set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/blender-4.5-windows/hdKanahebi")
  elseif(UNIX)
    set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/blender-4.5-linux/hdKanahebi")
  endif()
elseif (${BUILD_TYPE} STREQUAL "cli")
  if(WIN32)
    set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/cli-windows/hdKanahebi")
  elseif(UNIX)
    set(PLUGIN_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/dist/cli-linux/hdKanahebi")
  endif()
endif()

if(WIN32)
  set(PLUG_INFO_JSON "${CMAKE_CURRENT_SOURCE_DIR}/resources/plugInfo-windows.json")
elseif(UNIX)
  set(PLUG_INFO_JSON "${CMAKE_CURRENT_SOURCE_DIR}/resources/plugInfo-linux.json")
endif()

if (${BUILD_TYPE} STREQUAL "cli")
  add_custom_target(${TARGET_NAME}_post_build ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PLUGIN_OUTPUT_DIR}/resources"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:${TARGET_NAME}>"
      "${PLUGIN_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TARGET_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PLUG_INFO_JSON}"
      "${PLUGIN_OUTPUT_DIR}/resources/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/ptx"
      "${PLUGIN_OUTPUT_DIR}/resources/ptx"
    COMMENT "Deploying ${TARGET_NAME} plugin to ${PLUGIN_OUTPUT_DIR}"
  )
elseif (${BUILD_TYPE} STREQUAL "houdini-21.0")
  add_custom_target(${TARGET_NAME}_post_build ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PLUGIN_OUTPUT_DIR}/resources"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:${TARGET_NAME}>"
      "${PLUGIN_OUTPUT_DIR}/$<TARGET_FILE_NAME:${TARGET_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PLUG_INFO_JSON}"
      "${PLUGIN_OUTPUT_DIR}/resources/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/ptx"
      "${PLUGIN_OUTPUT_DIR}/resources/ptx"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/resources/soho"
      "${PLUGIN_OUTPUT_DIR}/resources/soho"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/resources/UsdRenderers.json"
      "${PLUGIN_OUTPUT_DIR}/resources/UsdRenderers.json"
    COMMENT "Deploying ${TARGET_NAME} plugin to ${PLUGIN_OUTPUT_DIR}"
  )
elseif (${BUILD_TYPE} STREQUAL "blender-4.5")
  add_custom_target(${TARGET_NAME}_post_build ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PLUGIN_OUTPUT_DIR}/hdKanahebi/resources"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_SOURCE_DIR}/blender-extension"
      "${PLUGIN_OUTPUT_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:${TARGET_NAME}>"
      "${PLUGIN_OUTPUT_DIR}/hdKanahebi/$<TARGET_FILE_NAME:${TARGET_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PLUG_INFO_JSON}"
      "${PLUGIN_OUTPUT_DIR}/hdKanahebi/resources/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_CURRENT_BINARY_DIR}/ptx"
      "${PLUGIN_OUTPUT_DIR}/hdKanahebi/resources/ptx"
    COMMAND ${CMAKE_COMMAND} -E chdir "${PLUGIN_OUTPUT_DIR}"
      ${CMAKE_COMMAND} -E tar "cfv"
        "${PLUGIN_OUTPUT_DIR}/../hdKanahebi.zip"
        --format=zip
        blender_manifest.toml
        __init__.py
        engine.py
        properties.py
        ui.py
        hdKanahebi
    COMMENT "Deploying ${TARGET_NAME} plugin to ${PLUGIN_OUTPUT_DIR}"
  )
endif()

add_dependencies(${TARGET_NAME}_post_build ${TARGET_NAME} ${TARGET_NAME}_ptx)
