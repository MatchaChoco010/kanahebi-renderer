if(NOT OPTIX_ROOT_DIR)
  if(DEFINED ENV{OPTIX_PATH})
    set(OPTIX_ROOT_DIR $ENV{OPTIX_PATH})
  endif()
endif()

find_path(OPTIX_INCLUDE_DIR
  NAMES optix.h
  HINTS ${OPTIX_ROOT_DIR}
  PATH_SUFFIXES include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
  REQUIRED_VARS OPTIX_INCLUDE_DIR
  FAIL_MESSAGE "OptiX SDK not found. Please set OPTIX_PATH or OPTIX_ROOT_DIR to the OptiX SDK installation directory."
)

if(OPTIX_FOUND AND NOT TARGET OptiX::OptiX)
  add_library(OptiX::OptiX INTERFACE IMPORTED)
  set_target_properties(OptiX::OptiX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OPTIX_INCLUDE_DIR}"
  )
endif()
