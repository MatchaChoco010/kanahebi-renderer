# COMPILE_PTX(
#   SOURCES file1.cu file2.cu
#   INCLUDE_DIRS /path/to/include1 /path/to/include2
#   DEPENDENCIES header1.h header2.h
#   TARGET_PATH /path/to/target
#   NVCC_OPTIONS <options>
#   TARGET_NAME target_name
# )

function(COMPILE_PTX)
  set(options "")
  set(oneValueArgs TARGET_PATH TARGET_NAME)
  set(multiValueArgs NVCC_OPTIONS SOURCES DEPENDENCIES INCLUDE_DIRS)

  CMAKE_PARSE_ARGUMENTS(
    COMPILE_PTX
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  file(MAKE_DIRECTORY ${COMPILE_PTX_TARGET_PATH})
  set(PTX_FILES "")

  # すべてのインクルードディレクトリ内のヘッダを再帰的に取得
  set(INCLUDE_HEADERS "")
  foreach(INC_DIR ${COMPILE_PTX_INCLUDE_DIRS})
    file(GLOB_RECURSE HEADERS "${INC_DIR}/*.h")
    list(APPEND INCLUDE_HEADERS ${HEADERS})
  endforeach()

  if (MSVC)
    set(_utf8_flag --compiler-options /utf-8)
  else()
    set(_utf8_flag "")
  endif()

  foreach(input ${COMPILE_PTX_SOURCES})
    get_filename_component(input_we ${input} NAME_WE)
    set(output ${COMPILE_PTX_TARGET_PATH}/${input_we}.ptx)
    list(APPEND PTX_FILES ${output})

    # include ディレクトリの引数を組み立て
    set(INCLUDE_ARGS "")
    foreach(INC_DIR ${COMPILE_PTX_INCLUDE_DIRS})
      list(APPEND INCLUDE_ARGS -I${INC_DIR})
    endforeach()

    add_custom_command(
      OUTPUT ${output}
      MAIN_DEPENDENCY ${input}
      DEPENDS ${COMPILE_PTX_DEPENDENCIES} ${INCLUDE_HEADERS}
      COMMAND ${CMAKE_CUDA_COMPILER}
      --machine=64
      --ptx
      --gpu-architecture=compute_86
      -std=c++${CMAKE_CUDA_STANDARD}
      -cudart static
      --use_fast_math
      ${_utf8_flag}
      ${COMPILE_PTX_NVCC_OPTIONS}
      -I${OPTIX_INCLUDE_DIR}
      ${INCLUDE_ARGS}
      ${input}
      -o ${output}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endforeach()

  add_custom_target(${COMPILE_PTX_TARGET_NAME} ALL DEPENDS ${PTX_FILES})
endfunction()
