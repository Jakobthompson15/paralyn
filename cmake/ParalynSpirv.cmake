# Optional SPIR-V import toolchain (PARALYN_ENABLE_SPIRV=ON).
#
# The exact upstream source archives are vendored in third_party/spirv/ and
# checked against the SHA-256 values below before extraction. No network access
# occurs during configuration or build. Each project is built in its own
# ExternalProject tree so Paralyn's warning flags never leak into third-party
# code and vice versa. The runtime-only build never includes this file.

include(ExternalProject)

set(PARALYN_SPIRV_RELEASE "vulkan-sdk-1.4.363.0")
set(PARALYN_SPIRV_HEADERS_COMMIT "496543121ce6419f23d6fa5d7194ba66c36212d2")
set(PARALYN_SPIRV_HEADERS_SHA256 "a9bb9c48713245eacf97cc539b6f1d45405a92d8813f8b82e635f0a085ee9898")
set(PARALYN_SPIRV_TOOLS_COMMIT "ef96ed763b43b59b33b31b362f09a02b729fa1c9")
set(PARALYN_SPIRV_TOOLS_SHA256 "82c62146083fd558735a3171cf97cfc47903ca7d368482e87f94bd44883c0f00")
set(PARALYN_SPIRV_CROSS_COMMIT "f11ba9f0b21ba8fc15153d50a2a1ae31ab1cf8f7")
set(PARALYN_SPIRV_CROSS_SHA256 "92b0458889ed77eac9892b3be5a41e3cf716849fd90f39632e159e555abf3d03")
set(PARALYN_SPIRV_BUILD_JOBS "4" CACHE STRING "Parallel jobs for the pinned SPIRV-Tools/SPIRV-Cross sub-builds")

set(_spirv_archives "${PROJECT_SOURCE_DIR}/third_party/spirv")
set(_spirv_root "${PROJECT_BINARY_DIR}/_spirv")

function(paralyn_spirv_extract name commit sha256 out_var)
  set(archive "${_spirv_archives}/${name}-${commit}.tar.gz")
  if(NOT EXISTS "${archive}")
    message(FATAL_ERROR "PARALYN_ENABLE_SPIRV: missing pinned archive ${archive}")
  endif()
  file(SHA256 "${archive}" actual)
  if(NOT actual STREQUAL sha256)
    message(FATAL_ERROR "PARALYN_ENABLE_SPIRV: SHA-256 mismatch for ${archive}\n  expected ${sha256}\n  actual   ${actual}")
  endif()
  set(source "${_spirv_root}/src/${name}-${commit}")
  set(stamp "${_spirv_root}/src/${name}-${commit}.sha256")
  set(previous "")
  if(EXISTS "${stamp}")
    file(READ "${stamp}" previous)
  endif()
  if(NOT EXISTS "${source}/CMakeLists.txt" OR NOT previous STREQUAL sha256)
    file(REMOVE_RECURSE "${source}")
    file(MAKE_DIRECTORY "${_spirv_root}/src")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${_spirv_root}/src")
    if(NOT EXISTS "${source}/CMakeLists.txt")
      message(FATAL_ERROR "PARALYN_ENABLE_SPIRV: ${archive} did not contain ${name}-${commit}/")
    endif()
    file(WRITE "${stamp}" "${sha256}")
  endif()
  set(${out_var} "${source}" PARENT_SCOPE)
endfunction()

paralyn_spirv_extract(SPIRV-Headers ${PARALYN_SPIRV_HEADERS_COMMIT} ${PARALYN_SPIRV_HEADERS_SHA256} PARALYN_SPIRV_HEADERS_DIR)
paralyn_spirv_extract(SPIRV-Tools ${PARALYN_SPIRV_TOOLS_COMMIT} ${PARALYN_SPIRV_TOOLS_SHA256} PARALYN_SPIRV_TOOLS_DIR)
paralyn_spirv_extract(SPIRV-Cross ${PARALYN_SPIRV_CROSS_COMMIT} ${PARALYN_SPIRV_CROSS_SHA256} PARALYN_SPIRV_CROSS_DIR)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_spirv_common_args
  -G ${CMAKE_GENERATOR}
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
  -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

set(_tools_bin "${_spirv_root}/build/SPIRV-Tools")
set(PARALYN_SPIRV_TOOLS_LIBRARY "${_tools_bin}/source/${CMAKE_STATIC_LIBRARY_PREFIX}SPIRV-Tools${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(PARALYN_SPIRV_AS "${_tools_bin}/tools/spirv-as${CMAKE_EXECUTABLE_SUFFIX}")
set(PARALYN_SPIRV_VAL "${_tools_bin}/tools/spirv-val${CMAKE_EXECUTABLE_SUFFIX}")
set(PARALYN_SPIRV_DIS "${_tools_bin}/tools/spirv-dis${CMAKE_EXECUTABLE_SUFFIX}")
ExternalProject_Add(paralyn_spirv_tools_build
  SOURCE_DIR "${PARALYN_SPIRV_TOOLS_DIR}"
  BINARY_DIR "${_tools_bin}"
  CMAKE_ARGS ${_spirv_common_args}
    -DSPIRV-Headers_SOURCE_DIR=${PARALYN_SPIRV_HEADERS_DIR}
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DSPIRV_SKIP_TESTS=ON -DSPIRV_WERROR=OFF -DSPIRV_TOOLS_BUILD_STATIC=ON
    -DSPIRV_BUILD_FUZZER=OFF -DSPIRV_BUILD_COMPRESSION=OFF -DSKIP_SPIRV_TOOLS_INSTALL=ON
    -DSPIRV_COLOR_TERMINAL=OFF -DBUILD_SHARED_LIBS=OFF
  # The extracted tree sits inside this repository's checkout; force the upstream
  # identity so SPIRV-Tools never embeds the enclosing Paralyn git revision.
  BUILD_COMMAND ${CMAKE_COMMAND} -E env "FORCED_BUILD_VERSION_DESCRIPTION=${PARALYN_SPIRV_RELEASE} ${PARALYN_SPIRV_TOOLS_COMMIT}"
    ${CMAKE_COMMAND} --build <BINARY_DIR> --target SPIRV-Tools-static spirv-as spirv-val spirv-dis -j ${PARALYN_SPIRV_BUILD_JOBS}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${PARALYN_SPIRV_TOOLS_LIBRARY}" "${PARALYN_SPIRV_AS}" "${PARALYN_SPIRV_VAL}" "${PARALYN_SPIRV_DIS}"
  USES_TERMINAL_BUILD ON)

set(_cross_bin "${_spirv_root}/build/SPIRV-Cross")
set(_cross_libs "")
foreach(component core glsl msl)
  list(APPEND _cross_libs "${_cross_bin}/${CMAKE_STATIC_LIBRARY_PREFIX}spirv-cross-${component}${CMAKE_STATIC_LIBRARY_SUFFIX}")
endforeach()
ExternalProject_Add(paralyn_spirv_cross_build
  SOURCE_DIR "${PARALYN_SPIRV_CROSS_DIR}"
  BINARY_DIR "${_cross_bin}"
  CMAKE_ARGS ${_spirv_common_args}
    -DSPIRV_CROSS_STATIC=ON -DSPIRV_CROSS_SHARED=OFF -DSPIRV_CROSS_CLI=OFF
    -DSPIRV_CROSS_ENABLE_TESTS=OFF -DSPIRV_CROSS_ENABLE_GLSL=ON -DSPIRV_CROSS_ENABLE_MSL=ON
    -DSPIRV_CROSS_ENABLE_HLSL=OFF -DSPIRV_CROSS_ENABLE_CPP=OFF -DSPIRV_CROSS_ENABLE_REFLECT=OFF
    -DSPIRV_CROSS_ENABLE_C_API=OFF -DSPIRV_CROSS_ENABLE_UTIL=OFF -DSPIRV_CROSS_SKIP_INSTALL=ON
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target spirv-cross-core spirv-cross-glsl spirv-cross-msl -j ${PARALYN_SPIRV_BUILD_JOBS}
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${_cross_libs}
  USES_TERMINAL_BUILD ON)

add_library(paralyn_spirv_tools_lib STATIC IMPORTED GLOBAL)
set_target_properties(paralyn_spirv_tools_lib PROPERTIES
  IMPORTED_LOCATION "${PARALYN_SPIRV_TOOLS_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${PARALYN_SPIRV_TOOLS_DIR}/include")
add_dependencies(paralyn_spirv_tools_lib paralyn_spirv_tools_build)

set(_previous "")
foreach(component msl glsl core)
  add_library(paralyn_spirv_cross_${component} STATIC IMPORTED GLOBAL)
  set_target_properties(paralyn_spirv_cross_${component} PROPERTIES
    IMPORTED_LOCATION "${_cross_bin}/${CMAKE_STATIC_LIBRARY_PREFIX}spirv-cross-${component}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    INTERFACE_INCLUDE_DIRECTORIES "${PARALYN_SPIRV_CROSS_DIR}")
  add_dependencies(paralyn_spirv_cross_${component} paralyn_spirv_cross_build)
endforeach()
set_property(TARGET paralyn_spirv_cross_msl APPEND PROPERTY INTERFACE_LINK_LIBRARIES paralyn_spirv_cross_glsl)
set_property(TARGET paralyn_spirv_cross_glsl APPEND PROPERTY INTERFACE_LINK_LIBRARIES paralyn_spirv_cross_core)

set(PARALYN_SPIRV_TOOLCHAIN
  "SPIRV-Tools ${PARALYN_SPIRV_RELEASE} (${PARALYN_SPIRV_TOOLS_COMMIT}); SPIRV-Cross ${PARALYN_SPIRV_RELEASE} (${PARALYN_SPIRV_CROSS_COMMIT}); SPIRV-Headers ${PARALYN_SPIRV_RELEASE} (${PARALYN_SPIRV_HEADERS_COMMIT})")
set(PARALYN_SPIRV_GENERATED_DIR "${PROJECT_BINARY_DIR}/generated/spirv")
file(CONFIGURE OUTPUT "${PARALYN_SPIRV_GENERATED_DIR}/paralyn_spirv_toolchain.h" CONTENT
"#pragma once
// Generated by cmake/ParalynSpirv.cmake from the pinned archive identities.
#define PARALYN_SPIRV_TOOLCHAIN \"@PARALYN_SPIRV_TOOLCHAIN@\"
" @ONLY)
message(STATUS "Paralyn SPIR-V import: ${PARALYN_SPIRV_TOOLCHAIN}")
