# Optional pinned GLSL (glslang) and HLSL (DXC) compiler workers.
#
# PARALYN_ENABLE_GLSL=ON builds the pinned glslang standalone compiler from the
# vendored archive in third_party/glslang/. PARALYN_ENABLE_HLSL=ON builds the
# pinned DirectXShaderCompiler (DXC) from archives that
# scripts/fetch_shader_compilers.py downloads into third_party/dxc/ (too large
# to vendor in the repository). Every archive is checked against its SHA-256
# before extraction and no network access occurs during configuration or build.
#
# Both compilers are built as isolated ExternalProjects and only ever run as
# separate worker processes (see compiler/shader/). DXC contains its own fork
# of LLVM/Clang 3.7; it is never linked into a Paralyn binary, so it never
# shares an LLVM ABI or address space with the Clang 21 CUDA frontend. Their
# SPIR-V output (Vulkan 1.1, GLCompute) goes through the existing pinned
# SPIR-V importer, which requires PARALYN_ENABLE_SPIRV=ON. The default,
# compiler-disabled and runtime-only builds never include this file.

include(ExternalProject)

set(PARALYN_SHADER_BUILD_JOBS "4" CACHE STRING "Parallel jobs for the pinned glslang/DXC sub-builds")
set(_shader_root "${PROJECT_BINARY_DIR}/_shader")

# Verify an archive, extract it once per SHA-256 into ${_shader_root}/src and
# return the extracted top-level directory (<name>-<commit>).
function(paralyn_shader_extract archive_dir name commit sha256 out_var)
  set(archive "${archive_dir}/${name}-${commit}.tar.gz")
  if(NOT EXISTS "${archive}")
    message(FATAL_ERROR "Paralyn shader compilers: missing pinned archive ${archive}\n"
      "  Run: python3 scripts/fetch_shader_compilers.py --restore")
  endif()
  file(SHA256 "${archive}" actual)
  if(NOT actual STREQUAL sha256)
    message(FATAL_ERROR "Paralyn shader compilers: SHA-256 mismatch for ${archive}\n  expected ${sha256}\n  actual   ${actual}")
  endif()
  set(source "${_shader_root}/src/${name}-${commit}")
  set(stamp "${_shader_root}/src/${name}-${commit}.sha256")
  set(previous "")
  if(EXISTS "${stamp}")
    file(READ "${stamp}" previous)
  endif()
  if(NOT EXISTS "${source}/CMakeLists.txt" OR NOT previous STREQUAL sha256)
    file(REMOVE_RECURSE "${source}")
    file(MAKE_DIRECTORY "${_shader_root}/src")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${_shader_root}/src")
    if(NOT EXISTS "${source}")
      message(FATAL_ERROR "Paralyn shader compilers: ${archive} did not contain ${name}-${commit}/")
    endif()
    file(WRITE "${stamp}" "${sha256}")
  endif()
  set(${out_var} "${source}" PARENT_SCOPE)
endfunction()

set(_shader_common_args
  -G ${CMAKE_GENERATOR}
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
  -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})

set(PARALYN_SHADER_TOOLCHAIN_GLSLANG "")
set(PARALYN_GLSLANG_EXECUTABLE "")
if(PARALYN_ENABLE_GLSL)
  # glslang tag vulkan-sdk-1.4.363.0: the same Khronos SDK release as the
  # pinned SPIRV-Tools/SPIRV-Headers (its known_good.json names exactly those
  # commits). Built without spirv-opt (ENABLE_OPT=OFF) and without glslang's
  # own HLSL frontend (ENABLE_HLSL=OFF; HLSL goes through DXC only).
  set(PARALYN_GLSLANG_RELEASE "vulkan-sdk-1.4.363.0")
  set(PARALYN_GLSLANG_COMMIT "e1b562a8bed273a02f30b59b66a5d499793cede5")
  set(PARALYN_GLSLANG_SHA256 "907174a24713c6202c146f164bf81783f1fbc79c8cb821a30f18f159eb980312")
  paralyn_shader_extract("${PROJECT_SOURCE_DIR}/third_party/glslang" glslang
    ${PARALYN_GLSLANG_COMMIT} ${PARALYN_GLSLANG_SHA256} PARALYN_GLSLANG_DIR)
  set(_glslang_bin "${_shader_root}/build/glslang")
  set(PARALYN_GLSLANG_EXECUTABLE "${_glslang_bin}/StandAlone/glslang${CMAKE_EXECUTABLE_SUFFIX}")
  ExternalProject_Add(paralyn_glslang_build
    SOURCE_DIR "${PARALYN_GLSLANG_DIR}"
    BINARY_DIR "${_glslang_bin}"
    CMAKE_ARGS ${_shader_common_args}
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DBUILD_EXTERNAL=OFF -DENABLE_OPT=OFF -DENABLE_HLSL=OFF -DGLSLANG_TESTS=OFF
      -DENABLE_GLSLANG_BINARIES=ON -DGLSLANG_ENABLE_INSTALL=OFF -DBUILD_SHARED_LIBS=OFF
      -DENABLE_PCH=OFF -DBUILD_WERROR=OFF -DENABLE_CTEST=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target glslang-standalone -j ${PARALYN_SHADER_BUILD_JOBS}
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${PARALYN_GLSLANG_EXECUTABLE}"
    USES_TERMINAL_BUILD ON)
  set(PARALYN_SHADER_TOOLCHAIN_GLSLANG "glslang ${PARALYN_GLSLANG_RELEASE} (${PARALYN_GLSLANG_COMMIT})")
  message(STATUS "Paralyn GLSL frontend: ${PARALYN_SHADER_TOOLCHAIN_GLSLANG}")
endif()

set(PARALYN_SHADER_TOOLCHAIN_DXC "")
set(PARALYN_DXC_EXECUTABLE "")
if(PARALYN_ENABLE_HLSL)
  # DXC release v1.9.2607 plus the exact submodule commits recorded in that
  # tag (external/SPIRV-Headers, external/SPIRV-Tools, external/DirectX-Headers).
  # DXC's own SPIRV-Tools copy is used only inside the DXC worker for HLSL
  # legalization; Paralyn validates DXC's output with its separately pinned
  # spirv-val in the importer.
  set(PARALYN_DXC_ARCHIVE_DIR "${PROJECT_SOURCE_DIR}/third_party/dxc" CACHE PATH
    "Directory holding the pinned DXC source archives (scripts/fetch_shader_compilers.py)")
  set(PARALYN_DXC_RELEASE "v1.9.2607")
  set(PARALYN_DXC_COMMIT "0d3ee6b551b8fa768fbf825300ebab81047ef6a8")
  set(PARALYN_DXC_SHA256 "36d9383cfcb1a189efbadf181c81719ab329bd940b2f417cc5ba2d39a2ab5aea")
  set(_dxc_submodules
    "SPIRV-Headers|29981f65241605e08b0ede4cfeb999fe3b723c6a|232899f1ad4104fb5bc377b94596c7621575eee62ad9a9e8f929b63a7dd8a7ad"
    "SPIRV-Tools|b707790a898e44038547df54580022fc1cf89c3d|05d8af89737bde57571c48dbd36714c9f520a69623e14de72c3be6b600e277d6"
    "DirectX-Headers|980971e835876dc0cde415e8f9bc646e64667bf7|b5a4b6d8806ff7f29f19879f83d015dbe8740676d4ca0b48647a789cc7773c4e")
  paralyn_shader_extract("${PARALYN_DXC_ARCHIVE_DIR}" DirectXShaderCompiler
    ${PARALYN_DXC_COMMIT} ${PARALYN_DXC_SHA256} PARALYN_DXC_DIR)
  foreach(entry IN LISTS _dxc_submodules)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 name)
    list(GET parts 1 commit)
    list(GET parts 2 sha256)
    paralyn_shader_extract("${PARALYN_DXC_ARCHIVE_DIR}" ${name} ${commit} ${sha256} extracted)
    # Place each pinned submodule where DXC's external/CMakeLists.txt expects it.
    set(target "${PARALYN_DXC_DIR}/external/${name}")
    set(target_stamp "${PARALYN_DXC_DIR}/external/${name}.paralyn-sha256")
    set(previous "")
    if(EXISTS "${target_stamp}")
      file(READ "${target_stamp}" previous)
    endif()
    if(NOT EXISTS "${target}/CMakeLists.txt" OR NOT previous STREQUAL sha256)
      file(REMOVE_RECURSE "${target}")
      file(COPY "${extracted}/" DESTINATION "${target}")
      file(WRITE "${target_stamp}" "${sha256}")
    endif()
  endforeach()
  set(_dxc_bin "${_shader_root}/build/dxc")
  set(PARALYN_DXC_EXECUTABLE "${_dxc_bin}/bin/dxc${CMAKE_EXECUTABLE_SUFFIX}")
  ExternalProject_Add(paralyn_dxc_build
    SOURCE_DIR "${PARALYN_DXC_DIR}"
    BINARY_DIR "${_dxc_bin}"
    # PredefinedParams.cmake is DXC's documented *nix configuration; the
    # explicit arguments below override its test and revision settings.
    CMAKE_ARGS -C "${PARALYN_DXC_DIR}/cmake/caches/PredefinedParams.cmake"
      ${_shader_common_args}
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DENABLE_SPIRV_CODEGEN=ON -DHLSL_INCLUDE_TESTS=OFF -DSPIRV_BUILD_TESTS=OFF
      -DLLVM_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_TESTS=OFF -DLLVM_BUILD_TESTS=OFF
      -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_WERROR=OFF
      # The extracted tree sits inside this checkout: never embed the
      # enclosing Paralyn git revision; use DXC's in-tree fixed version
      # (utils/version/version.inc, 1.9.2607.0) instead.
      # utils/GetCommitInfo.py would otherwise query the enclosing checkout's
      # revision (observed: the first build embedded the Paralyn commit).
      -DLLVM_APPEND_VC_REV=OFF -DHLSL_ENABLE_FIXED_VER=ON -DHLSL_OFFICIAL_BUILD=OFF
      -DHLSL_SUPPORT_QUERY_GIT_COMMIT_INFO=OFF
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    # DXC's bundled SPIRV-Tools copy likewise gets its pinned identity.
    BUILD_COMMAND ${CMAKE_COMMAND} -E env "FORCED_BUILD_VERSION_DESCRIPTION=DXC ${PARALYN_DXC_RELEASE} external SPIRV-Tools b707790a898e44038547df54580022fc1cf89c3d"
      ${CMAKE_COMMAND} --build <BINARY_DIR> --target dxc -j ${PARALYN_SHADER_BUILD_JOBS}
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${PARALYN_DXC_EXECUTABLE}"
    USES_TERMINAL_BUILD ON)
  set(PARALYN_SHADER_TOOLCHAIN_DXC "DXC ${PARALYN_DXC_RELEASE} (${PARALYN_DXC_COMMIT}; SPIRV-Tools b707790a898e44038547df54580022fc1cf89c3d, SPIRV-Headers 29981f65241605e08b0ede4cfeb999fe3b723c6a, DirectX-Headers 980971e835876dc0cde415e8f9bc646e64667bf7)")
  message(STATUS "Paralyn HLSL frontend: ${PARALYN_SHADER_TOOLCHAIN_DXC}")
endif()

set(PARALYN_SHADER_GENERATED_DIR "${PROJECT_BINARY_DIR}/generated/shader")
file(CONFIGURE OUTPUT "${PARALYN_SHADER_GENERATED_DIR}/paralyn_shader_toolchain.h" CONTENT
"#pragma once
// Generated by cmake/ParalynShaderCompilers.cmake from the pinned archive identities.
// Worker executables are build-tree paths; they are not installed or packaged.
#define PARALYN_GLSLANG_EXECUTABLE \"@PARALYN_GLSLANG_EXECUTABLE@\"
#define PARALYN_GLSLANG_TOOLCHAIN \"@PARALYN_SHADER_TOOLCHAIN_GLSLANG@\"
#define PARALYN_DXC_EXECUTABLE \"@PARALYN_DXC_EXECUTABLE@\"
#define PARALYN_DXC_TOOLCHAIN \"@PARALYN_SHADER_TOOLCHAIN_DXC@\"
" @ONLY)
