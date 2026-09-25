cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum deployment target (selected LLVM bottle requires macOS 26)")
project(Paralyn VERSION 0.0.1 LANGUAGES C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
if(NOT APPLE)
  message(FATAL_ERROR "Gate A requires macOS and Metal; no CPU fallback is provided")
endif()
find_package(LLVM CONFIG REQUIRED HINTS /opt/homebrew/opt/llvm@21/lib/cmake/llvm)
find_package(Clang CONFIG REQUIRED HINTS "${LLVM_INSTALL_PREFIX}/lib/cmake/clang")
set(PARALYN_CLANG_PATH "${LLVM_TOOLS_BINARY_DIR}/clang++")
execute_process(COMMAND "${PARALYN_CLANG_PATH}" -print-resource-dir OUTPUT_VARIABLE PARALYN_RESOURCE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE PARALYN_SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "Paralyn frontend: LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_INSTALL_PREFIX}")
add_library(paralyn_ir STATIC compiler/ir/ir.cpp compiler/codegen/metal.cpp)
target_include_directories(paralyn_ir PUBLIC include)
target_compile_options(paralyn_ir PRIVATE -Wall -Wextra -Wpedantic)
add_library(paralyn_runtime STATIC backends/metal/runtime.mm)
target_include_directories(paralyn_runtime PUBLIC include)
target_compile_options(paralyn_runtime PRIVATE -fobjc-arc -Wall -Wextra)
target_link_libraries(paralyn_runtime PUBLIC paralyn_ir "-framework Metal" "-framework Foundation")
add_library(paralyn_frontend STATIC compiler/frontend/frontend.cpp)
target_include_directories(paralyn_frontend SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_include_directories(paralyn_frontend PUBLIC include)
target_link_libraries(paralyn_frontend PUBLIC paralyn_ir PRIVATE clang-cpp LLVM)
target_compile_definitions(paralyn_frontend PRIVATE
  PARALYN_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/include"
  PARALYN_CLANG_PATH="${PARALYN_CLANG_PATH}"
  PARALYN_RESOURCE_DIR="${PARALYN_RESOURCE_DIR}"
  PARALYN_SDK_PATH="${PARALYN_SDK_PATH}")
add_executable(paralyn cli/main.cpp)
target_link_libraries(paralyn PRIVATE paralyn_frontend paralyn_runtime)
target_compile_definitions(paralyn PRIVATE
  PARALYN_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  PARALYN_BINARY_DIR="${PROJECT_BINARY_DIR}"
  PARALYN_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/include"
  PARALYN_HOST_CXX="${CMAKE_CXX_COMPILER}"
  PARALYN_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET}"
  PARALYN_RUNTIME_ARCHIVE="$<TARGET_FILE:paralyn_runtime>"
  PARALYN_IR_ARCHIVE="$<TARGET_FILE:paralyn_ir>"
  PARALYN_LLVM_VERSION="${LLVM_PACKAGE_VERSION}")
enable_testing()
add_executable(ir_tests tests/ir_tests.cpp)
target_link_libraries(ir_tests PRIVATE paralyn_ir)
add_test(NAME ir_and_codegen COMMAND ir_tests)
add_executable(frontend_tests tests/frontend_tests.cpp)
target_link_libraries(frontend_tests PRIVATE paralyn_frontend)
target_compile_definitions(frontend_tests PRIVATE PARALYN_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME frontend COMMAND frontend_tests)
add_executable(runtime_smoke tests/runtime_smoke.cpp)
target_link_libraries(runtime_smoke PRIVATE paralyn_runtime)
add_test(NAME metal_runtime COMMAND runtime_smoke)
set_tests_properties(metal_runtime PROPERTIES SKIP_RETURN_CODE 77)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME gate_a COMMAND ${CMAKE_COMMAND}
  -DPYTHON=${Python3_EXECUTABLE}
  -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
  -DPARALYN=$<TARGET_FILE:paralyn>
  -P ${PROJECT_SOURCE_DIR}/tests/gate_a.cmake)
set_tests_properties(gate_a PROPERTIES LABELS "gpu;gate-a" TIMEOUT 120)
