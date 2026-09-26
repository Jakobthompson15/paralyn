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
add_library(paralyn_ir STATIC compiler/ir/ir.cpp compiler/ir/artifact.cpp compiler/codegen/metal.cpp)
target_include_directories(paralyn_ir PUBLIC include)
target_compile_options(paralyn_ir PRIVATE -Wall -Wextra -Wpedantic)
add_library(paralyn_runtime STATIC backends/metal/runtime.mm backends/metal/engine.mm)
target_include_directories(paralyn_runtime PUBLIC include)
target_compile_options(paralyn_runtime PRIVATE -fobjc-arc -Wall -Wextra)
target_link_libraries(paralyn_runtime PUBLIC paralyn_ir "-framework Metal" "-framework Foundation")
add_library(paralyn_native SHARED runtime/native.cpp)
target_include_directories(paralyn_native PUBLIC include)
target_compile_options(paralyn_native PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(paralyn_native PRIVATE paralyn_runtime)
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
add_custom_command(TARGET paralyn POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink paralyn ${PROJECT_BINARY_DIR}/unicuda
  COMMENT "Preserving the deprecated unicuda command as a Paralyn alias")
add_library(unicuda_ir ALIAS paralyn_ir)
add_library(unicuda_frontend ALIAS paralyn_frontend)
add_library(unicuda_runtime ALIAS paralyn_runtime)
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
add_executable(artifact_tests tests/artifact_tests.cpp)
target_link_libraries(artifact_tests PRIVATE paralyn_ir)
add_test(NAME kernel_artifacts COMMAND artifact_tests)
add_executable(frontend_tests tests/frontend_tests.cpp)
target_link_libraries(frontend_tests PRIVATE paralyn_frontend)
target_compile_definitions(frontend_tests PRIVATE PARALYN_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME frontend COMMAND frontend_tests)
add_executable(runtime_smoke tests/runtime_smoke.cpp)
target_link_libraries(runtime_smoke PRIVATE paralyn_runtime)
add_test(NAME metal_runtime COMMAND runtime_smoke)
set_tests_properties(metal_runtime PROPERTIES LABELS "gpu")
add_test(NAME unobserved_runtime_failure COMMAND ${CMAKE_COMMAND}
  -DRUNTIME=$<TARGET_FILE:runtime_smoke>
  -P ${PROJECT_SOURCE_DIR}/tests/runtime_failure.cmake)
set_tests_properties(unobserved_runtime_failure PROPERTIES LABELS "gpu;gate-b" TIMEOUT 30)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME native_compile COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tests/compile_cli.py --paralyn $<TARGET_FILE:paralyn>
  --source ${PROJECT_SOURCE_DIR}/examples/native/kernels.cu)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/native-kernels.prk
  COMMAND ${CMAKE_COMMAND} -E rm -f ${PROJECT_BINARY_DIR}/native-kernels.prk
  COMMAND $<TARGET_FILE:paralyn> compile ${PROJECT_SOURCE_DIR}/examples/native/kernels.cu
    --output ${PROJECT_BINARY_DIR}/native-kernels.prk
  DEPENDS paralyn ${PROJECT_SOURCE_DIR}/examples/native/kernels.cu
  COMMENT "Compile native examples' verified kernel module without executing host source")
add_custom_target(native_kernels ALL DEPENDS ${PROJECT_BINARY_DIR}/native-kernels.prk)
add_executable(native_c examples/native/vector_add.c)
target_link_libraries(native_c PRIVATE paralyn_native)
add_dependencies(native_c native_kernels)
add_executable(native_cpp examples/native/vector_add.cpp)
target_link_libraries(native_cpp PRIVATE paralyn_native)
add_dependencies(native_cpp native_kernels)
add_executable(native_tests tests/native/native_tests.cpp)
target_link_libraries(native_tests PRIVATE paralyn_native)
add_dependencies(native_tests native_kernels)
set_property(TARGET native_c PROPERTY C_STANDARD 11)
foreach(target native_c native_cpp native_tests)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endforeach()
foreach(kind native-c native-cpp native-tests python-tests python-example python-finalizer)
  string(REPLACE "-" "_" test_name ${kind})
  add_test(NAME ${test_name} COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/native/run_example.py --kind ${kind} --build ${PROJECT_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES LABELS "gpu;native" TIMEOUT 120)
endforeach()

add_test(NAME gate_a COMMAND ${CMAKE_COMMAND}
  -DPYTHON=${Python3_EXECUTABLE}
  -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
  -DPARALYN=$<TARGET_FILE:paralyn>
  -P ${PROJECT_SOURCE_DIR}/tests/gate_a.cmake)
set_tests_properties(gate_a PROPERTIES LABELS "gpu;gate-a" TIMEOUT 120)
add_test(NAME gate_b_correctness COMMAND ${CMAKE_COMMAND}
  -DPYTHON=${Python3_EXECUTABLE}
  -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
  -DPARALYN=$<TARGET_FILE:paralyn>
  -P ${PROJECT_SOURCE_DIR}/tests/gate_b.cmake)
set_tests_properties(gate_b_correctness PROPERTIES LABELS "gpu;gate-b" TIMEOUT 300)
add_executable(gate_b_benchmark benchmarks/gate_b_benchmark.cpp)
target_link_libraries(gate_b_benchmark PRIVATE paralyn_frontend paralyn_runtime)
target_compile_options(gate_b_benchmark PRIVATE -O2 -fno-fast-math -ffp-contract=off)
