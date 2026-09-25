string(TIMESTAMP RUN_ID "%Y%m%d-%H%M%S")
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef RUN_SUFFIX)
execute_process(COMMAND "${PYTHON}" "${SOURCE_DIR}/scripts/verify_gate_a.py"
  --paralyn "${PARALYN}" --source "${SOURCE_DIR}/examples/vector_add.cu"
  --artifacts "${SOURCE_DIR}/artifacts/runs/ctest-${RUN_ID}-${RUN_SUFFIX}"
  RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
  message(FATAL_ERROR "Gate A physical-GPU test failed: ${RESULT}")
endif()
