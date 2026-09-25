string(TIMESTAMP RUN_ID "%Y%m%d-%H%M%S")
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef RUN_SUFFIX)
execute_process(COMMAND "${PYTHON}" "${SOURCE_DIR}/scripts/qualify_gate_b.py"
  --paralyn "${PARALYN}"
  --artifacts "${SOURCE_DIR}/artifacts/runs/gate-b-${RUN_ID}-${RUN_SUFFIX}"
  RESULT_VARIABLE RESULT)
if(NOT RESULT EQUAL 0)
  message(FATAL_ERROR "Gate B physical-GPU correctness qualification failed: ${RESULT}")
endif()
