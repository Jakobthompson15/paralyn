execute_process(COMMAND "${RUNTIME}" --unobserved-error
  RESULT_VARIABLE RESULT OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
if(NOT RESULT EQUAL 1 OR NOT ERR MATCHES "Paralyn shutdown failed")
  message(FATAL_ERROR "Unobserved runtime error did not prevent success: ${RESULT}\n${OUT}\n${ERR}")
endif()
