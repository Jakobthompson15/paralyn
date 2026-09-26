execute_process(COMMAND "${RUNTIME}" --unobserved-error
  RESULT_VARIABLE RESULT OUTPUT_VARIABLE OUT ERROR_VARIABLE ERR)
if(NOT RESULT EQUAL 1 OR NOT ERR MATCHES "Paralyn shutdown failed")
  message(FATAL_ERROR "Unobserved runtime error did not prevent success: ${RESULT}\n${OUT}\n${ERR}")
endif()
execute_process(COMMAND "${RUNTIME}" --throwing-log
  RESULT_VARIABLE LOG_RESULT OUTPUT_VARIABLE LOG_OUT ERROR_VARIABLE LOG_ERR TIMEOUT 15)
if(NOT LOG_RESULT EQUAL 0 OR NOT LOG_OUT MATCHES "GPU recovery passed")
  message(FATAL_ERROR "Throwing host log stranded GPU submission: ${LOG_RESULT}\n${LOG_OUT}\n${LOG_ERR}")
endif()
