#pragma once
#include "paralyn/native.h"

// Library-internal, not installed: sets the calling thread's pr_last_error record
// (and emits the usual api_failure runtime event) on behalf of an in-library
// provider that is built only on the public C ABI. Returns `code`. It acquires the
// same API lock as public calls, so never call it from inside a pr_* call.
pr_status paralyn_native_report_error(pr_status code, const char *operation,
                                      const char *message) noexcept;
