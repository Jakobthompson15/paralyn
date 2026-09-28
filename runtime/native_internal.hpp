#pragma once
#include "paralyn/native.h"

// Library-internal hooks for in-library operator providers that are built only on
// the public C ABI. Not installed and not exported from libparalyn_native (hidden
// visibility). Each hook acquires the same API lock as public calls, so never call
// one from inside a pr_* call.
#define PARALYN_NATIVE_INTERNAL __attribute__((visibility("hidden")))

// Sets the calling thread's pr_last_error record (and emits the usual api_failure
// runtime event) for `operation`. Returns `code`; PR_SUCCESS is reported as
// PR_INTERNAL_ERROR because a failure hook must never report success.
PARALYN_NATIVE_INTERNAL pr_status paralyn_native_report_error(pr_status code, const char *operation,
                                                              const char *message) noexcept;
// Clears the calling thread's pr_last_error record, as every successful ABI 1 call does.
PARALYN_NATIVE_INTERNAL void paralyn_native_clear_error() noexcept;

enum paralyn_native_kind { PARALYN_NATIVE_BUFFER = 1, PARALYN_NATIVE_MODULE, PARALYN_NATIVE_QUEUE };
// Resolves a live handle of exactly `kind` to an opaque identity of its owning
// context (equal identities mean the same context; never dereferenced). Unknown,
// released or wrong-kind handles fail with PR_INVALID_HANDLE through pr_last_error
// under operation "handle_context".
PARALYN_NATIVE_INTERNAL pr_status paralyn_native_handle_context(pr_handle handle,
                                                                paralyn_native_kind kind,
                                                                const void **context) noexcept;
