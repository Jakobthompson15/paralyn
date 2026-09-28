#ifndef PARALYN_NATIVE_H
#define PARALYN_NATIVE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define PR_ABI_VERSION 1u
/* Opaque, process-local, never-reused identities. Zero is invalid. */
typedef uint64_t pr_handle;
typedef pr_handle pr_context;
typedef pr_handle pr_buffer;
typedef pr_handle pr_view;
typedef pr_handle pr_module;
typedef pr_handle pr_kernel;
typedef pr_handle pr_queue;
typedef pr_handle pr_event;
typedef enum pr_status {
  PR_SUCCESS = 0,
  PR_INVALID_ARGUMENT = 1,
  PR_INVALID_HANDLE = 2,
  PR_CONTEXT_MISMATCH = 3,
  PR_OUT_OF_BOUNDS = 4,
  PR_UNSUPPORTED = 5,
  PR_OUT_OF_MEMORY = 6,
  PR_COMPILATION_FAILED = 7,
  PR_EXECUTION_FAILED = 8,
  PR_DEVICE_UNAVAILABLE = 9,
  PR_INTERNAL_ERROR = 10
} pr_status;
typedef enum pr_access { PR_READ = 1, PR_WRITE = 2, PR_READ_WRITE = 3 } pr_access;
typedef enum pr_type { PR_I32 = 1, PR_U32 = 2, PR_F32 = 3, PR_BUFFER = 4 } pr_type;
typedef struct pr_error {
  pr_status code;
  char operation[64];
  char message[1024];
} pr_error;
typedef struct pr_device_info {
  char name[256];
  char backend[32];
  char os[128];
  uint64_t registry_id;
  uint64_t max_buffer_bytes;
  uint32_t unified_memory;
} pr_device_info;
typedef struct pr_dim3 {
  uint32_t x, y, z;
} pr_dim3;
typedef struct pr_argument {
  pr_type type;
  pr_view view;
  int32_t i32;
  uint32_t u32;
  float f32;
} pr_argument;
typedef struct pr_parameter_info {
  /* Full artifact string limit plus terminator; reflection never truncates names. */
  char name[4097];
  pr_type type;
  uint32_t is_buffer;
  pr_access access;
} pr_parameter_info;
typedef struct pr_event_info {
  uint32_t completed;
  double gpu_start_seconds, gpu_end_seconds;
} pr_event_info;
/* Additive queries: initialize struct_size = sizeof(record), version = 1.
 * Existing ABI 1 records above never change layout. Unknown versions fail. */
#define PR_QUERY_VERSION_1 1u
#define PR_ARTIFACT_VERIFIED_IR (UINT64_C(1) << 0)
#define PR_ARTIFACT_MSL_SOURCE (UINT64_C(1) << 1)
#define PR_SCALAR_I32 (UINT64_C(1) << 0)
#define PR_SCALAR_U32 (UINT64_C(1) << 1)
#define PR_SCALAR_F32 (UINT64_C(1) << 2)
typedef struct pr_device_capabilities_v1 {
  uint32_t struct_size, version;
  /* Backend-qualified hardware identity, stable across enumeration order.
   * Metal registry IDs are local system identities, not portable UUIDs. */
  char stable_id[128];
  char backend[32];
  uint64_t artifact_formats, scalar_types;
  uint64_t max_buffer_bytes, max_threadgroup_memory_bytes;
  uint32_t max_block_x, max_block_y, max_block_z;
  uint32_t max_buffer_bindings;
  uint32_t unified_memory;
  uint32_t reserved;
} pr_device_capabilities_v1;
typedef enum pr_clock_domain {
  PR_CLOCK_UNAVAILABLE = 0,
  PR_CLOCK_DURATION_ONLY = 1,
  PR_CLOCK_METAL_SYSTEM_MACH = 2
} pr_clock_domain;
typedef struct pr_event_timing_v1 {
  uint32_t struct_size, version;
  uint32_t completed, duration_valid, timestamps_valid;
  pr_clock_domain clock_domain;
  double duration_seconds, start_seconds, end_seconds;
} pr_event_timing_v1;
uint32_t pr_abi_version(void);
pr_status pr_last_error(pr_error *out);
pr_status pr_device_count(uint32_t *out);
pr_status pr_device_get(uint32_t index, pr_device_info *out);
pr_status pr_device_capabilities_get(uint32_t index, pr_device_capabilities_v1 *out);
pr_status pr_context_create(const char *selector, pr_context *out);
pr_status pr_context_device(pr_context context, pr_device_info *out);
pr_status pr_context_capabilities(pr_context context, pr_device_capabilities_v1 *out);
pr_status pr_context_synchronize(pr_context context);
/* Export evidence into a new/empty directory. Waits and propagates GPU failures. */
pr_status pr_context_write_evidence(pr_context context, const char *directory);
/* Every acquired handle owns one reference. Release consumes a valid handle even
 * if completion fails. The first completion failure is returned; cleanup after
 * that failure was observed does not repeat it. Other operations remain failed. */
pr_status pr_release(pr_handle handle);
pr_status pr_buffer_create(pr_context context, uint64_t bytes, pr_buffer *out);
pr_status pr_buffer_size(pr_buffer buffer, uint64_t *out);
/* Views retain allocation/context after buffer/context handles are released. */
pr_status pr_view_create(pr_buffer buffer, uint64_t offset, uint64_t bytes, uint32_t alignment,
                         pr_access access, pr_view *out);
pr_status pr_buffer_write(pr_buffer buffer, uint64_t offset, const void *source, uint64_t bytes);
pr_status pr_buffer_read(pr_buffer buffer, uint64_t offset, void *destination, uint64_t bytes);
/* Load a versioned verified IR or public MSL executable artifact. No host source
 * is executed. MSL descriptors are checked against Metal argument reflection. */
pr_status pr_module_load(pr_context context, const void *data, uint64_t bytes, pr_module *out);
pr_status pr_module_load_file(pr_context context, const char *path, pr_module *out);
pr_status pr_module_kernel(pr_module module, const char *name, pr_kernel *out);
pr_status pr_kernel_parameter_count(pr_kernel kernel, uint32_t *out);
pr_status pr_kernel_parameter(pr_kernel kernel, uint32_t index, pr_parameter_info *out);
/* All queue handles for a context refer to its one ordered queue. */
pr_status pr_queue_get(pr_context context, pr_queue *out);
pr_status pr_queue_synchronize(pr_queue queue);
pr_status pr_launch(pr_queue queue, pr_kernel kernel, pr_dim3 grid, pr_dim3 block,
                    const pr_argument *arguments, uint32_t count, pr_event *out);
pr_status pr_event_wait(pr_event event, pr_event_info *out);
/* Waits and propagates command failure. Timestamp validity/domain are explicit;
 * duration-only backends must never invent absolute timestamps. */
pr_status pr_event_timing(pr_event event, pr_event_timing_v1 *out);
/* Never pretends already submitted GPU work can be cancelled. */
pr_status pr_event_cancel(pr_event event);
#ifdef __cplusplus
}
#endif
#endif
