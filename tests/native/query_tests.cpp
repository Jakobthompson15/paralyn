#include "paralyn/native.h"
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void ok(pr_status status) {
  if (status != PR_SUCCESS) {
    pr_error detail{};
    pr_last_error(&detail);
    throw std::runtime_error(detail.message);
  }
}
pr_device_capabilities_v1 record() {
  pr_device_capabilities_v1 result{};
  result.struct_size = sizeof(result);
  result.version = PR_QUERY_VERSION_1;
  return result;
}
}
int main() {
  try {
    // ABI 1 event layouts remain consumable by existing native/Python clients.
    static_assert(offsetof(pr_event_info, gpu_start_seconds) == 8);
    static_assert(offsetof(pr_event_info, gpu_end_seconds) == 16);
    static_assert(sizeof(pr_event_info) == 24);
    check(pr_abi_version() == 1, "Existing native ABI version changed");
    auto caps = record();
    caps.version = 99;
    check(pr_device_capabilities_get(0, &caps) == PR_UNSUPPORTED, "Unknown query version accepted");
    caps = record();
    caps.struct_size -= 1;
    check(pr_device_capabilities_get(0, &caps) == PR_INVALID_ARGUMENT, "Wrong record size accepted");
    check(pr_device_capabilities_get(0, nullptr) == PR_INVALID_ARGUMENT, "Null query accepted");
    pr_event_timing_v1 timing{};
    timing.struct_size = sizeof(timing);
    timing.version = PR_QUERY_VERSION_1;
    check(pr_event_timing(0, &timing) == PR_INVALID_HANDLE, "Invalid event timing accepted");
    timing.version = 2;
    check(pr_event_timing(0, &timing) == PR_UNSUPPORTED, "Unknown event record version accepted");
    uint32_t count = 0;
    ok(pr_device_count(&count));
    check(count > 0, "Physical Metal device required for selector/capability qualification");
    caps = record();
    ok(pr_device_capabilities_get(0, &caps));
    check(std::string(caps.stable_id).rfind("metal:registry:", 0) == 0,
          "Device identity is not backend-qualified");
    check(caps.artifact_formats == (PR_ARTIFACT_VERIFIED_IR | PR_ARTIFACT_MSL_SOURCE) &&
          caps.scalar_types == (PR_SCALAR_I32 | PR_SCALAR_U32 | PR_SCALAR_F32),
          "Implemented executable capabilities are inaccurate");
    check(caps.max_buffer_bytes && caps.max_threadgroup_memory_bytes && caps.max_block_x &&
          caps.max_block_y && caps.max_block_z && caps.max_buffer_bindings == 31,
          "Missing hardware limits");
    pr_device_info legacy{};
    ok(pr_device_get(0, &legacy));
    check(caps.max_buffer_bytes == legacy.max_buffer_bytes &&
          caps.unified_memory == legacy.unified_memory && !std::strcmp(caps.backend, legacy.backend),
          "New capability record disagrees with legacy device record");
    for (const char *selector : {"auto", "0", "metal:0"}) {
      pr_context context = 0;
      ok(pr_context_create(selector, &context));
      auto selected = record();
      ok(pr_context_capabilities(context, &selected));
      check(!std::strcmp(selected.stable_id, caps.stable_id), "Selectors choose inconsistent devices");
      ok(pr_release(context));
    }
    pr_context invalid = 0;
    check(pr_context_create("cuda:0", &invalid) == PR_DEVICE_UNAVAILABLE && invalid == 0,
          "Unavailable backend selector accepted");
    caps = record();
    check(pr_device_capabilities_get(count, &caps) == PR_DEVICE_UNAVAILABLE,
          "Unavailable device capability accepted");
    std::cout << "Versioned capabilities, ABI preservation and Metal selectors: PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Query qualification failed: " << e.what() << '\n';
    return 1;
  }
}
