#include "paralyn/native.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
unsigned negative_checks = 0, gpu_events = 0;
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
void success(pr_status status) {
  if (status == PR_SUCCESS)
    return;
  pr_error error{};
  pr_last_error(&error);
  throw std::runtime_error(std::string(error.operation) + ": " + error.message);
}
void failure(pr_status actual, pr_status expected, const char *operation) {
  require(actual == expected,
          std::string(operation) + " returned wrong status: " + std::to_string(actual));
  pr_error error{};
  require(pr_last_error(&error) == PR_SUCCESS, "Cannot retrieve error detail");
  require(error.code == expected && std::string(error.operation) == operation && *error.message,
          std::string(operation) + " lost structured error detail");
  ++negative_checks;
}
struct Handles {
  std::vector<pr_handle> values;
  pr_handle keep(pr_handle handle) {
    require(handle, "Successful API returned an invalid handle");
    values.push_back(handle);
    return handle;
  }
  void release(pr_handle handle) {
    for (auto &value : values)
      if (value == handle) {
        value = 0;
        success(pr_release(handle));
        return;
      }
    throw std::runtime_error("Test attempted to release an unowned handle");
  }
  ~Handles() {
    for (auto it = values.rbegin(); it != values.rend(); ++it)
      if (*it)
        pr_release(*it);
  }
};
pr_argument buffer_arg(pr_view view) {
  pr_argument result{};
  result.type = PR_BUFFER;
  result.view = view;
  return result;
}
pr_argument i32_arg(int32_t value) {
  pr_argument result{};
  result.type = PR_I32;
  result.i32 = value;
  return result;
}
pr_argument f32_arg(float value) {
  pr_argument result{};
  result.type = PR_F32;
  result.f32 = value;
  return result;
}
pr_buffer allocate(Handles &handles, pr_context context, uint64_t bytes) {
  pr_buffer handle = 0;
  success(pr_buffer_create(context, bytes, &handle));
  return handles.keep(handle);
}
pr_view view(Handles &handles, pr_buffer buffer, uint64_t offset, uint64_t bytes,
             pr_access access) {
  pr_view handle = 0;
  success(pr_view_create(buffer, offset, bytes, 4, access, &handle));
  return handles.keep(handle);
}
pr_event launch(Handles &handles, pr_queue queue, pr_kernel kernel, int n, unsigned block,
                const std::vector<pr_argument> &arguments) {
  pr_event event = 0;
  success(pr_launch(queue, kernel, {static_cast<uint32_t>((n + block - 1) / block), 1, 1},
                    {block, 1, 1}, arguments.data(), arguments.size(), &event));
  return handles.keep(event);
}
void completed(pr_event event) {
  pr_event_info info{};
  success(pr_event_wait(event, &info));
  require(info.completed && std::isfinite(info.gpu_start_seconds) &&
              std::isfinite(info.gpu_end_seconds) && info.gpu_start_seconds > 0 &&
              info.gpu_end_seconds > info.gpu_start_seconds,
          "Missing physical GPU completion/timestamp evidence");
  std::cout << std::setprecision(17) << "GPU event " << ++gpu_events
            << ": completed, start=" << info.gpu_start_seconds << ", end=" << info.gpu_end_seconds
            << '\n';
}
void verify(const std::vector<float> &actual, const std::vector<float> &reference,
            const char *label) {
  require(actual.size() == reference.size(), "Reference length mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(actual[i] == reference[i], std::string(label) + " mismatch at " + std::to_string(i));
}
void upload(pr_buffer buffer, const std::vector<float> &data) {
  success(pr_buffer_write(buffer, 0, data.data(), data.size() * sizeof(float)));
}
std::vector<float> download(pr_buffer buffer, std::size_t count) {
  std::vector<float> result(count);
  success(pr_buffer_read(buffer, 0, result.data(), result.size() * sizeof(float)));
  return result;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "Usage: native_tests MODULE NEW_EVIDENCE_DIRECTORY");
    require(pr_abi_version() == PR_ABI_VERSION, "C API ABI version mismatch");
    Handles handles;
    uint32_t device_count = 0;
    success(pr_device_count(&device_count));
    require(device_count > 0, "No physical GPU; native hardware qualification cannot skip");
    pr_device_info discovery{};
    success(pr_device_get(0, &discovery));
    require(std::string(discovery.backend) == "Metal" && *discovery.name && discovery.registry_id,
            "Missing genuine Metal device identity");
    pr_context context = 0;
    success(pr_context_create("auto", &context));
    handles.keep(context);
    pr_device_info selected{};
    success(pr_context_device(context, &selected));
    require(selected.registry_id == discovery.registry_id, "Auto selected a different device");
    std::cout << "Device: " << selected.name << "; backend: " << selected.backend << '\n';
    pr_queue queue = 0;
    success(pr_queue_get(context, &queue));
    handles.keep(queue);
    pr_module module = 0;
    success(pr_module_load_file(context, argv[1], &module));
    handles.keep(module);
    pr_kernel add = 0, affine = 0;
    success(pr_module_kernel(module, "vector_add", &add));
    success(pr_module_kernel(module, "affine", &affine));
    handles.keep(add);
    handles.keep(affine);
    uint32_t parameter_count = 0;
    success(pr_kernel_parameter_count(affine, &parameter_count));
    require(parameter_count == 5, "Affine reflection lost scalar parameter");
    for (uint32_t i = 0; i < parameter_count; ++i) {
      pr_parameter_info info{};
      success(pr_kernel_parameter(affine, i, &info));
      require(*info.name, "Missing reflected parameter name");
      require(info.type == (i == 3 ? PR_I32 : PR_F32), "Wrong reflected scalar type");
      require(info.is_buffer == (i < 3), "Wrong reflected buffer kind");
      if (i < 3)
        require(info.access == (i == 2 ? PR_WRITE : PR_READ), "Wrong inferred buffer access");
    }
    handles.release(module); // Each kernel retains its module.

    constexpr float sentinel = -16384.0f;
    for (int n : {17, 1003}) {
      const unsigned block = n == 17 ? 7 : 64;
      std::vector<float> a(n), b(n), expected(n + 2, sentinel);
      for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>((i * 13) % 97 - 48) * 0.25f;
        b[i] = static_cast<float>((i * 7) % 53 - 26) * 0.5f;
      }
      auto ba = allocate(handles, context, n * 4), bb = allocate(handles, context, n * 4);
      auto out = allocate(handles, context, (n + 2) * 4);
      upload(ba, a);
      upload(bb, b);
      upload(out, expected);
      auto va = view(handles, ba, 0, n * 4, PR_READ);
      auto vb = view(handles, bb, 0, n * 4, PR_READ);
      auto vo = view(handles, out, 4, n * 4, PR_WRITE);
      handles.release(ba); // Views retain both inputs even before submission.
      handles.release(bb);
      auto event = launch(handles, queue, add, n, block,
                          {buffer_arg(va), buffer_arg(vb), buffer_arg(vo), i32_arg(n)});
      completed(event);
      handles.release(event);
      for (int i = 0; i < n; ++i)
        expected[i + 1] = a[i] + b[i];
      verify(download(out, n + 2), expected, "vector_add with offset/canaries");

      const float scale = n == 17 ? 2.0f : -0.5f;
      auto args = std::vector<pr_argument>{buffer_arg(va), buffer_arg(vb), buffer_arg(vo),
                                           i32_arg(n), f32_arg(scale)};
      event = launch(handles, queue, affine, n, block, args);
      args[3].i32 = 0;
      args[4].f32 = 9999.0f; // Submitted scalar bytes must be an owned copy.
      handles.release(va);
      handles.release(vb);
      handles.release(vo); // Work retains allocations after all input owners/views disappear.
      completed(event);
      handles.release(event);
      for (int i = 0; i < n; ++i)
        expected[i + 1] = a[i] * scale + b[i];
      verify(download(out, n + 2), expected, "affine with copied scalar and retained inputs");
      handles.release(out);
    }

    // Two dependent dispatches must execute on the one ordered queue without a host wait.
    {
      constexpr int n = 257;
      std::vector<float> a(n), b(n), reference(n);
      for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 31 - 15) * 0.25f;
        b[i] = static_cast<float>(i % 19 - 9) * 0.5f;
        const float intermediate = a[i] * 2.0f + b[i];
        reference[i] = intermediate * -0.5f + b[i];
      }
      auto ba = allocate(handles, context, n * 4), bb = allocate(handles, context, n * 4);
      auto temporary = allocate(handles, context, n * 4), out = allocate(handles, context, n * 4);
      upload(ba, a);
      upload(bb, b);
      auto va = view(handles, ba, 0, n * 4, PR_READ);
      auto vb = view(handles, bb, 0, n * 4, PR_READ);
      auto vt = view(handles, temporary, 0, n * 4, PR_READ_WRITE);
      auto vo = view(handles, out, 0, n * 4, PR_WRITE);
      auto first = launch(handles, queue, affine, n, 32,
                          {buffer_arg(va), buffer_arg(vb), buffer_arg(vt), i32_arg(n), f32_arg(2)});
      auto second =
          launch(handles, queue, affine, n, 32,
                 {buffer_arg(vt), buffer_arg(vb), buffer_arg(vo), i32_arg(n), f32_arg(-0.5)});
      for (auto owned : {ba, bb, temporary, va, vb, vt, vo})
        handles.release(owned);
      completed(second); // Waiting for the later event also completes earlier commands.
      completed(first);
      verify(download(out, n), reference, "ordered dependent affine dispatches");
      handles.release(first);
      handles.release(second);
      handles.release(out);
    }

    // All three arguments share one allocation. A and output overlap exactly;
    // B is a disjoint offset view. No cross-thread race or out-of-bounds work.
    constexpr int n = 129;
    std::vector<float> aliased(2 * n + 3, sentinel), expected = aliased;
    for (int i = 0; i < n; ++i) {
      aliased[i + 1] = static_cast<float>(i % 23 - 11) * 0.5f;
      aliased[n + 2 + i] = static_cast<float>(i % 17 - 8) * 0.25f;
    }
    expected = aliased;
    auto storage = allocate(handles, context, aliased.size() * 4);
    upload(storage, aliased);
    auto va = view(handles, storage, 4, n * 4, PR_READ);
    auto vb = view(handles, storage, (n + 2) * 4, n * 4, PR_READ);
    auto vo = view(handles, storage, 4, n * 4, PR_READ_WRITE);
    auto args =
        std::vector<pr_argument>{buffer_arg(va), buffer_arg(vb), buffer_arg(vo), i32_arg(n)};
    auto first_alias = launch(handles, queue, add, n, 32, args);
    completed(first_alias);
    for (int i = 0; i < n; ++i)
      expected[i + 1] = aliased[i + 1] + aliased[n + 2 + i];
    verify(download(storage, expected.size()), expected, "same-allocation offset alias vector_add");

    // All invalid requests below must fail before submission, with structured errors.
    uint64_t size = 0;
    failure(pr_buffer_size(context, &size), PR_INVALID_HANDLE, "buffer_size");
    failure(pr_buffer_size(UINT64_MAX, &size), PR_INVALID_HANDLE, "buffer_size");
    failure(pr_event_wait(queue, nullptr), PR_INVALID_HANDLE, "event_wait");
    failure(pr_device_get(UINT32_MAX, &discovery), PR_DEVICE_UNAVAILABLE, "device_get");
    auto stale = allocate(handles, context, 4);
    handles.release(stale);
    failure(pr_release(stale), PR_INVALID_HANDLE, "release");
    failure(pr_buffer_size(stale, &size), PR_INVALID_HANDLE, "buffer_size");
    auto fresh = allocate(handles, context, 4);
    require(fresh != stale, "Released native handle identity was reused");
    handles.release(fresh);
    pr_view rejected_view = 999;
    failure(pr_view_create(storage, UINT64_MAX, 4, 4, PR_READ, &rejected_view), PR_OUT_OF_BOUNDS,
            "view_create");
    require(rejected_view == 0, "Failed view creation returned a live-looking handle");
    failure(pr_view_create(storage, 1, 4, 4, PR_READ, &rejected_view), PR_INVALID_ARGUMENT,
            "view_create");
    failure(pr_view_create(storage, 0, 3, 4, PR_READ, &rejected_view), PR_INVALID_ARGUMENT,
            "view_create");
    failure(pr_view_create(storage, 0, 4, 3, PR_READ, &rejected_view), PR_UNSUPPORTED,
            "view_create");
    failure(pr_view_create(storage, 0, 4, 4, static_cast<pr_access>(0), &rejected_view),
            PR_INVALID_ARGUMENT, "view_create");
    pr_buffer oversized = 999;
    failure(pr_buffer_create(context, UINT64_MAX, &oversized), PR_OUT_OF_MEMORY, "buffer_create");
    require(oversized == 0, "Failed allocation returned a live-looking handle");
    failure(pr_buffer_read(storage, UINT64_MAX, nullptr, 0), PR_OUT_OF_BOUNDS, "buffer_read");
    failure(pr_buffer_write(storage, 0, nullptr, 4), PR_INVALID_ARGUMENT, "buffer_write");
    failure(pr_buffer_read(storage, 0, nullptr, 4), PR_INVALID_ARGUMENT, "buffer_read");
    failure(pr_buffer_write(storage, aliased.size() * 4, aliased.data(), 4), PR_OUT_OF_BOUNDS,
            "buffer_write");
    failure(pr_event_cancel(first_alias), PR_UNSUPPORTED, "event_cancel");
    pr_event rejected_event = 999;
    auto bad_args = args;
    bad_args[3] = f32_arg(n);
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, bad_args.data(), 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    require(rejected_event == 0, "Failed launch returned a live-looking event");
    bad_args = args;
    bad_args[2] = buffer_arg(va); // read-only view cannot be a write target.
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, bad_args.data(), 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    auto write_only = view(handles, storage, 4, n * 4, PR_WRITE);
    bad_args = args;
    bad_args[0] = buffer_arg(write_only);
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, bad_args.data(), 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    handles.release(write_only);
    pr_view byte_view = 0;
    success(pr_view_create(storage, 4, n * 4, 1, PR_READ, &byte_view));
    handles.keep(byte_view);
    bad_args = args;
    bad_args[0] = buffer_arg(byte_view);
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, bad_args.data(), 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    handles.release(byte_view);
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, args.data(), 3, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, nullptr, 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    failure(pr_launch(queue, add, {0, 1, 1}, {32, 1, 1}, args.data(), 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    auto empty = allocate(handles, context, 0);
    success(pr_buffer_size(empty, &size));
    require(size == 0, "Empty buffer has nonzero size");
    success(pr_buffer_read(empty, 0, nullptr, 0));
    success(pr_buffer_write(empty, 0, nullptr, 0));
    auto empty_view = view(handles, empty, 0, 0, PR_READ);
    bad_args = args;
    bad_args[0] = buffer_arg(empty_view);
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, bad_args.data(), 4, &rejected_event),
            PR_INVALID_ARGUMENT, "launch");
    failure(pr_buffer_read(empty, 0, nullptr, 1), PR_OUT_OF_BOUNDS, "buffer_read");
    handles.release(empty);
    handles.release(empty_view);
    pr_context other = 0;
    success(pr_context_create("auto", &other));
    handles.keep(other);
    auto foreign_buffer = allocate(handles, other, n * 4);
    auto foreign_view = view(handles, foreign_buffer, 0, n * 4, PR_READ);
    bad_args = args;
    bad_args[0] = buffer_arg(foreign_view);
    failure(pr_launch(queue, add, {5, 1, 1}, {32, 1, 1}, bad_args.data(), 4, &rejected_event),
            PR_CONTEXT_MISMATCH, "launch");
    pr_queue foreign_queue = 0;
    success(pr_queue_get(other, &foreign_queue));
    handles.keep(foreign_queue);
    failure(pr_launch(foreign_queue, add, {5, 1, 1}, {32, 1, 1}, args.data(), 4, &rejected_event),
            PR_CONTEXT_MISMATCH, "launch");
    for (auto owned : {foreign_queue, foreign_view, foreign_buffer, other})
      handles.release(owned);
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<unsigned char> artifact{std::istreambuf_iterator<char>(input),
                                        std::istreambuf_iterator<char>()};
    require(artifact.size() > 20, "Test module is empty");
    artifact[0] ^= 0xff;
    pr_module rejected_module = 999;
    failure(pr_module_load(context, artifact.data(), artifact.size(), &rejected_module),
            PR_COMPILATION_FAILED, "module_load");
    require(rejected_module == 0, "Failed artifact load returned a live-looking module");
    failure(pr_module_load(context, artifact.data(), 16u * 1024u * 1024u + 1, &rejected_module),
            PR_COMPILATION_FAILED, "module_load");
    failure(pr_module_load(context, nullptr, 1, &rejected_module), PR_INVALID_ARGUMENT,
            "module_load");

    // Concurrent callers get thread-local error records despite one serialized runtime.
    std::atomic<unsigned> entered{0};
    std::exception_ptr thread_errors[2];
    std::thread threads[2];
    for (unsigned i = 0; i < 2; ++i)
      threads[i] = std::thread([&, i] {
        try {
          uint64_t unused = 0;
          auto result = i ? pr_buffer_size(0, &unused) : pr_context_synchronize(0);
          ++entered;
          while (entered.load() != 2)
            std::this_thread::yield();
          pr_error detail{};
          require(result == PR_INVALID_HANDLE && pr_last_error(&detail) == PR_SUCCESS &&
                      detail.code == PR_INVALID_HANDLE &&
                      std::string(detail.operation) == (i ? "buffer_size" : "context_synchronize"),
                  "Concurrent API calls overwrote thread-local error detail");
        } catch (...) {
          thread_errors[i] = std::current_exception();
        }
      });
    for (auto &thread : threads)
      thread.join();
    for (const auto &error : thread_errors)
      if (error)
        std::rethrow_exception(error);
    negative_checks += 2;
    success(pr_device_count(&device_count));
    pr_error cleared{};
    success(pr_last_error(&cleared));
    require(cleared.code == PR_SUCCESS && !*cleared.operation && !*cleared.message,
            "Successful call did not reset thread-local error detail");

    args.push_back(f32_arg(0.5f));
    auto final_event = launch(handles, queue, affine, n, 32, args);
    for (int i = 0; i < n; ++i)
      expected[i + 1] = expected[i + 1] * 0.5f + expected[n + 2 + i];
    handles.release(va);
    handles.release(vb);
    handles.release(vo);
    handles.release(add);
    handles.release(first_alias);
    completed(final_event);
    require(gpu_events == 8, "Unexpected number of real GPU dispatches");
    verify(download(storage, expected.size()), expected, "same-allocation affine alias update");
    success(pr_context_write_evidence(context, argv[2]));
    failure(pr_context_write_evidence(context, argv[2]), PR_INVALID_ARGUMENT,
            "context_write_evidence");
    handles.release(queue);
    handles.release(context);
    // Children retain their context/module/allocation after all parent handles close.
    success(pr_kernel_parameter_count(affine, &parameter_count));
    require(parameter_count == 5, "Kernel lost its released parent module/context");
    verify(download(storage, expected.size()), expected, "buffer outlives context handle");
    pr_event_info retained{};
    success(pr_event_wait(final_event, &retained));
    require(retained.completed, "Event lost completion after context/queue release");
    handles.release(affine);
    handles.release(final_event);
    handles.release(storage);
    failure(pr_context_synchronize(context), PR_INVALID_HANDLE, "context_synchronize");
    std::cout << "Independent CPU comparisons and allocation canaries: PASS\n"
              << "Ownership/lifetime, queue ordering, copied scalars: PASS\n"
              << "Structured negative checks: " << negative_checks << " PASS\n"
              << "Verified GPU executions: " << gpu_events << "\nVerification: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Native C API qualification failed: " << error.what() << '\n';
    return 1;
  }
}
