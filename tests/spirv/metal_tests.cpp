// SPIR-V-derived modules on the physical Metal GPU through the public C ABI.
// Every result is compared with an independent CPU reference computed here;
// no kernel runs on the CPU in place of the GPU.
// Usage: spirv_metal_tests VECTOR_ADD.prx REDUCE_SUM.prx NEW_EVIDENCE_DIR
#include "paralyn/executable.hpp"
#include "paralyn/native.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned negative_checks = 0, gpu_events = 0, compared_values = 0;
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
void success(pr_status status) {
  if (status == PR_SUCCESS)
    return;
  pr_error error{};
  pr_last_error(&error);
  throw std::runtime_error(std::string(error.operation) + ": " + error.message);
}
void failure(pr_status status, pr_status expected, const char *operation, const char *needle = nullptr) {
  pr_error error{};
  pr_last_error(&error);
  require(status == expected && error.code == expected && error.operation == std::string(operation) &&
              *error.message && (!needle || std::strstr(error.message, needle)),
          std::string("Incorrect failure for ") + operation + ": " + std::to_string(status) + " " +
              error.message);
  ++negative_checks;
}
struct Handles {
  std::vector<pr_handle> values;
  pr_handle keep(pr_handle h) {
    require(h != 0, "Invalid successful handle");
    values.push_back(h);
    return h;
  }
  ~Handles() {
    for (auto i = values.rbegin(); i != values.rend(); ++i)
      if (*i)
        pr_release(*i);
  }
};
pr_argument view_arg(pr_view v) {
  pr_argument a{};
  a.type = PR_BUFFER;
  a.view = v;
  return a;
}
pr_argument u32(std::uint32_t n) {
  pr_argument a{};
  a.type = PR_U32;
  a.u32 = n;
  return a;
}
pr_argument f32(float x) {
  pr_argument a{};
  a.type = PR_F32;
  a.f32 = x;
  return a;
}
std::vector<unsigned char> file(const char *path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), std::string("Cannot open ") + path);
  return {std::istreambuf_iterator<char>(in), {}};
}
pr_buffer buffer(Handles &h, pr_context c, const std::vector<float> &data) {
  pr_buffer b = 0;
  success(pr_buffer_create(c, data.size() * 4, &b));
  h.keep(b);
  success(pr_buffer_write(b, 0, data.data(), data.size() * 4));
  return b;
}
pr_view view(Handles &h, pr_buffer b, std::size_t offset_elements, std::size_t count, pr_access access) {
  pr_view v = 0;
  success(pr_view_create(b, offset_elements * 4, count * 4, 4, access, &v));
  return h.keep(v);
}
void completed(pr_event e) {
  pr_event_info info{};
  success(pr_event_wait(e, &info));
  require(info.completed && info.gpu_start_seconds > 0 && info.gpu_end_seconds >= info.gpu_start_seconds,
          "Missing real completed GPU event/timestamps");
  pr_event_timing_v1 timing{};
  timing.struct_size = sizeof(timing);
  timing.version = 1;
  success(pr_event_timing(e, &timing));
  require(timing.completed && timing.duration_valid && timing.timestamps_valid &&
              timing.clock_domain == PR_CLOCK_METAL_SYSTEM_MACH && timing.duration_seconds > 0,
          "Invalid versioned GPU timing contract");
  ++gpu_events;
}
pr_event launch(Handles &h, pr_queue q, pr_kernel k, pr_dim3 grid, pr_dim3 block,
                std::vector<pr_argument> args) {
  pr_event e = 0;
  success(pr_launch(q, k, grid, block, args.data(), static_cast<std::uint32_t>(args.size()), &e));
  return h.keep(e);
}
std::vector<float> readback(pr_buffer b, std::size_t n) {
  std::vector<float> out(n);
  success(pr_buffer_read(b, 0, out.data(), n * 4));
  return out;
}
// Exactly representable inputs: sums of these stay integral with |x| < 2^24,
// so the GPU and CPU results are exact regardless of summation order.
float value(std::size_t i, unsigned seed) { return float(int((i * 37 + seed) % 17) - 8); }
float fraction(std::size_t i, unsigned seed) { return float(int((i * 29 + seed) % 255) - 127) * 0.125f; }
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4, "Usage: spirv_metal_tests VECTOR_ADD.prx REDUCE_SUM.prx NEW_EVIDENCE_DIR");
    const std::filesystem::path evidence = argv[3];
    require(!std::filesystem::exists(evidence), "Evidence directory must be new");
    const auto add_bytes = file(argv[1]), reduce_bytes = file(argv[2]);
    const auto add_module = paralyn::deserialize_executable(add_bytes.data(), add_bytes.size());
    require(add_module.format == paralyn::ExecutableFormat::SpirvMsl, "vector_add.prx is not SPIR-V-derived");

    Handles h;
    pr_context context = 0;
    success(pr_context_create("auto", &context));
    h.keep(context);
    pr_device_capabilities_v1 caps{};
    caps.struct_size = sizeof(caps);
    caps.version = 1;
    success(pr_context_capabilities(context, &caps));
    require(caps.artifact_formats & PR_ARTIFACT_SPIRV_MSL, "Device does not advertise SPIR-V modules");
    pr_queue queue = 0;
    success(pr_queue_get(context, &queue));
    h.keep(queue);
    pr_module add_handle = 0, reduce_handle = 0;
    success(pr_module_load_file(context, argv[1], &add_handle));
    h.keep(add_handle);
    success(pr_module_load(context, reduce_bytes.data(), reduce_bytes.size(), &reduce_handle));
    h.keep(reduce_handle);
    pr_kernel add = 0, reduce = 0;
    success(pr_module_kernel(add_handle, "vector_add", &add));
    h.keep(add);
    success(pr_module_kernel(reduce_handle, "reduce_sum", &reduce));
    h.keep(reduce);

    std::uint32_t count = 0;
    success(pr_kernel_parameter_count(add, &count));
    require(count == 4, "vector_add reflected parameter count");
    const char *names[] = {"a", "b", "c", "n"};
    const pr_access access[] = {PR_READ, PR_READ, PR_WRITE, PR_READ};
    for (std::uint32_t i = 0; i < 4; ++i) {
      pr_parameter_info info{};
      success(pr_kernel_parameter(add, i, &info));
      require(info.name == std::string(names[i]) && info.access == access[i] &&
                  info.is_buffer == (i < 3) && info.type == (i < 3 ? PR_F32 : PR_U32),
              std::string("vector_add C ABI parameter reflection at ") + names[i]);
    }

    // Vector addition across boundary/partial/large sizes with canaries around the output view.
    for (std::size_t n : {std::size_t(1), std::size_t(63), std::size_t(64), std::size_t(65),
                          std::size_t(1000), std::size_t(262147), std::size_t(1000003)}) {
      std::vector<float> a(n), b(n), c(n + 2, -4242.0f);
      for (std::size_t i = 0; i < n; ++i) {
        a[i] = fraction(i, 3);
        b[i] = fraction(i, 101);
      }
      auto ba = buffer(h, context, a), bb = buffer(h, context, b), bc = buffer(h, context, c);
      auto va = view(h, ba, 0, n, PR_READ), vb = view(h, bb, 0, n, PR_READ),
           vc = view(h, bc, 1, n, PR_WRITE);
      const auto groups = static_cast<std::uint32_t>((n + 63) / 64);
      completed(launch(h, queue, add, {groups, 1, 1}, {64, 1, 1},
                       {view_arg(va), view_arg(vb), view_arg(vc), u32(static_cast<std::uint32_t>(n))}));
      const auto out = readback(bc, n + 2);
      require(out.front() == -4242.0f && out.back() == -4242.0f, "vector_add wrote outside its view");
      for (std::size_t i = 0; i < n; ++i)
        require(out[i + 1] == a[i] + b[i], "vector_add mismatch at " + std::to_string(i) + " of " +
                                              std::to_string(n));
      compared_values += static_cast<unsigned>(n + 2);
    }

    // Read-only aliases: a and b bound to the same allocation (both NonWritable).
    {
      const std::size_t n = 4099;
      std::vector<float> a(n), c(n, 0);
      for (std::size_t i = 0; i < n; ++i) a[i] = fraction(i, 7);
      auto ba = buffer(h, context, a), bc = buffer(h, context, c);
      auto va = view(h, ba, 0, n, PR_READ), vc = view(h, bc, 0, n, PR_WRITE);
      completed(launch(h, queue, add, {65, 1, 1}, {64, 1, 1},
                       {view_arg(va), view_arg(va), view_arg(vc), u32(static_cast<std::uint32_t>(n))}));
      const auto out = readback(bc, n);
      for (std::size_t i = 0; i < n; ++i)
        require(out[i] == a[i] + a[i], "read-only alias mismatch at " + std::to_string(i));
      compared_values += static_cast<unsigned>(n);
      // A writable binding aliasing a read binding is outside the profile.
      auto vrw = view(h, ba, 0, n, PR_READ_WRITE);
      std::vector<pr_argument> alias{view_arg(vrw), view_arg(va), view_arg(vrw),
                                     u32(static_cast<std::uint32_t>(n))};
      pr_event e = 0;
      failure(pr_launch(queue, add, {65, 1, 1}, {64, 1, 1}, alias.data(), 4, &e), PR_UNSUPPORTED,
              "launch", "repeated-allocation");
    }

    // Structured-loop + workgroup-barrier reduction with a two-member uniform block.
    struct Case { std::size_t n; std::uint32_t groups; float scale; };
    for (const Case &k : {Case{1, 1, 1.0f}, Case{64, 1, 0.5f}, Case{4097, 3, 0.5f},
                          Case{100000, 16, 1.0f}, Case{1000003, 128, 0.5f}, Case{777, 40, -2.0f}}) {
      std::vector<float> input(k.n), partial(k.groups + 1, -777.0f);
      for (std::size_t i = 0; i < k.n; ++i) input[i] = value(i, k.groups);
      auto bi = buffer(h, context, input), bp = buffer(h, context, partial);
      auto vi = view(h, bi, 0, k.n, PR_READ), vp = view(h, bp, 0, k.groups, PR_WRITE);
      completed(launch(h, queue, reduce, {k.groups, 1, 1}, {64, 1, 1},
                       {view_arg(vi), view_arg(vp), u32(static_cast<std::uint32_t>(k.n)), f32(k.scale)}));
      const auto out = readback(bp, k.groups + 1);
      // Independent reference: grid-stride ownership i mod (64 * groups) -> workgroup.
      std::vector<double> expected(k.groups, 0.0);
      const std::size_t total = std::size_t(64) * k.groups;
      for (std::size_t i = 0; i < k.n; ++i) expected[(i % total) / 64] += input[i];
      for (std::uint32_t w = 0; w < k.groups; ++w) {
        const double want = expected[w] * k.scale;
        require(std::fabs(want) < 16777216.0 && double(out[w]) == want,
                "reduce_sum mismatch at workgroup " + std::to_string(w) + " (n=" +
                    std::to_string(k.n) + "): " + std::to_string(out[w]) + " != " + std::to_string(want));
      }
      require(out.back() == -777.0f, "reduce_sum wrote past its output view");
      compared_values += k.groups + 1;
    }

    // Launch-time contract violations (no GPU work is submitted for any of these).
    {
      std::vector<float> one(64, 1.0f);
      auto b1 = buffer(h, context, one), b2 = buffer(h, context, one), b3 = buffer(h, context, one);
      auto r1 = view(h, b1, 0, 64, PR_READ), r2 = view(h, b2, 0, 64, PR_READ),
           w3 = view(h, b3, 0, 64, PR_WRITE), r3 = view(h, b3, 0, 64, PR_READ);
      pr_event e = 0;
      std::vector<pr_argument> args{view_arg(r1), view_arg(r2), view_arg(w3), u32(64)};
      failure(pr_launch(queue, add, {1, 1, 1}, {64, 1, 1}, args.data(), 3, &e), PR_INVALID_ARGUMENT,
              "launch", "argument count");
      args[3] = f32(64.0f);
      failure(pr_launch(queue, add, {1, 1, 1}, {64, 1, 1}, args.data(), 4, &e), PR_INVALID_ARGUMENT,
              "launch", "scalar type");
      args[3] = view_arg(r1);
      failure(pr_launch(queue, add, {1, 1, 1}, {64, 1, 1}, args.data(), 4, &e), PR_INVALID_ARGUMENT, "launch");
      args[3] = u32(64);
      failure(pr_launch(queue, add, {2, 1, 1}, {32, 1, 1}, args.data(), 4, &e), PR_INVALID_ARGUMENT,
              "launch", "workgroup");
      args[2] = view_arg(r3);
      failure(pr_launch(queue, add, {1, 1, 1}, {64, 1, 1}, args.data(), 4, &e), PR_INVALID_ARGUMENT,
              "launch", "access");
      std::vector<pr_argument> reduce_args{view_arg(r1), view_arg(w3), f32(1.0f), f32(1.0f)};
      failure(pr_launch(queue, reduce, {1, 1, 1}, {64, 1, 1}, reduce_args.data(), 4, &e),
              PR_INVALID_ARGUMENT, "launch", "scalar type");
    }

    // Tampered descriptors: rejected at load by the container verifier or by
    // Metal reflection of the generated pipeline, never at dispatch.
    auto tampered_load = [&](paralyn::ExecutableModule m, const char *needle) {
      const auto wire = paralyn::serialize_executable(m);
      pr_module handle = 0;
      failure(pr_module_load(context, wire.data(), wire.size(), &handle), PR_COMPILATION_FAILED,
              "module_load", needle);
      require(handle == 0, "failed module load returned a handle");
    };
    {
      auto m = add_module;
      m.entries[0].parameters[0].type = paralyn::ScalarType::I32; // declared i32, shader reads float
      tampered_load(m, "storage buffer layout differs");
      const auto reduce_module = paralyn::deserialize_executable(reduce_bytes.data(), reduce_bytes.size());
      m = reduce_module;
      m.entries[0].parameters[3].block_offset = 8; // scale lives at offset 4
      tampered_load(m, "scalar block member differs");
      m = reduce_module;
      m.entries[0].parameters[2].type = paralyn::ScalarType::F32; // n is u32
      tampered_load(m, "scalar block member differs");
      // A container that bypasses serialize-time checks: descriptor omits storage buffer c.
      m = add_module;
      m.entries[0].parameters.erase(m.entries[0].parameters.begin() + 2);
      bool refused = false;
      try {
        (void)paralyn::serialize_executable(m);
      } catch (const std::exception &) {
        refused = true;
      }
      require(refused, "descriptor omitting a used storage buffer was serialized");
      ++negative_checks;
      auto wire = paralyn::serialize_executable(add_module);
      wire[8] = 3; // unknown container version
      pr_module handle = 0;
      failure(pr_module_load(context, wire.data(), wire.size(), &handle), PR_COMPILATION_FAILED,
              "module_load", "container version");
    }

    success(pr_context_synchronize(context));
    success(pr_context_write_evidence(context, evidence.string().c_str()));
    std::cout << "SPIR-V on Metal: " << gpu_events << " GPU events, " << compared_values
              << " values independently compared, " << negative_checks << " rejection checks\n";
    std::cout << "Verification: PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "spirv_metal_tests failed: " << e.what() << "\n";
    return 1;
  }
}
