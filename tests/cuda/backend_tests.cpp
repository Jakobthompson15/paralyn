// CPU-only tests of the CUDA backend's host logic. The production loader is
// tested against missing/unsuitable libraries; engine behavior is tested with
// an explicitly labelled in-process test double of the driver/NVRTC tables
// (tests/cuda/fake_driver.cpp), which never executes a kernel. Nothing here is
// GPU evidence or CUDA qualification.
#include "engine.hpp"
#include "fake_driver.hpp"
#include "paralyn/cuda_codegen.hpp"
#include "paralyn/detail/cuda_backend.hpp"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

namespace be = paralyn::backend;
namespace cu = paralyn::backend::cuda;
using paralyn::ScalarType;
namespace {
int checks = 0;
void require(bool ok, const std::string &message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
bool contains(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }
void expect_error(const std::function<void()> &f, be::ErrorCode code, const std::string &text, const char *what) {
  try {
    f();
  } catch (const be::Error &e) {
    require(e.code == code, std::string(what) + ": wrong error code; message: " + e.what());
    require(text.empty() || contains(e.what(), text), std::string(what) + ": message lacks '" + text + "': " + e.what());
    return;
  }
  require(false, std::string(what) + ": no error raised");
}
void set_env(const char *key, const char *value) {
#ifdef _WIN32
  _putenv_s(key, value);
#else
  setenv(key, value, 1);
#endif
}
paralyn::Kernel vector_add() {
  using namespace paralyn;
  const Expr global{ExprKind::Binary, ScalarType::U32, "+",
                    {{ExprKind::Binary, ScalarType::U32, "*",
                      {{ExprKind::Builtin, ScalarType::U32, "blockIdx.x", {}},
                       {ExprKind::Builtin, ScalarType::U32, "blockDim.x", {}}}},
                     {ExprKind::Builtin, ScalarType::U32, "threadIdx.x", {}}}};
  const Expr i{ExprKind::Ref, ScalarType::U32, "i", {}};
  auto load = [&](const char *b) {
    return Expr{ExprKind::Load, ScalarType::F32, "", {{ExprKind::Ref, ScalarType::F32, b, {}}, i}};
  };
  Statement store{StmtKind::Store, "", ScalarType::F32,
                  {ExprKind::Binary, ScalarType::F32, "+", {load("a"), load("b")}}, load("out"), {}};
  Statement guard{StmtKind::If, "", ScalarType::Bool,
                  {ExprKind::Binary, ScalarType::Bool, "<", {i, {ExprKind::Ref, ScalarType::U32, "n", {}}}}, {}, {store}};
  return {"vector_add",
          {{"a", ScalarType::F32, true, true}, {"b", ScalarType::F32, true, true},
           {"out", ScalarType::F32, true, false}, {"n", ScalarType::U32, false, false}},
          {{StmtKind::Let, "i", ScalarType::U32, global, {}, {}}, guard}};
}
std::shared_ptr<const cu::Api> fake_api() {
  auto api = std::make_shared<cu::Api>();
  api->driver = fake_cuda::driver_table();
  api->nvrtc = fake_cuda::nvrtc_table();
  api->driver_path = "test-double:driver";
  api->nvrtc_path = "test-double:nvrtc";
  api->test_double = true;
  return api;
}
be::BoundArgument view(const std::shared_ptr<be::Buffer> &b, std::size_t offset, std::size_t size) {
  be::BoundArgument a;
  a.is_buffer = true;
  a.allocation = b;
  a.offset = offset;
  a.size = size;
  a.type = ScalarType::F32;
  return a;
}
be::BoundArgument u32(std::uint32_t value) {
  be::BoundArgument a;
  a.type = ScalarType::U32;
  std::memcpy(a.bytes.data(), &value, 4);
  return a;
}
const auto caller = reinterpret_cast<cu::CUcontext>(std::uintptr_t(0xCA11E4));
void caller_restored(const char *where) {
  const auto &stack = fake_cuda::context_stack();
  require(stack.size() == 1 && stack.back() == caller, std::string("caller context not restored after ") + where);
  require(fake_cuda::state().context_violations.empty(),
          std::string("driver call without primary context current before ") + where + ": " +
              (fake_cuda::state().context_violations.empty() ? "" : fake_cuda::state().context_violations[0]));
}

void loader_tests(const std::string &unsuitable_library) {
  // Missing libraries: explicit paths are tried exactly and never replaced.
  cu::LoadRequest missing{"/nonexistent/paralyn-test/libcuda-missing", "/nonexistent/paralyn-test/libnvrtc-missing"};
  auto result = cu::load(missing);
  require(!result.available() && !result.api && !result.driver_loaded && !result.nvrtc_loaded,
          "missing libraries reported as loaded");
  require(contains(result.reason, "CUDA driver library not found") && contains(result.reason, "NVRTC library not found"),
          "missing-library reason incomplete: " + result.reason);
  require(result.attempts.size() == 2 && contains(result.attempts[0], "libcuda-missing") &&
              contains(result.attempts[1], "libnvrtc-missing"),
          "attempted candidates not recorded");
  // A real library without CUDA symbols (Paralyn's own native library).
  auto wrong = cu::load({unsuitable_library, unsuitable_library});
  require(!wrong.available() && !wrong.driver_loaded && !wrong.nvrtc_loaded, "library without symbols accepted");
  require(contains(wrong.reason, "lacks required symbols: cuInit") && contains(wrong.reason, "nvrtcVersion"),
          "missing-symbol reason incomplete: " + wrong.reason);
  require(!cu::default_driver_candidates().empty() && !cu::default_nvrtc_candidates().empty(),
          "no default platform candidates");
#ifdef _WIN32
  require(cu::default_driver_candidates()[0] == "nvcuda.dll", "Windows driver name");
#elif !defined(__APPLE__)
  require(cu::default_driver_candidates()[0] == "libcuda.so.1", "Linux driver soname");
#endif

  // Production status/selection with the driver forced absent (set before first use).
  const auto status = cu::status();
  require(status.implemented && !status.available && !status.driver_loaded, "forced-absent driver reported available");
  require(contains(status.reason, "CUDA driver library not found"), "status reason: " + status.reason);
  require(cu::devices().empty(), "devices fabricated while unavailable");
  expect_error([] { cu::create_context("cuda:0"); }, be::ErrorCode::invalid_device, "CUDA backend unavailable",
               "unavailable create_context");
  expect_error([] { be::registry_create_context("cuda:0"); }, be::ErrorCode::invalid_device,
               "no CPU fallback", "unavailable registry cuda:0");
  const auto listed = be::registry_devices();
  for (const auto &d : listed) require(d.backend != "CUDA", "unavailable CUDA device listed");
}

void selector_tests() {
  auto s = be::parse_selector("cuda:0");
  require(s.kind == be::SelectorKind::cuda && s.index == 0, "cuda:0");
  s = be::parse_selector("cuda:12");
  require(s.kind == be::SelectorKind::cuda && s.index == 12, "cuda:12");
  for (const char *bad : {"cuda:", "cuda:-1", "cuda:+1", "cuda:1x", "cuda: 1", "cuda:99999999999", "cuda:2147483648"})
    expect_error([&] { be::parse_selector(bad); }, be::ErrorCode::invalid_device, "cuda:", bad);
  require(cu::parse_ordinal("cuda:7") == 7, "parse_ordinal");
  expect_error([] { cu::parse_ordinal("metal:0"); }, be::ErrorCode::invalid_device, "cuda:INDEX", "non-cuda ordinal");
  require(be::parse_selector("hip:0").kind == be::SelectorKind::hip, "hip selector");
  require(be::parse_selector("rocm:1").kind == be::SelectorKind::hip, "rocm selector");
  expect_error([] { be::registry_create_context("hip:0"); }, be::ErrorCode::invalid_device, "not implemented", "hip context");
  require(be::parse_selector("auto").kind == be::SelectorKind::automatic, "auto");
  for (const char *metal : {"metal:0", "0", "metal:x"})
    require(be::parse_selector(metal).kind == be::SelectorKind::metal && be::parse_selector(metal).text == metal,
            "Metal selectors must pass through unchanged");
}

void mapping_tests() {
  using E = be::ErrorCode;
  const std::pair<int, E> table[] = {
      {cu::CUDA_ERROR_INVALID_VALUE, E::invalid_value}, {cu::CUDA_ERROR_OUT_OF_MEMORY, E::out_of_memory},
      {cu::CUDA_ERROR_NOT_INITIALIZED, E::invalid_device}, {cu::CUDA_ERROR_INSUFFICIENT_DRIVER, E::invalid_device},
      {cu::CUDA_ERROR_NO_DEVICE, E::invalid_device}, {cu::CUDA_ERROR_INVALID_DEVICE, E::invalid_device},
      {cu::CUDA_ERROR_STUB_LIBRARY, E::invalid_device}, {cu::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, E::invalid_device},
      {cu::CUDA_ERROR_INVALID_PTX, E::compilation}, {cu::CUDA_ERROR_UNSUPPORTED_PTX_VERSION, E::compilation},
      {cu::CUDA_ERROR_NO_BINARY_FOR_GPU, E::compilation}, {cu::CUDA_ERROR_NOT_FOUND, E::compilation},
      {cu::CUDA_ERROR_INVALID_HANDLE, E::internal}, {cu::CUDA_ERROR_INVALID_CONTEXT, E::internal},
      {cu::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, E::invalid_value}, {cu::CUDA_ERROR_NOT_SUPPORTED, E::unsupported},
      {cu::CUDA_ERROR_ILLEGAL_ADDRESS, E::execution}, {cu::CUDA_ERROR_LAUNCH_FAILED, E::execution},
      {cu::CUDA_ERROR_LAUNCH_TIMEOUT, E::execution}, {cu::CUDA_ERROR_UNKNOWN, E::execution}};
  for (const auto &[code, expected] : table)
    require(cu::map_result(code) == expected, "error mapping for CUresult " + std::to_string(code));
  for (int sticky : {700, 702, 714, 715, 716, 717, 718, 719, 214, 999})
    require(cu::is_sticky(sticky), "sticky " + std::to_string(sticky));
  for (int recoverable : {1, 2, 200, 218, 222, 400, 701, 801})
    require(!cu::is_sticky(recoverable), "non-sticky " + std::to_string(recoverable));
}

void architecture_tests() {
  auto nvrtc = fake_cuda::nvrtc_table();
  require(cu::choose_architecture(nvrtc, 8, 6) == "compute_86", "exact supported architecture");
  require(cu::choose_architecture(nvrtc, 8, 9) == "compute_86", "highest supported below device");
  require(cu::choose_architecture(nvrtc, 12, 0) == "compute_90", "newer device than NVRTC list");
  expect_error([&] { cu::choose_architecture(nvrtc, 3, 5); }, be::ErrorCode::unsupported, "no virtual architecture",
               "device older than NVRTC support");
  nvrtc.nvrtcGetNumSupportedArchs = nullptr;
  require(cu::choose_architecture(nvrtc, 8, 9) == "compute_89", "fallback without supported-arch query");
}

void engine_tests() {
  fake_cuda::reset();
  auto &fake = fake_cuda::state();
  auto &stack = fake_cuda::context_stack();
  stack.push_back(caller); // the application's own current context
  auto api = fake_api();
  require(cu::devices(*api).size() == 1 && cu::devices(*api)[0].backend == "CUDA", "fake device enumeration");
  require(cu::devices(*api)[0].stable_id == "cuda:uuid:a0a1a2a3-a4a5-a6a7-a8a9-aaabacadaeaf", "UUID stable id");
  expect_error([&] { cu::create_context(api, 1); }, be::ErrorCode::invalid_device, "out of range", "ordinal range");
  {
    auto context = cu::create_context(api, 0);
    caller_restored("context creation");
    require(fake.primary_retained == 1 && fake.streams == 1, "primary context retained and stream created");
    const auto info = context->device_info();
    require(info.backend == "CUDA" && info.max_block.x == 1024 && info.max_threadgroup_memory_bytes == 49152,
            "device info from driver attributes");

    auto a = context->allocate(64), b = context->allocate(64), out = context->allocate(64);
    auto empty = context->allocate(0);
    caller_restored("allocation");
    require(fake.memory.size() == 3, "zero-byte allocation must not call cuMemAlloc");
    std::vector<float> ha(16), hb(16), sentinel(16, -7.0f), back(16, 0);
    for (int i = 0; i < 16; ++i) { ha[i] = float(i); hb[i] = float(2 * i); }
    context->write(a, 0, ha.data(), 64);
    context->write(b, 0, hb.data(), 64);
    context->write(out, 0, sentinel.data(), 64);
    context->read(a, 4, back.data(), 60);
    caller_restored("copies");
    require(back[0] == 1.0f && back[14] == 15.0f, "offset copy round trip through driver copy calls");
    expect_error([&] { context->write(a, 60, ha.data(), 8); }, be::ErrorCode::invalid_value, "exceeds", "copy range");

    // Compilation: generated source and explicit options reach NVRTC unchanged.
    const auto kernel = vector_add();
    context->prepare(kernel);
    caller_restored("prepare");
    require(fake.sources.size() == 1 && fake.sources[0] == paralyn::emit_cuda(kernel), "NVRTC source is not emit_cuda output");
    std::vector<std::string> expected{"--gpu-architecture=compute_86"};
    for (const auto &o : paralyn::cuda_numerical_options()) expected.push_back(o);
    require(fake.nvrtc_options == expected, "NVRTC options differ from explicit arch + numerical policy");
    require(fake.programs == 0, "NVRTC program leaked");
    require(fake.modules == 1 && fake.loaded_images.size() == 1, "module loaded once");

    // Launch: views lower to base + byte offset; scalars are copied; no computation happens.
    fake.parameter_sizes = {8, 8, 8, 4};
    auto event = context->submit(kernel, {2, 1, 1}, {8, 1, 1},
                                 {view(a, 0, 64), view(b, 4, 60), view(out, 0, 64), u32(15)});
    caller_restored("submit");
    require(fake.modules == 1, "prepared module must be reused");
    require(fake.launches.size() == 1, "one launch recorded");
    const auto &launch = fake.launches[0];
    require(launch.function == "uc_kernel_vector_add", "entrypoint name");
    require(launch.grid[0] == 2 && launch.block[0] == 8 && launch.grid[1] == 1 && launch.block[2] == 1, "geometry");
    cu::CUdeviceptr pa = 0, pb = 0, pout = 0;
    std::uint32_t n = 0;
    std::memcpy(&pa, launch.parameters[0].data(), 8);
    std::memcpy(&pb, launch.parameters[1].data(), 8);
    std::memcpy(&pout, launch.parameters[2].data(), 8);
    std::memcpy(&n, launch.parameters[3].data(), 4);
    require(pb % 256 == 4 && pa % 256 == 0 && pout % 256 == 0 && n == 15, "argument marshalling");
    const auto timing = event->wait();
    caller_restored("event wait");
    require(timing.completed && timing.duration_valid && !timing.timestamps_valid && timing.clock_domain == 1,
            "duration-only timing contract");
    require(timing.duration_seconds == 0.25 / 1000.0 && timing.gpu_start_seconds == 0 && timing.gpu_end_seconds == 0,
            "fabricated or wrong timing");
    require(fake.events == 0, "events leaked after completion");
    context->read(out, 0, back.data(), 64);
    require(back == sentinel, "test double must never compute kernel results");
    const auto stats = context->statistics();
    require(stats.completed_launches == 1 && stats.pipeline_compilations == 1 && stats.current_buffer_bytes == 192,
            "runtime statistics");

    // Validation before any driver launch.
    const auto launches = fake.launches.size();
    expect_error([&] { context->submit(kernel, {1, 1, 1}, {2048, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(out, 0, 64), u32(1)}); },
                 be::ErrorCode::invalid_value, "Block dimension", "block limit");
    expect_error([&] { context->submit(kernel, {0, 1, 1}, {1, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(out, 0, 64), u32(1)}); },
                 be::ErrorCode::invalid_value, "positive", "zero grid");
    expect_error([&] { context->submit(kernel, {1, 1, 1}, {1, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(out, 0, 64)}); },
                 be::ErrorCode::invalid_value, "count", "argument count");
    expect_error([&] { context->submit(kernel, {1, 1, 1}, {1, 1, 1}, {view(a, 2, 60), view(b, 0, 64), view(out, 0, 64), u32(1)}); },
                 be::ErrorCode::invalid_value, "aligned", "misaligned view");
    expect_error([&] { context->submit(kernel, {1, 1, 1}, {1, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(empty, 0, 0), u32(1)}); },
                 be::ErrorCode::invalid_value, "", "empty view");
    {
      using namespace paralyn;
      const Expr zero{ExprKind::Literal, ScalarType::U32, "0", {}};
      const Expr x0{ExprKind::Load, ScalarType::F32, "", {{ExprKind::Ref, ScalarType::F32, "x", {}}, zero}};
      Kernel alias{"mixed_alias",
                   {{"x", ScalarType::F32, true, false}, {"y", ScalarType::I32, true, true}},
                   {{StmtKind::Store, "", ScalarType::F32, x0, x0, {}}}};
      auto y = view(a, 0, 64);
      y.type = ScalarType::I32;
      expect_error([&] { context->submit(alias, {1, 1, 1}, {1, 1, 1}, {view(a, 0, 64), y}); },
                   be::ErrorCode::unsupported, "Mixed pointee", "mixed-type alias");
    }
    paralyn::ExecutableModule msl;
    expect_error([&] { context->prepare(msl); }, be::ErrorCode::unsupported, "Metal-specific", "MSL on CUDA");
    const std::string handwritten = "kernel void x() {}";
    expect_error([&] { context->submit(kernel, {1, 1, 1}, {1, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(out, 0, 64), u32(1)}, &handwritten); },
                 be::ErrorCode::unsupported, "Metal-only", "handwritten MSL on CUDA");
    require(fake.launches.size() == launches, "invalid launch reached the driver");
    caller_restored("validation failures");

    // Recoverable driver errors map precisely and leave the context usable.
    fake.fail_next["cuMemAlloc"] = cu::CUDA_ERROR_OUT_OF_MEMORY;
    expect_error([&] { context->allocate(1024); }, be::ErrorCode::out_of_memory, "CUDA_ERROR_OUT_OF_MEMORY", "OOM");
    caller_restored("allocation failure");
    fake.fail_next["cuLaunchKernel"] = cu::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    expect_error([&] { context->submit(kernel, {1, 1, 1}, {8, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(out, 0, 64), u32(8)}); },
                 be::ErrorCode::invalid_value, "LAUNCH_OUT_OF_RESOURCES", "launch resources");
    caller_restored("launch failure");
    require(fake.events == 0, "events leaked after failed launch");
    context->synchronize();

    auto other = kernel;
    other.name = "vector_add_bad_ptx";
    fake.fail_next["cuModuleLoadDataEx"] = cu::CUDA_ERROR_INVALID_PTX;
    expect_error([&] { context->prepare(other); }, be::ErrorCode::compilation, "JIT log: ptxas fatal", "invalid PTX");
    caller_restored("module load failure");
    fake.fail_next["nvrtcCompileProgram"] = 6;
    expect_error([&] { context->prepare(other); }, be::ErrorCode::compilation, "fake NVRTC log", "NVRTC failure");
    require(fake.programs == 0, "NVRTC program leaked after failure");
    fake.fail_next["cuCtxPushCurrent"] = cu::CUDA_ERROR_INVALID_CONTEXT;
    expect_error([&] { context->allocate(4); }, be::ErrorCode::internal, "cuCtxPushCurrent", "push failure");
    caller_restored("push failure");

    // Evidence names the test double and never invents timestamps.
    const auto directory = std::filesystem::temp_directory_path() /
                           ("paralyn-cuda-fake-" + std::to_string(reinterpret_cast<std::uintptr_t>(&fake)));
    std::filesystem::remove_all(directory);
    context->write_evidence(directory.string());
    std::ifstream in(directory / "execution.json");
    std::stringstream text;
    text << in.rdbuf();
    const auto evidence = text.str();
    require(contains(evidence, "\"test_double\": true") && contains(evidence, "\"gpu_timestamps_valid\": false") &&
                contains(evidence, "\"gpu_start_seconds\": null") && contains(evidence, "\"duration_only\"") &&
                contains(evidence, "\"cpu_fallback\": false") && contains(evidence, "compute_86"),
            "evidence contract");
    std::ifstream source(directory / "source-0.cu");
    std::stringstream cu_text;
    cu_text << source.rdbuf();
    require(cu_text.str() == paralyn::emit_cuda(kernel), "evidence source differs from launched source");
    std::filesystem::remove_all(directory);
    caller_restored("evidence");

    // Sticky fault: the first observation fails and the context stays failed.
    fake.parameter_sizes = {8, 8, 8, 4};
    auto faulted = context->submit(kernel, {1, 1, 1}, {8, 1, 1}, {view(a, 0, 64), view(b, 0, 64), view(out, 0, 64), u32(8)});
    fake.fail_next["cuEventSynchronize"] = cu::CUDA_ERROR_ILLEGAL_ADDRESS;
    expect_error([&] { faulted->wait(); }, be::ErrorCode::execution, "CUDA_ERROR_ILLEGAL_ADDRESS", "sticky fault");
    expect_error([&] { context->allocate(4); }, be::ErrorCode::execution, "ILLEGAL_ADDRESS", "context stays failed");
    expect_error([&] { context->synchronize(); }, be::ErrorCode::execution, "ILLEGAL_ADDRESS", "synchronize after fault");
    caller_restored("sticky fault");
    require(fake.events == 0, "events leaked after fault");
  }
  // Context and buffers released: every driver resource returned exactly once.
  caller_restored("destruction");
  require(fake.memory.empty(), "device allocations leaked");
  require(fake.modules == 0 && fake.streams == 0 && fake.events == 0, "modules/streams/events leaked");
  require(fake.primary_retained == 0 && fake.primary_retains == fake.primary_releases, "primary context retain/release imbalance");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: cuda_backend_tests UNSUITABLE_SHARED_LIBRARY");
    // Force deterministic absence for the production loader in this process.
    set_env("PARALYN_CUDA_DRIVER_LIBRARY", "/nonexistent/paralyn-test/libcuda-forced-absent");
    set_env("PARALYN_NVRTC_LIBRARY", "/nonexistent/paralyn-test/libnvrtc-forced-absent");
    loader_tests(argv[1]);
    selector_tests();
    mapping_tests();
    architecture_tests();
    engine_tests();
    std::cout << "CUDA backend host logic (loader, selectors, error mapping; engine via labelled "
                 "test double, no GPU execution): "
              << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "CUDA backend host test failed: " << e.what() << "\n";
    return 1;
  }
}
