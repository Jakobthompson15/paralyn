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
#ifdef _WIN32
  require(cu::default_driver_candidates() == std::vector<std::string>{"nvcuda.dll"}, "Windows driver name");
  for (const auto &candidate : cu::default_nvrtc_candidates())
    require(cu::is_absolute_library_path(candidate), "bare Windows NVRTC candidate " + candidate);
#elif defined(__APPLE__)
  // No CUDA driver exists for macOS: nothing may be loaded by default.
  require(cu::default_driver_candidates().empty() && cu::default_nvrtc_candidates().empty(),
          "macOS must have no default CUDA library candidates");
#else
  require(!cu::default_driver_candidates().empty() && cu::default_driver_candidates()[0] == "libcuda.so.1",
          "Linux driver soname");
  for (const auto &candidate : cu::default_nvrtc_candidates())
    require(candidate.rfind("libnvrtc.so", 0) == 0 || cu::is_absolute_library_path(candidate),
            "unexpected Linux NVRTC candidate " + candidate);
#endif
#ifndef _WIN32
  require(cu::is_absolute_library_path("/usr/lib/libcuda.so.1"), "absolute POSIX path");
  for (const char *relative : {"", "libcuda.dylib", "./libcuda.dylib", "lib/libcuda.so", "../libnvrtc.so"})
    require(!cu::is_absolute_library_path(relative), std::string("relative path accepted: ") + relative);
#else
  for (const char *absolute : {"C:\\Windows\\System32\\x.dll", "C:/CUDA/bin/x.dll", "\\\\server\\share\\x.dll"})
    require(cu::is_absolute_library_path(absolute), std::string("absolute Windows path: ") + absolute);
  for (const char *relative : {"", "x.dll", ".\\x.dll", "bin\\x.dll", "C:x.dll", "\\x.dll", "\\\\?\\C:\\x.dll"})
    require(!cu::is_absolute_library_path(relative), std::string("relative Windows path accepted: ") + relative);
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

// Regression: the loader must never load a library from the current working
// directory (macOS dyld and the default Windows search order both search it for
// bare names). A probe library is planted under every CUDA library name in a
// scratch directory that becomes the working directory.
void cwd_tests(const std::string &probe) {
  namespace fs = std::filesystem;
  const auto directory = fs::temp_directory_path() /
                         ("paralyn-cuda-cwd-" + std::to_string(reinterpret_cast<std::uintptr_t>(&probe)));
  fs::remove_all(directory);
  fs::create_directories(directory);
  for (const char *name : {"libcuda.dylib", "libnvrtc.dylib", "libcuda.so.1", "libcuda.so", "libnvrtc.so.13",
                           "libnvrtc.so.12", "libnvrtc.so", "nvcuda.dll", "nvrtc64_130_0.dll", "nvrtc64_120_0.dll"})
    fs::copy_file(probe, directory / name, fs::copy_options::overwrite_existing);
  const auto marker = directory / "PARALYN_CWD_PROBE_LOADED";
  const auto previous = fs::current_path();
  fs::current_path(directory);
  try {
    auto defaults = cu::load({});
    require(!fs::exists(marker), "default candidates loaded a library from the current directory");
    require(!contains(defaults.driver_path, directory.string()) && !contains(defaults.nvrtc_path, directory.string()),
            "default load chose a planted library");
#ifdef __APPLE__
    require(!defaults.driver_loaded && !defaults.nvrtc_loaded && defaults.attempts.empty() &&
                contains(defaults.reason, "PARALYN_CUDA_DRIVER_LIBRARY to an absolute path"),
            "macOS default load must attempt nothing: " + defaults.reason);
#endif
    // Relative explicit paths are refused before reaching the platform loader.
    for (const char *relative : {"libcuda.dylib", "./libcuda.so.1", "nvrtc64_120_0.dll"}) {
      auto refused = cu::load({relative, relative});
      require(!fs::exists(marker), std::string("relative explicit path loaded from the current directory: ") + relative);
      require(!refused.driver_loaded && !refused.nvrtc_loaded && refused.attempts.size() == 2 &&
                  contains(refused.attempts[0], "refused") && contains(refused.attempts[1], "refused"),
              std::string("relative explicit path not refused: ") + relative);
    }
    // Positive control: the same probe loaded by absolute path does run, so
    // the marker's absence above is meaningful.
    const auto absolute = (directory / "libcuda.dylib").string();
    auto control = cu::load({absolute, absolute});
    require(fs::exists(marker), "probe library did not signal when loaded by absolute path");
    require(!control.available() && contains(control.reason, "lacks required symbols"),
            "probe without CUDA symbols accepted: " + control.reason);
  } catch (...) {
    fs::current_path(previous);
    fs::remove_all(directory);
    throw;
  }
  fs::current_path(previous);
  fs::remove_all(directory);
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
      {cu::CUDA_ERROR_LAUNCH_TIMEOUT, E::execution}, {cu::CUDA_ERROR_UNKNOWN, E::execution},
      {cu::CUDA_ERROR_ASSERT, E::execution}, {cu::CUDA_ERROR_CONTEXT_IS_DESTROYED, E::execution},
      {cu::CUDA_ERROR_TENSOR_MEMORY_LEAK, E::execution}, {cu::CUDA_ERROR_CONTAINED, E::execution},
      {cu::CUDA_ERROR_MPS_CLIENT_TERMINATED, E::execution}, {cu::CUDA_ERROR_EXTERNAL_DEVICE, E::execution}};
  for (const auto &[code, expected] : table)
    require(cu::map_result(code) == expected, "error mapping for CUresult " + std::to_string(code));
  // Every code NVIDIA documents as leaving the context/process unusable.
  for (int sticky : {214, 226, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719, 721, 810, 911, 999})
    require(cu::is_sticky(sticky), "sticky " + std::to_string(sticky));
  for (int recoverable : {1, 2, 200, 218, 222, 400, 701, 711, 720, 801})
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
  // registry_id: nonzero (0 means "no physical device" to existing consumers),
  // backend-tagged, and distinct per ordinal.
  fake.device_count = 2;
  const auto two = cu::devices(*api);
  require(two.size() == 2 && two[0].registry_id == 0x4355444100000001ULL &&
              two[1].registry_id == 0x4355444100000002ULL && cu::cuda_registry_id(0) == two[0].registry_id,
          "CUDA registry_id must be nonzero, tagged and distinct");
  fake.device_count = 1;
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

// Every driver resource returned exactly once and the caller's context intact.
void balanced(const char *where) {
  const auto &fake = fake_cuda::state();
  caller_restored(where);
  require(fake.memory.empty(), std::string("device allocations leaked: ") + where);
  require(fake.modules == 0 && fake.streams == 0 && fake.events == 0 && fake.programs == 0,
          std::string("modules/streams/events/programs leaked: ") + where);
  require(fake.primary_retained == 0 && fake.primary_retains == fake.primary_releases,
          std::string("primary context retain/release imbalance: ") + where);
}
void fresh() {
  fake_cuda::reset();
  fake_cuda::context_stack().push_back(caller);
  fake_cuda::state().parameter_sizes = {8, 8, 8, 4};
}

// Negative paths of the engine through the labelled test double: injected
// driver failures map to precise error codes and leak nothing.
void failure_path_tests() {
  auto &fake = fake_cuda::state();
  const auto kernel = vector_add();
  auto args = [](const std::shared_ptr<be::Buffer> &a, const std::shared_ptr<be::Buffer> &b,
                 const std::shared_ptr<be::Buffer> &c) {
    return std::vector<be::BoundArgument>{view(a, 0, 64), view(b, 0, 64), view(c, 0, 64), u32(16)};
  };

  // Constructor failures: before, at and after the primary-context retain.
  fresh();
  fake.fail_next["cuDevicePrimaryCtxRetain"] = cu::CUDA_ERROR_INVALID_DEVICE;
  expect_error([&] { cu::create_context(fake_api(), 0); }, be::ErrorCode::invalid_device,
               "cuDevicePrimaryCtxRetain", "primary retain failure");
  require(fake.primary_retains == 0 && fake.primary_releases == 0, "failed retain must not be released");
  balanced("primary retain failure");

  fresh();
  fake.cc_major = 3;
  fake.cc_minor = 5; // older than every NVRTC architecture: choose_architecture throws after retain
  expect_error([&] { cu::create_context(fake_api(), 0); }, be::ErrorCode::unsupported, "no virtual architecture",
               "constructor failure after retain");
  require(fake.primary_retains == 1 && fake.primary_releases == 1, "primary context not released exactly once");
  balanced("architecture failure after retain");

  fresh();
  fake.fail_next["cuStreamCreate"] = cu::CUDA_ERROR_OUT_OF_MEMORY;
  expect_error([&] { cu::create_context(fake_api(), 0); }, be::ErrorCode::out_of_memory, "cuStreamCreate",
               "stream creation failure");
  require(fake.primary_retains == 1 && fake.primary_releases == 1, "primary context not released after stream failure");
  balanced("stream creation failure");

  // Module/function failures unload the module; the context stays usable.
  fresh();
  {
    auto context = cu::create_context(fake_api(), 0);
    fake.fail_next["cuModuleGetFunction"] = cu::CUDA_ERROR_NOT_FOUND;
    expect_error([&] { context->prepare(kernel); }, be::ErrorCode::compilation, "cuModuleGetFunction",
                 "function lookup failure");
    require(fake.modules == 0 && fake.programs == 0, "module leaked after cuModuleGetFunction failure");
    caller_restored("function lookup failure");
    fake.fail_next["cuFuncGetAttribute"] = cu::CUDA_ERROR_INVALID_VALUE;
    expect_error([&] { context->prepare(kernel); }, be::ErrorCode::invalid_value, "cuFuncGetAttribute",
                 "function attribute failure");
    require(fake.modules == 0, "module leaked after cuFuncGetAttribute failure");
    caller_restored("function attribute failure");
    context->prepare(kernel); // nothing was cached by the failures
    require(fake.modules == 1, "context unusable after recoverable module failures");

    // Copy failures map precisely and are recoverable.
    auto a = context->allocate(64), b = context->allocate(64), out = context->allocate(64);
    std::vector<float> host(16, 1.0f);
    fake.fail_next["cuMemcpyHtoD"] = cu::CUDA_ERROR_INVALID_VALUE;
    expect_error([&] { context->write(a, 0, host.data(), 64); }, be::ErrorCode::invalid_value, "cuMemcpyHtoD",
                 "host-to-device failure");
    caller_restored("host-to-device failure");
    fake.fail_next["cuMemcpyDtoH"] = cu::CUDA_ERROR_INVALID_VALUE;
    expect_error([&] { context->read(a, 0, host.data(), 64); }, be::ErrorCode::invalid_value, "cuMemcpyDtoH",
                 "device-to-host failure");
    caller_restored("device-to-host failure");
    context->write(a, 0, host.data(), 64);

    // Event creation and start-record failures: nothing enqueued, nothing leaked.
    fake.fail_after["cuEventCreate"] = 1; // the end event
    fake.fail_next["cuEventCreate"] = cu::CUDA_ERROR_OUT_OF_MEMORY;
    expect_error([&] { context->submit(kernel, {2, 1, 1}, {8, 1, 1}, args(a, b, out)); },
                 be::ErrorCode::out_of_memory, "cuEventCreate", "end event creation failure");
    require(fake.events == 0 && fake.launches.empty(), "event leaked or launch enqueued after cuEventCreate failure");
    caller_restored("event creation failure");
    fake.fail_next["cuEventRecord"] = cu::CUDA_ERROR_INVALID_VALUE;
    expect_error([&] { context->submit(kernel, {2, 1, 1}, {8, 1, 1}, args(a, b, out)); },
                 be::ErrorCode::invalid_value, "cuEventRecord", "start record failure");
    require(fake.events == 0 && fake.launches.empty(), "event leaked or launch enqueued after start record failure");
    caller_restored("start record failure");
    context->synchronize(); // still healthy

    // End-record failure after the launch is enqueued: the command cannot be
    // observed individually, so the context fails terminally, and the events
    // are destroyed with the primary context current.
    fake.fail_after["cuEventRecord"] = 1; // start succeeds, end fails
    fake.fail_next["cuEventRecord"] = cu::CUDA_ERROR_INVALID_VALUE;
    expect_error([&] { context->submit(kernel, {2, 1, 1}, {8, 1, 1}, args(a, b, out)); },
                 be::ErrorCode::invalid_value, "cuEventRecord after launch", "end record failure");
    require(fake.launches.size() == 1, "launch before end-record failure not recorded");
    caller_restored("end record failure");
    expect_error([&] { context->synchronize(); }, be::ErrorCode::execution, "cuEventRecord after launch",
                 "synchronize after end record failure");
    require(fake.events == 0, "events leaked after end-record failure");
    caller_restored("synchronize after end record failure"); // includes: no call without the primary current
    expect_error([&] { context->allocate(4); }, be::ErrorCode::execution, "cuEventRecord after launch",
                 "context stays failed after end-record failure");
  }
  balanced("module, copy and event failures");

  // CUDA_ERROR_ASSERT (710) returned directly by a copy call is sticky.
  fresh();
  {
    auto context = cu::create_context(fake_api(), 0);
    auto a = context->allocate(64);
    std::vector<float> host(16, 0.0f);
    fake.fail_next["cuMemcpyDtoH"] = cu::CUDA_ERROR_ASSERT;
    expect_error([&] { context->read(a, 0, host.data(), 64); }, be::ErrorCode::execution, "(710)",
                 "device assert on copy");
    expect_error([&] { context->allocate(4); }, be::ErrorCode::execution, "(710)",
                 "context stays failed after device assert");
    expect_error([&] { context->write(a, 0, host.data(), 64); }, be::ErrorCode::execution, "(710)",
                 "write after device assert");
    caller_restored("device assert");
  }
  balanced("device assert");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) throw std::runtime_error("usage: cuda_backend_tests UNSUITABLE_SHARED_LIBRARY CWD_PROBE_LIBRARY");
    // Force deterministic absence for the production loader in this process.
    set_env("PARALYN_CUDA_DRIVER_LIBRARY", "/nonexistent/paralyn-test/libcuda-forced-absent");
    set_env("PARALYN_NVRTC_LIBRARY", "/nonexistent/paralyn-test/libnvrtc-forced-absent");
    loader_tests(argv[1]);
    cwd_tests(std::filesystem::absolute(argv[2]).string());
    selector_tests();
    mapping_tests();
    architecture_tests();
    engine_tests();
    failure_path_tests();
    std::cout << "CUDA backend host logic (loader, selectors, error mapping; engine via labelled "
                 "test double, no GPU execution): "
              << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "CUDA backend host test failed: " << e.what() << "\n";
    return 1;
  }
}
