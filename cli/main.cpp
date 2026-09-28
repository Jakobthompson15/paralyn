#include "paralyn/artifact.hpp"
#include "paralyn/executable.hpp"
#include "paralyn/native.hpp"
#if PARALYN_HAS_COMPILER
#include "paralyn/frontend.hpp"
#endif
#include "process.hpp"
#include <chrono>
#include <cli11/CLI11.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;
using paralyn::cli::execute;
namespace {
struct Diagnostic : std::runtime_error {
  std::string id, stage;
  Diagnostic(std::string identifier, std::string where, std::string text)
      : std::runtime_error(std::move(text)), id(std::move(identifier)), stage(std::move(where)) {}
};
struct Options {
  std::string command, target, device = "auto", output, artifacts, manifest, report,
                               requires,
                               color = "auto";
  bool json = false, verbose = false;
  std::vector<std::string> arguments;
};
fs::path prefix;
std::string read(const fs::path &path, std::size_t limit = 16 * 1024 * 1024) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw Diagnostic("P-FILE", "input", "Cannot read " + path.string());
  auto size = fs::file_size(path);
  if (size > limit)
    throw Diagnostic("P-INPUT-LIMIT", "input", "File exceeds supported size: " + path.string());
  std::string text(static_cast<std::size_t>(size), '\0');
  file.read(text.data(), static_cast<std::streamsize>(size));
  if (!file)
    throw Diagnostic("P-FILE", "input", "Cannot read complete file: " + path.string());
  return text;
}
void write(const fs::path &path, const std::string &text) {
  std::ofstream file(path, std::ios::binary);
  if (!file || !file.write(text.data(), static_cast<std::streamsize>(text.size())))
    throw Diagnostic("P-FILE", "report", "Cannot write " + path.string());
}
void fresh_write(const fs::path &path, const std::string &text) {
  // Reserve destination atomically without truncating a previous artifact.
#ifdef _WIN32
  auto file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw Diagnostic("P-OUTPUT-EXISTS", "output",
                     "Cannot create " + path.string() + "; existing files are never overwritten");
  DWORD written = 0;
  bool okay = WriteFile(file, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) &&
              written == text.size();
  CloseHandle(file);
#else
  auto *file = std::fopen(path.c_str(), "wbx");
  if (!file)
    throw Diagnostic("P-OUTPUT-EXISTS", "output",
                     "Cannot create " + path.string() + "; existing files are never overwritten");
  bool okay = std::fwrite(text.data(), 1, text.size(), file) == text.size();
  if (std::fclose(file))
    okay = false;
#endif
  if (!okay) {
    fs::remove(path);
    throw Diagnostic("P-FILE", "output", "Cannot write complete output " + path.string());
  }
}
[[maybe_unused]] void set_environment(const char *key, const std::string &value) {
#ifdef _WIN32
  if (_putenv_s(key, value.c_str()))
    throw std::runtime_error("cannot set process environment");
#else
  if (setenv(key, value.c_str(), 1))
    throw std::runtime_error("cannot set process environment");
#endif
}
fs::path executable_path(const char *argv0) {
#ifdef __APPLE__
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> path(size);
  if (!_NSGetExecutablePath(path.data(), &size))
    return fs::weakly_canonical(path.data());
#elif defined(_WIN32)
  std::vector<wchar_t> path(32768);
  auto n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (n && n < path.size())
    return fs::path(std::wstring(path.data(), n));
#else
  std::error_code error;
  auto path = fs::read_symlink("/proc/self/exe", error);
  if (!error)
    return path;
#endif
  return fs::absolute(argv0);
}
fs::path includes() {
  auto path = prefix / "include";
  return fs::exists(path / "paralyn/native.h") ? path : fs::path(PARALYN_INCLUDE_DIR);
}
fs::path library(const char *name, const char *fallback) {
  auto p = prefix / "lib" / name;
  return fs::exists(p) ? p : fs::path(fallback);
}
std::string run_id() {
  return std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count()) +
         "-" + std::to_string(paralyn::cli::process_id());
}
Json report(const std::string &command) {
  return Json{{"schema", "paralyn.report"},
              {"schema_version", 1},
              {"paralyn_version", "0.0.1"},
              {"command", command},
              {"status", "pending"}};
}
void output(const Options &options, const Json &value, const std::string &human) {
  if (!options.report.empty())
    fresh_write(fs::absolute(options.report), value.dump(2) + "\n");
  if (options.json)
    std::cout << value.dump() << "\n";
  else
    std::cout << human;
}
Json device_info(const pr_device_info &d, const pr_device_capabilities_v1 &c,
                 const std::string &selector) {
  return Json{{"selector", selector},
              {"stable_id", c.stable_id},
              {"name", d.name},
              {"backend", d.backend},
              {"os", d.os},
              {"unified_memory", d.unified_memory != 0},
              {"max_buffer_bytes", c.max_buffer_bytes},
              {"max_threadgroup_memory_bytes", c.max_threadgroup_memory_bytes},
              {"max_block_dimensions", {c.max_block_x, c.max_block_y, c.max_block_z}},
              {"artifact_formats", c.artifact_formats},
              {"scalar_types", c.scalar_types}};
}
Json devices(const Options &options) {
  uint32_t n = 0;
  paralyn::native::check(pr_device_count(&n));
  auto result = Json::array();
  if (!options.requires.empty() && options.requires != "fp32" && options.requires != "fp64")
    throw Diagnostic("P-CAPABILITY-NAME", "device",
                     "Unknown capability; this version accepts fp32 or fp64");
  for (uint32_t i = 0; i < n; ++i) {
    pr_device_info d{};
    pr_device_capabilities_v1 c{};
    c.struct_size = sizeof(c);
    c.version = 1;
    paralyn::native::check(pr_device_get(i, &d));
    paralyn::native::check(pr_device_capabilities_get(i, &c));
    if (options.requires == "fp64")
      continue;
    if (options.requires == "fp32" && !(c.scalar_types & PR_SCALAR_F32))
      continue;
    result.push_back(device_info(d, c, "metal:" + std::to_string(i)));
  }
  return result;
}
Json context_info(const paralyn::native::Context &context, const std::string &selector) {
  pr_device_capabilities_v1 c{};
  c.struct_size = sizeof(c);
  c.version = 1;
  paralyn::native::check(pr_context_capabilities(context.get(), &c));
  return device_info(context.device(), c, selector);
}
Json timing(const paralyn::native::Event &event) {
  pr_event_timing_v1 t{};
  t.struct_size = sizeof(t);
  t.version = 1;
  paralyn::native::check(pr_event_timing(event.get(), &t));
  return Json{{"completed", t.completed != 0},
              {"duration_valid", t.duration_valid != 0},
              {"duration_seconds", t.duration_valid ? Json(t.duration_seconds) : Json(nullptr)},
              {"timestamps_valid", t.timestamps_valid != 0},
              {"clock_domain", static_cast<int>(t.clock_domain)},
              {"start_seconds", t.timestamps_valid ? Json(t.start_seconds) : Json(nullptr)},
              {"end_seconds", t.timestamps_valid ? Json(t.end_seconds) : Json(nullptr)}};
}
paralyn::Kernel probe_kernel() {
  using namespace paralyn;
  Expr i{ExprKind::Builtin, ScalarType::U32, "threadIdx.x", {}};
  Expr a{ExprKind::Load, ScalarType::F32, "", {{ExprKind::Ref, ScalarType::F32, "a", {}}, i}};
  Expr b{ExprKind::Load, ScalarType::F32, "", {{ExprKind::Ref, ScalarType::F32, "b", {}}, i}};
  Expr sum{ExprKind::Binary, ScalarType::F32, "+", {a, b}};
  Statement store;
  store.kind = StmtKind::Store;
  store.name = "out";
  store.type = ScalarType::F32;
  store.expression = sum;
  store.target = {
      ExprKind::Load, ScalarType::F32, "", {{ExprKind::Ref, ScalarType::F32, "out", {}}, i}};
  return Kernel{"paralyn_probe",
                {{"a", ScalarType::F32, true, true},
                 {"b", ScalarType::F32, true, true},
                 {"out", ScalarType::F32, true, false}},
                {store}};
}
int doctor(const Options &options) {
  auto result = report(options.command);
  result["compiler"] = {{"available", bool(PARALYN_HAS_COMPILER)},
                        {"llvm_version", PARALYN_LLVM_VERSION}};
  result["devices"] = devices(options);
  auto context = paralyn::native::Context(options.device);
  result["device"] = context_info(context, options.device);
  auto bytes = paralyn::serialize_module({probe_kernel()});
  pr_module handle = 0;
  paralyn::native::check(pr_module_load(context.get(), bytes.data(), bytes.size(), &handle));
  paralyn::native::Module module(handle);
  auto kernel = module.kernel("paralyn_probe");
  auto queue = context.queue();
  constexpr std::size_t n = 257;
  std::vector<float> a(n), b(n), out(n, -999);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = float(int(i % 31) - 15) * 0.5f;
    b[i] = float(int(i % 17) - 8) * 0.25f;
  }
  auto da = context.buffer(n * 4), db = context.buffer(n * 4), dc = context.buffer(n * 4);
  da.write(a.data(), n * 4);
  db.write(b.data(), n * 4);
  dc.write(out.data(), n * 4);
  auto va = da.view(0, n * 4, PR_READ), vb = db.view(0, n * 4, PR_READ),
       vc = dc.view(0, n * 4, PR_WRITE);
  auto event =
      queue.launch(kernel, {1, 1, 1}, {n, 1, 1},
                   {paralyn::native::Argument::buffer(va), paralyn::native::Argument::buffer(vb),
                    paralyn::native::Argument::buffer(vc)});
  result["timing"] = timing(event);
  dc.read(out.data(), n * 4);
  for (std::size_t i = 0; i < n; ++i)
    if (out[i] != a[i] + b[i])
      throw Diagnostic("P-VERIFY-MISMATCH", "verification",
                       "Bundled vector-add reference mismatch at " + std::to_string(i));
  fs::path artifact = options.artifacts.empty() ? fs::current_path() / ".paralyn/runs" / run_id()
                                                : fs::absolute(options.artifacts);
  context.evidence(artifact.string());
  write(artifact / "probe.prk", std::string(bytes.begin(), bytes.end()));
  write(artifact / "reference.json",
        Json{{"fixture", "builtin:vector-add-v1"},
             {"a", a},
             {"b", b},
             {"actual", out},
             {"comparison", "FP32 exact; host independently computes a[i]+b[i]"}}
                .dump(2) +
            "\n");
  result["status"] = "passed";
  result["verification"] = {
      {"status", "passed"}, {"reference", "builtin:vector-add-v1"}, {"elements", n}};
  result["cpu_fallback"] = false;
  result["artifacts"] = artifact.string();
  write(artifact / "report.json", result.dump(2) + "\n");
  output(options, result,
         "Paralyn\nDevice       " + std::string(context.device().name) +
             " / Metal\nGPU probe    completed\nVerification PASS (257 independently compared FP32 "
             "values)\nReport       " +
             (artifact / "report.json").string() + "\n");
  return 0;
}
void keys(const Json &object, std::initializer_list<const char *> allowed,
          const std::string &where) {
  if (!object.is_object())
    throw Diagnostic("P-MANIFEST", "input", where + " must be an object");
  for (auto it = object.begin(); it != object.end(); ++it) {
    bool found = false;
    for (auto key : allowed)
      found |= it.key() == key;
    if (!found)
      throw Diagnostic("P-MANIFEST", "input", "Unknown field " + where + "." + it.key());
  }
}
Json manifest_json(const std::string &text) {
  std::vector<std::set<std::string>> objects;
  return Json::parse(text, [&](int, Json::parse_event_t event, Json &value) {
    if (event == Json::parse_event_t::object_start)
      objects.emplace_back();
    else if (event == Json::parse_event_t::object_end)
      objects.pop_back();
    else if (event == Json::parse_event_t::key &&
             !objects.back().insert(value.get<std::string>()).second)
      throw Diagnostic("P-MANIFEST", "input",
                       "Duplicate manifest key: " + value.get<std::string>());
    return true;
  });
}
uint64_t integer(const Json &value, uint64_t maximum, const std::string &name) {
  if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
    throw Diagnostic("P-MANIFEST", "input", name + " requires a nonnegative integer");
  auto number = value.get<uint64_t>();
  if (number > maximum)
    throw Diagnostic("P-MANIFEST", "input", name + " is out of range");
  return number;
}
paralyn::ExecutableModule msl_module(const fs::path &source, const Options &options) {
  if (options.manifest.empty())
    throw Diagnostic("P-MANIFEST-REQUIRED", "input",
                     "Metal source requires --manifest FILE with typed resources and entrypoints");
  auto descriptor = manifest_json(read(options.manifest));
  keys(descriptor, {"target", "numerical_policy", "entries"}, "manifest");
  paralyn::ExecutableModule module;
  module.target = descriptor.at("target").get<std::string>();
  module.numerical_policy = static_cast<uint32_t>(
      integer(descriptor.at("numerical_policy"), UINT32_MAX, "numerical_policy"));
  module.producer = "paralyn";
  module.producer_version = "0.0.1+" PARALYN_BUILD_COMMIT ".dirty=" PARALYN_BUILD_DIRTY;
  module.source_name = source.filename().string();
  module.source = read(source);
  module.source_sha256 = paralyn::source_sha256(module.source);
  if (!descriptor.at("entries").is_array())
    throw Diagnostic("P-MANIFEST", "input", "entries must be an array");
  for (const auto &entry : descriptor.at("entries")) {
    keys(entry, {"name", "parameters", "required_block"}, "entry");
    paralyn::ExecutableEntry e;
    e.name = entry.at("name").get<std::string>();
    if (entry.contains("required_block")) {
      const auto &block = entry.at("required_block");
      if (!block.is_array() || block.size() != 3)
        throw Diagnostic("P-MANIFEST", "input",
                         "required_block must contain three integer dimensions");
      for (unsigned i = 0; i < 3; ++i)
        e.required_block[i] =
            static_cast<uint32_t>(integer(block[i], UINT32_MAX, "required_block"));
    }
    if (!entry.at("parameters").is_array())
      throw Diagnostic("P-MANIFEST", "input", "parameters must be an array");
    for (const auto &argument : entry.at("parameters")) {
      keys(argument, {"name", "type", "buffer", "access", "binding", "alignment", "minimum_bytes"},
           "parameter");
      paralyn::ExecutableParameter p;
      p.name = argument.at("name").get<std::string>();
      auto type = argument.at("type").get<std::string>();
      if (type == "f32")
        p.type = paralyn::ScalarType::F32;
      else if (type == "i32")
        p.type = paralyn::ScalarType::I32;
      else if (type == "u32")
        p.type = paralyn::ScalarType::U32;
      else
        throw Diagnostic("P-MANIFEST", "input", "Unsupported parameter type: " + type);
      p.buffer = argument.at("buffer").get<bool>();
      auto access = argument.at("access").get<std::string>();
      if (access == "read")
        p.access = paralyn::ResourceAccess::Read;
      else if (access == "write")
        p.access = paralyn::ResourceAccess::Write;
      else if (access == "read_write")
        p.access = paralyn::ResourceAccess::ReadWrite;
      else
        throw Diagnostic("P-MANIFEST", "input", "Unsupported access: " + access);
      p.binding = static_cast<uint32_t>(integer(argument.at("binding"), UINT32_MAX, "binding"));
      p.alignment = static_cast<uint32_t>(
          integer(argument.value("alignment", Json(4)), UINT32_MAX, "alignment"));
      p.minimum_bytes =
          integer(argument.value("minimum_bytes", Json(4)), UINT64_MAX, "minimum_bytes");
      e.parameters.push_back(p);
    }
    module.entries.push_back(e);
  }
  paralyn::verify_executable(module);
  return module;
}
#if PARALYN_HAS_COMPILER
std::string ir_text(const paralyn::FrontendResult &frontend) {
  std::string s;
  for (const auto &k : frontend.kernels)
    s += paralyn::dump_ir(k) + "\n";
  return s;
}
Json describe(const paralyn::FrontendResult &frontend) {
  Json result;
  result["kernels"] = Json::array();
  result["launches"] = Json::array();
  for (const auto &kernel : frontend.kernels) {
    Json k = {{"name", kernel.name}, {"parameters", Json::array()}};
    for (const auto &p : kernel.parameters)
      k["parameters"].push_back({{"name", p.name},
                                 {"type", paralyn::type_name(p.type)},
                                 {"buffer", p.buffer},
                                 {"read_only", p.read_only}});
    result["kernels"].push_back(k);
  }
  for (const auto &l : frontend.launches)
    result["launches"].push_back({{"kernel", l.kernel},
                                  {"line", l.line},
                                  {"grid_expression", l.grid_expression},
                                  {"block_expression", l.block_expression},
                                  {"runtime_dimensions_known", false}});
  result["ir"] = ir_text(frontend);
  return result;
}
#endif
std::string revision() { return PARALYN_BUILD_COMMIT; }
int source_command(const Options &options) {
  auto source = fs::absolute(options.target);
  auto extension = source.extension().string();
  auto result = report(options.command);
  result["source"] = source.string();
  result["source_sha256"] = paralyn::source_sha256(read(source));
  const bool msl = extension == ".metal", container = extension == ".prx",
             native = extension == ".py" || extension == ".cpp" || extension == ".cc" ||
                      extension == ".c";
  std::vector<unsigned char> module_bytes;
  std::string host_source, ir;
  if (msl || container) {
    auto module = msl ? msl_module(source, options) : [&] {
      auto b = read(source);
      return paralyn::deserialize_executable(b.data(), b.size());
    }();
    result["frontend"] = "metal_source";
    result["target"] = module.target;
    result["entries"] = Json::array();
    for (const auto &e : module.entries) {
      Json params = Json::array();
      for (const auto &p : e.parameters)
        params.push_back({{"name", p.name},
                          {"type", paralyn::type_name(p.type)},
                          {"buffer", p.buffer},
                          {"binding", p.binding},
                          {"access", static_cast<int>(p.access)}});
      result["entries"].push_back(
          {{"name", e.name}, {"parameters", params}, {"required_block", e.required_block}});
    }
    module_bytes = paralyn::serialize_executable(module);
    if (options.command == "run")
      throw Diagnostic("P-KERNEL-CASE-REQUIRED", "input",
                       "This is a kernel module, not a complete program. Load it using the native "
                       "C/C++/Python module API; CLI kernel cases are not implemented yet.");
  } else if (!native) {
    if (extension != ".cu")
      throw Diagnostic("P-FRONTEND-UNIMPLEMENTED", "input",
                       "No implemented input profile for " + extension +
                           ". Inspect available profiles with paralyn support");
#if PARALYN_HAS_COMPILER
    set_environment("PARALYN_INCLUDE_DIR", includes().string());
    auto frontend = paralyn::compile_source(source.string());
    result["frontend"] = "cuda_cpp";
    result["inspection"] = describe(frontend);
    host_source = frontend.rewritten_host;
    ir = ir_text(frontend);
    module_bytes = paralyn::serialize_module(frontend.kernels);
#else
    throw Diagnostic("P-COMPILER-MISSING", "compilation",
                     "This is a runtime-only installation. Install the Paralyn compiler component "
                     "to compile CUDA source.");
#endif
  } else if (options.command != "run")
    throw Diagnostic("P-PROFILE-UNIMPLEMENTED", "inspection",
                     "Static inspection of native applications is not implemented; use run for "
                     "applications using the native API");
  if (options.command == "compile") {
    if (options.output.empty())
      throw Diagnostic("P-OUTPUT-REQUIRED", "input", "compile requires --output FILE");
    fresh_write(fs::absolute(options.output),
                std::string(module_bytes.begin(), module_bytes.end()));
    result["status"] = "compiled";
    result["output"] = fs::absolute(options.output).string();
    auto count =
        (msl || container) ? result["entries"].size() : result["inspection"]["kernels"].size();
    result["payload_validation"] = (msl || container)
                                       ? "resource contract and source profile; Metal "
                                         "compilation/reflection occurs on check or load"
                                       : "verified scalar IR";
    output(
        options, result,
        (msl || container ? "Packaged " : "Compiled ") + std::to_string(count) +
            (msl || container ? " declared Metal entrypoint(s) to " : " verified kernel(s) to ") +
            options.output + "\n");
    return 0;
  }
  if (options.command == "inspect" || options.command == "check" || options.command == "explain") {
    if (options.command != "inspect") {
      auto context = paralyn::native::Context(options.device);
      result["device"] = context_info(context, options.device);
      pr_module h = 0;
      paralyn::native::check(
          pr_module_load(context.get(), module_bytes.data(), module_bytes.size(), &h));
      paralyn::native::Module compiled(h);
      result["backend_compilation"] = "passed";
    }
    result["status"] = "checked";
    result["host_code_executed"] = false;
    result["gpu_work_submitted"] = false;
    std::ostringstream text;
    text << "Detected kernels:\n";
    if (msl || container)
      for (const auto &e : result["entries"])
        text << "  " << e["name"].get<std::string>()
             << " (Metal source, reflected resources checked on module load)\n";
    else {
      for (const auto &k : result["inspection"]["kernels"]) {
        text << "  " << k["name"].get<std::string>() << "(";
        bool first = true;
        for (const auto &p : k["parameters"]) {
          if (!first)
            text << ", ";
          first = false;
          if (p["read_only"].get<bool>())
            text << "const ";
          text << p["type"].get<std::string>() << (p["buffer"].get<bool>() ? "* " : " ")
               << p["name"].get<std::string>();
        }
        text << ")\n";
      }
      text << "\nLaunches (expressions; runtime values are not evaluated):\n";
      for (const auto &l : result["inspection"]["launches"])
        text << "  " << l["kernel"].get<std::string>() << " at line " << l["line"]
             << ": grid=" << l["grid_expression"].get<std::string>()
             << ", block=" << l["block_expression"].get<std::string>() << "\n";
      text << "\nParalyn IR:\n" << ir;
    }
    if (options.command != "inspect")
      text << "\nBackend compilation checked; application main was not executed.\n";
    output(options, result, text.str());
    return 0;
  }
  auto id = run_id();
  fs::path artifact = options.artifacts.empty() ? fs::current_path() / ".paralyn/runs" / id
                                                : fs::absolute(options.artifacts);
  if (fs::exists(artifact) && !fs::is_empty(artifact))
    throw Diagnostic("P-OUTPUT-EXISTS", "report",
                     "Artifact directory is not empty; refusing to overwrite execution evidence");
  fs::create_directories(artifact);
  fs::copy_file(source, artifact / ("source" + extension));
  result["run_id"] = id;
  result["artifacts"] = artifact.string();
  result["device_selector"] = options.device;
  result["source_revision"] = revision();
  result["build_dirty"] = std::string(PARALYN_BUILD_DIRTY) == "true";
  result["llvm_version"] = PARALYN_LLVM_VERSION;
  result["verification"] = {{"status", "not_requested"}};
  result["application"] = {{"arguments", options.arguments},
                           {"stdout", (artifact / "application.stdout").string()},
                           {"stderr", (artifact / "application.stderr").string()}};
  result["runtime_log"] = (artifact / "runtime.log").string();
  result["events"] = Json::array();
  auto stage = [&](const char *name) {
    result["events"].push_back({{"stage", name}, {"status", "completed"}});
    if (!options.json)
      std::cerr << "Paralyn: " << name << "\n";
  };
  auto finish = [&](int status, const std::string &origin) {
    result["exit_code"] = status;
    result["status"] = status ? "failed" : "completed";
    result["failure_origin"] = status ? Json(origin) : Json(nullptr);
    write(artifact / "report.json", result.dump(2) + "\n");
    if (!options.report.empty())
      fresh_write(fs::absolute(options.report), result.dump(2) + "\n");
    if (options.json)
      std::cout << result.dump() << "\n";
    else
      std::cerr << "Paralyn: " << (status ? "failed" : "execution completed") << "; report "
                << (artifact / "report.json").string() << "\n";
    return status;
  };
  std::vector<std::string> command;
  try {
    if (extension == ".py") {
      const char *python = std::getenv("PARALYN_PYTHON");
      command = {python ? python : "python3", source.string()};
    } else {
      auto input = source;
      if (!native) {
        write(artifact / "host.cpp", host_source);
        write(artifact / "paralyn-ir.txt", ir);
        input = artifact / "host.cpp";
      }
      auto program = artifact / "program";
      std::vector<std::string> compile{extension == ".c" ? PARALYN_HOST_CC : PARALYN_HOST_CXX,
                                       extension == ".c" ? "-std=c11" : "-std=c++17",
                                       "-O0",
                                       "-g",
                                       "-fno-fast-math",
                                       "-ffp-contract=off",
                                       "-I",
                                       includes().string(),
                                       "-iquote",
                                       source.parent_path().string(),
                                       input.string()};
      if (native) {
        auto lib = library("libparalyn_native.dylib", PARALYN_NATIVE_LIBRARY);
        compile.push_back(lib.string());
        compile.push_back("-Wl,-rpath," + lib.parent_path().string());
      } else {
        compile.push_back(library("libparalyn_runtime.a", PARALYN_RUNTIME_ARCHIVE).string());
        compile.push_back(library("libparalyn_ir.a", PARALYN_IR_ARCHIVE).string());
      }
#if PARALYN_HAS_METAL
      compile.insert(compile.end(), {"-mmacosx-version-min=" PARALYN_DEPLOYMENT_TARGET,
                                     "-framework", "Metal", "-framework", "Foundation"});
#endif
      compile.insert(compile.end(), {"-o", program.string()});
      auto compiled = execute(compile);
      write(artifact / "compiler.stdout", compiled.out);
      write(artifact / "compiler.stderr", compiled.err);
      if (compiled.status) {
        if (!options.json)
          std::cerr << compiled.err;
        result["diagnostic"] = {{"id", "P-HOST-COMPILE"},
                                {"stage", "host_compilation"},
                                {"message", "Host compiler failed; see compiler.stderr"}};
        write(artifact / "verification.txt", "Host compilation failed\n" + compiled.err);
        return finish(compiled.status, "paralyn");
      }
      command = {program.string()};
    }
    // Resolve the requested backend before running arbitrary application code.
    {
      auto context = paralyn::native::Context(options.device);
      result["selected_device"] = context_info(context, options.device);
    }
    stage("executable prepared");
    command.insert(command.end(), options.arguments.begin(), options.arguments.end());
    std::vector<std::pair<std::string, std::string>> environment{
        {"PARALYN_DEVICE", options.device},
        {"PARALYN_ARTIFACT_DIR", artifact.string()},
        {"PARALYN_RUNTIME_LOG", (artifact / "runtime.log").string()},
        {"PARALYN_EVENT_LOG", (artifact / "runtime-events.ndjson").string()},
        {"PARALYN_COMMIT", result["source_revision"].get<std::string>()},
        {"PARALYN_LLVM_VERSION", PARALYN_LLVM_VERSION},
        {"PARALYN_SOURCE_DIRTY", PARALYN_BUILD_DIRTY}};
    auto installed_python = prefix / "share/paralyn/python";
    if (!fs::exists(installed_python / "paralyn"))
      installed_python = fs::path(PARALYN_SOURCE_DIR) / "bindings/python";
    if (fs::exists(installed_python / "paralyn")) {
      const char *old = std::getenv("PYTHONPATH");
      environment.emplace_back("PYTHONPATH",
                               installed_python.string() + (old ? std::string(":") + old : ""));
    }
    result["application"]["command"] = command;
    result["limitations"] = {"Host I/O inputs are not automatically traced",
                             "run does not verify results",
                             "Native applications may explicitly select a different context; "
                             "runtime events identify observed work"};
    auto executed = execute(command, !options.json, environment);
    write(artifact / "application.stdout", executed.out);
    write(artifact / "application.stderr", executed.err);
    const auto runtime_transcript =
        fs::exists(artifact / "runtime.log") ? read(artifact / "runtime.log") : "";
    write(artifact / "verification.txt",
          executed.out + executed.err + "\nRuntime progress:\n" + runtime_transcript +
              "\nHost exit status: " + std::to_string(executed.status) + "\n");
    result["application"]["exit_code"] = executed.status;
    result["application"]["interrupted"] = executed.interrupted;
    if (executed.interrupted)
      result["interruption"] =
          "Interruption forwarded to process group; GPU cancellation is not claimed";
    if (fs::exists(artifact / "execution.json")) {
      result["runtime_evidence"] = Json::parse(read(artifact / "execution.json"));
    } else
      result["runtime_evidence"] = nullptr;
    if (fs::exists(artifact / "runtime-events.ndjson")) {
      result["runtime_events"] = Json::array();
      std::istringstream events(read(artifact / "runtime-events.ndjson"));
      std::string line;
      while (std::getline(events, line))
        if (!line.empty())
          result["runtime_events"].push_back(Json::parse(line));
    }
    // Application output is data: printing PASS never constitutes a verified run.
    if (options.verbose && fs::exists(artifact / "runtime.log"))
      std::cerr << read(artifact / "runtime.log");
    if (!executed.status)
      stage("application exited successfully");
    bool runtime_failure = false, runtime_errors = false;
    if (result.contains("runtime_events"))
      for (const auto &event : result["runtime_events"]) {
        runtime_errors |= event.value("status", "") == "failed";
        runtime_failure |= event.value("category", "") == "process_failure" &&
                           event.value("status", "") == "failed";
      }
    result["runtime_errors_observed"] = runtime_errors;
    return finish(executed.status, runtime_failure ? "runtime" : "application");
  } catch (const std::exception &error) {
    result["diagnostic"] = {{"id", "P-RUN"}, {"stage", "execution"}, {"message", error.what()}};
    return finish(1, "paralyn");
  }
}
Json support() {
  Json rows = Json::array();
  for (auto name : {"CUDA C++", "Native C/C++", "Native Python", "HIP C++", "Triton", "OpenCL C",
                    "SYCL C++", "OpenMP target", "Slang", "HLSL compute", "GLSL compute", "WGSL",
                    "SPIR-V", "MLIR", "Metal source", "PTX", "SASS"}) {
    std::string n = name;
    bool implemented = n == "Native C/C++" || n == "Native Python" || n == "Metal source" ||
                       (n == "CUDA C++" && PARALYN_HAS_COMPILER);
    rows.push_back(
        {{"name", n},
         {"required", true},
         {"implementation", implemented ? "partial" : "not_implemented"},
         {"backend", implemented && PARALYN_HAS_METAL ? Json("metal") : Json(nullptr)},
         {"available_in_this_build", implemented && bool(PARALYN_HAS_METAL)},
         {"scope", n == "CUDA C++"       ? "documented scalar source profile"
                   : n == "Metal source" ? "declared 32-bit buffer/scalar resources; Metal-specific"
                   : n == "Native C/C++" || n == "Native Python"
                       ? "ABI1 buffers/modules/launch and contiguous FP32 arrays"
                       : "no executable profile"},
         {"full_profile_qualified", false}});
  }
  return Json{{"inputs", rows},
              {"clients",
               {{"Numba CUDA", "not_implemented"},
                {"CuPy", "not_implemented"},
                {"CUDA Python", "not_implemented"}}},
              {"backends",
               {{"metal", PARALYN_HAS_METAL ? "implemented_subset" : "not_built"},
                {"cuda", "not_implemented"},
                {"rocm", "not_implemented"}}},
              {"complete_portfolio", false}};
}
} // namespace
int main(int argc, char **argv) {
  Options options;
  prefix = executable_path(argv[0]).parent_path().parent_path();
  CLI::App app{"Paralyn: bring your computation, choose your GPU, understand the result"};
  app.set_version_flag("--version", "Paralyn v0.0.1 (LLVM " PARALYN_LLVM_VERSION ")");
  app.require_subcommand(1, 1);
  for (auto name : {"devices", "doctor", "support", "inspect", "check", "explain", "compile", "run",
                    "verify", "report"}) {
    auto sub = app.add_subcommand(name);
    std::string command = name;
    sub->callback([&, command] { options.command = command; });
    sub->add_flag("--json", options.json, "Emit one versioned JSON result on stdout");
    sub->add_option("--report-json", options.report,
                    "Write a JSON report without changing application streams");
    sub->add_option("--color", options.color,
                    "Color policy (current output uses stable plain text)")
        ->check(CLI::IsMember({"auto", "always", "never"}));
    if (command == "devices")
      sub->add_option("--requires", options.requires, "Filter by fp32 or fp64");
    if (command == "doctor" || command == "run" || command == "check" || command == "explain" ||
        command == "verify")
      sub->add_option("--device", options.device, "auto, metal:INDEX, or legacy numeric index");
    if (command == "doctor" || command == "run" || command == "verify")
      sub->add_option("--artifacts", options.artifacts, "New/empty evidence directory");
    if (command == "run") {
      sub->add_flag("--verbose", options.verbose, "Show captured runtime progress");
    }
    if (command == "compile")
      sub->add_option("--output", options.output, "New module artifact file")->required();
    if (command == "compile" || command == "inspect" || command == "check" || command == "explain")
      sub->add_option("--manifest", options.manifest, "Typed Metal resource manifest (JSON)");
    if (command == "run" || command == "compile" || command == "inspect" || command == "check" ||
        command == "explain" || command == "verify" || command == "report")
      sub->add_option("TARGET", options.target)->required();
  }
  // The separator belongs to the complete application, including flags and unicode.
  std::vector<char *> cli_args;
  for (int i = 0; i < argc; ++i) {
    if (i > 1 && std::string(argv[i]) == "--" && argc > 1 && std::string(argv[1]) == "run") {
      for (++i; i < argc; ++i)
        options.arguments.emplace_back(argv[i]);
      break;
    }
    cli_args.push_back(argv[i]);
  }
  try {
    app.parse(static_cast<int>(cli_args.size()), cli_args.data());
    if (options.device.rfind("cuda:", 0) == 0 || options.device.rfind("rocm:", 0) == 0)
      throw Diagnostic("P-BACKEND-UNIMPLEMENTED", "device",
                       "The requested vendor backend is not implemented. Installing its SDK alone "
                       "cannot enable this path. Run paralyn support for current profiles.");
    if (!options.report.empty() && fs::exists(options.report))
      throw Diagnostic("P-OUTPUT-EXISTS", "report",
                       "Report already exists; existing files are never overwritten");
    if (options.command == "devices") {
      auto result = report("devices");
      result["devices"] = devices(options);
      result["status"] = "completed";
      std::ostringstream text;
      for (const auto &d : result["devices"])
        text << d["selector"].get<std::string>() << "  " << d["name"].get<std::string>() << " / "
             << d["backend"].get<std::string>() << "  (" << d["stable_id"].get<std::string>()
             << ")\n";
      if (result["devices"].empty())
        text << "No matching GPU devices available. Run paralyn doctor for diagnostics.\n";
      output(options, result, text.str());
      return 0;
    }
    if (options.command == "support") {
      auto result = report("support");
      result["status"] = "completed";
      result["support"] = support();
      std::ostringstream text;
      for (const auto &row : result["support"]["inputs"])
        text << row["name"].get<std::string>() << ": " << row["implementation"].get<std::string>()
             << " — " << row["scope"].get<std::string>() << "\n";
      text << "\nNVIDIA/CUDA and AMD/HIP backends, Numba CUDA, CuPy and CUDA Python: not "
              "implemented.\nAll tracks remain required. No complete-portfolio claim.\n";
      output(options, result, text.str());
      return 0;
    }
    if (options.command == "doctor")
      return doctor(options);
    if (options.command == "verify") {
      if (options.target != "builtin:vector-add")
        throw Diagnostic("P-REFERENCE-REQUIRED", "verification",
                         "This version verifies builtin:vector-add only. Arbitrary targets need a "
                         "declared independent reference protocol, which is not implemented yet.");
      return doctor(options);
    }
    if (options.command == "report") {
      auto result = Json::parse(read(options.target));
      if (result.value("schema", "") != "paralyn.report" || result.value("schema_version", 0) != 1)
        throw Diagnostic("P-REPORT-VERSION", "report", "Unsupported report schema/version");
      output(options, result, result.dump(2) + "\n");
      return 0;
    }
    return source_command(options);
  } catch (const CLI::ParseError &error) {
    if (!error.get_exit_code())
      return app.exit(error);
    bool wants_json = options.json;
    for (auto arg : cli_args)
      wants_json = wants_json || std::string(arg) == "--json";
    if (!wants_json)
      return app.exit(error);
    auto result = report(options.command);
    result["status"] = "failed";
    result["failure_origin"] = "paralyn";
    result["exit_code"] = error.get_exit_code();
    result["diagnostic"] = {
        {"id", "P-ARGUMENT"}, {"stage", "arguments"}, {"message", error.what()}};
    std::cout << result.dump() << "\n";
    return error.get_exit_code();
  } catch (const std::exception &error) {
    auto d = dynamic_cast<const Diagnostic *>(&error);
    auto result = report(options.command);
    result["status"] = "failed";
    result["failure_origin"] = "paralyn";
    result["exit_code"] = 1;
    result["diagnostic"] = {{"id", d ? d->id : "P-RUNTIME"},
                            {"stage", d ? d->stage : "runtime"},
                            {"message", error.what()},
                            {"device_selector", options.device}};
    if (options.json)
      std::cout << result.dump() << "\n";
    else
      std::cerr << "error[" << (d ? d->id : "P-RUNTIME") << "]: " << error.what() << "\n";
    if (!options.report.empty()) {
      try {
        fresh_write(options.report, result.dump(2) + "\n");
      } catch (const std::exception &writing) {
        std::cerr << "error[P-REPORT]: " << writing.what() << "\n";
      }
    }
    return 1;
  }
}
