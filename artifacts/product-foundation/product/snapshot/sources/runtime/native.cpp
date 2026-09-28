#include "paralyn/native.h"
#include "paralyn/artifact.hpp"
#include "paralyn/detail/backend.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {
namespace be = paralyn::backend;
struct Failure : std::runtime_error {
  pr_status code;
  Failure(pr_status c, const std::string &m) : std::runtime_error(m), code(c) {}
};
void require(bool value, pr_status code, const char *message) {
  if (!value)
    throw Failure(code, message);
}
thread_local pr_error last_error{};
std::mutex api_mutex;
enum class Kind { context, buffer, view, module, kernel, queue, event };
struct Object {
  Kind kind;
  explicit Object(Kind k) : kind(k) {}
  virtual ~Object() = default;
};
struct Context : Object {
  std::shared_ptr<be::Context> backend;
  bool completion_failure_observed = false;
  explicit Context(std::shared_ptr<be::Context> b) : Object(Kind::context), backend(std::move(b)) {}
};
struct Buffer : Object {
  std::shared_ptr<Context> owner;
  std::shared_ptr<be::Buffer> allocation;
  Buffer(std::shared_ptr<Context> c, std::shared_ptr<be::Buffer> b)
      : Object(Kind::buffer), owner(std::move(c)), allocation(std::move(b)) {}
};
struct View : Object {
  std::shared_ptr<Buffer> buffer;
  std::size_t offset, size;
  std::uint32_t alignment;
  pr_access access;
  View(std::shared_ptr<Buffer> b, std::size_t o, std::size_t s, std::uint32_t a, pr_access mode)
      : Object(Kind::view), buffer(std::move(b)), offset(o), size(s), alignment(a), access(mode) {}
};
struct Module : Object {
  std::shared_ptr<Context> owner;
  std::vector<paralyn::Kernel> kernels;
  std::vector<paralyn::ExecutableEntry> entries;
  std::shared_ptr<be::CompiledExecutable> executable;
  explicit Module(std::shared_ptr<Context> c) : Object(Kind::module), owner(std::move(c)) {}
};
struct Kernel : Object {
  std::shared_ptr<Module> module;
  std::size_t index;
  Kernel(std::shared_ptr<Module> m, std::size_t i)
      : Object(Kind::kernel), module(std::move(m)), index(i) {}
  const paralyn::Kernel &ir() const { return module->kernels[index]; }
  const paralyn::ExecutableEntry &entry() const { return module->entries[index]; }
};
struct Queue : Object {
  std::shared_ptr<Context> owner;
  explicit Queue(std::shared_ptr<Context> c) : Object(Kind::queue), owner(std::move(c)) {}
};
struct Event : Object {
  std::shared_ptr<Context> owner;
  std::shared_ptr<be::Event> completion;
  Event(std::shared_ptr<Context> c, std::shared_ptr<be::Event> e)
      : Object(Kind::event), owner(std::move(c)), completion(std::move(e)) {}
};
// Handles are numeric capabilities, never exposed as host/device addresses. Erasing a
// registry entry destroys resources when dependent references finish; IDs never recycle.
std::unordered_map<pr_handle, std::shared_ptr<Object>> objects;
pr_handle next_handle = 1;
void shutdown_native() {
  try {
    std::lock_guard<std::mutex> lock(api_mutex);
    std::set<std::shared_ptr<Context>> contexts;
    for (const auto &entry : objects) {
      const auto &o = entry.second;
      switch (o->kind) {
      case Kind::context:
        contexts.insert(std::static_pointer_cast<Context>(o));
        break;
      case Kind::buffer:
        contexts.insert(std::static_pointer_cast<Buffer>(o)->owner);
        break;
      case Kind::view:
        contexts.insert(std::static_pointer_cast<View>(o)->buffer->owner);
        break;
      case Kind::module:
        contexts.insert(std::static_pointer_cast<Module>(o)->owner);
        break;
      case Kind::kernel:
        contexts.insert(std::static_pointer_cast<Kernel>(o)->module->owner);
        break;
      case Kind::queue:
        contexts.insert(std::static_pointer_cast<Queue>(o)->owner);
        break;
      case Kind::event:
        contexts.insert(std::static_pointer_cast<Event>(o)->owner);
        break;
      }
    }
    std::string failures;
    for (const auto &context : contexts) {
      try {
        context->backend->synchronize();
      } catch (const be::Error &e) {
        if (e.code != be::ErrorCode::execution || !context->completion_failure_observed)
          failures += std::string(e.what()) + "\n";
      } catch (const std::exception &e) {
        failures += std::string(e.what()) + "\n";
      }
    }
    if (!failures.empty())
      throw std::runtime_error(failures);
  } catch (const std::exception &e) {
    try { be::runtime_event("process_failure", "failed", e.what()); }
    catch (...) { std::fputs("Paralyn native shutdown event log also failed\n", stderr); }
    std::fprintf(stderr, "Paralyn native shutdown failed: %s\n", e.what());
    std::fflush(nullptr);
    std::_Exit(EXIT_FAILURE);
  }
}
void ensure_shutdown() {
  // Registered after static registry construction, so this runs before its destructor.
  static const bool installed = std::atexit(shutdown_native) == 0;
  require(installed, PR_INTERNAL_ERROR, "Cannot install native shutdown handler");
}
template <class T> std::shared_ptr<T> get(pr_handle h, Kind kind) {
  auto it = objects.find(h);
  require(it != objects.end() && it->second->kind == kind, PR_INVALID_HANDLE,
          "Unknown, released, or wrong-kind handle");
  return std::static_pointer_cast<T>(it->second);
}
pr_handle add(std::shared_ptr<Object> object) {
  require(next_handle != 0, PR_OUT_OF_MEMORY, "Handle identity space exhausted");
  const auto h = next_handle++;
  objects.emplace(h, std::move(object));
  return h;
}
template <std::size_t N> void text(char (&out)[N], std::string_view value) {
  const auto n = std::min(N - 1, value.size());
  std::memcpy(out, value.data(), n);
  out[n] = 0;
}
pr_status translate(be::ErrorCode code) {
  switch (code) {
  case be::ErrorCode::invalid_value:
    return PR_INVALID_ARGUMENT;
  case be::ErrorCode::invalid_device:
    return PR_DEVICE_UNAVAILABLE;
  case be::ErrorCode::invalid_handle:
    return PR_INVALID_HANDLE;
  case be::ErrorCode::out_of_memory:
    return PR_OUT_OF_MEMORY;
  case be::ErrorCode::unsupported:
    return PR_UNSUPPORTED;
  case be::ErrorCode::compilation:
    return PR_COMPILATION_FAILED;
  case be::ErrorCode::execution:
    return PR_EXECUTION_FAILED;
  case be::ErrorCode::internal:
    return PR_INTERNAL_ERROR;
  }
  return PR_INTERNAL_ERROR;
}
template <class Function> pr_status api(const char *operation, Function &&fn) noexcept {
  try {
    std::lock_guard<std::mutex> lock(api_mutex);
    fn();
    last_error = {};
    return PR_SUCCESS;
  } catch (const Failure &e) {
    last_error.code = e.code;
    text(last_error.message, e.what());
  } catch (const be::Error &e) {
    last_error.code = translate(e.code);
    text(last_error.message, e.what());
  } catch (const std::bad_alloc &) {
    last_error.code = PR_OUT_OF_MEMORY;
    text(last_error.message, "Allocation failed");
  } catch (const std::exception &e) {
    last_error.code = PR_INTERNAL_ERROR;
    text(last_error.message, e.what());
  } catch (...) {
    last_error.code = PR_INTERNAL_ERROR;
    text(last_error.message, "Unknown native runtime failure");
  }
  text(last_error.operation, operation);
  try {
    be::runtime_event("api_failure", "failed", std::string(operation) + ": " + last_error.message);
  } catch (...) {
    // Preserve the primary API failure and make a broken diagnostic channel
    // explicit without allocating or throwing across the C ABI.
    constexpr const char suffix[] = " [runtime event log also failed]";
    const auto length = std::strlen(last_error.message);
    if (length + sizeof(suffix) <= sizeof(last_error.message))
      std::memcpy(last_error.message + length, suffix, sizeof(suffix));
  }
  return last_error.code;
}
std::size_t bytes(uint64_t n) {
  require(n <= std::numeric_limits<std::size_t>::max(), PR_OUT_OF_BOUNDS,
          "Byte count does not fit platform size_t");
  return static_cast<std::size_t>(n);
}
void range(std::size_t capacity, uint64_t offset, uint64_t count) {
  require(offset <= capacity && count <= capacity - offset, PR_OUT_OF_BOUNDS,
          "Byte range exceeds allocation");
}
pr_type type(paralyn::ScalarType t) {
  switch (t) {
  case paralyn::ScalarType::I32:
    return PR_I32;
  case paralyn::ScalarType::U32:
    return PR_U32;
  case paralyn::ScalarType::F32:
    return PR_F32;
  default:
    throw Failure(PR_UNSUPPORTED, "Predicate parameters are not supported");
  }
}
void collect_reads(const paralyn::Expr &e, std::set<std::string> &names) {
  if (e.kind == paralyn::ExprKind::Load)
    names.insert(e.operands[0].text);
  for (const auto &child : e.operands)
    collect_reads(child, names);
}
void accesses(const std::vector<paralyn::Statement> &body, std::set<std::string> &read,
              std::set<std::string> &write) {
  for (const auto &s : body) {
    collect_reads(s.expression, read);
    if (s.kind == paralyn::StmtKind::Store) {
      write.insert(s.target.operands[0].text);
      collect_reads(s.target.operands[1], read);
    }
    accesses(s.body, read, write);
  }
}
pr_access access(const paralyn::Kernel &kernel, std::size_t index) {
  std::set<std::string> read, write;
  accesses(kernel.body, read, write);
  const auto &p = kernel.parameters[index];
  unsigned bits = (read.count(p.name) ? PR_READ : 0) | (write.count(p.name) ? PR_WRITE : 0);
  return static_cast<pr_access>(bits ? bits : PR_READ);
}
void device(const be::DeviceInfo &d, pr_device_info *out) {
  require(out, PR_INVALID_ARGUMENT, "Missing device-info output");
  *out = {};
  text(out->name, d.name);
  text(out->backend, d.backend);
  text(out->os, d.os);
  out->registry_id = d.registry_id;
  out->max_buffer_bytes = d.max_buffer_bytes;
  out->unified_memory = d.unified_memory;
}
template <class T> void query_record(T *out) {
  require(out, PR_INVALID_ARGUMENT, "Missing versioned query output");
  require(out->version == PR_QUERY_VERSION_1, PR_UNSUPPORTED, "Unsupported query record version");
  require(out->struct_size == sizeof(T), PR_INVALID_ARGUMENT, "Query record size mismatch");
}
void capabilities(const be::DeviceInfo &d, pr_device_capabilities_v1 *out) {
  query_record(out);
  *out = {};
  out->struct_size = sizeof(*out);
  out->version = PR_QUERY_VERSION_1;
  text(out->stable_id, d.stable_id);
  text(out->backend, d.backend);
  out->artifact_formats = PR_ARTIFACT_VERIFIED_IR | PR_ARTIFACT_MSL_SOURCE;
  out->scalar_types = PR_SCALAR_I32 | PR_SCALAR_U32 | PR_SCALAR_F32;
  out->max_buffer_bytes = d.max_buffer_bytes;
  out->max_threadgroup_memory_bytes = d.max_threadgroup_memory_bytes;
  out->max_block_x = d.max_block.x;
  out->max_block_y = d.max_block.y;
  out->max_block_z = d.max_block.z;
  out->max_buffer_bindings = 31;
  out->unified_memory = d.unified_memory;
}
std::shared_ptr<Module> load(const std::shared_ptr<Context> &ctx, const void *data,
                             std::size_t count) {
  require(data || !count, PR_INVALID_ARGUMENT, "Null module data");
  auto result = std::make_shared<Module>(ctx);
  try {
    if (paralyn::is_executable_artifact(data, count)) {
      auto module = paralyn::deserialize_executable(data, count);
      result->executable = ctx->backend->prepare(module);
      result->entries = std::move(module.entries);
      return result;
    }
    result->kernels = paralyn::deserialize_module(data, count);
  } catch (const std::bad_alloc &) {
    throw;
  } catch (const be::Error &) {
    throw;
  } catch (const std::exception &e) {
    throw Failure(PR_COMPILATION_FAILED, e.what());
  }
  for (const auto &k : result->kernels) {
    ctx->backend->prepare(k);
    paralyn::ExecutableEntry entry;
    entry.name = k.name;
    for (std::size_t i = 0; i < k.parameters.size(); ++i) {
      const auto &p = k.parameters[i];
      entry.parameters.push_back({p.name, p.type, p.buffer,
          static_cast<paralyn::ResourceAccess>(access(k, i)), static_cast<std::uint32_t>(i), 4, 4});
    }
    result->entries.push_back(std::move(entry));
  }
  return result;
}
// A terminal command failure remains sticky for operations. Once a caller has
// observed it, cleanup may consume handles without reporting that same failure
// again. The first observation (including release itself) always returns an error.
template <class F> auto completion(const std::shared_ptr<Context> &ctx, F &&fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const be::Error &e) {
    if (e.code == be::ErrorCode::execution)
      ctx->completion_failure_observed = true;
    throw;
  }
}
template <class F> void release_completion(const std::shared_ptr<Context> &ctx, F &&fn) {
  const bool observed = ctx->completion_failure_observed;
  try {
    completion(ctx, std::forward<F>(fn));
  } catch (const be::Error &e) {
    if (!observed || e.code != be::ErrorCode::execution)
      throw;
  }
}
} // namespace

extern "C" {
uint32_t pr_abi_version(void) { return PR_ABI_VERSION; }
pr_status pr_last_error(pr_error *out) {
  if (!out)
    return PR_INVALID_ARGUMENT;
  *out = last_error;
  return PR_SUCCESS;
}
pr_status pr_device_count(uint32_t *out) {
  return api("device_count", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing count output");
    *out = static_cast<uint32_t>(be::devices().size());
  });
}
pr_status pr_device_get(uint32_t index, pr_device_info *out) {
  return api("device_get", [&] {
    auto list = be::devices();
    require(index < list.size(), PR_DEVICE_UNAVAILABLE, "Device index unavailable");
    device(list[index], out);
  });
}
pr_status pr_device_capabilities_get(uint32_t index, pr_device_capabilities_v1 *out) {
  return api("device_capabilities_get", [&] {
    query_record(out);
    auto list = be::devices();
    require(index < list.size(), PR_DEVICE_UNAVAILABLE, "Device index unavailable");
    capabilities(list[index], out);
  });
}
pr_status pr_context_create(const char *selector, pr_context *out) {
  return api("context_create", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing context output");
    *out = 0;
    ensure_shutdown();
    *out = add(std::make_shared<Context>(be::create_context(selector ? selector : "auto")));
  });
}
pr_status pr_context_device(pr_context h, pr_device_info *out) {
  return api("context_device",
             [&] { device(get<Context>(h, Kind::context)->backend->device_info(), out); });
}
pr_status pr_context_capabilities(pr_context h, pr_device_capabilities_v1 *out) {
  return api("context_capabilities", [&] {
    capabilities(get<Context>(h, Kind::context)->backend->device_info(), out);
  });
}
pr_status pr_context_synchronize(pr_context h) {
  return api("context_synchronize", [&] {
    auto ctx = get<Context>(h, Kind::context);
    completion(ctx, [&] { ctx->backend->synchronize(); });
  });
}
pr_status pr_context_write_evidence(pr_context h, const char *directory) {
  return api("context_write_evidence", [&] {
    auto ctx = get<Context>(h, Kind::context);
    require(directory && *directory, PR_INVALID_ARGUMENT, "Missing evidence directory");
    const std::filesystem::path path(directory);
    require(!std::filesystem::exists(path) ||
                (std::filesystem::is_directory(path) && std::filesystem::is_empty(path)),
            PR_INVALID_ARGUMENT, "Evidence directory must be new or empty");
    completion(ctx, [&] { ctx->backend->write_evidence(path.string()); });
  });
}
pr_status pr_release(pr_handle h) {
  return api("release", [&] {
    auto it = objects.find(h);
    require(it != objects.end(), PR_INVALID_HANDLE, "Unknown or released handle");
    auto object = std::move(it->second);
    objects.erase(it); // consumed even if completion reports failure
    if (object->kind == Kind::context) {
      auto ctx = std::static_pointer_cast<Context>(object);
      release_completion(ctx, [&] { ctx->backend->synchronize(); });
    }
    if (object->kind == Kind::queue) {
      auto ctx = std::static_pointer_cast<Queue>(object)->owner;
      release_completion(ctx, [&] { ctx->backend->synchronize(); });
    }
    if (object->kind == Kind::event) {
      auto event = std::static_pointer_cast<Event>(object);
      release_completion(event->owner, [&] { event->completion->wait(); });
    }
  });
}
pr_status pr_buffer_create(pr_context context, uint64_t n, pr_buffer *out) {
  return api("buffer_create", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing buffer output");
    *out = 0;
    auto ctx = get<Context>(context, Kind::context);
    *out = add(std::make_shared<Buffer>(ctx, ctx->backend->allocate(bytes(n))));
  });
}
pr_status pr_buffer_size(pr_buffer h, uint64_t *out) {
  return api("buffer_size", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing size output");
    *out = get<Buffer>(h, Kind::buffer)->allocation->size();
  });
}
pr_status pr_view_create(pr_buffer h, uint64_t offset, uint64_t n, uint32_t alignment,
                         pr_access mode, pr_view *out) {
  return api("view_create", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing view output");
    *out = 0;
    auto buffer = get<Buffer>(h, Kind::buffer);
    require(mode == PR_READ || mode == PR_WRITE || mode == PR_READ_WRITE, PR_INVALID_ARGUMENT,
            "Invalid view access mode");
    require(alignment == 1 || alignment == 2 || alignment == 4, PR_UNSUPPORTED,
            "Supported element alignments are 1, 2 and 4 bytes");
    range(buffer->allocation->size(), offset, n);
    require(offset % alignment == 0 && n % alignment == 0, PR_INVALID_ARGUMENT,
            "Misaligned view offset or size");
    *out = add(std::make_shared<View>(buffer, bytes(offset), bytes(n), alignment, mode));
  });
}
pr_status pr_buffer_write(pr_buffer h, uint64_t offset, const void *source, uint64_t n) {
  return api("buffer_write", [&] {
    auto b = get<Buffer>(h, Kind::buffer);
    range(b->allocation->size(), offset, n);
    require(source || !n, PR_INVALID_ARGUMENT, "Null nonempty upload source");
    completion(b->owner,
               [&] { b->owner->backend->write(b->allocation, bytes(offset), source, bytes(n)); });
  });
}
pr_status pr_buffer_read(pr_buffer h, uint64_t offset, void *destination, uint64_t n) {
  return api("buffer_read", [&] {
    auto b = get<Buffer>(h, Kind::buffer);
    range(b->allocation->size(), offset, n);
    require(destination || !n, PR_INVALID_ARGUMENT, "Null nonempty download destination");
    completion(b->owner, [&] {
      b->owner->backend->read(b->allocation, bytes(offset), destination, bytes(n));
    });
  });
}
pr_status pr_module_load(pr_context context, const void *data, uint64_t n, pr_module *out) {
  return api("module_load", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing module output");
    *out = 0;
    *out = add(load(get<Context>(context, Kind::context), data, bytes(n)));
  });
}
pr_status pr_module_load_file(pr_context context, const char *path, pr_module *out) {
  return api("module_load_file", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing module output");
    *out = 0;
    auto ctx = get<Context>(context, Kind::context);
    require(path && *path, PR_INVALID_ARGUMENT, "Missing module path");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), PR_COMPILATION_FAILED, "Cannot open module artifact");
    auto length = input.tellg();
    require(length >= 0 && static_cast<std::uint64_t>(length) <= paralyn::artifact_max_bytes,
            PR_COMPILATION_FAILED, "Module artifact exceeds 16 MiB limit");
    std::vector<unsigned char> data(static_cast<std::size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(data.data()), data.size());
    require(bool(input), PR_COMPILATION_FAILED, "Cannot read module artifact");
    *out = add(load(ctx, data.data(), data.size()));
  });
}
pr_status pr_module_kernel(pr_module h, const char *name, pr_kernel *out) {
  return api("module_kernel", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing kernel output");
    *out = 0;
    require(name, PR_INVALID_ARGUMENT, "Missing kernel name");
    auto module = get<Module>(h, Kind::module);
    for (std::size_t i = 0; i < module->entries.size(); ++i)
      if (module->entries[i].name == name) {
        *out = add(std::make_shared<Kernel>(module, i));
        return;
      }
    throw Failure(PR_INVALID_ARGUMENT, "Module has no entrypoint with that name");
  });
}
pr_status pr_kernel_parameter_count(pr_kernel h, uint32_t *out) {
  return api("kernel_parameter_count", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing count output");
    *out = get<Kernel>(h, Kind::kernel)->entry().parameters.size();
  });
}
pr_status pr_kernel_parameter(pr_kernel h, uint32_t index, pr_parameter_info *out) {
  return api("kernel_parameter", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing parameter output");
    auto k = get<Kernel>(h, Kind::kernel);
    require(index < k->entry().parameters.size(), PR_OUT_OF_BOUNDS, "Parameter index out of range");
    const auto &p = k->entry().parameters[index];
    *out = {};
    text(out->name, p.name);
    out->type = type(p.type);
    out->is_buffer = p.buffer;
    out->access = static_cast<pr_access>(p.access);
  });
}
pr_status pr_queue_get(pr_context h, pr_queue *out) {
  return api("queue_get", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing queue output");
    *out = 0;
    *out = add(std::make_shared<Queue>(get<Context>(h, Kind::context)));
  });
}
pr_status pr_queue_synchronize(pr_queue h) {
  return api("queue_synchronize", [&] {
    auto ctx = get<Queue>(h, Kind::queue)->owner;
    completion(ctx, [&] { ctx->backend->synchronize(); });
  });
}
pr_status pr_launch(pr_queue q, pr_kernel h, pr_dim3 grid, pr_dim3 block, const pr_argument *args,
                    uint32_t count, pr_event *out) {
  return api("launch", [&] {
    require(out, PR_INVALID_ARGUMENT, "Missing event output");
    *out = 0;
    auto queue = get<Queue>(q, Kind::queue);
    auto kernel = get<Kernel>(h, Kind::kernel);
    require(queue->owner == kernel->module->owner, PR_CONTEXT_MISMATCH,
            "Queue and kernel belong to different contexts");
    require(count == kernel->entry().parameters.size(), PR_INVALID_ARGUMENT,
            "Kernel argument count mismatch");
    require(args || !count, PR_INVALID_ARGUMENT, "Missing kernel arguments");
    std::vector<be::BoundArgument> bound;
    bound.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto &p = kernel->entry().parameters[i];
      const auto &a = args[i];
      be::BoundArgument value;
      value.type = p.type;
      value.is_buffer = p.buffer;
      if (p.buffer) {
        require(a.type == PR_BUFFER, PR_INVALID_ARGUMENT, "Expected a buffer-view argument");
        auto view = get<View>(a.view, Kind::view);
        require(view->buffer->owner == queue->owner, PR_CONTEXT_MISMATCH,
                "View belongs to a different context");
        require(view->alignment == 4 && view->offset % 4 == 0 && view->size % 4 == 0,
                PR_INVALID_ARGUMENT, "Typed kernel view must have four-byte alignment and size");
        require(view->size >= p.minimum_bytes && view->offset % p.alignment == 0,
                PR_INVALID_ARGUMENT, "View violates kernel minimum size or alignment");
        auto needed = static_cast<pr_access>(p.access);
        require((view->access & needed) == needed, PR_INVALID_ARGUMENT,
                "View access does not allow the kernel's reads/writes");
        value.allocation = view->buffer->allocation;
        value.offset = view->offset;
        value.size = view->size;
      } else {
        require(a.type == type(p.type), PR_INVALID_ARGUMENT, "Kernel scalar type mismatch");
        if (a.type == PR_I32)
          std::memcpy(value.bytes.data(), &a.i32, 4);
        if (a.type == PR_U32)
          std::memcpy(value.bytes.data(), &a.u32, 4);
        if (a.type == PR_F32)
          std::memcpy(value.bytes.data(), &a.f32, 4);
      }
      bound.push_back(std::move(value));
    }
    // Publish the event identity before committing work: allocation failure must
    // never leave a submitted command without a completion handle for its caller.
    auto event = std::make_shared<Event>(queue->owner, nullptr);
    const auto event_handle = add(event);
    try {
      if (kernel->module->executable)
        event->completion = queue->owner->backend->submit(kernel->module->executable, kernel->index,
            {grid.x, grid.y, grid.z}, {block.x, block.y, block.z}, bound);
      else
        event->completion = queue->owner->backend->submit(kernel->ir(), {grid.x, grid.y, grid.z},
                                                          {block.x, block.y, block.z}, bound);
    } catch (...) {
      objects.erase(event_handle);
      throw;
    }
    *out = event_handle;
  });
}
pr_status pr_event_wait(pr_event h, pr_event_info *out) {
  return api("event_wait", [&] {
    auto event = get<Event>(h, Kind::event);
    auto info = completion(event->owner, [&] { return event->completion->wait(); });
    if (out)
      *out = {static_cast<uint32_t>(info.completed), info.gpu_start_seconds, info.gpu_end_seconds};
  });
}
pr_status pr_event_timing(pr_event h, pr_event_timing_v1 *out) {
  return api("event_timing", [&] {
    query_record(out);
    auto event = get<Event>(h, Kind::event);
    auto info = completion(event->owner, [&] { return event->completion->wait(); });
    *out = {};
    out->struct_size = sizeof(*out);
    out->version = PR_QUERY_VERSION_1;
    out->completed = info.completed;
    out->duration_valid = info.duration_valid;
    out->timestamps_valid = info.timestamps_valid;
    out->clock_domain = static_cast<pr_clock_domain>(info.clock_domain);
    out->duration_seconds = info.duration_valid ? info.duration_seconds : 0;
    out->start_seconds = info.timestamps_valid ? info.gpu_start_seconds : 0;
    out->end_seconds = info.timestamps_valid ? info.gpu_end_seconds : 0;
  });
}
pr_status pr_event_cancel(pr_event h) {
  return api("event_cancel", [&] {
    (void)get<Event>(h, Kind::event);
    throw Failure(PR_UNSUPPORTED,
                  "Cancellation of submitted GPU work is unsupported; wait for completion");
  });
}
} // extern C
