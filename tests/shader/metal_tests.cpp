// GLSL- and HLSL-derived modules on the physical Metal GPU through the public
// C ABI. The modules were produced by `paralyn compile` from examples/glsl and
// examples/hlsl (pinned glslang/DXC worker -> SPIR-V -> importer -> MSL).
// Every output is compared with an independent CPU reference computed here;
// no kernel runs on the CPU in place of the GPU.
// Usage: shader_metal_tests LANGUAGES GLSL_VECTOR_ADD.prx BLUR_ROWS.prx BLUR_COLUMNS.prx
//                           HLSL_VECTOR_ADD.prx TRANSPOSE.prx REDUCE_SUM.prx NEW_EVIDENCE_DIR
// LANGUAGES is glsl, hlsl or glsl,hlsl (the frontends in the build); the
// modules of a language that is not built are given as "-".
#include "paralyn/executable.hpp"
#include "paralyn/native.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned negative_checks = 0, gpu_events = 0;
std::uint64_t compared_values = 0;
double blur_max_abs_error_vs_double = 0; // informational bound, see the blur section
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
pr_argument i32(std::int32_t n) {
  pr_argument a{};
  a.type = PR_I32;
  a.i32 = n;
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
bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
std::uint32_t lcg(std::uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  return state >> 8;
}
// Multiples of 1/8 with |x| <= 16: sums of up to 2^16 of them are exact in FP32.
float fraction(std::size_t i, unsigned seed) { return float(int((i * 29 + seed) % 255) - 127) * 0.125f; }
float value(std::size_t i, unsigned seed) { return float(int((i * 37 + seed) % 17) - 8); }

void check_parameters(pr_kernel k, const char *kernel, const std::vector<const char *> &names,
                      const std::vector<pr_access> &access, const std::vector<pr_type> &types,
                      std::uint32_t buffers) {
  std::uint32_t count = 0;
  success(pr_kernel_parameter_count(k, &count));
  require(count == names.size(), std::string(kernel) + " reflected parameter count");
  for (std::uint32_t i = 0; i < count; ++i) {
    pr_parameter_info info{};
    success(pr_kernel_parameter(k, i, &info));
    require(info.name == std::string(names[i]) && info.access == access[i] &&
                bool(info.is_buffer) == (i < buffers) && info.type == types[i],
            std::string(kernel) + " C ABI parameter reflection at " + names[i]);
  }
}

pr_kernel load(Handles &h, pr_context context, const char *path, const char *entry) {
  const auto bytes = file(path);
  const auto module = paralyn::deserialize_executable(bytes.data(), bytes.size());
  require(module.format == paralyn::ExecutableFormat::SpirvMsl, std::string(path) + " is not SPIR-V-derived");
  pr_module handle = 0;
  success(pr_module_load(context, bytes.data(), bytes.size(), &handle));
  h.keep(handle);
  pr_kernel k = 0;
  success(pr_module_kernel(handle, entry, &k));
  return h.keep(k);
}

void vector_add(Handles &h, pr_context context, pr_queue queue, pr_kernel add, const char *label) {
  for (std::size_t n : {std::size_t(1), std::size_t(63), std::size_t(64), std::size_t(65),
                        std::size_t(1000), std::size_t(262147), std::size_t(1000003)}) {
    std::vector<float> a(n), b(n), c(n + 2, -4242.0f);
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = fraction(i, 3);
      b[i] = fraction(i, 101);
    }
    auto ba = buffer(h, context, a), bb = buffer(h, context, b), bc = buffer(h, context, c);
    auto va = view(h, ba, 0, n, PR_READ), vb = view(h, bb, 0, n, PR_READ), vc = view(h, bc, 1, n, PR_WRITE);
    const auto groups = static_cast<std::uint32_t>((n + 63) / 64);
    completed(launch(h, queue, add, {groups, 1, 1}, {64, 1, 1},
                     {view_arg(va), view_arg(vb), view_arg(vc), u32(static_cast<std::uint32_t>(n))}));
    const auto out = readback(bc, n + 2);
    require(out.front() == -4242.0f && out.back() == -4242.0f, std::string(label) + " wrote outside its view");
    for (std::size_t i = 0; i < n; ++i)
      require(out[i + 1] == a[i] + b[i], std::string(label) + " mismatch at " + std::to_string(i) + " of " +
                                             std::to_string(n));
    compared_values += n + 2;
  }
}

// CPU references for one separable blur pass, clamp-to-edge. `ordered` uses
// FP32 with taps accumulated k = -r..r, separate multiply and add (this file
// is compiled with -ffp-contract=off), i.e. the exact operation order the
// shaders specify; `precise` accumulates in double for an error estimate.
template <typename T>
std::vector<T> blur_pass(const std::vector<T> &src, const std::vector<float> &w, std::uint32_t width,
                         std::uint32_t height, int radius, bool rows) {
  std::vector<T> dst(src.size());
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x) {
      T sum = 0;
      for (int k = -radius; k <= radius; ++k) {
        const int sx = rows ? std::clamp(int(x) + k, 0, int(width) - 1) : int(x);
        const int sy = rows ? int(y) : std::clamp(int(y) + k, 0, int(height) - 1);
        const T product = T(w[std::size_t(k + radius)]) * src[std::size_t(sy) * width + std::size_t(sx)];
        sum = sum + product;
      }
      dst[std::size_t(y) * width + x] = sum;
    }
  return dst;
}

void blur_pipeline(Handles &h, pr_context context, pr_queue queue, pr_kernel rows, pr_kernel columns) {
  struct Shape {
    std::uint32_t width, height;
    int radius, iterations;
    bool binomial;
  };
  for (const Shape &s : {Shape{1, 1, 3, 1, false}, Shape{2, 3, 3, 2, false}, Shape{17, 5, 2, 1, true},
                         Shape{64, 64, 2, 3, true}, Shape{333, 257, 3, 2, false},
                         Shape{1920, 1080, 2, 1, true}, Shape{1023, 769, 4, 3, false}}) {
    const std::size_t pixels = std::size_t(s.width) * s.height;
    std::vector<float> weights(std::size_t(2 * s.radius + 1));
    if (s.binomial) { // exact: 1 4 6 4 1 / 16
      const float b[] = {1, 4, 6, 4, 1};
      for (int k = 0; k < 5; ++k)
        weights[std::size_t(k)] = b[k] / 16.0f;
    } else { // a normalized Gaussian, rounded to FP32 once
      double total = 0;
      std::vector<double> g(weights.size());
      for (int k = -s.radius; k <= s.radius; ++k)
        total += g[std::size_t(k + s.radius)] = std::exp(-0.5 * k * k / (0.6 * s.radius * 0.6 * s.radius));
      for (std::size_t k = 0; k < g.size(); ++k)
        weights[k] = float(g[k] / total);
    }
    std::uint32_t state = s.width * 7919u + s.height;
    std::vector<float> image(pixels);
    for (auto &p : image)
      p = float(lcg(state) % 256) / 256.0f; // exactly representable pixels in [0, 1)
    // Ping-pong: image -> tmp (rows) -> out (columns) -> tmp (rows) -> image ...
    // Every pass is queued without host synchronization; ordering comes from the queue.
    std::vector<float> sentinel(pixels + 2, -1.0f);
    auto b_image = buffer(h, context, image), b_tmp = buffer(h, context, sentinel),
         b_out = buffer(h, context, sentinel), b_w = buffer(h, context, weights);
    auto vw = view(h, b_w, 0, weights.size(), PR_READ);
    const pr_dim3 grid{(s.width + 15) / 16, (s.height + 15) / 16, 1}, block{16, 16, 1};
    std::vector<pr_event> events;
    pr_buffer source = b_image, target = b_out;
    std::vector<float> ordered = image;
    std::vector<double> precise(image.begin(), image.end());
    for (int it = 0; it < s.iterations; ++it) {
      auto src_view = view(h, source, source == b_image ? 0 : 1, pixels, PR_READ);
      auto tmp_write = view(h, b_tmp, 1, pixels, PR_WRITE);
      events.push_back(launch(h, queue, rows, grid, block,
                              {view_arg(src_view), view_arg(vw), view_arg(tmp_write), u32(s.width),
                               u32(s.height), i32(s.radius)}));
      auto tmp_read = view(h, b_tmp, 1, pixels, PR_READ);
      auto dst_write = view(h, target, target == b_image ? 0 : 1, pixels, PR_WRITE);
      events.push_back(launch(h, queue, columns, grid, block,
                              {view_arg(tmp_read), view_arg(vw), view_arg(dst_write), u32(s.width),
                               u32(s.height), i32(s.radius)}));
      ordered = blur_pass(blur_pass(ordered, weights, s.width, s.height, s.radius, true), weights, s.width,
                          s.height, s.radius, false);
      precise = blur_pass(blur_pass(precise, weights, s.width, s.height, s.radius, true), weights, s.width,
                          s.height, s.radius, false);
      std::swap(source, target);
    }
    for (auto e : events)
      completed(e);
    const bool in_image = source == b_image; // last written buffer is now `source`
    const auto raw = readback(source, in_image ? pixels : pixels + 2);
    const std::size_t base = in_image ? 0 : 1;
    if (!in_image)
      require(raw.front() == -1.0f && raw.back() == -1.0f, "blur wrote outside its output view");
    double max_error = 0;
    for (std::size_t i = 0; i < pixels; ++i) {
      require(same_bits(raw[base + i], ordered[i]),
              "blur pipeline mismatch at pixel " + std::to_string(i) + " (" + std::to_string(s.width) + "x" +
                  std::to_string(s.height) + ", radius " + std::to_string(s.radius) + ", " +
                  std::to_string(s.iterations) + " iterations): GPU " + std::to_string(raw[base + i]) +
                  " != CPU " + std::to_string(ordered[i]));
      max_error = std::max(max_error, std::fabs(double(raw[base + i]) - precise[i]));
    }
    if (s.binomial && s.iterations == 1) // <= 24 significant bits: no rounding anywhere
      require(max_error == 0, "binomial blur must be exact against the double reference");
    require(max_error <= 1e-5, "blur deviates from the double reference by " + std::to_string(max_error));
    blur_max_abs_error_vs_double = std::max(blur_max_abs_error_vs_double, max_error);
    compared_values += pixels + (in_image ? 0 : 2);
  }
}

void transpose(Handles &h, pr_context context, pr_queue queue, pr_kernel k) {
  struct Shape {
    std::uint32_t width, height;
  };
  for (const Shape &s : {Shape{1, 1}, Shape{16, 16}, Shape{17, 33}, Shape{1000, 3}, Shape{3, 1000},
                         Shape{513, 1025}, Shape{2048, 1024}}) {
    const std::size_t n = std::size_t(s.width) * s.height;
    std::vector<float> src(n), dst(n + 2, -9.0f);
    std::uint32_t state = s.width + 31u * s.height;
    for (auto &x : src)
      x = float(int(lcg(state) % 2001) - 1000) * 0.25f;
    auto bs = buffer(h, context, src), bd = buffer(h, context, dst);
    auto vs = view(h, bs, 0, n, PR_READ), vd = view(h, bd, 1, n, PR_WRITE);
    completed(launch(h, queue, k, {(s.width + 15) / 16, (s.height + 15) / 16, 1}, {16, 16, 1},
                     {view_arg(vs), view_arg(vd), u32(s.width), u32(s.height)}));
    const auto out = readback(bd, n + 2);
    require(out.front() == -9.0f && out.back() == -9.0f, "transpose wrote outside its view");
    for (std::uint32_t y = 0; y < s.height; ++y)
      for (std::uint32_t x = 0; x < s.width; ++x)
        require(same_bits(out[1 + std::size_t(x) * s.height + y], src[std::size_t(y) * s.width + x]),
                "transpose mismatch at (" + std::to_string(x) + ", " + std::to_string(y) + ") of " +
                    std::to_string(s.width) + "x" + std::to_string(s.height));
    compared_values += n + 2;
  }
}

void reduce(Handles &h, pr_context context, pr_queue queue, pr_kernel k) {
  struct Case {
    std::size_t n;
    float scale;
    bool fractional;
  };
  for (const Case &c : {Case{1, 1.0f, false}, Case{255, 0.5f, true}, Case{256, 1.0f, false},
                        Case{257, -2.0f, true}, Case{100000, 0.5f, false}, Case{1000003, 1.0f, true}}) {
    const auto groups = static_cast<std::uint32_t>((c.n + 255) / 256);
    std::vector<float> input(c.n), partial(groups + 1, -777.0f);
    for (std::size_t i = 0; i < c.n; ++i)
      input[i] = c.fractional ? fraction(i, 11) : value(i, groups);
    auto bi = buffer(h, context, input), bp = buffer(h, context, partial);
    auto vi = view(h, bi, 0, c.n, PR_READ), vp = view(h, bp, 0, groups, PR_WRITE);
    completed(launch(h, queue, k, {groups, 1, 1}, {256, 1, 1},
                     {view_arg(vi), view_arg(vp), u32(static_cast<std::uint32_t>(c.n)), f32(c.scale)}));
    const auto out = readback(bp, groups + 1);
    for (std::uint32_t g = 0; g < groups; ++g) {
      double want = 0;
      for (std::size_t i = std::size_t(g) * 256; i < std::min(c.n, std::size_t(g + 1) * 256); ++i)
        want += input[i];
      want *= c.scale;
      // Exact: every partial sum of these inputs is representable in FP32.
      require(std::fabs(want) < 16777216.0 && double(out[g]) == want,
              "reduce_sum mismatch at group " + std::to_string(g) + " (n=" + std::to_string(c.n) +
                  "): " + std::to_string(out[g]) + " != " + std::to_string(want));
    }
    require(out.back() == -777.0f, "reduce_sum wrote past its output view");
    compared_values += groups + 1;
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 9, "Usage: shader_metal_tests LANGUAGES GLSL_VECTOR_ADD.prx BLUR_ROWS.prx BLUR_COLUMNS.prx "
                       "HLSL_VECTOR_ADD.prx TRANSPOSE.prx REDUCE_SUM.prx NEW_EVIDENCE_DIR");
    const std::string languages = argv[1];
    const bool glsl = languages == "glsl" || languages == "glsl,hlsl";
    const bool hlsl = languages == "hlsl" || languages == "glsl,hlsl";
    require(glsl || hlsl, "LANGUAGES must be glsl, hlsl or glsl,hlsl");
    for (int i = 2; i <= 7; ++i)
      require((std::string(argv[i]) == "-") == (i <= 4 ? !glsl : !hlsl),
              "a module is given exactly for each built language ('-' otherwise)");
    const std::filesystem::path evidence = argv[8];
    require(!std::filesystem::exists(evidence), "Evidence directory must be new");
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

    const std::vector<pr_access> add_access{PR_READ, PR_READ, PR_WRITE, PR_READ};
    const std::vector<pr_type> add_types{PR_F32, PR_F32, PR_F32, PR_U32};
    pr_kernel glsl_add = 0, rows = 0, columns = 0, hlsl_add = 0, tiled = 0, sum = 0;
    if (glsl) {
      glsl_add = load(h, context, argv[2], "vector_add");
      rows = load(h, context, argv[3], "blur_rows");
      columns = load(h, context, argv[4], "blur_columns");
      check_parameters(glsl_add, "GLSL vector_add", {"a", "b", "c", "n"}, add_access, add_types, 3);
      for (auto k : {rows, columns})
        check_parameters(k, "blur", {"src", "weights", "dst", "width", "height", "radius"},
                         {PR_READ, PR_READ, PR_WRITE, PR_READ, PR_READ, PR_READ},
                         {PR_F32, PR_F32, PR_F32, PR_U32, PR_U32, PR_I32}, 3);
    }
    if (hlsl) {
      hlsl_add = load(h, context, argv[5], "vector_add");
      tiled = load(h, context, argv[6], "transpose_tiled");
      sum = load(h, context, argv[7], "reduce_sum");
      check_parameters(hlsl_add, "HLSL vector_add", {"a", "b", "c", "n"}, add_access, add_types, 3);
      check_parameters(tiled, "transpose_tiled", {"src", "dst", "width", "height"},
                       {PR_READ, PR_WRITE, PR_READ, PR_READ}, {PR_F32, PR_F32, PR_U32, PR_U32}, 2);
      check_parameters(sum, "reduce_sum", {"input", "partial", "n", "scale"},
                       {PR_READ, PR_WRITE, PR_READ, PR_READ}, {PR_F32, PR_F32, PR_U32, PR_F32}, 2);
    }

    unsigned glsl_events = 0, hlsl_events = 0;
    if (glsl) {
      const auto before = gpu_events;
      vector_add(h, context, queue, glsl_add, "GLSL vector_add");
      blur_pipeline(h, context, queue, rows, columns);
      glsl_events = gpu_events - before;
    }
    if (hlsl) {
      const auto before = gpu_events;
      vector_add(h, context, queue, hlsl_add, "HLSL vector_add");
      transpose(h, context, queue, tiled);
      reduce(h, context, queue, sum);
      hlsl_events = gpu_events - before;
    }

    // Launch-contract violations: refused before dispatch, no event returned.
    {
      std::vector<float> ones(256, 1.0f);
      auto b1 = buffer(h, context, ones), b2 = buffer(h, context, ones);
      auto r1 = view(h, b1, 0, 256, PR_READ), w2 = view(h, b2, 0, 256, PR_WRITE),
           rw1 = view(h, b1, 0, 256, PR_READ_WRITE), r2 = view(h, b2, 0, 256, PR_READ);
      pr_event e = 0;
      if (glsl) {
        std::vector<pr_argument> blur{view_arg(r1), view_arg(r1), view_arg(w2), u32(16), u32(16), i32(1)};
        failure(pr_launch(queue, rows, {2, 2, 1}, {8, 8, 1}, blur.data(), 6, &e), PR_INVALID_ARGUMENT,
                "launch", "workgroup");
        blur[5] = u32(1); // radius is i32
        failure(pr_launch(queue, rows, {1, 1, 1}, {16, 16, 1}, blur.data(), 6, &e), PR_INVALID_ARGUMENT,
                "launch", "scalar type");
        blur[5] = i32(1);
        blur[2] = view_arg(r2); // dst needs write access
        failure(pr_launch(queue, rows, {1, 1, 1}, {16, 16, 1}, blur.data(), 6, &e), PR_INVALID_ARGUMENT,
                "launch", "access");
        blur[0] = view_arg(rw1);
        blur[2] = view_arg(rw1); // in-place blur: a writable alias of src
        failure(pr_launch(queue, columns, {1, 1, 1}, {16, 16, 1}, blur.data(), 6, &e), PR_UNSUPPORTED,
                "launch", "repeated-allocation");
      }
      if (hlsl) {
        std::vector<pr_argument> t{view_arg(r1), view_arg(w2), u32(16)};
        failure(pr_launch(queue, tiled, {1, 1, 1}, {16, 16, 1}, t.data(), 3, &e), PR_INVALID_ARGUMENT,
                "launch", "argument count");
        std::vector<pr_argument> r{view_arg(r1), view_arg(w2), u32(256), f32(1.0f)};
        failure(pr_launch(queue, sum, {1, 1, 1}, {64, 1, 1}, r.data(), 4, &e), PR_INVALID_ARGUMENT,
                "launch", "workgroup");
      }
      require(e == 0, "refused launch returned an event");
      require(readback(b1, 256) == ones, "a refused launch modified an input allocation");
      compared_values += 256;
    }

    success(pr_context_synchronize(context));
    success(pr_context_write_evidence(context, evidence.string().c_str()));
    std::cout << "Languages: " << languages << "; GLSL on Metal: " << glsl_events
              << " GPU events; HLSL on Metal: " << hlsl_events << " GPU events; " << compared_values
              << " values independently compared; " << negative_checks << " rejection checks";
    if (glsl)
      std::cout << "; blur max |GPU - double reference| = " << blur_max_abs_error_vs_double;
    std::cout << "\n";
    std::cout << "Verification: PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "shader_metal_tests failed: " << e.what() << "\n";
    return 1;
  }
}
