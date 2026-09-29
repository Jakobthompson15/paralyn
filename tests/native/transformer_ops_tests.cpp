// Real-GPU qualification of the transformer-block operators of the
// paralyn.msl.tensor provider (batched strided matmul, row sum/max, softmax,
// LayerNorm, GELU, residual add) through the C ABI and C++ wrappers.
//
// References are independent CPU code (never reported as GPU output):
//   * bit-exact FP32 emulation of the documented order where the contract
//     fixes every rounding (matmul, sum, max, add, bias add);
//   * a float64 reference with a-priori running error bounds for every
//     operator (examples/native/transformer_reference.hpp).
#include "../../examples/native/transformer_reference.hpp"
#include "paralyn/executable.hpp"
#include "paralyn/tensor.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace n = paralyn::native;
namespace t = paralyn::tensors;
namespace r = paralyn_reference;
namespace {
constexpr float inf = std::numeric_limits<float>::infinity();
const float nan = std::numeric_limits<float>::quiet_NaN();

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
void status(pr_status actual, pr_status expected, const std::string &label, const char *operation) {
  pr_error e{};
  pr_last_error(&e);
  require(actual == expected, label + ": expected status " + std::to_string(expected) + ", got " +
                                  std::to_string(actual) + " (" + e.operation + ": " + e.message + ")");
  if (expected != PR_SUCCESS)
    require(!std::strcmp(e.operation, operation),
            label + ": wrong error operation " + e.operation + " (expected " + operation + ")");
}
struct Rng { // xorshift32; 16-bit dyadic values in [-1, 1) times a scale
  std::uint32_t state;
  float next(float scale = 1.0f) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int(state >> 16) - 32768) / 32768.0f * scale;
  }
};
std::vector<float> random(std::size_t count, std::uint32_t seed, float scale = 1.0f) {
  Rng g{seed};
  std::vector<float> v(count);
  for (auto &x : v) x = g.next(scale);
  return v;
}
std::string shape_text(std::initializer_list<std::uint64_t> dims) {
  std::string s = "[";
  bool first = true;
  for (auto d : dims) {
    s += (first ? "" : ",") + std::to_string(d);
    first = false;
  }
  return s + "]";
}

struct Suite {
  n::Context context;
  n::Queue queue;
  n::Module ops;
  unsigned events = 0;
  std::map<std::string, unsigned> per_kernel;
  std::map<std::string, double> worst;
  Suite() : context("auto"), queue(context.queue()), ops(t::load_operators(context)) {}
  // Buffer holding `values` after `pad` canary floats (descriptor byte offset pad*4).
  n::Buffer upload(const std::vector<float> &values, std::size_t pad = 0, std::size_t after = 2) {
    std::vector<float> all(pad, -777.0f);
    all.insert(all.end(), values.begin(), values.end());
    all.insert(all.end(), after, -777.0f);
    auto b = context.buffer(all.size() * 4);
    b.write(all.data(), all.size() * 4);
    return b;
  }
  n::Buffer output(std::size_t count, std::size_t pad = 0) {
    return upload(std::vector<float>(count, 12345.0f), pad, 3);
  }
  std::vector<float> read(const n::Buffer &b) {
    std::vector<float> out(b.size() / 4);
    b.read(out.data(), out.size() * 4);
    return out;
  }
  void canaries(const std::vector<float> &all, std::size_t pad, std::size_t count,
                const std::string &label) {
    for (std::size_t i = 0; i < pad; ++i) require(all[i] == -777.0f, label + ": leading canary overwritten");
    for (std::size_t i = pad + count; i < all.size(); ++i)
      require(all[i] == -777.0f || all[i] == 12345.0f, label + ": trailing canary overwritten");
  }
  void completed(const std::optional<n::Event> &e, const std::string &kernel, const std::string &label) {
    require(bool(e), label + ": missing GPU event");
    const auto timing = e->timing();
    require(timing.completed && timing.duration_valid && timing.timestamps_valid &&
                timing.clock_domain == PR_CLOCK_METAL_SYSTEM_MACH && timing.duration_seconds > 0,
            label + ": event is not a completed physical GPU interval");
    ++events;
    ++per_kernel[kernel];
  }
  void note(const std::string &op, double ratio) { worst[op] = std::max(worst[op], ratio); }
};

// ---------------------------------------------------------------- batched matmul
struct View {
  std::vector<std::uint64_t> shape;
  std::vector<std::int64_t> strides;
  std::uint64_t offset = 0; // elements
  float at(const std::vector<float> &data, std::uint64_t b, std::uint64_t i, std::uint64_t j) const {
    return data[offset + b * strides[0] + i * strides[1] + j * strides[2]];
  }
  std::uint64_t index(std::uint64_t b, std::uint64_t i, std::uint64_t j) const {
    return offset + b * strides[0] + i * strides[1] + j * strides[2];
  }
};
View contiguous3(std::uint64_t d0, std::uint64_t d1, std::uint64_t d2) {
  return {{d0, d1, d2}, {std::int64_t(d1 * d2), std::int64_t(d2), 1}, 0};
}
// One batched matmul over (possibly strided) views into three storage vectors.
void batched_case(Suite &s, const std::string &label, std::uint64_t batch, std::uint64_t m,
                  std::uint64_t nn, std::uint64_t k, const View &av, const std::vector<float> &a,
                  const View &bv, const std::vector<float> &b, const View &cv, std::size_t c_storage) {
  auto ab = s.upload(a), bb = s.upload(b);
  auto cb = s.output(c_storage);
  const auto ad = t::strided(ab, av.shape, av.strides, av.offset * 4);
  const auto bd = t::strided(bb, bv.shape, bv.strides, bv.offset * 4);
  const auto cd = t::strided(cb, cv.shape, cv.strides, cv.offset * 4);
  auto event = t::batched_matmul(s.queue, s.ops, batch, m, nn, k, ad, bd, cd);
  const auto out = s.read(cb);
  if (!batch || !m || !nn) {
    require(!event, label + ": empty output fabricated a GPU event");
    for (std::size_t i = 0; i < c_storage; ++i) require(out[i] == 12345.0f, label + ": empty output wrote");
    return;
  }
  s.completed(event, k ? "paralyn_batched_matmul_f32" : "paralyn_batched_fill_f32", label);
  std::vector<bool> written(c_storage, false);
  for (std::uint64_t z = 0; z < batch; ++z)
    for (std::uint64_t i = 0; i < m; ++i)
      for (std::uint64_t j = 0; j < nn; ++j) {
        float seq = 0.0f; // documented order: FP32 products, increasing k
        std::vector<r::B> x(k), y(k);
        for (std::uint64_t p = 0; p < k; ++p) {
          const float pa = av.at(a, z, i, p), pb = bv.at(b, z, p, j);
          const float product = pa * pb;
          seq = seq + product;
          x[p] = r::exact(pa);
          y[p] = r::exact(pb);
        }
        const auto idx = cv.index(z, i, j);
        written[idx] = true;
        const float gpu = out[idx];
        require(r::same_bits(gpu, seq), label + ": GPU differs from sequential FP32 reference at (" +
                                            std::to_string(z) + "," + std::to_string(i) + "," +
                                            std::to_string(j) + ")");
        const auto ref = r::dot(x, y);
        require(r::within(gpu, ref), label + ": float64 error bound exceeded");
        s.note("batched_matmul", r::ratio(gpu, ref));
      }
  for (std::size_t i = 0; i < c_storage; ++i)
    if (!written[i]) require(out[i] == 12345.0f, label + ": element outside the strided output was written");
}
void batched_matmul(Suite &s) {
  std::uint32_t seed = 100;
  const std::uint64_t shapes[][4] = {{1, 1, 1, 1},  {3, 17, 13, 5},  {4, 33, 31, 47}, {2, 1, 65, 16},
                                     {1, 16, 16, 16}, {2, 129, 70, 257}, {5, 16, 17, 0}, {0, 4, 4, 4},
                                     {3, 0, 5, 2},   {3, 5, 0, 2},    {1, 1, 1, 300}};
  for (const auto &sh : shapes) {
    const auto batch = sh[0], m = sh[1], nn = sh[2], k = sh[3];
    const std::string label = "batched " + shape_text({batch, m, nn, k});
    // Contiguous with element offsets.
    auto av = contiguous3(batch, m, k), bv = contiguous3(batch, k, nn), cv = contiguous3(batch, m, nn);
    av.offset = 3;
    bv.offset = 1;
    cv.offset = 2;
    const auto a = random(3 + batch * m * k, seed++), b = random(1 + batch * k * nn, seed++);
    batched_case(s, label + " contiguous", batch, m, nn, k, av, a, bv, b, cv, 2 + batch * m * nn);
    // Transposed B view: B[z] stored as [n,k] and viewed as [k,n] via strides (no copy).
    View bt{{batch, k, nn}, {std::int64_t(nn * k), 1, std::int64_t(k)}, 0};
    batched_case(s, label + " transposed-B view", batch, m, nn, k, contiguous3(batch, m, k), a, bt, b,
                 contiguous3(batch, m, nn), batch * m * nn);
    if (batch > 1 && k) {
      // Broadcast B over the batch (batch stride 0).
      View bb{{batch, k, nn}, {0, std::int64_t(nn), 1}, 0};
      batched_case(s, label + " broadcast-B", batch, m, nn, k, contiguous3(batch, m, k), a, bb, b,
                   contiguous3(batch, m, nn), batch * m * nn);
    }
  }
  // Attention-style head split: QKV stored [T, 3*H*D]; Q_h = [H,T,D] view with strides
  // [D, 3HD, 1]; K_h^T = [H,D,T] view with strides [D, 1, 3HD] at offset HD; scores
  // contiguous [H,T,T]; then probs x V_h written into a head-concatenated [T,H*D]
  // output via C strides [D, HD, 1].
  const std::uint64_t T = 19, H = 3, D = 7, W = 3 * H * D;
  const auto qkv = random(T * W, 77);
  View q{{H, T, D}, {std::int64_t(D), std::int64_t(W), 1}, 0};
  View kt{{H, D, T}, {std::int64_t(D), 1, std::int64_t(W)}, H * D};
  batched_case(s, "attention Q K^T", H, T, T, D, q, qkv, kt, qkv, contiguous3(H, T, T), H * T * T);
  const auto probs = random(H * T * T, 78);
  View v{{H, T, D}, {std::int64_t(D), std::int64_t(W), 1}, 2 * H * D};
  View concat{{H, T, D}, {std::int64_t(D), std::int64_t(H * D), 1}, 0};
  batched_case(s, "attention P V into concatenated heads", H, T, D, T, contiguous3(H, T, T), probs, v, qkv,
               concat, T * H * D);
  // k == 0 into a strided output: the GPU fill writes +0.0 only at addressed elements.
  View sparse{{2, 3, 4}, {40, 10, 2}, 1};
  batched_case(s, "k == 0 strided fill", 2, 3, 4, 0, contiguous3(2, 3, 0), {0.0f}, contiguous3(2, 0, 4),
               {0.0f}, sparse, 2 * 40);
}

// ---------------------------------------------------------------- reductions
void reductions(Suite &s) {
  const std::uint64_t shapes[][2] = {{1, 1},   {3, 255}, {5, 256}, {2, 257},    {7, 1000},
                                     {64, 3},  {4, 0},   {0, 5},   {1, 100003}, {2, 513}};
  std::uint32_t seed = 300;
  for (const auto &sh : shapes) {
    const auto rows = sh[0], cols = sh[1];
    const auto x = random(rows * cols, seed++, 4.0f);
    for (int op = 0; op < 2; ++op) {
      const std::string label = std::string(op ? "row max " : "row sum ") + shape_text({rows, cols});
      auto xb = s.upload(x, 5), ob = s.output(rows, 1);
      const auto xd = t::contiguous(xb, {rows, cols}, 20), od = t::contiguous(ob, {rows}, 4);
      auto e = t::reduce_rows(s.queue, s.ops, xd, od, op ? PR_REDUCE_MAX : PR_REDUCE_SUM);
      const auto all = s.read(ob);
      if (!rows) {
        require(!e, label + ": empty output fabricated an event");
        continue;
      }
      s.completed(e, cols ? "paralyn_reduce_rows_f32" : "paralyn_fill_f32", label);
      s.canaries(s.read(xb), 5, rows * cols, label + " input");
      for (std::uint64_t i = 0; i < rows; ++i) {
        const float *row = x.data() + i * cols;
        const float gpu = all[1 + i];
        if (op) {
          const float expected = r::tree_max_f32(row, cols);
          require(r::same_bits(gpu, expected), label + ": max differs at row " + std::to_string(i));
          if (!cols) require(gpu == -inf, label + ": empty-row max is not -inf");
        } else {
          const float expected = r::tree_sum_f32(row, cols);
          require(r::same_bits(gpu, expected), label + ": sum differs from fixed-order FP32 at row " +
                                                   std::to_string(i));
          const auto ref = r::tree_sum(r::exact_values(std::vector<float>(row, row + cols)));
          require(r::within(gpu, ref), label + ": sum exceeds float64 bound");
          s.note("row_sum", r::ratio(gpu, ref));
          if (!cols) require(r::same_bits(gpu, 0.0f), label + ": empty sum is not +0.0");
        }
      }
    }
  }
  // Rounding-heavy sums: non-dyadic values over a 2^40 dynamic range with cancellation,
  // so the fixed order and the float64 bound are both exercised (dyadic inputs sum exactly).
  for (std::uint64_t cols : {1000ull, 70001ull}) {
    const std::uint64_t rows = 3;
    auto x = random(rows * cols, 390 + unsigned(cols), 1.0f);
    for (std::uint64_t i = 0; i < x.size(); ++i) x[i] = x[i] / 3.0f * std::ldexp(1.0f, int(i % 41) - 20);
    auto xb = s.upload(x), ob = s.output(rows);
    auto e = t::reduce_rows(s.queue, s.ops, t::contiguous(xb, {rows, cols}), t::contiguous(ob, {rows}), PR_REDUCE_SUM);
    const std::string label = "row sum mixed magnitude " + shape_text({rows, cols});
    s.completed(e, "paralyn_reduce_rows_f32", label);
    const auto out = s.read(ob);
    for (std::uint64_t i = 0; i < rows; ++i) {
      const float *row = x.data() + i * cols;
      require(r::same_bits(out[i], r::tree_sum_f32(row, cols)), label + ": differs from fixed-order FP32");
      const auto ref = r::tree_sum(r::exact_values(std::vector<float>(row, row + cols)));
      require(r::within(out[i], ref), label + ": sum exceeds float64 bound");
      s.note("row_sum", r::ratio(out[i], ref));
    }
  }
  // NaN/Inf and signed-zero policy.
  const std::vector<float> special{1.0f,  nan,  -2.0f, 3.0f,  // row 0: NaN
                                   -0.0f, 0.0f, -0.0f, -1.0f, // row 1: +0.0 preferred
                                   -0.0f, -0.0f, -3.0f, -1.0f, // row 2: -0.0 max
                                   inf,   -inf, 1.0f,  2.0f,  // row 3: +inf, sum NaN
                                   -inf,  -inf, -inf,  -inf,  // row 4: -inf
                                   inf,   1.0f, inf,   0.0f}; // row 5: +inf sum
  auto xb = s.upload(special);
  const auto xd = t::contiguous(xb, {6, 4});
  for (int op = 0; op < 2; ++op) {
    auto ob = s.output(6);
    auto e = t::reduce_rows(s.queue, s.ops, xd, t::contiguous(ob, {6}), op ? PR_REDUCE_MAX : PR_REDUCE_SUM);
    s.completed(e, "paralyn_reduce_rows_f32", "special reductions");
    const auto out = s.read(ob);
    if (op) {
      require(std::isnan(out[0]), "max: NaN not propagated");
      require(r::same_bits(out[1], 0.0f), "max: +0.0 not preferred over -0.0");
      require(r::same_bits(out[2], -0.0f), "max: all-nonpositive row max is not -0.0");
      require(out[3] == inf && out[4] == -inf && out[5] == inf, "max: infinities");
    } else {
      require(std::isnan(out[0]), "sum: NaN not propagated");
      require(out[1] == -1.0f && out[2] == -4.0f, "sum: signed zeros");
      require(std::isnan(out[3]), "sum: inf + -inf is not NaN");
      require(out[4] == -inf && out[5] == inf, "sum: infinities");
    }
  }
}

// ---------------------------------------------------------------- softmax
void softmax_case(Suite &s, const std::string &label, std::uint64_t batch, std::uint64_t rows,
                  std::uint64_t cols, float scale, bool causal, const std::vector<float> &x, bool rank3) {
  auto xb = s.upload(x, 2), ob = s.output(batch * rows * cols, 1);
  const t::Shape shape = rank3 ? t::Shape{batch, rows, cols} : t::Shape{rows, cols};
  auto e = t::softmax_rows(s.queue, s.ops, t::contiguous(xb, shape, 8), t::contiguous(ob, shape, 4), scale,
                           causal);
  const auto all = s.read(ob);
  if (!(batch * rows * cols)) {
    require(!e, label + ": empty softmax fabricated an event");
    return;
  }
  s.completed(e, "paralyn_softmax_rows_f32", label);
  for (std::uint64_t z = 0; z < batch; ++z)
    for (std::uint64_t i = 0; i < rows; ++i) {
      const auto base = (z * rows + i) * cols;
      const std::vector<float> row(x.begin() + base, x.begin() + base + cols);
      const std::size_t visible = causal ? i + (cols - rows) + 1 : cols;
      const auto ref = r::softmax_row(r::exact_values(row), double(scale), visible);
      for (std::size_t j = 0; j < cols; ++j) {
        const float gpu = all[1 + base + j];
        if (j >= visible) {
          require(r::same_bits(gpu, 0.0f), label + ": masked position is not +0.0");
          continue;
        }
        require(r::within(gpu, ref[j]), label + ": float64 bound exceeded at row " + std::to_string(i) +
                                            " column " + std::to_string(j) + " (gpu " +
                                            std::to_string(gpu) + ", ref " + std::to_string(ref[j].v) + ")");
        s.note("softmax", r::ratio(gpu, ref[j]));
      }
    }
}
void softmax(Suite &s) {
  std::uint32_t seed = 500;
  struct Case {
    std::uint64_t batch, rows, cols;
    float scale, magnitude;
    bool causal, rank3;
  };
  const Case cases[] = {{1, 1, 1, 1.0f, 4.0f, false, false},     {1, 3, 17, 1.0f, 4.0f, false, false},
                        {1, 5, 256, 0.125f, 16.0f, false, false}, {1, 2, 1000, 1.0f, 8.0f, false, false},
                        {1, 1, 100003, 1.0f, 30.0f, false, false}, {4, 9, 9, 0.35355339f, 8.0f, true, true},
                        {2, 5, 12, 1.0f, 8.0f, true, true},        {3, 7, 7, 3.0f, 4.0f, false, true},
                        {1, 1, 300, 1.0f, 100.0f, true, false},    {1, 17, 17, 1.0f, 60.0f, true, false},
                        {1, 0, 5, 1.0f, 1.0f, false, false},       {1, 3, 0, 1.0f, 1.0f, false, false},
                        {0, 3, 3, 1.0f, 1.0f, true, true}};
  for (const auto &c : cases) {
    const auto x = random(c.batch * c.rows * c.cols, seed++, c.magnitude);
    softmax_case(s, "softmax " + shape_text({c.batch, c.rows, c.cols}) + " scale " + std::to_string(c.scale) +
                        (c.causal ? " causal" : ""),
                 c.batch, c.rows, c.cols, c.scale, c.causal, x, c.rank3);
  }
  // NaN/Inf policy (see docs): any NaN or +inf among unmasked inputs, or an all -inf
  // row, gives an all-NaN row; -inf entries alone give exact zeros; masked NaN is ignored.
  const std::vector<float> special{1.0f, nan,  2.0f,  0.5f,  // row 0: NaN -> NaN row
                                   1.0f, inf,  2.0f,  0.5f,  // row 1: +inf -> NaN row
                                   -inf, -inf, -inf,  -inf,  // row 2: all -inf -> NaN row
                                   0.0f, -inf, 0.0f,  -inf}; // row 3: [0.5, 0, 0.5, 0]
  auto xb = s.upload(special);
  auto ob = s.output(16);
  auto e = t::softmax_rows(s.queue, s.ops, t::contiguous(xb, {4, 4}), t::contiguous(ob, {4, 4}));
  s.completed(e, "paralyn_softmax_rows_f32", "softmax specials");
  auto out = s.read(ob);
  for (int j = 0; j < 12; ++j) require(std::isnan(out[j]), "softmax: non-finite row is not all NaN");
  require(r::same_bits(out[13], 0.0f) && r::same_bits(out[15], 0.0f), "softmax: -inf entries are not exact +0.0");
  const auto half = r::softmax_row(r::exact_values({0.0f, 0.0f}), 1.0, 2);
  require(r::within(out[12], half[0]) && r::within(out[14], half[1]), "softmax: finite entries next to -inf");
  // Causal: a NaN in a masked position never enters the row.
  const std::vector<float> masked{1.0f, nan, 2.0f, 3.0f};
  auto mb = s.upload(masked);
  auto mo = s.output(4);
  e = t::softmax_rows(s.queue, s.ops, t::contiguous(mb, {2, 2}), t::contiguous(mo, {2, 2}), 1.0f, true);
  s.completed(e, "paralyn_softmax_rows_f32", "softmax masked NaN");
  out = s.read(mo);
  const auto single = r::softmax_row(r::exact_values({1.0f}), 1.0, 1);
  require(r::within(out[0], single[0]) && r::same_bits(out[1], 0.0f) && !std::isnan(out[2]) && !std::isnan(out[3]),
          "softmax: masked NaN leaked into the row");
  // Overflow of scale * x is +inf and therefore a NaN row (documented, not detected).
  const std::vector<float> big{3e38f, 1.0f};
  auto bb = s.upload(big);
  auto bo = s.output(2);
  e = t::softmax_rows(s.queue, s.ops, t::contiguous(bb, {1, 2}), t::contiguous(bo, {1, 2}), 4.0f);
  s.completed(e, "paralyn_softmax_rows_f32", "softmax overflow");
  out = s.read(bo);
  require(std::isnan(out[0]) && std::isnan(out[1]), "softmax: overflowed row is not NaN");
}

// ---------------------------------------------------------------- layer norm
void layer_norm(Suite &s) {
  std::uint32_t seed = 700;
  struct Case {
    std::uint64_t rows, cols;
    float center, magnitude, epsilon;
  };
  const Case cases[] = {{1, 1, 0.0f, 1.0f, 1e-5f},      {3, 17, 0.5f, 2.0f, 1e-5f},  {4, 256, 0.0f, 1.0f, 1e-5f},
                        {2, 257, -3.0f, 4.0f, 1e-6f},   {5, 1000, 0.0f, 1.0f, 1e-5f}, {1, 100003, 0.0f, 2.0f, 1e-5f},
                        {3, 64, 1000.0f, 1.0f, 1e-5f},  {2, 48, 0.0f, 1e-3f, 1e-2f}, {0, 8, 0.0f, 1.0f, 1e-5f},
                        {3, 0, 0.0f, 1.0f, 1e-5f}};
  for (const auto &c : cases) {
    auto x = random(c.rows * c.cols, seed++, c.magnitude);
    for (auto &v : x) v += c.center;
    const auto g = random(c.cols, seed++, 2.0f), b = random(c.cols, seed++, 0.5f);
    const std::string label = "layer_norm " + shape_text({c.rows, c.cols}) + " center " + std::to_string(c.center);
    auto xb = s.upload(x, 1), gb = s.upload(g), bb = s.upload(b, 3), ob = s.output(c.rows * c.cols, 2);
    auto e = t::layer_norm(s.queue, s.ops, t::contiguous(xb, {c.rows, c.cols}, 4), t::contiguous(gb, {c.cols}),
                           t::contiguous(bb, {c.cols}, 12), t::contiguous(ob, {c.rows, c.cols}, 8), c.epsilon);
    const auto all = s.read(ob);
    if (!(c.rows * c.cols)) {
      require(!e, label + ": empty output fabricated an event");
      continue;
    }
    s.completed(e, "paralyn_layer_norm_f32", label);
    const auto gr = r::exact_values(g), br = r::exact_values(b);
    for (std::uint64_t i = 0; i < c.rows; ++i) {
      const std::vector<float> row(x.begin() + i * c.cols, x.begin() + (i + 1) * c.cols);
      const auto ref = r::layer_norm_row(r::exact_values(row), gr, br, double(c.epsilon));
      for (std::uint64_t j = 0; j < c.cols; ++j) {
        const float gpu = all[2 + i * c.cols + j];
        require(r::within(gpu, ref[j]), label + ": float64 bound exceeded at (" + std::to_string(i) + "," +
                                            std::to_string(j) + ") gpu " + std::to_string(gpu) + " ref " +
                                            std::to_string(ref[j].v) + " bound " + std::to_string(ref[j].e));
        s.note("layer_norm", r::ratio(gpu, ref[j]));
      }
    }
  }
  // Non-finite policy: any NaN/Inf in a row makes the whole row NaN; a NaN in gamma or
  // beta affects only its column; a constant row gives exactly beta.
  const std::vector<float> x{1.0f, nan, 2.0f, 3.0f,   // row 0
                             1.0f, inf, 2.0f, 3.0f,   // row 1
                             5.0f, 5.0f, 5.0f, 5.0f,  // row 2 (constant)
                             1.0f, 2.0f, 3.0f, 4.0f}; // row 3
  const std::vector<float> g{1.0f, 2.0f, nan, 1.0f}, b{0.25f, -0.5f, 0.0f, nan};
  auto xb = s.upload(x), gb = s.upload(g), bb = s.upload(b), ob = s.output(16);
  auto e = t::layer_norm(s.queue, s.ops, t::contiguous(xb, {4, 4}), t::contiguous(gb, {4}), t::contiguous(bb, {4}),
                         t::contiguous(ob, {4, 4}), 1e-5f);
  s.completed(e, "paralyn_layer_norm_f32", "layer_norm specials");
  const auto out = s.read(ob);
  for (int j = 0; j < 8; ++j) require(std::isnan(out[j]), "layer_norm: non-finite row is not all NaN");
  require(out[8] == 0.25f && out[9] == -0.5f && std::isnan(out[10]) && std::isnan(out[11]),
          "layer_norm: constant row is not beta (NaN gamma/beta columns excepted)");
  require(!std::isnan(out[12]) && !std::isnan(out[13]) && std::isnan(out[14]) && std::isnan(out[15]),
          "layer_norm: NaN gamma/beta must affect only their columns");
}

// ---------------------------------------------------------------- GELU, bias+GELU, add
void gelu_and_add(Suite &s) {
  std::uint32_t seed = 900;
  for (std::uint64_t count : {1ull, 255ull, 1000ull, 100003ull}) {
    const auto x = random(count, seed++, 8.0f);
    const std::string label = "gelu count " + std::to_string(count);
    auto xb = s.upload(x, 3), ob = s.output(count, 1);
    auto e = t::bias_activation(s.queue, s.ops, t::contiguous(xb, {1, count}, 12), nullptr,
                                t::contiguous(ob, {1, count}, 4), PR_ACTIVATION_GELU_TANH);
    s.completed(e, "paralyn_activation_f32", label);
    const auto out = s.read(ob);
    s.canaries(out, 1, count, label);
    for (std::uint64_t i = 0; i < count; ++i) {
      const auto ref = r::gelu(r::exact(x[i]));
      require(r::within(out[1 + i], ref), label + ": float64 bound exceeded at " + std::to_string(i) +
                                              " x=" + std::to_string(x[i]));
      s.note("gelu", r::ratio(out[1 + i], ref));
    }
  }
  // Bias + GELU (fused): out = gelu(fl(x + bias)).
  const std::uint64_t rows = 33, cols = 70;
  const auto x = random(rows * cols, 950, 4.0f), bias = random(cols, 951, 2.0f);
  auto xb = s.upload(x), bb = s.upload(bias), ob = s.output(rows * cols);
  auto bd = t::contiguous(bb, {cols});
  auto e = t::bias_activation(s.queue, s.ops, t::contiguous(xb, {rows, cols}), &bd, t::contiguous(ob, {rows, cols}),
                              PR_ACTIVATION_GELU_TANH);
  s.completed(e, "paralyn_bias_activation_f32", "bias+gelu");
  auto out = s.read(ob);
  for (std::uint64_t i = 0; i < rows * cols; ++i) {
    const float sum = x[i] + bias[i % cols];
    const auto ref = r::gelu(r::exact(sum));
    require(r::within(out[i], ref), "bias+gelu: float64 bound exceeded at " + std::to_string(i));
    s.note("gelu", r::ratio(out[i], ref));
  }
}
void gelu_specials(Suite &s) {
  const std::vector<float> special{nan, inf, -inf, 1e20f, -1e20f, 0.0f, -0.0f, 30.0f, -30.0f};
  const std::uint64_t count = special.size();
  auto sb = s.upload(special), so = s.output(count);
  auto e = t::bias_activation(s.queue, s.ops, t::contiguous(sb, {1, count}), nullptr,
                              t::contiguous(so, {1, count}), PR_ACTIVATION_GELU_TANH);
  s.completed(e, "paralyn_activation_f32", "gelu specials");
  const auto o = s.read(so);
  require(std::isnan(o[0]), "gelu(NaN) is not NaN");
  require(o[1] == inf, "gelu(+inf) is not +inf");
  require(std::isnan(o[2]), "gelu(-inf) is not NaN (documented: -inf * 0)");
  require(o[3] == 1e20f, "gelu(1e20) is not 1e20");
  require(r::same_bits(o[4], -0.0f), "gelu(-1e20) is not -0.0");
  require(r::same_bits(o[5], 0.0f) && r::same_bits(o[6], -0.0f), "gelu(+-0) does not keep the zero sign");
  // Finite saturation is still bounded (tanh may be up to 5 ulp from +-1).
  require(r::within(o[7], r::gelu(r::exact(30.0))) && r::within(o[8], r::gelu(r::exact(-30.0))),
          "gelu(+-30) outside the float64 bound");
}
void residual_add(Suite &s) {
  std::uint32_t seed = 1100;
  const std::vector<t::Shape> shapes{{}, {1}, {17}, {2, 3, 5}, {1000003}, {0, 4}, {2, 1, 1, 1, 1, 1, 1, 3}};
  for (const auto &shape : shapes) {
    std::uint64_t count = 1;
    for (auto d : shape) count *= d;
    const auto x = random(count, seed++, 1000.0f), y = random(count, seed++, 3.0f);
    const std::string label = "add rank " + std::to_string(shape.size()) + " count " + std::to_string(count);
    auto xb = s.upload(x, 1), yb = s.upload(y), ob = s.output(count, 2);
    auto e = t::add(s.queue, s.ops, t::contiguous(xb, shape, 4), t::contiguous(yb, shape), t::contiguous(ob, shape, 8));
    const auto out = s.read(ob);
    if (!count) {
      require(!e, label + ": empty add fabricated an event");
      continue;
    }
    s.completed(e, "paralyn_add_f32", label);
    s.canaries(out, 2, count, label);
    for (std::uint64_t i = 0; i < count; ++i) {
      const float expected = x[i] + y[i];
      require(r::same_bits(out[2 + i], expected), label + ": add differs from FP32 x + y at " + std::to_string(i));
    }
  }
  const std::vector<float> x{nan, inf, inf, -0.0f, 3e38f}, y{1.0f, -inf, 1.0f, -0.0f, 3e38f};
  auto xb = s.upload(x), yb = s.upload(y), ob = s.output(5);
  auto e = t::add(s.queue, s.ops, t::contiguous(xb, {5}), t::contiguous(yb, {5}), t::contiguous(ob, {5}));
  s.completed(e, "paralyn_add_f32", "add specials");
  const auto o = s.read(ob);
  require(std::isnan(o[0]) && std::isnan(o[1]) && o[2] == inf && r::same_bits(o[3], -0.0f) && o[4] == inf,
          "add: IEEE NaN/Inf/signed-zero/overflow propagation");
}

// ---------------------------------------------------------------- negative tests
template <class Record> Record rec() {
  Record op{};
  op.struct_size = sizeof(op);
  op.version = PR_TENSOR_VERSION_1;
  return op;
}
void negative(Suite &s) {
  pr_event e = 99;
  auto x = s.upload(random(64, 1)), y = s.upload(random(64, 2)), o = s.output(64), g = s.upload(random(8, 3));
  const auto x2 = t::contiguous(x, {2, 8}), y2 = t::contiguous(y, {2, 8}), o2 = t::contiguous(o, {2, 8});
  const auto g8 = t::contiguous(g, {8}), o_rows = t::contiguous(o, {2});
  n::Context other("auto");
  auto other_queue = other.queue();
  auto other_ops = t::load_operators(other);
  auto foreign = other.buffer(256);
  const auto f2 = t::contiguous(foreign, {2, 8}), f8 = t::contiguous(foreign, {8});
  const pr_queue Q = s.queue.get();
  const pr_module M = s.ops.get();
  // Provider identity: the genuine artifact loaded with plain pr_module_load, and a released provider.
  const auto genuine = t::operators_artifact();
  pr_module plain = 0;
  status(pr_module_load(s.context.get(), genuine.data(), genuine.size(), &plain), PR_SUCCESS, "plain load", "");
  n::Module plain_owner(plain);
  pr_module released = 0;
  status(pr_tensor_operators_load(s.context.get(), &released), PR_SUCCESS, "provider load", "");
  status(pr_release(released), PR_SUCCESS, "provider release", "");

  // Generic checks shared by every operator: handles, provider identity, contexts.
  struct Op {
    const char *name;
    std::function<pr_status(pr_queue, pr_module, pr_event *)> call;
  };
  auto common = [&](const Op &op) {
    const std::string name = op.name;
    auto run = [&](pr_queue q, pr_module m, pr_status expected, const std::string &label) {
      e = 99;
      status(op.call(q, m, &e), expected, name + ": " + label, op.name);
      require(e == 0, name + ": " + label + ": event output not cleared");
    };
    run(0xdeadbeefull, M, PR_INVALID_HANDLE, "invalid queue");
    run(M, M, PR_INVALID_HANDLE, "queue handle of wrong kind");
    run(Q, 0xffffffffull, PR_INVALID_HANDLE, "invalid module");
    run(Q, released, PR_INVALID_HANDLE, "released provider");
    run(Q, plain, PR_INVALID_ARGUMENT, "artifact loaded without pr_tensor_operators_load");
    run(other_queue.get(), M, PR_CONTEXT_MISMATCH, "foreign queue");
    run(Q, other_ops.get(), PR_CONTEXT_MISMATCH, "foreign module");
    status(op.call(Q, M, nullptr), PR_INVALID_ARGUMENT, name + ": null event", op.name);
  };

  // --- batched matmul
  const auto a3 = t::contiguous(x, {2, 2, 4}), b3 = t::contiguous(y, {2, 4, 3}), c3 = t::contiguous(o, {2, 2, 3});
  auto bm = [&](const pr_tensor_desc_v1 *a, const pr_tensor_desc_v1 *b, const pr_tensor_desc_v1 *c,
                std::uint64_t batch = 2, std::uint64_t m = 2, std::uint64_t nn = 3, std::uint64_t k = 4) {
    auto op = rec<pr_batched_matmul_v1>();
    op.batch = batch;
    op.m = m;
    op.n = nn;
    op.k = k;
    op.a = a;
    op.b = b;
    op.c = c;
    return op;
  };
  auto bm_run = [&](pr_batched_matmul_v1 op, pr_status expected, const std::string &label, pr_queue q = 0) {
    e = 99;
    status(pr_batched_matmul_f32(q ? q : Q, M, &op, &e), expected, "batched: " + label, "batched_matmul_f32");
    if (expected == PR_SUCCESS && e) {
      s.completed(n::Event(e), "paralyn_batched_matmul_f32", label);
    } else
      require(e == 0, label + ": event not cleared");
  };
  common({"batched_matmul_f32", [&](pr_queue q, pr_module m, pr_event *ev) {
            auto op = bm(&a3, &b3, &c3);
            return pr_batched_matmul_f32(q, m, &op, ev);
          }});
  bm_run(bm(&a3, &b3, &c3), PR_SUCCESS, "valid");
  bm_run(bm(&a3, &b3, &c3, 1), PR_INVALID_ARGUMENT, "batch mismatch");
  bm_run(bm(&a3, &b3, &c3, 2, 3), PR_INVALID_ARGUMENT, "m mismatch");
  bm_run(bm(&a3, &b3, &c3, 2, 2, 4), PR_INVALID_ARGUMENT, "n mismatch");
  bm_run(bm(&a3, &b3, &c3, 2, 2, 3, 3), PR_INVALID_ARGUMENT, "k mismatch");
  bm_run(bm(&x2, &b3, &c3), PR_INVALID_ARGUMENT, "rank-2 A");
  bm_run(bm(&a3, &b3, nullptr), PR_INVALID_ARGUMENT, "missing C");
  bm_run(bm(&a3, &b3, &c3, 0x80000000ull), PR_UNSUPPORTED, "batch > INT32_MAX");
  auto bad = bm(&a3, &b3, &c3);
  bad.version = 2;
  bm_run(bad, PR_UNSUPPORTED, "record version");
  bad = bm(&a3, &b3, &c3);
  bad.struct_size = 8;
  bm_run(bad, PR_INVALID_ARGUMENT, "record size");
  auto self_overlap = t::strided(o, {2, 2, 3}, {0, 3, 1}); // batch stride 0 on the output
  bm_run(bm(&a3, &b3, &self_overlap), PR_INVALID_ARGUMENT, "self-overlapping output");
  auto interleaved = t::strided(o, {2, 2, 3}, {6, 1, 1}); // rows collide with columns
  bm_run(bm(&a3, &b3, &interleaved), PR_INVALID_ARGUMENT, "interleaved self-overlapping output");
  auto negative_stride = c3;
  negative_stride.strides[1] = -3;
  bm_run(bm(&a3, &b3, &negative_stride), PR_UNSUPPORTED, "negative stride");
  auto c_over_a = t::contiguous(x, {2, 2, 3}, 8);
  bm_run(bm(&a3, &b3, &c_over_a), PR_INVALID_ARGUMENT, "output overlapping A");
  auto c_share_b = t::contiguous(y, {2, 2, 3}, 4 * 40);
  bm_run(bm(&a3, &b3, &c_share_b), PR_UNSUPPORTED, "output sharing B's allocation");
  const auto fa = t::contiguous(foreign, {2, 2, 4});
  bm_run(bm(&fa, &b3, &c3), PR_CONTEXT_MISMATCH, "foreign A");
  const auto a0 = t::contiguous(x, {0, 2, 4}), c0 = t::contiguous(o, {0, 2, 3}), fc0 = t::contiguous(foreign, {0, 2, 3});
  bm_run(bm(&a0, &b3, &c0, 0), PR_INVALID_ARGUMENT, "zero batch with mismatched B");
  const auto b0 = t::contiguous(y, {0, 4, 3});
  bm_run(bm(&a0, &b0, &c0, 0), PR_SUCCESS, "zero batch");
  bm_run(bm(&a0, &b0, &fc0, 0), PR_CONTEXT_MISMATCH, "zero batch foreign C");
  bm_run(bm(&a0, &b0, &c0, 0), PR_INVALID_HANDLE, "zero batch invalid queue", 0xdeadbeefull);
  auto nan_a = s.upload({nan, 1.0f, 2.0f, 3.0f}), nan_b = s.upload({1.0f, 1.0f, 1.0f, 1.0f}), nan_c = s.output(4);
  auto na = t::contiguous(nan_a, {1, 2, 2}), nb = t::contiguous(nan_b, {1, 2, 2}), nc = t::contiguous(nan_c, {1, 2, 2});
  bm_run(bm(&na, &nb, &nc, 1, 2, 2, 2), PR_SUCCESS, "NaN propagation");
  const auto nan_out = s.read(nan_c);
  require(std::isnan(nan_out[0]) && std::isnan(nan_out[1]) && nan_out[2] == 5.0f && nan_out[3] == 5.0f,
          "batched: NaN must propagate to exactly the outputs whose dot product contains it");

  // --- reductions
  auto rd = [&](const pr_tensor_desc_v1 *in, const pr_tensor_desc_v1 *out, pr_reduce_op op = PR_REDUCE_SUM) {
    auto r0 = rec<pr_reduce_rows_v1>();
    r0.op = op;
    r0.x = in;
    r0.out = out;
    return r0;
  };
  auto rd_run = [&](pr_reduce_rows_v1 op, pr_status expected, const std::string &label) {
    e = 99;
    status(pr_reduce_rows_f32(Q, M, &op, &e), expected, "reduce: " + label, "reduce_rows_f32");
    require(e == 0, label + ": event not cleared");
  };
  common({"reduce_rows_f32", [&](pr_queue q, pr_module m, pr_event *ev) {
            auto op = rd(&x2, &o_rows);
            return pr_reduce_rows_f32(q, m, &op, ev);
          }});
  const auto o3 = t::contiguous(o, {3});
  rd_run(rd(&x2, &o3), PR_INVALID_ARGUMENT, "output length");
  rd_run(rd(&x2, &o2), PR_INVALID_ARGUMENT, "output rank");
  rd_run(rd(&x2, &o_rows, static_cast<pr_reduce_op>(2)), PR_INVALID_ARGUMENT, "unknown op");
  auto reserved = rd(&x2, &o_rows);
  reserved.reserved = 1;
  rd_run(reserved, PR_INVALID_ARGUMENT, "reserved");
  const auto over = t::contiguous(x, {2}, 8);
  rd_run(rd(&x2, &over), PR_INVALID_ARGUMENT, "output overlapping input");
  const auto share = t::contiguous(x, {2}, 4 * 60);
  rd_run(rd(&x2, &share), PR_UNSUPPORTED, "output sharing input allocation");
  auto column_major = x2;
  column_major.strides[0] = 1;
  column_major.strides[1] = 2;
  rd_run(rd(&column_major, &o_rows), PR_UNSUPPORTED, "non-contiguous input");
  const auto fo = t::contiguous(foreign, {2});
  rd_run(rd(&x2, &fo), PR_CONTEXT_MISMATCH, "foreign output");
  const auto x0 = t::contiguous(x, {0, 8}), o0 = t::contiguous(o, {0}), fo0 = t::contiguous(foreign, {0});
  rd_run(rd(&x0, &fo0), PR_CONTEXT_MISMATCH, "empty foreign output");
  auto i32 = x2;
  i32.dtype = PR_I32;
  rd_run(rd(&i32, &o_rows), PR_UNSUPPORTED, "non-FP32 input");
  e = 99;
  auto ok0 = rd(&x0, &o0);
  status(pr_reduce_rows_f32(Q, M, &ok0, &e), PR_SUCCESS, "reduce zero rows", "reduce_rows_f32");
  require(e == 0, "zero-row reduction fabricated an event");

  // --- softmax
  auto sm = [&](const pr_tensor_desc_v1 *in, const pr_tensor_desc_v1 *out, float scale = 1.0f, std::uint32_t causal = 0) {
    auto op = rec<pr_softmax_rows_v1>();
    op.scale = scale;
    op.causal = causal;
    op.x = in;
    op.out = out;
    return op;
  };
  auto sm_run = [&](pr_softmax_rows_v1 op, pr_status expected, const std::string &label) {
    e = 99;
    status(pr_softmax_rows_f32(Q, M, &op, &e), expected, "softmax: " + label, "softmax_rows_f32");
    require(e == 0, label + ": event not cleared");
  };
  common({"softmax_rows_f32", [&](pr_queue q, pr_module m, pr_event *ev) {
            auto op = sm(&x2, &o2);
            return pr_softmax_rows_f32(q, m, &op, ev);
          }});
  const auto o_t = t::contiguous(o, {8, 2});
  sm_run(sm(&x2, &o_t), PR_INVALID_ARGUMENT, "output shape");
  sm_run(sm(&x2, &o2, 0.0f), PR_INVALID_ARGUMENT, "zero scale");
  sm_run(sm(&x2, &o2, -1.0f), PR_INVALID_ARGUMENT, "negative scale");
  sm_run(sm(&x2, &o2, nan), PR_INVALID_ARGUMENT, "NaN scale");
  sm_run(sm(&x2, &o2, inf), PR_INVALID_ARGUMENT, "infinite scale");
  sm_run(sm(&x2, &o2, 1.0f, 2), PR_INVALID_ARGUMENT, "causal flag");
  const auto tall = t::contiguous(x, {8, 2}), tall_o = t::contiguous(o, {8, 2});
  sm_run(sm(&tall, &tall_o, 1.0f, 1), PR_INVALID_ARGUMENT, "causal with rows > columns");
  const auto r1 = t::contiguous(x, {16}), r1o = t::contiguous(o, {16});
  sm_run(sm(&r1, &r1o), PR_INVALID_ARGUMENT, "rank-1 input");
  sm_run(sm(&x2, &x2), PR_INVALID_ARGUMENT, "in-place");
  sm_run(sm(&x2, &f2), PR_CONTEXT_MISMATCH, "foreign output");
  auto sm_version = sm(&x2, &o2);
  sm_version.version = 9;
  sm_run(sm_version, PR_UNSUPPORTED, "record version");
  const auto e0 = t::contiguous(x, {0, 8}), eo0 = t::contiguous(foreign, {0, 8});
  sm_run(sm(&e0, &eo0), PR_CONTEXT_MISMATCH, "empty foreign output");

  // --- layer norm
  auto ln = [&](const pr_tensor_desc_v1 *in, const pr_tensor_desc_v1 *gamma_, const pr_tensor_desc_v1 *beta,
                const pr_tensor_desc_v1 *out, float eps = 1e-5f) {
    auto op = rec<pr_layer_norm_v1>();
    op.epsilon = eps;
    op.x = in;
    op.gamma = gamma_;
    op.beta = beta;
    op.out = out;
    return op;
  };
  auto ln_run = [&](pr_layer_norm_v1 op, pr_status expected, const std::string &label) {
    e = 99;
    status(pr_layer_norm_f32(Q, M, &op, &e), expected, "layer_norm: " + label, "layer_norm_f32");
    require(e == 0, label + ": event not cleared");
  };
  const auto beta8 = t::contiguous(g, {8});
  common({"layer_norm_f32", [&](pr_queue q, pr_module m, pr_event *ev) {
            auto op = ln(&x2, &g8, &beta8, &o2);
            return pr_layer_norm_f32(q, m, &op, ev);
          }});
  const auto g7 = t::contiguous(g, {7});
  ln_run(ln(&x2, &g7, &beta8, &o2), PR_INVALID_ARGUMENT, "gamma length");
  ln_run(ln(&x2, &g8, &g7, &o2), PR_INVALID_ARGUMENT, "beta length");
  ln_run(ln(&x2, nullptr, &beta8, &o2), PR_INVALID_ARGUMENT, "missing gamma");
  ln_run(ln(&x2, &g8, &beta8, &o_t), PR_INVALID_ARGUMENT, "output shape");
  ln_run(ln(&x2, &g8, &beta8, &o2, 0.0f), PR_INVALID_ARGUMENT, "zero epsilon");
  ln_run(ln(&x2, &g8, &beta8, &o2, nan), PR_INVALID_ARGUMENT, "NaN epsilon");
  auto ln_reserved = ln(&x2, &g8, &beta8, &o2);
  ln_reserved.reserved = 3;
  ln_run(ln_reserved, PR_INVALID_ARGUMENT, "reserved");
  ln_run(ln(&x2, &g8, &beta8, &x2), PR_INVALID_ARGUMENT, "in-place");
  ln_run(ln(&x2, &g8, &f8, &o2), PR_CONTEXT_MISMATCH, "foreign beta");
  const auto lx0 = t::contiguous(x, {0, 8}), lfo = t::contiguous(foreign, {0, 8});
  ln_run(ln(&lx0, &g8, &beta8, &lfo), PR_CONTEXT_MISMATCH, "empty foreign output");
  {
    const std::uint64_t wide = (std::uint64_t(1) << 24) + 1; // needs the 2^24 column limit
    auto wb = s.context.buffer(wide * 4), wg = s.context.buffer(wide * 4), wo = s.context.buffer(wide * 4);
    const auto wx = t::contiguous(wb, {1, wide}), wgd = t::contiguous(wg, {wide}), wod = t::contiguous(wo, {1, wide});
    ln_run(ln(&wx, &wgd, &wgd, &wod), PR_UNSUPPORTED, "more than 2^24 columns");
  }

  // --- add
  auto ad = [&](const pr_tensor_desc_v1 *a, const pr_tensor_desc_v1 *b, const pr_tensor_desc_v1 *out) {
    auto op = rec<pr_add_v1>();
    op.x = a;
    op.y = b;
    op.out = out;
    return op;
  };
  auto ad_run = [&](pr_add_v1 op, pr_status expected, const std::string &label) {
    e = 99;
    status(pr_add_f32(Q, M, &op, &e), expected, "add: " + label, "add_f32");
    require(e == 0, label + ": event not cleared");
  };
  common({"add_f32", [&](pr_queue q, pr_module m, pr_event *ev) {
            auto op = ad(&x2, &y2, &o2);
            return pr_add_f32(q, m, &op, ev);
          }});
  const auto y_t = t::contiguous(y, {8, 2});
  ad_run(ad(&x2, &y_t, &o2), PR_INVALID_ARGUMENT, "shape mismatch");
  const auto y16 = t::contiguous(y, {16});
  ad_run(ad(&x2, &y16, &o2), PR_INVALID_ARGUMENT, "rank mismatch");
  ad_run(ad(&x2, &y2, &x2), PR_INVALID_ARGUMENT, "in-place");
  const auto o_share = t::contiguous(y, {2, 8}, 4 * 32);
  ad_run(ad(&x2, &y2, &o_share), PR_UNSUPPORTED, "output sharing y allocation");
  ad_run(ad(&x2, &f2, &o2), PR_CONTEXT_MISMATCH, "foreign y");
  ad_run(ad(&x2, &y2, nullptr), PR_INVALID_ARGUMENT, "missing output");
  const auto ax0 = t::contiguous(x, {0, 3}), ay0 = t::contiguous(y, {0, 3}), afo = t::contiguous(foreign, {0, 3});
  ad_run(ad(&ax0, &ay0, &afo), PR_CONTEXT_MISMATCH, "empty foreign output");

  // --- activation value outside the enum
  auto bop = rec<pr_bias_activation_v1>();
  bop.activation = static_cast<pr_activation>(3);
  bop.x = &x2;
  bop.out = &o2;
  e = 99;
  status(pr_bias_activation_f32(Q, M, &bop, &e), PR_INVALID_ARGUMENT, "activation 3", "bias_activation_f32");
}

// ---------------------------------------------------------------- capabilities and row limits
// The capability record must agree with what the operators actually accept.
unsigned capabilities(Suite &s) {
  const auto caps = t::operators_capabilities();
  require(caps.struct_size == sizeof(caps) && caps.version == PR_TENSOR_VERSION_1, "capability header");
  require(caps.operations == (PR_TENSOR_OP_MATMUL | PR_TENSOR_OP_BIAS_ACTIVATION | PR_TENSOR_OP_BATCHED_MATMUL |
                              PR_TENSOR_OP_REDUCE_ROWS | PR_TENSOR_OP_SOFTMAX_ROWS | PR_TENSOR_OP_LAYER_NORM |
                              PR_TENSOR_OP_ADD),
          "capability operations");
  require(caps.activations == 0x7 && (caps.activations & PR_TENSOR_ACTIVATION_BIT(PR_ACTIVATION_GELU_TANH)),
          "capability activations (NONE, RELU, GELU_TANH)");
  require(caps.reductions == 0x3, "capability reductions (SUM, MAX)");
  require(caps.max_tensor_elements == 2147483647ull && caps.max_row_operator_rows == 16777215ull &&
              caps.max_row_operator_rows == PR_TENSOR_ROW_OPERATOR_MAX_ROWS &&
              caps.max_layer_norm_columns == (1ull << 24),
          "capability limits");
  require(t::supports_activation(PR_ACTIVATION_GELU_TANH) && !t::supports_activation(static_cast<pr_activation>(3)),
          "supports_activation");
  pr_tensor_operators_capabilities_v1 bad{};
  bad.struct_size = sizeof(bad);
  bad.version = 2;
  status(pr_tensor_operators_capabilities(&bad), PR_UNSUPPORTED, "capabilities: version 2", "tensor_operators_capabilities");
  bad.version = PR_TENSOR_VERSION_1;
  bad.struct_size = sizeof(bad) - 8;
  status(pr_tensor_operators_capabilities(&bad), PR_INVALID_ARGUMENT, "capabilities: short record",
         "tensor_operators_capabilities");
  status(pr_tensor_operators_capabilities(nullptr), PR_INVALID_ARGUMENT, "capabilities: null",
         "tensor_operators_capabilities");
  // Advertised activations run on the GPU; every other value is rejected (0..63).
  const auto x = random(16, 4242, 3.0f);
  auto xb = s.upload(x), ob = s.output(16);
  const auto xd = t::contiguous(xb, {2, 8}), od = t::contiguous(ob, {2, 8});
  unsigned accepted = 0;
  for (unsigned v = 0; v < 64; ++v) {
    auto op = rec<pr_bias_activation_v1>();
    op.activation = static_cast<pr_activation>(v);
    op.x = &xd;
    op.out = &od;
    pr_event e = 0;
    const bool advertised = caps.activations >> v & 1;
    const std::string label = "activation " + std::to_string(v);
    status(pr_bias_activation_f32(s.queue.get(), s.ops.get(), &op, &e),
           advertised ? PR_SUCCESS : PR_INVALID_ARGUMENT, label, "bias_activation_f32");
    if (!advertised) {
      require(e == 0, label + ": event not cleared");
      continue;
    }
    s.completed(n::Event(e), "paralyn_activation_f32", label);
    const auto out = s.read(ob);
    for (std::size_t i = 0; i < 16; ++i) {
      const float want = v == PR_ACTIVATION_NONE ? x[i] : v == PR_ACTIVATION_RELU ? (x[i] >= 0.0f ? x[i] : 0.0f) : 0.0f;
      if (v == PR_ACTIVATION_GELU_TANH)
        require(r::within(out[i], r::gelu(r::exact(x[i]))), label + ": GELU outside its bound");
      else
        require(r::same_bits(out[i], want), label + ": wrong value");
    }
    ++accepted;
  }
  require(accepted == 3, "exactly the three advertised activations must run");
  return accepted;
}
// One 256-lane threadgroup per row: rows <= floor((2^32-1)/256). The exact boundary runs
// on the GPU (columns == 1, where every operator has an exact answer); one more row, and
// any empty-column shape with too many rows, is PR_UNSUPPORTED before the zero-work decision.
void row_limits(Suite &s) {
  const std::uint64_t limit = PR_TENSOR_ROW_OPERATOR_MAX_ROWS, over = limit + 1;
  require(limit == 16777215ull && limit * 256 <= 0xffffffffull && over * 256 > 0xffffffffull, "row limit arithmetic");
  const auto Q = s.queue.get();
  const auto M = s.ops.get();
  auto xb = s.context.buffer(over * 4), ob = s.context.buffer(over * 4);
  const auto gamma_ = s.upload({1.5f}), beta = s.upload({-0.25f});
  const auto gd = t::contiguous(gamma_, {1}), bd = t::contiguous(beta, {1});
  const auto gd0 = t::contiguous(gamma_, {0}), bd0 = t::contiguous(beta, {0});
  pr_event e = 99;
  auto reduce = [&](std::uint64_t rows, std::uint64_t cols, pr_reduce_op op) {
    const auto xd = t::contiguous(xb, {rows, cols}), od = t::contiguous(ob, {rows});
    auto r0 = rec<pr_reduce_rows_v1>();
    r0.op = op;
    r0.x = &xd;
    r0.out = &od;
    e = 99;
    return pr_reduce_rows_f32(Q, M, &r0, &e);
  };
  auto softmax_ = [&](std::vector<std::uint64_t> shape) {
    const auto xd = t::contiguous(xb, shape), od = t::contiguous(ob, shape);
    auto op = rec<pr_softmax_rows_v1>();
    op.scale = 1.0f;
    op.x = &xd;
    op.out = &od;
    e = 99;
    return pr_softmax_rows_f32(Q, M, &op, &e);
  };
  auto norm = [&](std::uint64_t rows, std::uint64_t cols) {
    const auto xd = t::contiguous(xb, {rows, cols}), od = t::contiguous(ob, {rows, cols});
    auto op = rec<pr_layer_norm_v1>();
    op.epsilon = 1e-5f;
    op.x = &xd;
    op.gamma = cols ? &gd : &gd0;
    op.beta = cols ? &bd : &bd0;
    op.out = &od;
    e = 99;
    return pr_layer_norm_f32(Q, M, &op, &e);
  };
  auto rejected = [&](pr_status st, const std::string &label, const char *operation) {
    status(st, PR_UNSUPPORTED, label, operation);
    require(e == 0, label + ": event not cleared");
    pr_error err{};
    pr_last_error(&err);
    require(std::string(err.message).find("16777215") != std::string::npos, label + ": message lacks the limit");
  };
  rejected(reduce(over, 1, PR_REDUCE_SUM), "reduce sum [2^24,1]", "reduce_rows_f32");
  rejected(reduce(over, 1, PR_REDUCE_MAX), "reduce max [2^24,1]", "reduce_rows_f32");
  rejected(reduce(over, 0, PR_REDUCE_SUM), "reduce sum [2^24,0]", "reduce_rows_f32");
  rejected(softmax_({over, 1}), "softmax [2^24,1]", "softmax_rows_f32");
  rejected(softmax_({over, 0}), "softmax [2^24,0] (no elements)", "softmax_rows_f32");
  rejected(softmax_({2, over / 2, 1}), "softmax [2,2^23,1] (batch*rows)", "softmax_rows_f32");
  rejected(softmax_({1ull << 32, 1ull << 20, 0}), "softmax [2^32,2^20,0] (no elements, batch*rows = 2^52)", "softmax_rows_f32");
  rejected(norm(over, 1), "layer_norm [2^24,1]", "layer_norm_f32");
  rejected(norm(over, 0), "layer_norm [2^24,0] (no elements)", "layer_norm_f32");

  // The exact boundary executes on the GPU.
  const auto x = random(limit, 5150, 8.0f);
  xb.write(x.data(), limit * 4);
  std::vector<float> out(limit);
  const std::string rows_text = "[" + std::to_string(limit) + ",1]";
  for (auto op : {PR_REDUCE_SUM, PR_REDUCE_MAX}) {
    const std::string label = std::string(op == PR_REDUCE_SUM ? "reduce sum " : "reduce max ") + rows_text;
    status(reduce(limit, 1, op), PR_SUCCESS, label, "");
    s.completed(n::Event(e), "paralyn_reduce_rows_f32", label);
    ob.read(out.data(), limit * 4);
    // With one column the documented order reduces to +0.0 + x (sum) and x (max); the
    // full 256-lane emulation re-derives that on every 4099th row and the last row.
    for (std::uint64_t i = 0; i < limit; ++i) {
      const float closed = op == PR_REDUCE_SUM ? 0.0f + x[i] : x[i];
      const bool sampled = i % 4099 == 0 || i + 1 == limit;
      if (!r::same_bits(out[i], closed) ||
          (sampled && !r::same_bits(out[i], op == PR_REDUCE_SUM ? r::tree_sum_f32(&x[i], 1)
                                                                : r::tree_max_f32(&x[i], 1))))
        require(false, label + ": row " + std::to_string(i));
    }
  }
  status(softmax_({limit, 1}), PR_SUCCESS, "softmax " + rows_text, "");
  s.completed(n::Event(e), "paralyn_softmax_rows_f32", "softmax " + rows_text);
  ob.read(out.data(), limit * 4);
  for (std::uint64_t i = 0; i < limit; ++i)
    if (!r::same_bits(out[i], 1.0f))
      require(false, "softmax " + rows_text + ": row " + std::to_string(i) + " is not exactly 1");
  status(norm(limit, 1), PR_SUCCESS, "layer_norm " + rows_text, "");
  s.completed(n::Event(e), "paralyn_layer_norm_f32", "layer_norm " + rows_text);
  ob.read(out.data(), limit * 4);
  for (std::uint64_t i = 0; i < limit; ++i) // x - mean == 0 exactly, so out == 0*rstd*gamma + beta == beta
    if (!r::same_bits(out[i], -0.25f))
      require(false, "layer_norm " + rows_text + ": row " + std::to_string(i) + " is not beta");
}

// ---------------------------------------------------------------- high-level wrappers
unsigned high_level() {
  t::Context ctx("auto");
  const auto a = random(2 * 5 * 7, 1300), b = random(2 * 7 * 3, 1301), g = random(7, 1302), bias = random(7, 1303);
  auto ta = t::from_host(ctx, {2, 5, 7}, a), tb = t::from_host(ctx, {2, 7, 3}, b);
  auto c = t::batched_matmul(ta, tb);
  auto x = t::from_host(ctx, {10, 7}, a), tg = t::from_host(ctx, {7}, g), tbias = t::from_host(ctx, {7}, bias);
  auto sums = t::row_sum(x), maxes = t::row_max(x);
  auto probs = t::softmax(x, 0.5f, false);
  auto norm = t::layer_norm(x, tg, tbias, 1e-5f);
  auto act = t::gelu(x);
  auto fused = t::bias_gelu(x, tbias);
  auto sum2 = t::add(x, act);
  const auto ch = c.to_host();
  for (std::uint64_t z = 0; z < 2; ++z)
    for (std::uint64_t i = 0; i < 5; ++i)
      for (std::uint64_t j = 0; j < 3; ++j) {
        float seq = 0.0f;
        for (std::uint64_t p = 0; p < 7; ++p) seq = seq + a[(z * 5 + i) * 7 + p] * b[(z * 7 + p) * 3 + j];
        require(r::same_bits(ch[(z * 5 + i) * 3 + j], seq), "high-level batched_matmul");
      }
  const auto sh = sums.to_host(), mh = maxes.to_host(), ph = probs.to_host(), nh = norm.to_host(),
             gh = act.to_host(), fh = fused.to_host(), ah = sum2.to_host();
  for (std::uint64_t i = 0; i < 10; ++i) {
    const float *row = a.data() + i * 7;
    require(r::same_bits(sh[i], r::tree_sum_f32(row, 7)) && r::same_bits(mh[i], r::tree_max_f32(row, 7)),
            "high-level reductions");
    const std::vector<float> rv(row, row + 7);
    const auto sref = r::softmax_row(r::exact_values(rv), 0.5, 7);
    const auto lref = r::layer_norm_row(r::exact_values(rv), r::exact_values(g), r::exact_values(bias), 1e-5f);
    for (std::uint64_t j = 0; j < 7; ++j) {
      require(r::within(ph[i * 7 + j], sref[j]) && r::within(nh[i * 7 + j], lref[j]), "high-level softmax/layer_norm");
      require(r::within(gh[i * 7 + j], r::gelu(r::exact(row[j]))), "high-level gelu");
      require(r::within(fh[i * 7 + j], r::gelu(r::exact(row[j] + bias[j]))), "high-level bias_gelu");
      require(r::same_bits(ah[i * 7 + j], row[j] + gh[i * 7 + j]), "high-level add");
    }
  }
  unsigned events = 0;
  for (const auto *tensor : {&c, &sums, &maxes, &probs, &norm, &act, &fused, &sum2}) {
    const auto timing = tensor->timing();
    require(timing && timing->completed && timing->duration_valid, "high-level event timing");
    ++events;
  }
  bool rejected = false;
  try { t::batched_matmul(ta, ta); } catch (const std::exception &) { rejected = true; }
  require(rejected, "high-level batched_matmul accepted mismatched shapes");
  rejected = false;
  try { t::add(x, c); } catch (const std::exception &) { rejected = true; }
  require(rejected, "high-level add accepted mismatched shapes");
  t::Context other("auto");
  auto foreign = t::from_host(other, {10, 7}, a);
  rejected = false;
  try { t::add(x, foreign); } catch (const std::exception &) { rejected = true; }
  require(rejected, "high-level add accepted a foreign-context tensor");
  return events;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: transformer_ops_tests NEW_EVIDENCE_DIRECTORY");
    const fs::path evidence(argv[1]);
    require(!fs::exists(evidence), "Evidence directory must be new");
    Suite s;
    const auto artifact = t::operators_artifact();
    const auto module = paralyn::deserialize_executable(artifact.data(), artifact.size());
    require(module.producer == "paralyn.msl.tensor" && module.entries.size() == 10, "provider artifact identity");
    batched_matmul(s);
    reductions(s);
    softmax(s);
    layer_norm(s);
    gelu_and_add(s);
    gelu_specials(s);
    residual_add(s);
    negative(s);
    const unsigned activations = capabilities(s);
    row_limits(s);
    s.context.evidence(evidence.string());
    std::ifstream input(evidence / "execution.json");
    const auto execution = nlohmann::json::parse(input);
    require(execution["backend"] == "Metal" && execution["cpu_fallback"] == false, "wrong execution policy");
    const auto &launches = execution["launches"];
    require(launches.size() == s.events, "launch count " + std::to_string(launches.size()) +
                                             " differs from observed events " + std::to_string(s.events));
    std::map<std::string, unsigned> kernels;
    for (const auto &launch : launches) {
      require(launch["command_status"] == "completed" && launch["error"] == "", "failed launch");
      require(fs::is_regular_file(evidence / launch["source_file"].get<std::string>()), "missing source");
      ++kernels[launch["kernel"].get<std::string>()];
    }
    require(kernels == s.per_kernel, "per-kernel launch counts differ from observed events");
    const unsigned high = high_level();
    std::cout << "Transformer operators: " << s.events << " source-linked GPU commands (";
    bool first = true;
    for (const auto &[name, count] : kernels) {
      std::cout << (first ? "" : ", ") << name << " " << count;
      first = false;
    }
    std::cout << "), " << high << " high-level events\n";
    std::cout << "Capability record matches accepted activations (" << activations
              << " advertised, 61 rejected); row limit " << PR_TENSOR_ROW_OPERATOR_MAX_ROWS
              << " executed at the boundary and rejected one row above it\n";
    for (const auto &[op, ratio] : s.worst) std::cout << "worst error/bound " << op << ": " << ratio << '\n';
    std::cout << "Verification: PASS batched strided matmul, row sum/max, softmax, LayerNorm, GELU, residual add "
                 "(bitwise FP32 order references and float64 running-error bounds)\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Transformer operator qualification failed: " << e.what() << '\n';
    return 1;
  }
}
