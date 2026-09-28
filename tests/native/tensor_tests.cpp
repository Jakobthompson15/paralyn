// Real-GPU qualification of the FP32 tensor descriptor and paralyn.msl.tensor
// provider through the C ABI and C++ wrappers. Every numeric result is compared
// with independent CPU references: (1) a sequential FP32 reference that follows
// the documented accumulation order and must match bit-for-bit, and (2) a
// float64-accumulated reference with an a-priori componentwise error bound.
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
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace n = paralyn::native;
namespace t = paralyn::tensors;
namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
void status(pr_status actual, pr_status expected, const std::string &label,
            const char *operation = nullptr) {
  pr_error e{};
  pr_last_error(&e);
  require(actual == expected, label + ": expected status " + std::to_string(expected) + ", got " +
                                  std::to_string(actual) + " (" + e.operation + ": " + e.message + ")");
  if (operation && expected != PR_SUCCESS)
    require(!std::strcmp(e.operation, operation), label + ": wrong error operation " + e.operation);
}
struct Rng {
  std::uint32_t state;
  float next() { // Deterministic xorshift32; 16-bit dyadic values in [-1, 1).
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int(state >> 16) - 32768) / 32768.0f;
  }
};
std::vector<float> random(std::size_t count, std::uint32_t seed) {
  Rng r{seed};
  std::vector<float> v(count);
  for (auto &x : v) x = r.next();
  return v;
}
float at_a(const std::vector<float> &a, bool ta, std::uint64_t m, std::uint64_t k, std::uint64_t i,
           std::uint64_t p) {
  return ta ? a[p * m + i] : a[i * k + p];
}
float at_b(const std::vector<float> &b, bool tb, std::uint64_t n, std::uint64_t k, std::uint64_t p,
           std::uint64_t j) {
  return tb ? b[j * k + p] : b[p * n + j];
}
struct Reference {
  std::vector<float> sequential; // documented FP32 order, no contraction (host -ffp-contract=off)
  std::vector<double> exact, bound;
};
Reference reference(const std::vector<float> &a, const std::vector<float> &b, std::uint64_t m,
                    std::uint64_t n, std::uint64_t k, bool ta, bool tb) {
  Reference r;
  r.sequential.resize(m * n);
  r.exact.resize(m * n);
  r.bound.resize(m * n);
  const double u = std::ldexp(1.0, -24), gamma = k * u / (1 - k * u);
  for (std::uint64_t i = 0; i < m; ++i)
    for (std::uint64_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      double exact = 0, magnitude = 0;
      for (std::uint64_t p = 0; p < k; ++p) {
        const float x = at_a(a, ta, m, k, i, p), y = at_b(b, tb, n, k, p, j);
        const float product = x * y;
        sum = sum + product;
        exact += double(x) * double(y);
        magnitude += std::fabs(double(x) * double(y));
      }
      r.sequential[i * n + j] = sum;
      r.exact[i * n + j] = exact;
      r.bound[i * n + j] = gamma * magnitude;
    }
  return r;
}
bool same_bits(float x, float y) { return std::memcmp(&x, &y, 4) == 0; }

struct Suite {
  n::Context context;
  n::Queue queue;
  n::Module ops;
  unsigned events = 0;
  double worst_ratio = 0;
  Suite() : context("auto"), queue(context.queue()), ops(t::load_operators(context)) {}
  n::Buffer upload(const std::vector<float> &values, std::size_t pad_before = 0,
                   std::size_t pad_after = 0, float canary = -777.0f) {
    std::vector<float> all(pad_before, canary);
    all.insert(all.end(), values.begin(), values.end());
    all.insert(all.end(), pad_after, canary);
    auto buffer = context.buffer(all.size() * 4);
    buffer.write(all.data(), all.size() * 4);
    return buffer;
  }
  std::vector<float> read(const n::Buffer &b) {
    std::vector<float> out(b.size() / 4);
    b.read(out.data(), out.size() * 4);
    return out;
  }
  void completed(const std::optional<n::Event> &e, const std::string &label) {
    require(bool(e), label + ": missing GPU event");
    const auto timing = e->timing();
    require(timing.completed && timing.duration_valid && timing.timestamps_valid &&
                timing.clock_domain == PR_CLOCK_METAL_SYSTEM_MACH && timing.duration_seconds > 0 &&
                timing.start_seconds > 0 && timing.start_seconds < timing.end_seconds,
            label + ": event timing is not a completed physical GPU interval");
    const auto info = e->wait();
    require(info.completed && info.gpu_start_seconds == timing.start_seconds, label + ": event mismatch");
    ++events;
  }
  // One matmul with offsets/canaries, checked against both references.
  void matmul(std::uint64_t m, std::uint64_t nn, std::uint64_t k, bool ta, bool tb, std::uint32_t seed) {
    const std::string label = "matmul m=" + std::to_string(m) + " n=" + std::to_string(nn) +
                              " k=" + std::to_string(k) + " ta=" + std::to_string(ta) +
                              " tb=" + std::to_string(tb);
    const auto a = random(m * k, seed), b = random(k * nn, seed * 7 + 1);
    const std::size_t pad = 3; // 12-byte offsets exercise descriptor byte_offset.
    auto ab = upload(a, pad, 2), bb = upload(b, 1, 5);
    auto cb = upload(std::vector<float>(m * nn, 12345.0f), pad, 4);
    const auto ad = t::contiguous(ab, ta ? t::Shape{k, m} : t::Shape{m, k}, pad * 4);
    const auto bd = t::contiguous(bb, tb ? t::Shape{nn, k} : t::Shape{k, nn}, 4);
    const auto cd = t::contiguous(cb, {m, nn}, pad * 4);
    require(t::required_bytes(cd) == m * nn * 4, label + ": wrong required bytes");
    auto event = t::matmul(queue, ops, m, nn, k, ad, bd, cd, ta, tb);
    const auto out = read(cb);
    for (std::size_t i = 0; i < pad; ++i)
      require(out[i] == -777.0f && out[out.size() - 1 - i] == -777.0f, label + ": canary overwritten");
    if (!m || !nn) {
      require(!event, label + ": empty output fabricated a GPU event");
      return;
    }
    completed(event, label);
    const auto ref = reference(a, b, m, nn, k, ta, tb);
    for (std::uint64_t i = 0; i < m * nn; ++i) {
      const float gpu = out[pad + i];
      require(same_bits(gpu, ref.sequential[i]),
              label + ": GPU differs from sequential FP32 reference at " + std::to_string(i));
      const double error = std::fabs(double(gpu) - ref.exact[i]);
      require(error <= ref.bound[i], label + ": float64 error bound exceeded at " + std::to_string(i));
      if (ref.bound[i] > 0) worst_ratio = std::max(worst_ratio, error / ref.bound[i]);
    }
  }
};

pr_matmul_v1 record(std::uint64_t m, std::uint64_t nn, std::uint64_t k, const pr_tensor_desc_v1 *a,
                    const pr_tensor_desc_v1 *b, const pr_tensor_desc_v1 *c) {
  pr_matmul_v1 op{};
  op.struct_size = sizeof(op);
  op.version = PR_TENSOR_VERSION_1;
  op.m = m;
  op.n = nn;
  op.k = k;
  op.a = a;
  op.b = b;
  op.c = c;
  return op;
}
pr_bias_activation_v1 bias_record(const pr_tensor_desc_v1 *x, const pr_tensor_desc_v1 *bias,
                                  const pr_tensor_desc_v1 *out, pr_activation act) {
  pr_bias_activation_v1 op{};
  op.struct_size = sizeof(op);
  op.version = PR_TENSOR_VERSION_1;
  op.activation = act;
  op.x = x;
  op.bias = bias;
  op.out = out;
  return op;
}

void descriptors(Suite &s) {
  auto buffer = s.context.buffer(64);
  pr_tensor_desc_v1 d{};
  const std::uint64_t shape[] = {2, 3, 2};
  status(pr_tensor_desc_contiguous(buffer.get(), 8, PR_F32, 3, shape, &d), PR_SUCCESS, "contiguous");
  require(d.struct_size == sizeof(d) && d.version == 1 && d.strides[0] == 6 && d.strides[1] == 2 &&
              d.strides[2] == 1 && d.shape[3] == 0, "wrong contiguous strides");
  std::uint64_t bytes = 0;
  status(pr_tensor_desc_validate(&d, &bytes), PR_SUCCESS, "validate");
  require(bytes == 48, "wrong extent");
  auto bad = d;
  bad.version = 2;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_UNSUPPORTED, "unknown version", "tensor_desc_validate");
  bad = d;
  bad.struct_size -= 1;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_INVALID_ARGUMENT, "struct size");
  bad = d;
  bad.dtype = PR_I32;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_UNSUPPORTED, "dtype");
  bad = d;
  bad.rank = 9;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_INVALID_ARGUMENT, "rank");
  bad = d;
  bad.shape[5] = 1;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_INVALID_ARGUMENT, "unused dims");
  bad = d;
  bad.byte_offset = 6;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_INVALID_ARGUMENT, "misaligned offset");
  bad = d;
  bad.byte_offset = 20;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_OUT_OF_BOUNDS, "extent beyond buffer");
  bad = d;
  bad.strides[1] = -2;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_UNSUPPORTED, "negative stride");
  bad = d;
  bad.shape[0] = std::uint64_t(1) << 40;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_UNSUPPORTED, "element limit");
  bad = d;
  bad.buffer = 0;
  status(pr_tensor_desc_validate(&bad, nullptr), PR_INVALID_HANDLE, "buffer handle");
  status(pr_tensor_desc_validate(nullptr, nullptr), PR_INVALID_ARGUMENT, "null");
  // A validated non-contiguous (column-major) layout is legal but operators reject it.
  auto strided = d;
  strided.rank = 2;
  strided.shape[0] = 2;
  strided.shape[1] = 3;
  strided.shape[2] = 0;
  strided.strides[0] = 1;
  strided.strides[1] = 2;
  strided.strides[2] = 0;
  status(pr_tensor_desc_validate(&strided, &bytes), PR_SUCCESS, "strided validate");
  require(bytes == 24, "wrong strided extent");
  const std::uint64_t empty_shape[] = {4, 0};
  status(pr_tensor_desc_contiguous(buffer.get(), 64, PR_F32, 2, empty_shape, &d), PR_SUCCESS, "empty");
  status(pr_tensor_desc_validate(&d, &bytes), PR_SUCCESS, "empty validate");
  require(bytes == 0, "empty extent is not zero");
  status(pr_tensor_desc_contiguous(buffer.get(), 0, PR_F32, 9, shape, &d), PR_INVALID_ARGUMENT, "rank 9");
  // Artifact query.
  std::uint64_t size = 0;
  status(pr_tensor_operators_artifact(nullptr, 0, &size), PR_SUCCESS, "artifact size");
  std::vector<unsigned char> artifact(size);
  status(pr_tensor_operators_artifact(artifact.data(), size - 1, &size), PR_OUT_OF_BOUNDS, "capacity");
  const auto bytes_copy = t::operators_artifact();
  require(bytes_copy.size() == size && !std::memcmp(bytes_copy.data(), "PARALYNX", 8), "artifact magic");
  const auto module = paralyn::deserialize_executable(bytes_copy.data(), bytes_copy.size());
  require(module.producer == "paralyn.msl.tensor" && module.numerical_policy == 1 &&
              module.entries.size() == 4 && module.entries[0].name == "paralyn_matmul_f32",
          "provider artifact identity");
}

void errors(Suite &s) {
  auto ab = s.upload(random(6, 3)), bb = s.upload(random(12, 4)), cb = s.upload(std::vector<float>(8));
  auto a = t::contiguous(ab, {2, 3}), b = t::contiguous(bb, {3, 4}), c = t::contiguous(cb, {2, 4});
  pr_event e = 99;
  auto run = [&](pr_matmul_v1 op, pr_status expected, const std::string &label,
                 pr_queue q = 0, pr_module m = 0) {
    e = 99;
    status(pr_matmul_f32(q ? q : s.queue.get(), m ? m : s.ops.get(), &op, &e), expected, label,
           "matmul_f32");
    require(e == 0 || expected == PR_SUCCESS, label + ": event output not cleared");
  };
  run(record(2, 4, 3, &a, &b, &c), PR_SUCCESS, "valid");
  n::Event valid(e);
  valid.wait();
  run(record(3, 4, 3, &a, &b, &c), PR_INVALID_ARGUMENT, "m mismatch");
  run(record(2, 4, 2, &a, &b, &c), PR_INVALID_ARGUMENT, "k mismatch");
  run(record(2, 5, 3, &a, &b, &c), PR_INVALID_ARGUMENT, "n mismatch");
  auto op = record(2, 4, 3, &a, &b, &c);
  op.transpose_a = 1; // A would need [3,2] storage.
  run(op, PR_INVALID_ARGUMENT, "transpose shape mismatch");
  op = record(2, 4, 3, &a, &b, &c);
  op.transpose_b = 2;
  run(op, PR_INVALID_ARGUMENT, "transpose flag");
  op = record(2, 4, 3, &a, &b, nullptr);
  run(op, PR_INVALID_ARGUMENT, "missing C");
  op = record(2, 4, 3, &a, &b, &c);
  op.version = 3;
  run(op, PR_UNSUPPORTED, "record version");
  op = record(2, 4, 3, &a, &b, &c);
  op.struct_size = 8;
  run(op, PR_INVALID_ARGUMENT, "record size");
  auto rank3 = t::contiguous(ab, {1, 2, 3});
  run(record(2, 4, 3, &rank3, &b, &c), PR_INVALID_ARGUMENT, "rank 3 input");
  auto column_major = a;
  column_major.strides[0] = 1;
  column_major.strides[1] = 2;
  run(record(2, 4, 3, &column_major, &b, &c), PR_UNSUPPORTED, "non-contiguous input");
  auto small = t::contiguous(cb, {2, 4}, 4);
  run(record(2, 4, 3, &a, &b, &small), PR_OUT_OF_BOUNDS, "output extent");
  status(pr_matmul_f32(s.queue.get(), s.ops.get(), nullptr, &e), PR_INVALID_ARGUMENT, "null record");
  auto valid_op = record(2, 4, 3, &a, &b, &c);
  status(pr_matmul_f32(s.queue.get(), s.ops.get(), &valid_op, nullptr), PR_INVALID_ARGUMENT, "null event");
  run(valid_op, PR_INVALID_HANDLE, "module handle", s.queue.get(), 0xffffffffull);
  // Alias rules: output overlapping an input is rejected; disjoint same allocation
  // is outside the MSL profile; read-only input aliasing is valid (tested below).
  auto shared = s.upload(random(16, 5));
  auto a_in = t::contiguous(shared, {2, 3}), c_over = t::contiguous(shared, {2, 4}, 8);
  run(record(2, 4, 3, &a_in, &b, &c_over), PR_INVALID_ARGUMENT, "overlapping output");
  auto c_disjoint = t::contiguous(shared, {2, 4}, 32);
  run(record(2, 4, 3, &a_in, &b, &c_disjoint), PR_UNSUPPORTED, "same-allocation output");
  // Wrong provider module: same entry name, different resource schema.
  paralyn::ExecutableModule fake;
  fake.producer = "test";
  fake.producer_version = "1";
  fake.source_name = "fake.metal";
  fake.source = "#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void paralyn_matmul_f32(device float* out [[buffer(0)]],"
                " uint i [[thread_position_in_grid]]) { out[i] = 1.0f; }\n";
  fake.source_sha256 = paralyn::source_sha256(fake.source);
  fake.entries.push_back({"paralyn_matmul_f32",
      {{"out", paralyn::ScalarType::F32, true, paralyn::ResourceAccess::ReadWrite, 0, 4, 4}}, {0, 0, 0}});
  const auto fake_bytes = paralyn::serialize_executable(fake);
  pr_module fake_module = 0;
  status(pr_module_load(s.context.get(), fake_bytes.data(), fake_bytes.size(), &fake_module), PR_SUCCESS,
         "fake module");
  n::Module fake_owner(fake_module);
  run(valid_op, PR_INVALID_ARGUMENT, "wrong provider module", s.queue.get(), fake_module);
  // Context mismatch is detected by the public launch contract.
  n::Context other("auto");
  auto other_queue = other.queue();
  run(valid_op, PR_CONTEXT_MISMATCH, "foreign queue", other_queue.get());
  auto other_ops = t::load_operators(other);
  run(valid_op, PR_CONTEXT_MISMATCH, "foreign module", s.queue.get(), other_ops.get());
  // Bias/activation errors.
  auto bias_buffer = s.upload(random(4, 6));
  auto bias = t::contiguous(bias_buffer, {4}), wrong_bias = t::contiguous(bias_buffer, {3});
  auto out = s.upload(std::vector<float>(8));
  auto od = t::contiguous(out, {2, 4});
  auto brun = [&](pr_bias_activation_v1 bop, pr_status expected, const std::string &label) {
    e = 99;
    status(pr_bias_activation_f32(s.queue.get(), s.ops.get(), &bop, &e), expected, label,
           "bias_activation_f32");
    if (expected == PR_SUCCESS && e) n::Event(e).wait();
  };
  brun(bias_record(&c, &bias, &od, PR_ACTIVATION_RELU), PR_SUCCESS, "valid bias");
  brun(bias_record(&c, &wrong_bias, &od, PR_ACTIVATION_RELU), PR_INVALID_ARGUMENT, "bias shape");
  brun(bias_record(&c, &bias, &c, PR_ACTIVATION_RELU), PR_INVALID_ARGUMENT, "in-place bias");
  brun(bias_record(&c, &bias, &od, static_cast<pr_activation>(7)), PR_INVALID_ARGUMENT, "activation");
  auto reserved = bias_record(&c, &bias, &od, PR_ACTIVATION_NONE);
  reserved.reserved = 1;
  brun(reserved, PR_INVALID_ARGUMENT, "reserved");
  auto od_wrong = t::contiguous(out, {4, 2});
  brun(bias_record(&c, &bias, &od_wrong, PR_ACTIVATION_NONE), PR_INVALID_ARGUMENT, "output shape");
  s.events += 2; // the valid matmul and valid bias commands above
}

void alias_and_bias(Suite &s) {
  // Read-only input aliasing: G = A * A^T with one allocation bound twice.
  const std::uint64_t m = 19, k = 23;
  const auto a = random(m * k, 42);
  auto ab = s.upload(a);
  auto gb = s.upload(std::vector<float>(m * m));
  const auto ad = t::contiguous(ab, {m, k}), gd = t::contiguous(gb, {m, m});
  auto event = t::matmul(s.queue, s.ops, m, m, k, ad, ad, gd, false, true);
  s.completed(event, "A*A^T alias");
  const auto ref = reference(a, a, m, m, k, false, true);
  const auto g = s.read(gb);
  for (std::size_t i = 0; i < g.size(); ++i)
    require(same_bits(g[i], ref.sequential[i]), "A*A^T alias result differs");
  // Bias/ReLU exceptional values: NaN propagates, -0.0 is kept, negatives become +0.0.
  const float inf = std::numeric_limits<float>::infinity(), nan = std::nanf("");
  const std::vector<float> x{-1.5f, 0.0f, -0.0f, 2.25f, nan, -inf, inf, 3.0f, -3.0f, 0.5f, 1e-3f, -7.0f};
  const std::vector<float> bias{0.5f, -0.0f, 0.0f, -2.25f, 1.0f, 0.0f};
  auto xb = s.upload(x), bb = s.upload(bias), ob = s.upload(std::vector<float>(12, 9.0f));
  const auto xd = t::contiguous(xb, {2, 6}), bd = t::contiguous(bb, {6}), od = t::contiguous(ob, {2, 6});
  for (int mode = 0; mode < 4; ++mode) {
    const bool relu = mode & 1, with_bias = mode & 2;
    auto e = t::bias_activation(s.queue, s.ops, xd, with_bias ? &bd : nullptr, od,
                                relu ? PR_ACTIVATION_RELU : PR_ACTIVATION_NONE);
    s.completed(e, "bias/activation mode " + std::to_string(mode));
    const auto out = s.read(ob);
    for (std::size_t i = 0; i < x.size(); ++i) {
      float expected = with_bias ? x[i] + bias[i % 6] : x[i];
      if (relu && expected < 0.0f) expected = 0.0f;
      require(same_bits(out[i], expected) || (std::isnan(out[i]) && std::isnan(expected)),
              "bias/activation mismatch at mode " + std::to_string(mode) + " index " + std::to_string(i));
    }
  }
  // Empty bias/activation: no event.
  auto empty_in = s.context.buffer(0), empty_out = s.context.buffer(0);
  const auto ed = t::contiguous(empty_in, {0, 6}), eo = t::contiguous(empty_out, {0, 6});
  auto none = t::bias_activation(s.queue, s.ops, ed, &bd, eo, PR_ACTIVATION_RELU);
  require(!none, "empty bias/activation fabricated an event");
}

void high_level(Suite &s, unsigned &events) {
  t::Context ctx("auto");
  const auto caps = ctx.capabilities();
  require(std::string(caps.stable_id).rfind("metal:registry:", 0) == 0, "C++ capability wrapper");
  const auto caps0 = n::device_capabilities(0);
  require(caps0.struct_size == sizeof(caps0) && caps0.max_buffer_bindings == 31, "device capabilities");
  const std::uint64_t m = 5, k = 7, nn = 3;
  const auto a = random(m * k, 11), w = random(k * nn, 12), bias = random(nn, 13);
  auto ta = t::from_host(ctx, {m, k}, a), tw = t::from_host(ctx, {k, nn}, w), tb = t::from_host(ctx, {nn}, bias);
  require(!ta.wait() && !ta.timing(), "upload fabricated an event");
  auto y = t::matmul(ta, tw);
  auto z = t::bias_add(y, tb, true);
  auto r = t::relu(z);
  auto wt = t::from_host(ctx, {nn, k}, [&] {
    std::vector<float> v(nn * k);
    for (std::uint64_t i = 0; i < k; ++i)
      for (std::uint64_t j = 0; j < nn; ++j) v[j * k + i] = w[i * nn + j];
    return v;
  }());
  auto y2 = t::matmul(ta, wt, false, true);
  ta.close();
  tw.close();
  const auto ref = reference(a, w, m, nn, k, false, false);
  const auto yh = y.to_host(), zh = z.to_host(), rh = r.to_host(), y2h = y2.to_host();
  for (std::uint64_t i = 0; i < m * nn; ++i) {
    float expected = ref.sequential[i] + bias[i % nn];
    if (expected < 0.0f) expected = 0.0f;
    require(same_bits(yh[i], ref.sequential[i]) && same_bits(y2h[i], ref.sequential[i]) &&
                same_bits(zh[i], expected) && same_bits(rh[i], expected),
            "high-level tensor chain differs from CPU reference");
  }
  for (const auto *tensor : {&y, &z, &r, &y2}) {
    const auto timing = tensor->timing();
    require(timing && timing->completed && timing->duration_valid, "high-level event timing");
    ++events;
  }
  require(y.shape() == t::Shape({m, nn}) && y.nbytes() == m * nn * 4, "high-level metadata");
  bool rejected = false;
  try { t::matmul(y, y); } catch (const std::exception &) { rejected = true; }
  require(rejected, "inner-dimension mismatch accepted");
  rejected = false;
  try { t::from_host(ctx, {2, 2}, {1, 2, 3}); } catch (const std::exception &) { rejected = true; }
  require(rejected, "shape/value mismatch accepted");
  auto zero = t::matmul(t::from_host(ctx, {0, 4}, {}), t::from_host(ctx, {4, 2}, std::vector<float>(8)));
  require(zero.shape() == t::Shape({0, 2}) && !zero.wait(), "empty high-level matmul event");
  ctx.close();
  require(rh.size() == m * nn && r.to_host() == rh, "retained result lost after context close");
  rejected = false;
  try { t::relu(r); } catch (const std::exception &) { rejected = true; }
  require(rejected, "closed context accepted work");
  (void)s;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: tensor_tests NEW_EVIDENCE_DIRECTORY");
    const fs::path evidence(argv[1]);
    require(!fs::exists(evidence), "Evidence directory must be new");
    Suite s;
    descriptors(s);
    const std::uint64_t shapes[][3] = {{1, 1, 1},   {1, 7, 1},    {5, 1, 9},   {1, 1, 300}, {16, 16, 16},
                                       {17, 33, 15}, {31, 17, 47}, {64, 48, 80}, {129, 65, 257},
                                       {3, 5, 0},   {0, 5, 3},    {5, 0, 3},   {0, 0, 0}};
    std::uint32_t seed = 1;
    for (const auto &shape : shapes)
      for (int flags = 0; flags < 4; ++flags)
        s.matmul(shape[0], shape[1], shape[2], flags & 1, flags & 2, seed++);
    errors(s);
    alias_and_bias(s);
    s.context.evidence(evidence.string());
    std::ifstream input(evidence / "execution.json");
    const auto execution = nlohmann::json::parse(input);
    require(execution["backend"] == "Metal" && execution["cpu_fallback"] == false, "wrong execution policy");
    const auto &launches = execution["launches"];
    require(launches.size() == s.events, "launch count " + std::to_string(launches.size()) +
                                             " differs from observed events " + std::to_string(s.events));
    unsigned matmuls = 0;
    for (const auto &launch : launches) {
      require(launch["command_status"] == "completed" && launch["error"] == "", "failed launch");
      require(fs::is_regular_file(evidence / launch["source_file"].get<std::string>()), "missing source");
      matmuls += launch["kernel"] == "paralyn_matmul_f32";
    }
    unsigned high = 0;
    high_level(s, high);
    std::cout << "Tensor provider: " << s.events << " source-linked GPU commands (" << matmuls
              << " tiled matmul), " << high << " high-level events; worst error/bound ratio "
              << s.worst_ratio << '\n';
    std::cout << "Verification: PASS FP32 tensor descriptors, matmul, bias/ReLU (bitwise sequential "
                 "FP32 reference and float64 bound)\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Tensor qualification failed: " << e.what() << '\n';
    return 1;
  }
}
