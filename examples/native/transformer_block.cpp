// One pre-LN transformer encoder block, FP32 inference, every stage on the GPU:
//   ln1  = LayerNorm(x)                          layer_norm
//   qkv  = ln1 Wqkv + bqkv                       matmul, bias_activation
//   S_h  = Q_h K_h^T        (strided head views) batched_matmul
//   P_h  = softmax(S_h / sqrt(d_head), causal)   softmax_rows
//   A    = concat_h(P_h V_h) (strided output)    batched_matmul
//   h1   = x + (A Wo + bo)                       matmul, bias_activation, add
//   ln2  = LayerNorm(h1)                         layer_norm
//   f    = gelu_tanh(ln2 W1 + b1) W2 + b2        matmul, bias_activation x2, matmul
//   y    = h1 + f                                add
// 15 launches through the paralyn.msl.tensor provider; the host only uploads
// seeded inputs/weights and reads results back. There is no CPU fallback. All
// audited stages are verified against an independent float64 reference with
// a-priori running error bounds (examples/native/transformer_reference.hpp).
//
// This is not PyTorch integration, not cuDNN/cuBLAS compatibility and not training.
#include "transformer_reference.hpp"
#include <paralyn/executable.hpp> // SHA-256 (paralyn_ir) for report hashes
#include <paralyn/tensor.hpp>
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace n = paralyn::native;
namespace t = paralyn::tensors;
namespace r = paralyn_reference;
namespace {
using Matrix = std::vector<r::B>; // row-major bounded reference values

struct Rng { // xorshift32; identical sequence to transformer_block.py
  std::uint32_t state;
  float next(float scale) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int(state >> 16) - 32768) / 32768.0f * scale;
  }
};
std::vector<float> values(std::size_t count, Rng &rng, float scale, float offset = 0.0f) {
  std::vector<float> v(count);
  for (auto &x : v) x = rng.next(scale) + offset;
  return v;
}
std::string sha256(const std::vector<float> &v) {
  return paralyn::source_sha256(std::string(reinterpret_cast<const char *>(v.data()), v.size() * 4));
}
std::uint64_t argument(int &i, int argc, char **argv) {
  if (++i >= argc) throw std::runtime_error("missing option value");
  return std::stoull(argv[i]);
}
Matrix bounded(const std::vector<float> &v) { return r::exact_values(v); }
// out[t, j] = sum_p in[t, p] W[p, j] (+ bias[j]) with the provider's FP32 dot order.
Matrix linear(const Matrix &in, std::uint64_t rows, std::uint64_t k, const Matrix &w, std::uint64_t cols,
              const Matrix &bias) {
  Matrix out(rows * cols);
  std::vector<r::B> a(k), b(k);
  for (std::uint64_t i = 0; i < rows; ++i)
    for (std::uint64_t j = 0; j < cols; ++j) {
      for (std::uint64_t p = 0; p < k; ++p) {
        a[p] = in[i * k + p];
        b[p] = w[p * cols + j];
      }
      out[i * cols + j] = r::add(r::dot(a, b), bias[j]);
    }
  return out;
}
Matrix layer_norm(const Matrix &x, std::uint64_t rows, std::uint64_t cols, const Matrix &g, const Matrix &b,
                  double eps) {
  Matrix out;
  for (std::uint64_t i = 0; i < rows; ++i) {
    const auto row = r::layer_norm_row(Matrix(x.begin() + i * cols, x.begin() + (i + 1) * cols), g, b, eps);
    out.insert(out.end(), row.begin(), row.end());
  }
  return out;
}
struct Check {
  double worst = 0, max_abs = 0;
};
Check verify(const std::string &stage, const std::vector<float> &gpu, const Matrix &ref) {
  if (gpu.size() != ref.size()) throw std::runtime_error(stage + ": size mismatch");
  Check c;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    if (!r::within(gpu[i], ref[i]))
      throw std::runtime_error(stage + ": GPU value " + std::to_string(gpu[i]) + " at " + std::to_string(i) +
                               " differs from float64 reference " + std::to_string(ref[i].v) + " beyond bound " +
                               std::to_string(ref[i].e));
    c.worst = std::max(c.worst, r::ratio(gpu[i], ref[i]));
    c.max_abs = std::max(c.max_abs, std::fabs(double(gpu[i]) - ref[i].v));
  }
  return c;
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::uint64_t tokens = 24, model = 48, heads = 4, ff = 192, seed = 2026;
    bool causal = true;
    std::string artifacts;
    if (const char *env = std::getenv("PARALYN_ARTIFACT_DIR")) artifacts = env;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--artifacts" && i + 1 < argc) artifacts = argv[++i];
      else if (option == "--tokens") tokens = argument(i, argc, argv);
      else if (option == "--model") model = argument(i, argc, argv);
      else if (option == "--heads") heads = argument(i, argc, argv);
      else if (option == "--ff") ff = argument(i, argc, argv);
      else if (option == "--seed") seed = argument(i, argc, argv);
      else if (option == "--no-causal") causal = false;
      else throw std::runtime_error("usage: native_transformer_block [--artifacts DIR] [--tokens T] [--model D] "
                                    "[--heads H] [--ff F] [--seed S] [--no-causal]");
    }
    if (!seed || seed > 0xffffffffu) throw std::runtime_error("seed must be in [1, 2^32-1]");
    if (!tokens || !model || !heads || !ff || model % heads)
      throw std::runtime_error("tokens, model, heads, ff must be positive and heads must divide model");
    const std::uint64_t T = tokens, D = model, H = heads, Dh = D / H, F = ff, W3 = 3 * D;
    const float eps = 1e-5f;
    const float scale = float(1.0 / std::sqrt(double(Dh))); // FP32 scale passed to softmax

    // Seeded weights; dyadic values scaled by powers of two are exact in FP32 (same order as .py).
    Rng rng{static_cast<std::uint32_t>(seed)};
    const auto x = values(T * D, rng, 1.0f), g1 = values(D, rng, 0.25f, 1.0f), be1 = values(D, rng, 0.25f),
               wqkv = values(D * W3, rng, 0.125f), bqkv = values(W3, rng, 0.25f), wo = values(D * D, rng, 0.125f),
               bo = values(D, rng, 0.25f), g2 = values(D, rng, 0.25f, 1.0f), be2 = values(D, rng, 0.25f),
               w1 = values(D * F, rng, 0.125f), b1 = values(F, rng, 0.25f), w2 = values(F * D, rng, 0.0625f),
               b2 = values(D, rng, 0.25f);

    // ---- GPU: every stage is one provider launch on the context's ordered queue.
    n::Context context("auto");
    auto queue = context.queue();
    auto ops = t::load_operators(context);
    auto upload = [&](const std::vector<float> &v) {
      auto b = context.buffer(v.size() * 4);
      b.write(v.data(), v.size() * 4);
      return b;
    };
    auto scratch = [&](std::uint64_t count) { return context.buffer(count * 4); };
    auto bx = upload(x), bg1 = upload(g1), bbe1 = upload(be1), bwqkv = upload(wqkv), bbqkv = upload(bqkv),
         bwo = upload(wo), bbo = upload(bo), bg2 = upload(g2), bbe2 = upload(be2), bw1 = upload(w1),
         bb1 = upload(b1), bw2 = upload(w2), bb2 = upload(b2);
    auto ln1 = scratch(T * D), qkv_raw = scratch(T * W3), qkv = scratch(T * W3), scores = scratch(H * T * T),
         probs = scratch(H * T * T), attn = scratch(T * D), proj_raw = scratch(T * D), proj = scratch(T * D),
         h1 = scratch(T * D), ln2 = scratch(T * D), f1_raw = scratch(T * F), f1 = scratch(T * F),
         f2_raw = scratch(T * D), f2 = scratch(T * D), y = scratch(T * D);
    auto c2 = [](const n::Buffer &b, std::uint64_t r0, std::uint64_t c0) { return t::contiguous(b, {r0, c0}); };
    auto c1 = [](const n::Buffer &b, std::uint64_t c0) { return t::contiguous(b, {c0}); };
    std::vector<std::pair<std::string, n::Event>> events;
    auto keep = [&](const char *stage, std::optional<n::Event> e) {
      if (!e) throw std::runtime_error(std::string(stage) + ": no GPU event for nonempty work");
      events.emplace_back(stage, std::move(*e));
    };
    const std::int64_t sD = std::int64_t(D), sDh = std::int64_t(Dh), sW3 = std::int64_t(W3);
    keep("ln1", t::layer_norm(queue, ops, c2(bx, T, D), c1(bg1, D), c1(bbe1, D), c2(ln1, T, D), eps));
    keep("qkv_matmul", t::matmul(queue, ops, T, W3, D, c2(ln1, T, D), c2(bwqkv, D, W3), c2(qkv_raw, T, W3)));
    const auto bqkv_d = c1(bbqkv, W3);
    keep("qkv_bias", t::bias_activation(queue, ops, c2(qkv_raw, T, W3), &bqkv_d, c2(qkv, T, W3)));
    // Head views into qkv [T, 3D]: Q_h = [H,T,Dh] strides [Dh,3D,1]; K_h^T = [H,Dh,T]
    // strides [Dh,1,3D] at column D; V_h = [H,T,Dh] strides [Dh,3D,1] at column 2D.
    const auto q_view = t::strided(qkv, {H, T, Dh}, {sDh, sW3, 1}, 0);
    const auto kt_view = t::strided(qkv, {H, Dh, T}, {sDh, 1, sW3}, D * 4);
    const auto v_view = t::strided(qkv, {H, T, Dh}, {sDh, sW3, 1}, 2 * D * 4);
    keep("scores", t::batched_matmul(queue, ops, H, T, T, Dh, q_view, kt_view, t::contiguous(scores, {H, T, T})));
    keep("softmax", t::softmax_rows(queue, ops, t::contiguous(scores, {H, T, T}), t::contiguous(probs, {H, T, T}),
                                    scale, causal));
    // A[t, h*Dh + c] = sum_s P_h[t, s] V_h[s, c]: strided output writes concatenated heads.
    keep("attention", t::batched_matmul(queue, ops, H, T, Dh, T, t::contiguous(probs, {H, T, T}), v_view,
                                        t::strided(attn, {H, T, Dh}, {sDh, sD, 1}, 0)));
    keep("out_matmul", t::matmul(queue, ops, T, D, D, c2(attn, T, D), c2(bwo, D, D), c2(proj_raw, T, D)));
    const auto bo_d = c1(bbo, D), b1_d = c1(bb1, F), b2_d = c1(bb2, D);
    keep("out_bias", t::bias_activation(queue, ops, c2(proj_raw, T, D), &bo_d, c2(proj, T, D)));
    keep("residual1", t::add(queue, ops, c2(bx, T, D), c2(proj, T, D), c2(h1, T, D)));
    keep("ln2", t::layer_norm(queue, ops, c2(h1, T, D), c1(bg2, D), c1(bbe2, D), c2(ln2, T, D), eps));
    keep("ff1_matmul", t::matmul(queue, ops, T, F, D, c2(ln2, T, D), c2(bw1, D, F), c2(f1_raw, T, F)));
    keep("ff1_bias_gelu", t::bias_activation(queue, ops, c2(f1_raw, T, F), &b1_d, c2(f1, T, F),
                                             PR_ACTIVATION_GELU_TANH));
    keep("ff2_matmul", t::matmul(queue, ops, T, D, F, c2(f1, T, F), c2(bw2, F, D), c2(f2_raw, T, D)));
    keep("ff2_bias", t::bias_activation(queue, ops, c2(f2_raw, T, D), &b2_d, c2(f2, T, D)));
    keep("residual2", t::add(queue, ops, c2(h1, T, D), c2(f2, T, D), c2(y, T, D)));
    nlohmann::json stages = nlohmann::json::array();
    for (auto &[stage, event] : events) {
      const auto timing = event.timing();
      if (!timing.completed || !timing.duration_valid || timing.duration_seconds <= 0)
        throw std::runtime_error(stage + ": GPU event did not complete with a valid duration");
      stages.push_back({{"stage", stage}, {"gpu_duration_seconds", timing.duration_seconds}});
    }
    auto read = [](const n::Buffer &b) {
      std::vector<float> v(b.size() / 4);
      b.read(v.data(), b.size());
      return v;
    };
    const std::map<std::string, std::vector<float>> gpu{
        {"ln1", read(ln1)},   {"qkv", read(qkv)},   {"scores", read(scores)}, {"probs", read(probs)},
        {"attention", read(attn)}, {"projection", read(proj)}, {"h1", read(h1)}, {"ln2", read(ln2)},
        {"ff1", read(f1)},    {"ff2", read(f2)},    {"output", read(y)}};
    const auto device = context.device();
    if (!artifacts.empty()) {
      context.evidence(artifacts);
      std::ifstream input(std::filesystem::path(artifacts) / "execution.json");
      const auto execution = nlohmann::json::parse(input);
      if (execution["launches"].size() != events.size() || execution["cpu_fallback"] != false)
        throw std::runtime_error("exported evidence does not match the observed GPU launches");
    }

    // ---- Independent float64 reference with running error bounds (CPU, verification only).
    const auto G1 = bounded(g1), BE1 = bounded(be1), Wqkv = bounded(wqkv), Bqkv = bounded(bqkv), Wo = bounded(wo),
               Bo = bounded(bo), G2 = bounded(g2), BE2 = bounded(be2), W1 = bounded(w1), B1 = bounded(b1),
               W2 = bounded(w2), B2 = bounded(b2), X = bounded(x);
    const auto rLn1 = layer_norm(X, T, D, G1, BE1, double(eps));
    const auto rQkv = linear(rLn1, T, D, Wqkv, W3, Bqkv);
    Matrix rScores(H * T * T), rProbs(H * T * T), rAttn(T * D);
    for (std::uint64_t h = 0; h < H; ++h)
      for (std::uint64_t i = 0; i < T; ++i) {
        std::vector<r::B> a(Dh), b(Dh);
        for (std::uint64_t j = 0; j < T; ++j) {
          for (std::uint64_t c = 0; c < Dh; ++c) {
            a[c] = rQkv[i * W3 + h * Dh + c];
            b[c] = rQkv[j * W3 + D + h * Dh + c];
          }
          rScores[(h * T + i) * T + j] = r::dot(a, b);
        }
        const auto row = r::softmax_row(Matrix(rScores.begin() + (h * T + i) * T, rScores.begin() + (h * T + i + 1) * T),
                                        double(scale), causal ? i + 1 : T);
        std::copy(row.begin(), row.end(), rProbs.begin() + (h * T + i) * T);
      }
    for (std::uint64_t h = 0; h < H; ++h)
      for (std::uint64_t i = 0; i < T; ++i)
        for (std::uint64_t c = 0; c < Dh; ++c) {
          std::vector<r::B> a(T), b(T);
          for (std::uint64_t s = 0; s < T; ++s) {
            a[s] = rProbs[(h * T + i) * T + s];
            b[s] = rQkv[s * W3 + 2 * D + h * Dh + c];
          }
          rAttn[i * D + h * Dh + c] = r::dot(a, b);
        }
    const auto rProj = linear(rAttn, T, D, Wo, D, Bo);
    Matrix rH1(T * D);
    for (std::uint64_t i = 0; i < T * D; ++i) rH1[i] = r::add(X[i], rProj[i]);
    const auto rLn2 = layer_norm(rH1, T, D, G2, BE2, double(eps));
    auto rF1 = linear(rLn2, T, D, W1, F, B1);
    for (auto &v : rF1) v = r::gelu(v);
    const auto rF2 = linear(rF1, T, F, W2, D, B2);
    Matrix rY(T * D);
    for (std::uint64_t i = 0; i < T * D; ++i) rY[i] = r::add(rH1[i], rF2[i]);
    const std::map<std::string, const Matrix *> refs{
        {"ln1", &rLn1},   {"qkv", &rQkv}, {"scores", &rScores}, {"probs", &rProbs}, {"attention", &rAttn},
        {"projection", &rProj}, {"h1", &rH1}, {"ln2", &rLn2}, {"ff1", &rF1}, {"ff2", &rF2}, {"output", &rY}};
    nlohmann::json checks = nlohmann::json::object(), hashes = nlohmann::json::object();
    double worst = 0, max_abs = 0;
    for (const auto &[stage, values] : gpu) {
      const auto c = verify(stage, values, *refs.at(stage));
      checks[stage] = {{"worst_error_to_bound", c.worst}, {"max_abs_error", c.max_abs}};
      hashes[stage] = sha256(values);
      worst = std::max(worst, c.worst);
      if (stage == "output") max_abs = c.max_abs;
    }

    std::cout << "Device: " << device.name << "; provider paralyn.msl.tensor; pre-LN encoder block T=" << T
              << " d_model=" << D << " heads=" << H << " d_head=" << Dh << " d_ff=" << F << " causal="
              << (causal ? "yes" : "no") << " seed " << seed << '\n';
    for (const auto &s : stages)
      std::cout << "GPU " << s["stage"].get<std::string>() << ": " << std::fixed << std::setprecision(3)
                << s["gpu_duration_seconds"].get<double>() * 1e6 << " us\n";
    std::cout << std::scientific << std::setprecision(3) << "Output max |error| vs float64 reference: " << max_abs
              << std::fixed << "; worst error/bound over " << gpu.size() << " audited stages: " << worst << '\n';
    if (!artifacts.empty()) {
      const auto artifact = t::operators_artifact();
      nlohmann::json report{
          {"application", "pre-LN transformer encoder block, FP32 inference (C++)"},
          {"provider", "paralyn.msl.tensor"},
          {"provider_artifact_sha256",
           paralyn::source_sha256(std::string(artifact.begin(), artifact.end()))},
          {"shapes", {{"tokens", T}, {"model", D}, {"heads", H}, {"head", Dh}, {"ff", F}}},
          {"seed", seed},
          {"causal", causal},
          {"epsilon", double(eps)},
          {"softmax_scale", double(scale)},
          {"gelu", "tanh approximation, FP32 constants"},
          {"gpu_commands", events.size()},
          {"stages", stages},
          {"reference", "float64 from exact FP32 inputs with a-priori running error bounds"},
          {"tolerance", "docs/transformer-operators.md"},
          {"checks", checks},
          {"worst_error_to_bound", worst},
          {"max_abs_error_output", max_abs},
          {"cpu_fallback", false},
          {"stage_sha256", hashes},
          {"output_sha256", hashes["output"]}};
      std::ofstream(std::filesystem::path(artifacts) / "transformer-report.json") << report.dump(2) << '\n';
    }
    std::cout << "Verification: PASS pre-LN transformer block (" << gpu.size() << " stages, " << T * D
              << " outputs within float64 running-error bounds; " << events.size() << " GPU commands)\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Transformer block failed: " << e.what() << '\n';
    return 1;
  }
}
