// Two-layer FP32 MLP inference on the GPU: y = relu(x W1 + b1) W2 + b2.
// Every layer (two matmuls, two bias/activation passes) runs through the
// paralyn.msl.tensor provider on the native runtime; there is no CPU fallback.
// The result is compared with an independent float64-accumulated CPU reference
// under the a-priori error bound documented in docs/tensors-matmul.md.
#include <paralyn/executable.hpp> // SHA-256 (paralyn_ir) for report hashes
#include <paralyn/tensor.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace t = paralyn::tensors;
namespace {
struct Rng { // xorshift32; identical sequence to examples/native/mlp.py
  std::uint32_t state;
  float next(float scale) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(int(state >> 16) - 32768) / 32768.0f * scale;
  }
};
std::vector<float> values(std::size_t count, Rng &rng, float scale) {
  std::vector<float> v(count);
  for (auto &x : v) x = rng.next(scale);
  return v;
}
// Hex SHA-256 of the exact little-endian FP32 bytes, as Python's array('f').tobytes().
std::string sha256(const void *data, std::size_t bytes) {
  return paralyn::source_sha256(std::string(static_cast<const char *>(data), bytes));
}
std::uint64_t argument(int &i, int argc, char **argv) {
  if (++i >= argc) throw std::runtime_error("missing option value");
  return std::stoull(argv[i]);
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::uint64_t batch = 64, in = 257, hidden = 130, out = 11, seed = 2026;
    std::string artifacts;
    if (const char *env = std::getenv("PARALYN_ARTIFACT_DIR")) artifacts = env;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--artifacts" && i + 1 < argc) artifacts = argv[++i];
      else if (option == "--batch") batch = argument(i, argc, argv);
      else if (option == "--in") in = argument(i, argc, argv);
      else if (option == "--hidden") hidden = argument(i, argc, argv);
      else if (option == "--out") out = argument(i, argc, argv);
      else if (option == "--seed") seed = argument(i, argc, argv);
      else throw std::runtime_error("usage: native_mlp [--artifacts DIR] [--batch B] [--in I] "
                                    "[--hidden H] [--out O] [--seed S]");
    }
    if (!seed || seed > 0xffffffffu) throw std::runtime_error("seed must be in [1, 2^32-1]");
    Rng rng{static_cast<std::uint32_t>(seed)};
    // Dyadic weights scaled by powers of two keep inputs exact in FP32.
    const auto x = values(batch * in, rng, 1.0f), w1 = values(in * hidden, rng, 0.0625f),
               b1 = values(hidden, rng, 0.25f), w2 = values(hidden * out, rng, 0.0625f),
               b2 = values(out, rng, 0.25f);

    t::Context context;
    auto tx = t::from_host(context, {batch, in}, x), tw1 = t::from_host(context, {in, hidden}, w1),
         tb1 = t::from_host(context, {hidden}, b1), tw2 = t::from_host(context, {hidden, out}, w2),
         tb2 = t::from_host(context, {out}, b2);
    auto z1 = t::matmul(tx, tw1);           // GPU: x W1
    auto h = t::bias_add(z1, tb1, true);    // GPU: relu(. + b1)
    auto z2 = t::matmul(h, tw2);            // GPU: h W2
    auto y = t::bias_add(z2, tb2, false);   // GPU: . + b2
    const auto gpu_h = h.to_host(), gpu_y = y.to_host();

    // Independent CPU reference: float64 accumulation from the same FP32 inputs.
    const double u = std::ldexp(1.0, -24);
    auto gamma = [&](std::uint64_t n) { return n * u / (1 - n * u); };
    std::vector<double> h64(batch * hidden), e1(batch * hidden), y64(batch * out), e2(batch * out);
    for (std::uint64_t i = 0; i < batch; ++i)
      for (std::uint64_t j = 0; j < hidden; ++j) {
        double sum = 0, magnitude = 0;
        for (std::uint64_t p = 0; p < in; ++p) {
          sum += double(x[i * in + p]) * double(w1[p * hidden + j]);
          magnitude += std::fabs(double(x[i * in + p]) * double(w1[p * hidden + j]));
        }
        h64[i * hidden + j] = std::max(0.0, sum + double(b1[j]));
        e1[i * hidden + j] = gamma(in + 1) * (magnitude + std::fabs(double(b1[j])));
      }
    for (std::uint64_t i = 0; i < batch; ++i)
      for (std::uint64_t j = 0; j < out; ++j) {
        double sum = 0, propagated = 0, magnitude = 0;
        for (std::uint64_t p = 0; p < hidden; ++p) {
          const double wv = double(w2[p * out + j]), hv = h64[i * hidden + p], ev = e1[i * hidden + p];
          sum += hv * wv;
          propagated += ev * std::fabs(wv);
          magnitude += (std::fabs(hv) + ev) * std::fabs(wv);
        }
        y64[i * out + j] = sum + double(b2[j]);
        e2[i * out + j] = propagated + gamma(hidden + 1) * (magnitude + std::fabs(double(b2[j])));
      }
    double worst_h = 0, worst_y = 0, max_error = 0;
    for (std::size_t i = 0; i < gpu_h.size(); ++i) {
      const double error = std::fabs(double(gpu_h[i]) - h64[i]);
      if (!(error <= e1[i])) throw std::runtime_error("hidden layer exceeds bound at " + std::to_string(i));
      if (e1[i] > 0) worst_h = std::max(worst_h, error / e1[i]);
    }
    for (std::size_t i = 0; i < gpu_y.size(); ++i) {
      const double error = std::fabs(double(gpu_y[i]) - y64[i]);
      if (!(error <= e2[i])) throw std::runtime_error("output exceeds bound at " + std::to_string(i));
      if (e2[i] > 0) worst_y = std::max(worst_y, error / e2[i]);
      max_error = std::max(max_error, error);
    }
    std::cout << "Device: " << context.device().name << "; provider paralyn.msl.tensor; shapes x["
              << batch << "," << in << "] W1[" << in << "," << hidden << "] W2[" << hidden << ","
              << out << "] seed " << seed << '\n';
    const char *labels[] = {"matmul1", "bias_relu1", "matmul2", "bias2"};
    std::vector<double> durations;
    int index = 0;
    for (const auto *stage : {&z1, &h, &z2, &y}) {
      const auto timing = stage->timing();
      if (!timing || !timing->completed || !timing->duration_valid || !(timing->duration_seconds > 0))
        throw std::runtime_error("missing completed GPU event for a layer");
      durations.push_back(timing->duration_seconds);
      std::cout << "GPU " << labels[index++] << ": " << std::setprecision(6)
                << timing->duration_seconds * 1e6 << " us\n";
    }
    std::cout << "Max |error| vs float64 reference: " << max_error << "; worst error/bound: hidden "
              << worst_h << ", output " << worst_y << '\n';
    for (auto *owner : {&tx, &tw1, &tb1, &tw2, &tb2, &z1, &h, &z2, &y}) owner->close();
    if (!artifacts.empty()) {
      context.evidence(artifacts);
      const auto provider = t::operators_artifact();
      std::ofstream report(artifacts + "/mlp-report.json");
      report << std::setprecision(17) << "{\n  \"application\": \"two-layer FP32 MLP inference (C++)\",\n"
             << "  \"provider\": \"paralyn.msl.tensor\",\n"
             << "  \"provider_artifact_sha256\": \"" << sha256(provider.data(), provider.size()) << "\",\n"
             << "  \"shapes\": {\"batch\": " << batch
             << ", \"in\": " << in << ", \"hidden\": " << hidden << ", \"out\": " << out << "},\n"
             << "  \"seed\": " << seed << ",\n  \"gpu_commands\": 4,\n  \"stages\": [";
      for (int i = 0; i < 4; ++i)
        report << (i ? ", " : "") << "{\"stage\": \"" << labels[i] << "\", \"gpu_duration_seconds\": "
               << durations[i] << "}";
      report << "],\n"
             << "  \"reference\": \"float64 accumulation from FP32 inputs\",\n"
             << "  \"tolerance\": \"componentwise a-priori bound, docs/tensors-matmul.md\",\n"
             << "  \"max_abs_error\": " << max_error << ",\n  \"worst_error_to_bound_hidden\": "
             << worst_h << ",\n  \"worst_error_to_bound_output\": " << worst_y << ",\n"
             << "  \"cpu_fallback\": false,\n"
             << "  \"hidden_sha256\": \"" << sha256(gpu_h.data(), gpu_h.size() * 4) << "\",\n"
             << "  \"output_sha256\": \"" << sha256(gpu_y.data(), gpu_y.size() * 4) << "\"\n}\n";
      if (!report) throw std::runtime_error("cannot write mlp-report.json");
    }
    std::cout << "Verification: PASS two-layer FP32 MLP (" << batch * out
              << " outputs within float64-reference bound; 4 GPU commands)\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "MLP failed: " << e.what() << '\n';
    return 1;
  }
}
