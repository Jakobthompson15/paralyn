#pragma once
// Independent float64 reference with a-priori running error bounds for the
// paralyn.msl.tensor transformer operators (docs/transformer-operators.md).
//
// This is CPU test/verification code only: it never produces results that are
// reported as GPU output. Every quantity is a pair (value, error): `value` is
// the exact-math result evaluated in float64 from the exact FP32 inputs, and
// `error` bounds |GPU FP32 result - value| given these stated assumptions:
//   * FP32 +, -, * and sqrt round to nearest (relative error <= u = 2^-24);
//   * division is budgeted at 2.5 ulp (the MSL fast-math figure, deliberately
//     looser than the precise-mode table), exp at 4 ulp and tanh at 5 ulp
//     (MSL specification, Table 8.1, precise math); k ulp <= k * 2^-23 * |v|;
//   * a result may be flushed to zero if subnormal: each rounding adds 2^-126;
//   * no overflow (callers exercise overflow and non-finite values separately
//     with exact expectations).
// Composition uses first-order-plus-product propagation that is valid for any
// errors within the stated bounds (it ignores error cancellation, so it is
// conservative). The float64 evaluation itself contributes ~1e-16 relative
// error; comparisons add `reference_slack`.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace paralyn_reference {
inline constexpr double u = 1.0 / 16777216.0;                      // 2^-24
inline constexpr double ulp = 2.0 * u;                             // 2^-23 relative per ulp
inline constexpr double tiny = 1.1754943508222875e-38;             // 2^-126 (flush floor)
inline constexpr double eps_div = 2.5 * ulp, eps_exp = 4.0 * ulp, eps_tanh = 5.0 * ulp;
inline constexpr double reference_slack = 1.0 / 35184372088832.0;  // 2^-45 relative
inline constexpr unsigned lanes = 256;                             // provider row width

inline double gamma(double n) { return n * u / (1.0 - n * u); }
// Additions any element passes through in the provider's fixed row order:
// the lane's sequential chain (ceil(n/256), counting the exact 0 + x) plus 8 tree levels.
inline double tree_depth(std::uint64_t n) { return double((n + lanes - 1) / lanes + 8); }

struct B {
  double v = 0, e = 0;
};
inline B exact(double v) { return {v, 0.0}; }
inline double rounded(double magnitude, double eps = u) { return eps * magnitude + tiny; }
inline B add(B a, B b) {
  const double v = a.v + b.v, e0 = a.e + b.e;
  return {v, e0 + rounded(std::fabs(v) + e0)};
}
inline B sub(B a, B b) {
  const double v = a.v - b.v, e0 = a.e + b.e;
  return {v, e0 + rounded(std::fabs(v) + e0)};
}
inline B mul(B a, B b) {
  const double v = a.v * b.v;
  const double e0 = std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e;
  return {v, e0 + rounded(std::fabs(v) + e0)};
}
inline B div(B a, B b) {
  const double d = std::fabs(b.v);
  if (!(d > b.e)) return {a.v / b.v, std::numeric_limits<double>::infinity()};
  const double v = a.v / b.v;
  const double e0 = (std::fabs(a.v) * b.e + d * a.e) / (d * (d - b.e));
  return {v, e0 + rounded(std::fabs(v) + e0, eps_div)};
}
inline B sqrt_(B a) {
  const double v = std::sqrt(a.v);
  const double e0 = a.e / (v + std::sqrt(std::max(a.v - a.e, 0.0)));
  return {v, e0 + rounded(v + e0)};
}
inline B exp_(B a) {
  const double v = std::exp(a.v), e0 = v * std::expm1(a.e);
  return {v, e0 + rounded(v + e0, eps_exp)};
}
inline B tanh_(B a) {
  const double v = std::tanh(a.v);
  const double t = std::tanh(std::max(std::fabs(a.v) - a.e, 0.0));
  const double e0 = (1.0 - t * t) * a.e; // sech^2 is decreasing in |x|
  return {v, e0 + rounded(std::fabs(v) + e0, eps_tanh)};
}
// Max is exact and 1-Lipschitz in the max norm.
inline B max_(const std::vector<B> &x) {
  B r{-std::numeric_limits<double>::infinity(), 0};
  for (const auto &b : x) {
    r.v = std::max(r.v, b.v);
    r.e = std::max(r.e, b.e);
  }
  return r;
}
// Any summation tree whose deepest path has L additions: sum|x| * gamma_L (Higham 4.2).
inline B tree_sum(const std::vector<B> &x) {
  double v = 0, e0 = 0, magnitude = 0;
  for (const auto &b : x) {
    v += b.v;
    e0 += b.e;
    magnitude += std::fabs(b.v) + b.e;
  }
  const double depth = tree_depth(x.size());
  return {v, e0 + gamma(depth) * magnitude + depth * tiny};
}
// FP32 dot product with one accumulator in increasing order (Higham 3.1: gamma_k).
inline B dot(const std::vector<B> &a, const std::vector<B> &b) {
  double v = 0, e0 = 0, magnitude = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    v += a[i].v * b[i].v;
    const double p = std::fabs(a[i].v) * b[i].e + std::fabs(b[i].v) * a[i].e + a[i].e * b[i].e;
    e0 += p;
    magnitude += std::fabs(a[i].v * b[i].v) + p;
  }
  return {v, e0 + gamma(double(a.size())) * magnitude + double(a.size()) * tiny};
}

// ---- Operator references (row-major) ----
// The GELU constants are the FP32 values the provider uses.
inline const double gelu_c0 = double(0.7978845608028654f), gelu_c1 = double(0.044715f);
inline B gelu(B x) {
  const B cube = mul(mul(x, x), x);
  const B inner = add(x, mul(exact(gelu_c1), cube));
  const B t = tanh_(mul(exact(gelu_c0), inner));
  return mul(mul(exact(0.5), x), add(exact(1.0), t));
}
// softmax(scale * x) over `visible` leading entries of a row; the rest are exact zeros.
inline std::vector<B> softmax_row(const std::vector<B> &x, double scale, std::size_t visible) {
  std::vector<B> v(visible);
  for (std::size_t j = 0; j < visible; ++j) v[j] = mul(exact(scale), x[j]);
  const B m = max_(v);
  std::vector<B> e(visible);
  for (std::size_t j = 0; j < visible; ++j) e[j] = exp_(sub(v[j], m));
  const B s = tree_sum(e);
  std::vector<B> y(x.size(), exact(0.0));
  for (std::size_t j = 0; j < visible; ++j) y[j] = div(e[j], s);
  return y;
}
inline std::vector<B> layer_norm_row(const std::vector<B> &x, const std::vector<B> &gamma_,
                                     const std::vector<B> &beta, double epsilon) {
  const B n = exact(double(x.size()));
  const B mean = div(tree_sum(x), n);
  std::vector<B> c(x.size()), q(x.size());
  for (std::size_t j = 0; j < x.size(); ++j) {
    c[j] = sub(x[j], mean);
    q[j] = mul(c[j], c[j]);
  }
  const B variance = div(tree_sum(q), n);
  const B rstd = div(exact(1.0), sqrt_(add(variance, exact(epsilon))));
  std::vector<B> y(x.size());
  for (std::size_t j = 0; j < x.size(); ++j) y[j] = add(mul(mul(c[j], rstd), gamma_[j]), beta[j]);
  return y;
}
inline std::vector<B> exact_values(const std::vector<float> &x) {
  std::vector<B> r(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) r[i] = exact(double(x[i]));
  return r;
}
// |gpu - value| <= error (+ float64 slack); NaN never passes.
inline bool within(float gpu, const B &ref) {
  const double error = std::fabs(double(gpu) - ref.v);
  return error <= ref.e + reference_slack * std::fabs(ref.v) + tiny;
}
inline double ratio(float gpu, const B &ref) {
  const double bound = ref.e + reference_slack * std::fabs(ref.v) + tiny;
  return std::fabs(double(gpu) - ref.v) / bound;
}

// ---- Bit-exact FP32 emulation of the provider's fixed reduction order ----
// (The including translation unit must be compiled with -ffp-contract=off.)
inline float max2(float a, float b) {
  if (std::isnan(a)) return a;
  if (std::isnan(b)) return b;
  if (b > a) return b;
  if (b == a && std::signbit(a) && !std::signbit(b)) return b;
  return a;
}
inline float tree_sum_f32(const float *x, std::size_t n) {
  float p[lanes];
  for (unsigned t = 0; t < lanes; ++t) {
    p[t] = 0.0f;
    for (std::size_t j = t; j < n; j += lanes) p[t] = p[t] + x[j];
  }
  for (unsigned w = lanes / 2; w > 0; w >>= 1)
    for (unsigned t = 0; t < w; ++t) p[t] = p[t] + p[t + w];
  return p[0];
}
inline float tree_max_f32(const float *x, std::size_t n) {
  float p[lanes];
  for (unsigned t = 0; t < lanes; ++t) {
    p[t] = -std::numeric_limits<float>::infinity();
    for (std::size_t j = t; j < n; j += lanes) p[t] = max2(p[t], x[j]);
  }
  for (unsigned w = lanes / 2; w > 0; w >>= 1)
    for (unsigned t = 0; t < w; ++t) p[t] = max2(p[t], p[t + w]);
  return p[0];
}
inline bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }
} // namespace paralyn_reference
