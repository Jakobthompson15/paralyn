"""Independent float64 reference with a-priori running error bounds for the
paralyn.msl.tensor transformer operators (docs/transformer-operators.md).

CPU verification code only; nothing here is reported as GPU output. Each value
is a pair (v, e): v is exact math evaluated in float64 from the exact FP32
inputs and e bounds |GPU FP32 result - v| under the stated assumptions:
FP32 +, -, *, sqrt round to nearest (u = 2^-24); division budgeted at 2.5 ulp,
exp at 4 ulp, tanh at 5 ulp (MSL spec Table 8.1, precise math; k ulp <= k *
2^-23 * |v|); each rounding may flush a subnormal (adds 2^-126); no overflow.
This mirrors examples/native/transformer_reference.hpp but is written
separately so the C++ and Python applications verify independently.
"""
import math
import struct

U = 2.0 ** -24
ULP = 2.0 * U
TINY = 2.0 ** -126
EPS_DIV, EPS_EXP, EPS_TANH = 2.5 * ULP, 4.0 * ULP, 5.0 * ULP
REFERENCE_SLACK = 2.0 ** -45
LANES = 256


def f32(value):
    """Round a float64 to the nearest FP32 value (innocuous double rounding for +-*/)."""
    return struct.unpack("f", struct.pack("f", value))[0]


def gamma(n):
    return n * U / (1.0 - n * U)


def tree_depth(n):
    return (n + LANES - 1) // LANES + 8


def exact(v):
    return (float(v), 0.0)


def _rounded(magnitude, eps=U):
    return eps * magnitude + TINY


def add(a, b):
    v, e0 = a[0] + b[0], a[1] + b[1]
    return (v, e0 + _rounded(abs(v) + e0))


def sub(a, b):
    v, e0 = a[0] - b[0], a[1] + b[1]
    return (v, e0 + _rounded(abs(v) + e0))


def mul(a, b):
    v = a[0] * b[0]
    e0 = abs(a[0]) * b[1] + abs(b[0]) * a[1] + a[1] * b[1]
    return (v, e0 + _rounded(abs(v) + e0))


def div(a, b):
    d = abs(b[0])
    if not d > b[1]:
        return (a[0] / b[0], math.inf)
    v = a[0] / b[0]
    e0 = (abs(a[0]) * b[1] + d * a[1]) / (d * (d - b[1]))
    return (v, e0 + _rounded(abs(v) + e0, EPS_DIV))


def sqrt(a):
    v = math.sqrt(a[0])
    e0 = a[1] / (v + math.sqrt(max(a[0] - a[1], 0.0)))
    return (v, e0 + _rounded(v + e0))


def exp(a):
    v = math.exp(a[0])
    e0 = v * math.expm1(a[1])
    return (v, e0 + _rounded(v + e0, EPS_EXP))


def tanh(a):
    v = math.tanh(a[0])
    t = math.tanh(max(abs(a[0]) - a[1], 0.0))
    e0 = (1.0 - t * t) * a[1]
    return (v, e0 + _rounded(abs(v) + e0, EPS_TANH))


def maximum(values):
    return (max(v for v, _ in values), max(e for _, e in values))


def tree_sum(values):
    v = sum(x for x, _ in values)
    e0 = sum(e for _, e in values)
    magnitude = sum(abs(x) + e for x, e in values)
    depth = tree_depth(len(values))
    return (v, e0 + gamma(depth) * magnitude + depth * TINY)


def dot(a, b):
    """FP32 products accumulated in one FP32 sum in increasing order (Higham 3.1)."""
    v = e0 = magnitude = 0.0
    for (av, ae), (bv, be) in zip(a, b):
        v += av * bv
        p = abs(av) * be + abs(bv) * ae + ae * be
        e0 += p
        magnitude += abs(av * bv) + p
    k = len(a)
    return (v, e0 + gamma(k) * magnitude + k * TINY)


GELU_C0 = f32(0.7978845608028654)
GELU_C1 = f32(0.044715)


def gelu(x):
    cube = mul(mul(x, x), x)
    inner = add(x, mul(exact(GELU_C1), cube))
    t = tanh(mul(exact(GELU_C0), inner))
    return mul(mul(exact(0.5), x), add(exact(1.0), t))


def softmax_row(x, scale, visible):
    v = [mul(exact(scale), x[j]) for j in range(visible)]
    m = maximum(v)
    e = [exp(sub(value, m)) for value in v]
    s = tree_sum(e)
    return [div(value, s) for value in e] + [exact(0.0)] * (len(x) - visible)


def layer_norm_row(x, gamma_, beta, epsilon):
    n = exact(float(len(x)))
    mean = div(tree_sum(x), n)
    c = [sub(value, mean) for value in x]
    variance = div(tree_sum([mul(value, value) for value in c]), n)
    rstd = div(exact(1.0), sqrt(add(variance, exact(epsilon))))
    return [add(mul(mul(c[j], rstd), gamma_[j]), beta[j]) for j in range(len(x))]


def bound(reference):
    return reference[1] + REFERENCE_SLACK * abs(reference[0]) + TINY


def within(gpu, reference):
    """|gpu - v| <= e (+ float64 slack); NaN never passes."""
    return abs(gpu - reference[0]) <= bound(reference)


def ratio(gpu, reference):
    return abs(gpu - reference[0]) / bound(reference)


# ---- Bit-exact FP32 emulation of the provider's fixed row order ----
def max2(a, b):
    if math.isnan(a):
        return a
    if math.isnan(b):
        return b
    if b > a:
        return b
    if b == a and math.copysign(1.0, a) < 0 and math.copysign(1.0, b) > 0:
        return b
    return a


def tree_sum_f32(row):
    p = [0.0] * LANES
    for t in range(LANES):
        acc = 0.0
        for j in range(t, len(row), LANES):
            acc = f32(acc + row[j])
        p[t] = acc
    w = LANES // 2
    while w:
        for t in range(w):
            p[t] = f32(p[t] + p[t + w])
        w //= 2
    return p[0]


def tree_max_f32(row):
    p = [-math.inf] * LANES
    for t in range(LANES):
        acc = -math.inf
        for j in range(t, len(row), LANES):
            acc = max2(acc, row[j])
        p[t] = acc
    w = LANES // 2
    while w:
        for t in range(w):
            p[t] = max2(p[t], p[t + w])
        w //= 2
    return p[0]


def same_bits(a, b):
    return struct.pack("f", a) == struct.pack("f", b)
