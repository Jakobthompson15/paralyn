#pragma STDC FP_CONTRACT OFF
#include <metal_stdlib>
using namespace metal;

// C[m,n] = op(A)[m,k] * op(B)[k,n]. One thread per output element, 16x16 tiles.
// FP32 products are added to one FP32 accumulator in strictly increasing k
// order (0, 1, ..., k-1); padding lanes never enter the sum. The executable's
// numerical policy prepends FP_CONTRACT OFF, so product and sum round separately.
kernel void paralyn_matmul_f32(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device float* c [[buffer(2)]],
                               constant uint& m [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               constant uint& k [[buffer(5)]],
                               constant uint& transpose_a [[buffer(6)]],
                               constant uint& transpose_b [[buffer(7)]],
                               uint2 lane [[thread_position_in_threadgroup]],
                               uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup float tile_a[16][17];
  threadgroup float tile_b[16][17];
  uint row = group.y * 16 + lane.y;
  uint column = group.x * 16 + lane.x;
  float sum = 0.0f;
  for (uint base = 0; base < k; base += 16) {
    uint ka = base + lane.x;
    float va = 0.0f;
    if (row < m && ka < k) va = transpose_a != 0 ? a[ka * m + row] : a[row * k + ka];
    tile_a[lane.y][lane.x] = va;
    uint kb = base + lane.y;
    float vb = 0.0f;
    if (kb < k && column < n) vb = transpose_b != 0 ? b[column * k + kb] : b[kb * n + column];
    tile_b[lane.y][lane.x] = vb;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = min(16u, k - base);
    for (uint p = 0; p < count; ++p) {
      float product = tile_a[lane.y][p] * tile_b[p][lane.x];
      sum = sum + product;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < m && column < n) c[row * n + column] = sum;
}

// out[i] = act(x[i] + bias[i % columns]); ReLU replaces only values less than zero.
kernel void paralyn_bias_activation_f32(device const float* x [[buffer(0)]],
                                        device const float* bias [[buffer(1)]],
                                        device float* out [[buffer(2)]],
                                        constant uint& count [[buffer(3)]],
                                        constant uint& columns [[buffer(4)]],
                                        constant uint& activation [[buffer(5)]],
                                        uint i [[thread_position_in_grid]]) {
  if (i < count) {
    float value = x[i] + bias[i % columns];
    if (activation == 1u && value < 0.0f) value = 0.0f;
    out[i] = value;
  }
}

kernel void paralyn_activation_f32(device const float* x [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   constant uint& count [[buffer(2)]],
                                   constant uint& activation [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i < count) {
    float value = x[i];
    if (activation == 1u && value < 0.0f) value = 0.0f;
    out[i] = value;
  }
}

// Used for k == 0: the empty sum is +0.0 in every output element.
kernel void paralyn_fill_f32(device float* out [[buffer(0)]],
                             constant uint& count [[buffer(1)]],
                             constant float& value [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < count) out[i] = value;
}
