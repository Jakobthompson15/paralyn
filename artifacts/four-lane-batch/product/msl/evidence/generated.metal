#pragma STDC FP_CONTRACT OFF
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(2)]],
                       device float* out [[buffer(5)]],
                       constant uint& n [[buffer(7)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = a[i] + b[i];
}

// One independently checkable partial sum per 64-element group. Every lane
// reaches every barrier, including the final partial input group.
kernel void block_reduce(device const float* input [[buffer(1)]],
                         device float* output [[buffer(3)]],
                         constant uint& n [[buffer(6)]],
                         uint lane [[thread_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[64];
  uint i = group * 64 + lane;
  scratch[lane] = i < n ? input[i] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride > 0; stride >>= 1) {
    if (lane < stride) scratch[lane] += scratch[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) output[group] = scratch[0];
}

// A padded shared tile exercises 2D workgroup indexing, barriers and transpose.
kernel void tiled_transpose(device const float* input [[buffer(0)]],
                            device float* output [[buffer(4)]],
                            constant uint& width [[buffer(8)]],
                            constant uint& height [[buffer(9)]],
                            uint2 lane [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup float tile[16][17];
  uint x = group.x * 16 + lane.x;
  uint y = group.y * 16 + lane.y;
  if (x < width && y < height) tile[lane.y][lane.x] = input[y * width + x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint tx = group.y * 16 + lane.x;
  uint ty = group.x * 16 + lane.y;
  if (tx < height && ty < width) output[ty * height + tx] = tile[lane.x][lane.y];
}
