
#include <metal_stdlib>
#pragma STDC FP_CONTRACT OFF
using namespace metal;
kernel void handwritten_vector_add(const device float* a [[buffer(0)]],
                                   const device float* b [[buffer(1)]],
                                   device float* c [[buffer(2)]],
                                   constant int& n [[buffer(3)]],
                                   uint i [[thread_position_in_grid]]) {
  if (i < uint(n)) c[i] = a[i] + b[i];
}
