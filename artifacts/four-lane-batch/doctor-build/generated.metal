#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Generated exclusively from verified Paralyn IR.
kernel void uc_kernel_paralyn_probe(
  const device float* uc_slot_0 [[buffer(0)]],
  const device float* uc_slot_1 [[buffer(1)]],
  device float* uc_slot_2 [[buffer(2)]],
  uint3 uc_tid [[thread_position_in_threadgroup]],
  uint3 uc_bid [[threadgroup_position_in_grid]],
  uint3 uc_bdim [[threads_per_threadgroup]],
  uint3 uc_gdim [[threadgroups_per_grid]]) {
  const device float* uc_arg_0 = uc_slot_0;
  const device float* uc_arg_1 = uc_slot_1;
  device float* uc_arg_2 = uc_slot_2;
  uc_arg_2[uc_tid.x] = (uc_arg_0[uc_tid.x] + uc_arg_1[uc_tid.x]);
}
