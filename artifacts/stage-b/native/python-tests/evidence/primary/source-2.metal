#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Generated exclusively from verified Paralyn IR.
kernel void uc_kernel_vector_add(
  device float* uc_slot_0 [[buffer(0)]],
  const device float* uc_slot_1 [[buffer(1)]],
  constant int& uc_slot_2 [[buffer(2)]],
  uint3 uc_tid [[thread_position_in_threadgroup]],
  uint3 uc_bid [[threadgroup_position_in_grid]],
  uint3 uc_bdim [[threads_per_threadgroup]],
  uint3 uc_gdim [[threadgroups_per_grid]]) {
  const device float* uc_arg_0 = uc_slot_0 + 5u;
  const device float* uc_arg_1 = uc_slot_1 + 5u;
  device float* uc_arg_2 = uc_slot_0 + 5u;
  int uc_arg_3 = uc_slot_2;
  int uc_local_0 = int(((uc_bid.x * uc_bdim.x) + uc_tid.x));
  if ((uc_local_0 < uc_arg_3)) {
    uc_arg_2[uc_local_0] = (uc_arg_0[uc_local_0] + uc_arg_1[uc_local_0]);
  }
}
