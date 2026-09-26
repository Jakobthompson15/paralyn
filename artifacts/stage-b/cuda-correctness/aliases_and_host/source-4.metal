#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Generated exclusively from verified Paralyn IR.
kernel void uc_kernel_unsigned_values(
  device uint* uc_slot_0 [[buffer(0)]],
  constant uint& uc_slot_1 [[buffer(1)]],
  uint3 uc_tid [[thread_position_in_threadgroup]],
  uint3 uc_bid [[threadgroup_position_in_grid]],
  uint3 uc_bdim [[threads_per_threadgroup]],
  uint3 uc_gdim [[threadgroups_per_grid]]) {
  device uint* uc_arg_0 = uc_slot_0;
  uint uc_arg_1 = uc_slot_1;
  int uc_local_0 = int(uc_tid.x);
  uc_arg_0[uc_local_0] = (uc_tid.x + uc_arg_1);
}
