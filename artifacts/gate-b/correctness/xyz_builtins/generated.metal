#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Generated exclusively from verified Paralyn IR.
kernel void uc_kernel_record_coordinates(
  device int* uc_slot_0 [[buffer(0)]],
  constant int& uc_slot_1 [[buffer(1)]],
  uint3 uc_tid [[thread_position_in_threadgroup]],
  uint3 uc_bid [[threadgroup_position_in_grid]],
  uint3 uc_bdim [[threads_per_threadgroup]],
  uint3 uc_gdim [[threadgroups_per_grid]]) {
  device int* uc_arg_0 = uc_slot_0;
  int uc_arg_1 = uc_slot_1;
  int uc_local_0 = int(((uc_bid.x * uc_bdim.x) + uc_tid.x));
  int uc_local_1 = int(((uc_bid.y * uc_bdim.y) + uc_tid.y));
  int uc_local_2 = int(((uc_bid.z * uc_bdim.z) + uc_tid.z));
  int uc_local_3 = int((uc_gdim.x * uc_bdim.x));
  int uc_local_4 = int((uc_gdim.y * uc_bdim.y));
  int uc_local_5 = ((((((uc_local_2 * uc_local_4) + uc_local_1) * uc_local_3) + uc_local_0) * int(12)) + uc_arg_1);
  uc_arg_0[(uc_local_5 + int(0))] = int(uc_tid.x);
  uc_arg_0[(uc_local_5 + int(1))] = int(uc_tid.y);
  uc_arg_0[(uc_local_5 + int(2))] = int(uc_tid.z);
  uc_arg_0[(uc_local_5 + int(3))] = int(uc_bid.x);
  uc_arg_0[(uc_local_5 + int(4))] = int(uc_bid.y);
  uc_arg_0[(uc_local_5 + int(5))] = int(uc_bid.z);
  uc_arg_0[(uc_local_5 + int(6))] = int(uc_bdim.x);
  uc_arg_0[(uc_local_5 + int(7))] = int(uc_bdim.y);
  uc_arg_0[(uc_local_5 + int(8))] = int(uc_bdim.z);
  uc_arg_0[(uc_local_5 + int(9))] = int(uc_gdim.x);
  uc_arg_0[(uc_local_5 + int(10))] = int(uc_gdim.y);
  uc_arg_0[(uc_local_5 + int(11))] = int(uc_gdim.z);
}
