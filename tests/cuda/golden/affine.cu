// Generated exclusively from verified Paralyn IR (CUDA C++ device code for NVRTC).
// Numerical policy 1: __fadd_rn/__fmul_rn, --fmad=false, --ftz=false, IEEE div/sqrt.
extern "C" __global__ void uc_kernel_affine(
    const float *uc_arg_0,
    const float *uc_arg_1,
    float *uc_arg_2,
    int uc_arg_3,
    float uc_arg_4) {
  int uc_local_0 = static_cast<int>(((blockIdx.x * blockDim.x) + threadIdx.x));
  if ((uc_local_0 < uc_arg_3)) {
    uc_arg_2[uc_local_0] = __fadd_rn(__fmul_rn(uc_arg_0[uc_local_0], uc_arg_4), uc_arg_1[uc_local_0]);
  }
}
