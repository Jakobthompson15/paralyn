// HLSL compute shader compiled by the pinned DXC worker with -spirv
// (Vulkan 1.1 / SPIR-V 1.3) and imported through the Paralyn SPIR-V profile.
// c[i] = a[i] + b[i] for every i < n, one thread per element.
//   paralyn compile examples/hlsl/vector_add.hlsl --entry vector_add --profile cs_6_0 --output vector_add.prx
[[vk::binding(0, 0)]] StructuredBuffer<float> a : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float> b : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> c : register(u0);

struct Params {
  uint n;
};
[[vk::push_constant]] Params params;

[numthreads(64, 1, 1)]
void vector_add(uint3 id : SV_DispatchThreadID) {
  if (id.x < params.n)
    c[id.x] = a[id.x] + b[id.x];
}
