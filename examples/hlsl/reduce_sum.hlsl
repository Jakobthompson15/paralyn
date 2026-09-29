// Workgroup tree reduction: partial[g] = scale * sum(input[i]) over the 256
// elements i of group g that are < n. Pairwise tree order in groupshared
// memory, one partial sum per group; dispatch ceil(n / 256) groups.
[[vk::binding(0, 0)]] StructuredBuffer<float> input : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> partial : register(u0);

struct Params {
  uint n;
  float scale;
};
[[vk::push_constant]] Params params;

groupshared float scratch[256];

[numthreads(256, 1, 1)]
void reduce_sum(uint3 id : SV_DispatchThreadID, uint3 local : SV_GroupThreadID,
                uint3 group : SV_GroupID) {
  scratch[local.x] = id.x < params.n ? input[id.x] : 0.0f;
  GroupMemoryBarrierWithGroupSync();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (local.x < stride)
      scratch[local.x] = scratch[local.x] + scratch[local.x + stride];
    GroupMemoryBarrierWithGroupSync();
  }
  if (local.x == 0)
    partial[group.x] = scratch[0] * params.scale;
}
