// Negative fixture: atomics are outside the profile.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> counter : register(u0);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
  InterlockedAdd(counter[0], 1u);
}
