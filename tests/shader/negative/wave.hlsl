// Negative fixture: wave (subgroup) intrinsics are outside the profile.
[[vk::binding(0, 0)]] RWStructuredBuffer<float> data : register(u0);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
  data[id.x] = WaveActiveSum(data[id.x]);
}
