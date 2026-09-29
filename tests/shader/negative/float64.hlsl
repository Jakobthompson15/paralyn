// Negative fixture: 64-bit floating point (Float64 capability) is outside the profile.
[[vk::binding(0, 0)]] RWStructuredBuffer<double> data : register(u0);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
  data[id.x] = data[id.x] * 2.0;
}
