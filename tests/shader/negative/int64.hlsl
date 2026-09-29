// Negative fixture: 64-bit integers (Int64 capability) are outside the profile.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint64_t> data : register(u0);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
  data[id.x] = data[id.x] + 1;
}
