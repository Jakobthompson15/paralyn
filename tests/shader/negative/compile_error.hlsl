// Negative fixture: `scale` is not declared (line 7, column 24).
[[vk::binding(0, 0)]] RWStructuredBuffer<float> data : register(u0);

[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
  float value = data[id.x];
  data[id.x] = value * scale;
}
