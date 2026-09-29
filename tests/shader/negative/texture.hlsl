// Negative fixture: textures/images are outside the buffer/scalar profile.
[[vk::binding(0, 0)]] Texture2D<float> source : register(t0);
[[vk::binding(1, 0)]] RWTexture2D<float> target : register(u0);

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {
  target[id.xy] = source[id.xy] * 2.0f;
}
