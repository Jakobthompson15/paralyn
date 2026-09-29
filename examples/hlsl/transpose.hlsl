// Tiled matrix transpose through groupshared memory. src is row-major
// height x width; dst is row-major width x height. 16x16 threads per group
// load one tile, synchronize, and store it transposed; dispatch
// ceil(width / 16) x ceil(height / 16) groups. Pure data movement: the result
// must match an independent CPU transpose bit for bit.
[[vk::binding(0, 0)]] StructuredBuffer<float> src : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dst : register(u0);

[[vk::binding(2, 0)]] cbuffer Shape : register(b0) {
  uint width;
  uint height;
};

groupshared float tile[16][17];

[numthreads(16, 16, 1)]
void transpose_tiled(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
  uint x = group.x * 16 + local.x;
  uint y = group.y * 16 + local.y;
  if (x < width && y < height)
    tile[local.y][local.x] = src[y * width + x];
  GroupMemoryBarrierWithGroupSync();
  uint tx = group.y * 16 + local.x; // output column = input row
  uint ty = group.x * 16 + local.y; // output row = input column
  if (tx < height && ty < width)
    dst[ty * height + tx] = tile[local.x][local.y];
}
