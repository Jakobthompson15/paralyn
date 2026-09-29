// Negative fixture: #include cannot be resolved; only the exact source bytes are compiled.
#include "common.hlsli"
[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {}
