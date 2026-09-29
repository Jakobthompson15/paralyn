// Negative fixture: a backslash-newline splice between # and include.
#\
include "/nonexistent/secret.hlsl"
[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {}
