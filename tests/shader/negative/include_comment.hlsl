// Negative fixture: a comment between # and include still includes a file (clang and glslang).
#/**/include "/nonexistent/secret.hlsl"
[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {}
