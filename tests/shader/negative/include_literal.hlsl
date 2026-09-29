// Negative fixture: '/*' is a character literal for DXC, so line 3 is a live #include.
#define QUOTE '/*'
/**/ #include "secret.hlsl"
// */
[numthreads(64, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {}
