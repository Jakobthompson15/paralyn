// Negative fixture: a vertex-stage entry point compiled with a compute profile
// (stage/profile mismatch; no [numthreads] attribute).
float4 vs_main(float4 position : POSITION) : SV_Position {
  return position;
}
