#pragma STDC FP_CONTRACT OFF
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct In
{
    float data[1];
};

struct Out
{
    float data[1];
};

struct Params
{
    uint n;
};

kernel void vector_add(device In& a [[buffer(0)]], device In& b [[buffer(1)]], device Out& c [[buffer(2)]], constant Params& params [[buffer(30)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    if (gl_GlobalInvocationID.x < params.n)
    {
        c.data[gl_GlobalInvocationID.x] = a.data[gl_GlobalInvocationID.x] + b.data[gl_GlobalInvocationID.x];
    }
}

