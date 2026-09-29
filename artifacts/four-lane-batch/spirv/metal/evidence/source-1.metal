#pragma STDC FP_CONTRACT OFF
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct In
{
    float data[1];
};

struct InOut
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

kernel void modf_write(device In& a [[buffer(0)]], device InOut& b [[buffer(1)]], device Out& c [[buffer(2)]], constant Params& params [[buffer(30)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    if (gl_GlobalInvocationID.x < params.n)
    {
        float _38 = a.data[gl_GlobalInvocationID.x];
        float _39 = b.data[gl_GlobalInvocationID.x];
        float _40 = _38 + _39;
        float _43;
        float _41 = modf(_40, _43);
        b.data[gl_GlobalInvocationID.x] = _43;
        c.data[gl_GlobalInvocationID.x] = _40;
    }
}

