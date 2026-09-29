#pragma STDC FP_CONTRACT OFF
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

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
    float scale;
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(64u, 1u, 1u);

kernel void reduce_sum(device In& _input [[buffer(0)]], device Out& partial [[buffer(1)]], constant Params& params [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_NumWorkGroups [[threadgroups_per_grid]])
{
    threadgroup spvUnsafeArray<float, 64> _shared;
    uint _52 = gl_NumWorkGroups.x * 64u;
    float _61;
    _61 = 0.0;
    float _62;
    for (uint _58 = gl_GlobalInvocationID.x; _58 < params.n; _58 += _52, _61 = _62)
    {
        _62 = _61 + _input.data[_58];
    }
    _shared[gl_LocalInvocationID.x] = _61;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint _71 = 32u; _71 > 0u; _71 = _71 >> 1u)
    {
        if (gl_LocalInvocationID.x < _71)
        {
            _shared[gl_LocalInvocationID.x] += _shared[gl_LocalInvocationID.x + _71];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gl_LocalInvocationID.x == 0u)
    {
        partial.data[gl_WorkGroupID.x] = _shared[0u] * params.scale;
    }
}

