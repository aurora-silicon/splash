#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

#ifndef SPLASH_LOCAL_SIZE_DEFAULT
#define SPLASH_LOCAL_SIZE_DEFAULT 1
#endif
layout(local_size_x=SPLASH_LOCAL_SIZE_DEFAULT,local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
#ifdef SPLASH_DIRECT_LINEAR_PUSH
layout(push_constant,std430) uniform DirectLinearPush {
    uint64_t addresses[18];
    uint parameters[18];
} direct_linear;
#else
layout(set = 0, binding = 0, std430) readonly buffer DispatchTable {
    uint64_t addresses[];
} dispatch_table;
layout(push_constant) uniform DispatchIndex { uint address_index; } dispatch_index;
#endif
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer R16 { uint16_t values[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) writeonly buffer W16 { uint16_t values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RF32x4 { vec4 values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RF32 { float values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer WF32 { float values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RU32x4 { uvec4 values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RU32x2 { uvec2 values[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer RU64 { uint64_t values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer RU32 { uint values[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer WU32 { uint values[]; };

uint64_t binding_address(uint index) {
#ifdef SPLASH_DIRECT_LINEAR_PUSH
    return direct_linear.addresses[index];
#else
    return dispatch_table.addresses[dispatch_index.address_index + index];
#endif
}

float bf16_float(uint16_t value) {
    return uintBitsToFloat(uint(value) << 16);
}

uint16_t float_bf16(float value) {
    uint bits = floatBitsToUint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
        return uint16_t((bits >> 16) | 0x40u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
