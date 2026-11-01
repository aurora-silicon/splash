bool q_affine(uint f) { return f >= 19 && f <= 36; }
uint q_bits(uint f) {
    const uint widths[6] = uint[6](2, 3, 4, 5, 6, 8);
    return widths[(f - 19) / 3];
}
uint q_meta_groups(uint f) {
    return q_affine(f) ? 1u << ((f - 19) % 3) : f == 2 || f == 6 ? 1 : 8;
}
uvec3 q_fields(uint f) {
    if (q_affine(f)) {
        uint bits = q_bits(f), low = bits == 8 ? 8 : bits <= 3 ? 2 : 4;
        return uvec3(4 * low, 4 * (bits - low), 4);
    }
    if (f == 0) return uvec3(16, 0, 16);
    if (f == 1) return uvec3(16, 0, 8);
    if (f == 2) return uvec3(16, 0, 2);
    if (f == 3) return uvec3(16, 4, 16);
    if (f == 4) return uvec3(16, 8, 20);
    if (f == 6) return uvec3(32, 0, 2);
    return uvec3(64, 0, 4);
}
uint q_tile32(uint row,uint group,uint groups) {
 return ((row/256)*groups+group)*256+row%256;
}
uint64_t q_tile(uint row, uint group, uint groups) {
    return (uint64_t(row / 256) * groups + group) * 256 + row % 256;
}
uint q_byte(uint64_t address, uint at) {
    return (uint(R16(address + (at & ~1u)).values[0]) >> (8 * (at & 1))) & 255;
}
float q_half(uint64_t address) { return unpackHalf2x16(uint(R16(address).values[0])).x; }
int q_signed(uint byte_value) { return int(byte_value << 24) >> 24; }
float q_iq4(uint code) {
    uint packed=code<4?0xbfad9881u:code<8?0xf6eaddcfu:code<12?0x26190d01u:0x71594535u;
    return float(q_signed((packed>>(8*(code&3)))&255));
}
float q_apply(uint f, uint code, uint64_t meta, uint j, uint e) {
    if (q_affine(f)) {
        float scale = bf16_float(R16(meta).values[0]);
        float bias = bf16_float(R16(meta + 2).values[0]);
        return fma(scale, float(code), bias);
    }
    if (f == 0 || f == 3) {
        uint scale, minimum;
        if (j < 4) {
            scale = q_byte(meta, 4 + j) & 63;
            minimum = q_byte(meta, 8 + j) & 63;
        } else {
            scale = (q_byte(meta, 8 + j) & 15) | ((q_byte(meta, j) >> 6) << 4);
            minimum = (q_byte(meta, 8 + j) >> 4) | ((q_byte(meta, 4 + j) >> 6) << 4);
        }
        return fma(q_half(meta) * float(scale), float(code), -q_half(meta + 2) * float(minimum));
    }
    if (f == 4)
        return q_half(meta + 16) * float(q_signed(q_byte(meta, 2 * j + e / 16))) * float(int(code) - 32);
    if (f == 6) return q_half(meta) * float(q_signed(code));
    if (f == 41) return RF32(meta).values[0] * bf16_float(uint16_t(code));
    if (f == 42) return RF32(meta).values[0] * unpackHalf2x16(code).x;
    float scale = q_half(meta);
    if (f == 1) {
        uint lo = (q_byte(meta, 4 + j / 2) >> (4 * (j & 1))) & 15;
        uint hi = (uint(R16(meta + 2).values[0]) >> (2 * j)) & 3;
        scale *= float(int(lo | (hi << 4)) - 32);
    }
    return scale * q_iq4(code);
}

uint q_code(uint f,uint64_t w0,uint64_t w1,uint row,uint column,uint plane_width){
    uvec3 fields = q_fields(f);
    uint groups = plane_width / 32, g = column / 32, e = column % 32;
    uint slot = 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3);
    uint64_t a0 = w0 + q_tile(row, g, groups) * fields.x;
    uint64_t a1 = w1 + q_tile(row, g, groups) * fields.y;
    uint low_bits = fields.x / 4;
    bool pairs = f == 0 || f == 3 || f == 4 || (q_affine(f) && low_bits == 4);
    uint word = RU32(a0 + 4 * (pairs ? slot / 8 : slot / (32 / low_bits))).values[0];
    uint shift = pairs ? 4 * ((slot / 2) % 4) + 16 * (slot & 1) : low_bits * (slot % (32 / low_bits));
    uint code = (word >> shift) & ((1u << low_bits) - 1);
    if (fields.y != 0) {
        uint high_bits = fields.y / 4;
        uint hi = RU32(a1 + 4 * (slot / (32 / high_bits))).values[0];
        code |= ((hi >> (high_bits * (slot % (32 / high_bits)))) & ((1u << high_bits) - 1)) << low_bits;
    }
    return code;
}

float q_value(uint f, uint64_t w0, uint64_t w1, uint64_t metadata,
              uint row, uint column, uint plane_width) {
    if (f == 0xffffffffu)
        return RF32(w0 + 4 * (uint64_t(row) * plane_width + column)).values[0];
    uvec3 fields = q_fields(f);
    uint groups = plane_width / 32, g = column / 32, e = column % 32;
    uint mg = q_meta_groups(f), j = g % mg;
    uint slot = 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3);
    uint64_t a0 = w0 + q_tile(row, g, groups) * fields.x;
    uint64_t a1 = w1 + q_tile(row, g, groups) * fields.y;
    uint64_t meta = metadata + q_tile(row, g / mg, groups / mg) * fields.z;
    uint low_bits = fields.x / 4;
    bool pairs = f == 0 || f == 3 || f == 4 || (q_affine(f) && low_bits == 4);
    uint word = RU32(a0 + 4 * (pairs ? slot / 8 : slot / (32 / low_bits))).values[0];
    uint shift = pairs ? 4 * ((slot / 2) % 4) + 16 * (slot & 1) : low_bits * (slot % (32 / low_bits));
    uint code = (word >> shift) & ((1u << low_bits) - 1);
    if (fields.y != 0) {
        uint high_bits = fields.y / 4;
        uint hi = RU32(a1 + 4 * (slot / (32 / high_bits))).values[0];
        code |= ((hi >> (high_bits * (slot % (32 / high_bits)))) & ((1u << high_bits) - 1)) << low_bits;
    }
    return q_apply(f, code, meta, j, e);
}
