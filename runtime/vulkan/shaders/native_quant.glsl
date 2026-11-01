bool affine(uint f) { return f >= 19 && f <= 36; }
uint affine_bits(uint f) {
    const uint widths[6] = uint[6](2, 3, 4, 5, 6, 8);
    return widths[(f - 19) / 3];
}
uvec4 fields(uint f) {
    if (affine(f)) {
        uint bits = affine_bits(f), group = 32u << ((f - 19) % 3);
        uint low = bits == 8 ? 8 : bits <= 3 ? 2 : 4;
        return uvec4(4 + group * bits / 8, 4 * low, 4 * (bits - low), 4);
    }
    if (f == 0) return uvec4(144, 16, 0, 16);
    if (f == 1) return uvec4(136, 16, 0, 8);
    if (f == 2) return uvec4(18, 16, 0, 2);
    if (f == 3) return uvec4(176, 16, 4, 16);
    if (f == 4) return uvec4(210, 16, 8, 20);
    if (f == 6) return uvec4(34, 32, 0, 2);
    return uvec4(512, 64, 0, 4);
}
uint meta_groups(uint f) {
    if (affine(f)) return 1u << ((f - 19) % 3);
    return f == 2 || f == 6 ? 1 : 8;
}
uint64_t tile_index(uint row, uint group, uint groups) {
    return (uint64_t(row / 256) * groups + group) * 256 + row % 256;
}
uint byte_at(uint64_t base, uint index) {
    uint word = uint(R16(base + (index & ~1u)).values[0]);
    return (word >> (8 * (index & 1))) & 255;
}
uint element_of_slot(uint slot) {
    uint chunk = slot / 8, pair = (slot / 2) % 4;
    return 4 * chunk + 16 * (pair / 2) + 2 * (pair % 2) + slot % 2;
}
uint code_at(uint f, uint64_t base, uint j, uint e) {
    if (affine(f)) {
        uint bits = affine_bits(f), at = (32 * j + e) * bits;
        uint word = byte_at(base, 4 + at / 8);
        if (at % 8 + bits > 8) word |= byte_at(base, 5 + at / 8) << 8;
        return (word >> (at % 8)) & ((1u << bits) - 1);
    }
    if (f == 0 || f == 3) {
        uint lo = (byte_at(base, (f == 0 ? 16 : 48) + (j / 2) * 32 + e) >> (4 * (j % 2))) & 15;
        return f == 3 ? lo | (((byte_at(base, 16 + e) >> j) & 1) << 4) : lo;
    }
    if (f == 1 || f == 2) {
        uint q = byte_at(base, (f == 1 ? 8 + 16 * j : 2) + e % 16);
        return (q >> (4 * (e / 16))) & 15;
    }
    if (f == 4) {
        uint quarter = j % 4, hb = j / 4;
        uint lo = (byte_at(base, 64 * hb + 32 * (quarter & 1) + e) >> (4 * (quarter >> 1))) & 15;
        uint hi = (byte_at(base, 128 + 32 * hb + e) >> (2 * quarter)) & 3;
        return lo | (hi << 4);
    }
    if (f == 6) return byte_at(base, 2 + e);
    return uint(R16(base + 2 * (32 * j + e)).values[0]);
}


float native_value(uint f, uint64_t base, uint row, uint col, uint width) {
    uint mg = meta_groups(f), g = col / 32, j = g % mg, e = col % 32;
    uint64_t block = base + (uint64_t(row) * (width / (32 * mg)) + g / mg) * fields(f).x;
    if (f == 41) return bf16_float(R16(block + 2 * (32 * j + e)).values[0]);
    if (f == 42) return q_half(block + 2 * (32 * j + e));
    return q_apply(f, code_at(f, block, j, e), block + (f == 4 ? 192 : 0), j, e);
}
