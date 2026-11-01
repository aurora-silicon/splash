#pragma once

#include <cstdint>

// Host/shader contract for the registered GGUF projection kernels.  The
// layout is deliberately scalar: it is copied into a 72-byte Vulkan storage
// block and read by GLSL without padding-dependent C++ types.
struct VulkanLinearParams final {
  uint32_t rows = 0;
  uint32_t input_size = 0;
  uint32_t out_stride = 0;
  uint32_t plane_input_size = 0;
  uint32_t cols[3] = {};
  uint32_t fmt[3] = {};
  uint32_t offset[3] = {};
  uint32_t epilogue = 0;
  uint32_t output_f32 = 0;
  uint32_t row_start = 0;
  uint32_t active_rows = 0;
  uint32_t input_f32 = 0;
};
static_assert(sizeof(VulkanLinearParams) == 72);

// The split law used by the qualified registered kernels.  Large K with a
// narrow output is split to keep register pressure bounded; wide projections
// stay single-pass because their tile count already exposes enough work.
inline constexpr uint32_t vulkanRegisterSplits(uint32_t input_size,
                                               uint32_t out_stride = 0) {
  if (out_stride >= 65536) return 1;
  return input_size >= 8192 ? 8 : 4;
}

inline constexpr uint32_t vulkanLinearKSplits(uint32_t input_size) {
  return input_size >= 8192 ? 4 : 1;
}

inline constexpr uint32_t vulkanQ6MatrixTile(bool all_q6) {
  return all_q6 ? 16 : 8;
}
