// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal (MIT notice in THIRD_PARTY_NOTICES). Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output (fp32 from
// the _a_f32 instances, the logits).
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/split_reduce.h"

// A tile's sums over every K partition, handed to store(row, column, sum). One partition stores its own; more publish
// fp32 partials [split][Rows][destination column] (kernels/common/split_reduce.h) and the last arriving partition adds
// them in split order. `column0` is the simdgroup's first destination column, `counter` its threadgroup's (one per 64
// destination columns: the segments of a projection never share one).
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

// ---------------- prefill tiles: a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is not
// a multiple of the tile) skip their matmuls and stores, so a chunk costs its rows rounded up to RowsPerSG rather than
// to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile). Full tiles run this
// loop too; a branch-free copy for them is no faster (M3 Max, M5 Max).
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, GgufEpilogue Ep = EpNone>
inline void gguf_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint input_size, uint output_origin, uint rows, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint simd_group, uint out_stride, uint out_offset, device bfloat *aux = nullptr) {
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  device bfloat *rows_input = input + ulong(simd_group) * RowsPerSG * input_size;
  auto acc = staged_accumulator<RowsPerSG, TileN, KS>(rows_input, input_size, stage);
  gguf_zero(acc);
  gguf_staged_steps<F, RowsPerSG, TileN, KS, Simdgroups * 32>(rows_input, w0, w1, meta, input_size, output_origin, stage, tl,
                                                              simd_group * 32 + simd_lane, 0, input_size / KS, 0, owns_rows,
                                                              acc);
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = gguf_epilogue<Ep>(acc[i], aux, o);
  }
}

// The two stages of a prefill tile.
constant constexpr uint kPrefillStages = 2 * GGUF_TILE_COLUMNS * GGUF_PREFILL_STEP;

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows (these kernels and the fused ones): a 3-lane step
// runs the 32-row tile over the storage of four lanes (LinearPlan::storageRows) and the padding lane's rows are
// computed and discarded. Rows are independent, so every active row is the bits of any other tile height
// (gguf-projection checks it); the 32-row tile at three lanes costs what it costs at four, 3-15% less than a 16-row
// plus an 8-row matmul per stage (DRAM-cold, 16- and 20-core M5 Pro).
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do. The destination's type Out is bf16, or fp32 for the plain epilogue's
// logits (a_f32, ops::Projection::destination). Each column tile walks its partition from the step staged_first_step
// gives it (kernels/common/gguf_staged_tile.h): in lockstep, or spread when p.spread is set.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device Out *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, sb, se,
                                          staged_first_step(group.x, sb, se, p.spread), acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// The staged decode kernels: grid (column tiles, K partitions), two simdgroups.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                        device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                        device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                        device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  gguf_decode_tile<F, Rows, Ep>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group,
                                stage, tl, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE(F, f, R, ep, Ep, Out) \
  template [[host_name("gguf_decode_" #f "_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode<F, R, Ep, Out>;

GGUF_DECODE(FmtQ6K,q6k,8,a,EpNone,bfloat)

GGUF_DECODE(FmtQ6K,q6k,8,a_f32,EpNone,float)

GGUF_DECODE(FmtQ6K,q6k,8,r,EpResidual,bfloat)

GGUF_DECODE(FmtQ6K,q6k,8,g,EpUpWithGate,bfloat)

GGUF_DECODE(FmtQ5K,q5k,8,a,EpNone,bfloat)

GGUF_DECODE(FmtQ5K,q5k,8,a_f32,EpNone,float)

GGUF_DECODE(FmtQ5K,q5k,8,r,EpResidual,bfloat)

GGUF_DECODE(FmtQ5K,q5k,8,g,EpUpWithGate,bfloat)
