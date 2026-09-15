import { describe, expect, it } from 'vitest'
import { prefillAttentionFlops, averageDecodeAttentionFlops } from '../../src/engine/attention'
import type { AttentionConfig, ModelArch } from '../../src/engine/types'
import { testModel } from '../fixtures'

// Enumerate scalar multiply/add work in QK and AV independently of the
// production closed forms. Each row lists visible slots in one layer by query.
function operations(rows: number[][], heads = 2, qkWidth = 2, valueWidth = 2): number {
  let flops = 0
  for (const row of rows) for (const slots of row) {
    for (let h = 0; h < heads; h++) for (let slot = 0; slot < slots; slot++) {
      for (let d = 0; d < qkWidth; d++) flops += 2
      for (let d = 0; d < valueWidth; d++) flops += 2
    }
  }
  return flops
}
function model(attention: AttentionConfig, layers = 2): ModelArch {
  return { ...testModel, attention, layers }
}
const mla = { kvLoraRank: 512, qkRopeHeadDim: 64, qkNopeHeadDim: 128, vHeadDim: 128 }

describe('attention matrix work', () => {
  it('counts both matrices and each query head with causal prefill', () => {
    expect(prefillAttentionFlops(testModel, 3)).toBe(operations([[1, 2, 3], [1, 2, 3]]))
    expect(averageDecodeAttentionFlops(testModel, 3, 3)).toBe(operations([[3, 4, 5], [3, 4, 5]]) / 3)
  })

  it.each([
    { prompt: 2, output: 1, prefill: [1, 2], decode: [2] },
    { prompt: 2, output: 4, prefill: [1, 2], decode: [2, 3, 3, 3] },
    { prompt: 5, output: 3, prefill: [1, 2, 3, 3, 3], decode: [3, 3, 3] },
  ])('caps sliding work per query at prompt=$prompt output=$output', ({ prompt, output, prefill, decode }) => {
    const m = model({ type: 'sliding', window: 3 })
    expect(prefillAttentionFlops(m, prompt)).toBe(operations([prefill, prefill]))
    expect(averageDecodeAttentionFlops(m, prompt, output)).toBe(operations([decode, decode]) / output)
  })

  it('combines sliding and global layers without capping global queries', () => {
    const m = model({ type: 'hybrid', slidingWindow: 2, numSlidingLayers: 1, numGlobalLayers: 1 })
    expect(prefillAttentionFlops(m, 4)).toBe(operations([[1, 2, 2, 2], [1, 2, 3, 4]]))
    expect(averageDecodeAttentionFlops(m, 1, 4)).toBe(operations([[1, 2, 2, 2], [1, 2, 3, 4]]) / 4)
  })

  it('retains all 128 MLA query heads and uses phase-specific widths', () => {
    const m = { ...model({ type: 'mla', ...mla }), numHeads: 128 }
    expect(prefillAttentionFlops(m, 2)).toBe(operations([[1, 2], [1, 2]], 128, 192, 128))
    expect(averageDecodeAttentionFlops(m, 2, 2)).toBe(operations([[2, 3], [2, 3]], 128, 576, 512) / 2)
  })

  it('caps DSA selection at available causal tokens in both phases', () => {
    const m = model({ type: 'mla-dsa', ...mla, topK: 3 })
    expect(prefillAttentionFlops(m, 5)).toBe(operations([[1, 2, 3, 3, 3], [1, 2, 3, 3, 3]], 2, 192, 128))
    expect(averageDecodeAttentionFlops(m, 2, 4)).toBe(operations([[2, 3, 3, 3], [2, 3, 3, 3]], 2, 576, 512) / 4)
  })

  it('limits MSA sparse layers by selected block capacity', () => {
    const m = model({ type: 'msa-hybrid', numFullLayers: 1, numSparseLayers: 1, blockSize: 2, topKBlocks: 2, indexHeadDim: 4 })
    expect(prefillAttentionFlops(m, 5)).toBe(operations([[1, 2, 3, 4, 5], [1, 2, 3, 4, 4]]))
    expect(averageDecodeAttentionFlops(m, 3, 3)).toBe(operations([[3, 4, 5], [3, 4, 4]]) / 3)
  })

  it.each<AttentionConfig>([
    { type: 'partial', numFullLayers: 1 },
    { type: 'delta-hybrid', numDeltaNetLayers: 1, numFullLayers: 1, numDeltaNetHeads: 4, deltaHeadDim: 8, ropeDim: 2 },
    { type: 'mamba2-hybrid', numMambaLayers: 1, numFullLayers: 1, numFfnLayers: 0, numMambaHeads: 4, mambaHeadDim: 8, ssmStateSize: 16 },
  ])('counts only attention-bearing layers for $type', attention => {
    const m = model(attention)
    expect(prefillAttentionFlops(m, 3)).toBe(operations([[1, 2, 3]]))
    expect(averageDecodeAttentionFlops(m, 2, 2)).toBe(operations([[2, 3]]) / 2)
  })

  it('excludes recurrent work from linear-MLA hybrids', () => {
    const m = model({ type: 'linear-mla-hybrid', ...mla, numLinearLayers: 1, numFullLayers: 1, numLinearHeads: 8, linearHeadDim: 128 })
    expect(prefillAttentionFlops(m, 2)).toBe(operations([[1, 2]], 2, 192, 128))
    expect(averageDecodeAttentionFlops(m, 2, 2)).toBe(operations([[2, 3]], 2, 576, 512) / 2)
  })

  it('counts integer compressed slots, limits CSA availability, and bounds every sliding branch', () => {
    const m = model({ type: 'csa-hca-hybrid', numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1, slidingWindow: 3, csaCompressionM: 2, csaTopK: 2, csaIndexerHeads: 2, csaIndexerHeadDim: 2, hcaCompressionM: 4 }, 3)
    const slots = [[1, 2, 3, 3, 3, 3, 3, 3], [1, 3, 4, 5, 5, 5, 5, 5], [1, 2, 3, 4, 4, 4, 4, 5]]
    expect(prefillAttentionFlops(m, 8)).toBe(operations(slots))
    expect(averageDecodeAttentionFlops(m, 3, 6)).toBe(operations(slots.map(row => row.slice(2))) / 6)
  })

  it('keeps the zero-output probe at the prompt position and empty prefill at zero', () => {
    expect(prefillAttentionFlops(testModel, 0)).toBe(0)
    expect(averageDecodeAttentionFlops(testModel, 3, 0)).toBe(operations([[3], [3]]))
  })
})
