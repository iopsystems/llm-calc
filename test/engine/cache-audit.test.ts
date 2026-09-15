import { describe, expect, it } from 'vitest'
import { MODELS } from '../../src/data/models'
import * as memory from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'
import { testInput, testModel } from '../fixtures'

const model = (id: string) => MODELS.find(m => m.id === id)!
const compressed: ModelArch = { ...testModel, layers: 3, attention: {
  type: 'csa-hca-hybrid', numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
  slidingWindow: 4, csaCompressionM: 2, csaTopK: 2,
  csaIndexerHeads: 2, csaIndexerHeadDim: 2, hcaCompressionM: 4,
} }

describe('audited cache geometry', () => {
  it('Qwen 27B uses dense parameters and 48 Delta value heads', () => {
    expect(model('qwen3.5-27b').architecture.type).toBe('dense')
    expect(memory.activeParams(model('qwen3.5-27b'))).toBe(27e9)
    expect(model('qwen3.5-27b').attention).toMatchObject({ numDeltaNetHeads: 48 })
  })
  it('Qwen 35B uses 32 Delta value heads and 512 expert intermediate units', () => {
    expect(model('qwen3.5-35b-a3b').attention).toMatchObject({ numDeltaNetHeads: 32 })
    expect(model('qwen3.5-35b-a3b').intermediateDim).toBe(512)
  })
  it.each(['bf16', 'fp8', 'int4'] as const)('Delta recurrent state remains FP32 with %s KV', dtype => {
    // 24 matrices, each with 32 heads of 128 by 128 FP32 elements.
    expect(memory.deltaStateBytes(model('qwen3.5-4b'), dtype)).toBe(50_331_648)
  })
  it('V4 stores one head_dim-wide vector shared by keys and values', () => {
    // 12 local slots + 4 CSA slots + 2 HCA slots; each shared vector is 4 bytes.
    expect(memory.computeMemory({ ...testInput, model: compressed,
      workload: { ...testInput.workload, promptTokens: 7, outputTokens: 1 } }).kvCachePerRequest).toBe(72)
  })
  it('exports peak and average cache helpers', () => {
    expect(memory.cacheBytesAtSequence).toBeTypeOf('function')
    expect(memory.averageDecodeCacheBytes).toBeTypeOf('function')
  })
})

describe('peak and mean cache bytes', () => {
  it('averages contexts before each generated token, including zero-output convention', () => {
    expect(memory.averageDecodeCacheBytes(testModel, 'fp16', 1, 9)).toEqual({ bytes: 80, recurrentBytes: 0 })
    expect(memory.cacheBytesAtSequence(testModel, 'fp16', 10).bytes).toBe(160)
    expect(memory.averageDecodeCacheBytes(testModel, 'fp16', 3, 0).bytes).toBe(48)
  })
  it.each([[1, 3, 2], [3, 4, 3.75], [8, 3, 4]])('averages windows at prompt %i/output %i', (prompt, output, average) => {
    const sliding: ModelArch = { ...testModel, attention: { type: 'sliding', window: 4 } }
    expect(memory.averageDecodeCacheBytes(sliding, 'fp16', prompt, output).bytes).toBe(16 * average)
  })
  it('averages each compressed layer local window across the cap', () => {
    // Contexts 2,3,4,5,6: local window average 3.4, completed compressed-slot averages 1.8 and 0.6.
    expect(memory.averageDecodeCacheBytes(compressed, 'fp16', 2, 5).bytes).toBeCloseTo(50.4)
    expect(memory.cacheBytesAtSequence(compressed, 'fp16', 2).bytes).toBe(28)
  })
  it('shards recurrent heads independently of full-attention KV heads', () => {
    const qwen = model('qwen3.5-4b')
    const rank = memory.cacheBytesAtSequence(qwen, 'int4', 10, 8, 2)
    expect(rank.recurrentBytes).toBe(50_331_648 / 8 / 2)
    const full = memory.cacheBytesAtSequence(qwen, 'int4', 10)
    expect(rank.bytes - rank.recurrentBytes).toBe((full.bytes - full.recurrentBytes) / qwen.numKvHeads / 2)
  })
  it('replicates shared MLA latents but shards independent linear-state heads', () => {
    const hybrid: ModelArch = { ...testModel, attention: {
      type: 'linear-mla-hybrid', kvLoraRank: 5, qkRopeHeadDim: 1, qkNopeHeadDim: 1, vHeadDim: 1,
      numFullLayers: 1, numLinearLayers: 1, numLinearHeads: 4, linearHeadDim: 2,
    } }
    expect(memory.cacheBytesAtSequence(hybrid, 'fp16', 10, 4, 2)).toEqual({ bytes: 64, recurrentBytes: 4 })
  })
  it('replicates V4 shared vectors across TP and partitions layers across PP', () => {
    expect(memory.cacheBytesAtSequence(compressed, 'fp16', 8, 8, 2)).toEqual({ bytes: 36, recurrentBytes: 0 })
  })
  it('retains MSA shared index keys separately from sharded main cache', () => {
    const msa: ModelArch = { ...testModel, numKvHeads: 4, attention: {
      type: 'msa-hybrid', numFullLayers: 1, numSparseLayers: 1, blockSize: 2, topKBlocks: 3, indexHeadDim: 4,
    } }
    expect(memory.cacheBytesAtSequence(msa, 'fp16', 15, 4, 2).bytes).toBe(180)
    expect(memory.averageDecodeCacheBytes(msa, 'fp16', 10, 5, 4, 2).bytes).toBe(144)
  })
  it('uses rank cache geometry for peak capacity', () => {
    const qwen = model('qwen3.5-4b')
    const result = memory.computeMemory({ ...testInput, model: qwen, multiDevice: {
      system: {} as never, parallelism: ['tp', 'pp'], parallelismDegrees: { tp: 8, pp: 2 },
    } })
    expect(result.perRank!.kvCachePerRequest).toBe(memory.cacheBytesAtSequence(qwen, 'fp16', 15, 8, 2).bytes)
  })
})

describe('cache architecture regression', () => {
  // Synthetic two-layer models with 4 KV heads, dim=2, FP16. Reference counts
  // count stored elements independently for one TP4/PP2 rank.
  const cases: { attention: ModelArch['attention']; bytes: (s: number) => number; state: number }[] = [
    { attention: { type: 'full' }, bytes: s => 8 * s, state: 0 },
    { attention: { type: 'sliding', window: 4 }, bytes: s => 8 * Math.min(s, 4), state: 0 },
    { attention: { type: 'hybrid', numGlobalLayers: 1, numSlidingLayers: 1, slidingWindow: 4 },
      bytes: s => 4 * (s + Math.min(s, 4)), state: 0 },
    { attention: { type: 'partial', numFullLayers: 1 }, bytes: s => 4 * s, state: 0 },
    { attention: { type: 'mla', kvLoraRank: 5, qkRopeHeadDim: 1, qkNopeHeadDim: 2, vHeadDim: 2 },
      bytes: s => 12 * s, state: 0 },
    { attention: { type: 'mla-dsa', kvLoraRank: 5, qkRopeHeadDim: 1, qkNopeHeadDim: 2, vHeadDim: 2, topK: 3 },
      bytes: s => 12 * s, state: 0 },
    { attention: { type: 'linear-mla-hybrid', kvLoraRank: 5, qkRopeHeadDim: 1, qkNopeHeadDim: 2, vHeadDim: 2,
      numLinearLayers: 1, numFullLayers: 1, numLinearHeads: 4, linearHeadDim: 2 },
      bytes: s => 6 * s + 4, state: 4 },
    { attention: { type: 'delta-hybrid', numDeltaNetLayers: 1, numFullLayers: 1,
      numDeltaNetHeads: 8, deltaHeadDim: 2, ropeDim: 2 }, bytes: s => 4 * s + 16, state: 16 },
    { attention: { type: 'mamba2-hybrid', numMambaLayers: 1, numFullLayers: 1, numFfnLayers: 0,
      numMambaHeads: 8, mambaHeadDim: 2, ssmStateSize: 3 }, bytes: s => 4 * s + 24, state: 24 },
  ]
  it.each(cases)('$attention.type peak and mean match independent element counts', ({ attention, bytes, state }) => {
    const m: ModelArch = { ...testModel, numKvHeads: 4, attention }
    for (const prompt of [0, 1, 3, 4, 8]) {
      for (const output of [0, 1, 5]) {
        const contexts = Array.from({ length: Math.max(output, 1) }, (_, i) => prompt + i)
        const expected = contexts.reduce((sum, s) => sum + bytes(s), 0) / contexts.length
        expect(memory.averageDecodeCacheBytes(m, 'fp16', prompt, output, 4, 2))
          .toEqual({ bytes: expected, recurrentBytes: state })
        expect(memory.cacheBytesAtSequence(m, 'fp16', prompt + output, 4, 2))
          .toEqual({ bytes: bytes(prompt + output), recurrentBytes: state })
      }
    }
  })
  it('compressed slot means match explicit integer-block enumeration across boundaries', () => {
    for (const prompt of [0, 1, 3, 4, 7, 15]) {
      for (const output of [0, 1, 3, 8]) {
        const contexts = Array.from({ length: Math.max(output, 1) }, (_, i) => prompt + i)
        const expected = contexts.reduce((sum, s) => sum + 4 * (
          3 * Math.min(s, 4) + Math.floor(s / 2) + Math.floor(s / 4)
        ), 0) / contexts.length
        expect(memory.averageDecodeCacheBytes(compressed, 'fp16', prompt, output).bytes).toBeCloseTo(expected)
      }
    }
  })
  it('allocates whole requests on the most-loaded DP replica', () => {
    const result = memory.computeMemory({ ...testInput, workload: { ...testInput.workload, concurrency: 3 },
      multiDevice: { system: {} as never, parallelism: ['dp'], parallelismDegrees: { dp: 2 } },
    })
    expect(result.perRank!.kvCacheTotal).toBe(480)
  })
})
