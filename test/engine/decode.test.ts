import { describe, it, expect } from 'vitest'
import { computeDecode } from '../../src/engine/decode'
import { testInput } from '../fixtures'
import { computeMemory } from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'

describe('computeDecode', () => {
  const opPoint = testInput.accelerator.variants[0].operatingPoints[0]
  const memory = computeMemory(testInput)

  // testInput: prompt=10, output=5, concurrency=2
  // Decode positions are 10, 11, 12, 13, 14: average visible length is 12.

  it('counts both attention matrices at the average generated position', () => {
    // Linear work: 2 * 1000 = 2000 per request.
    // QK+AV: 2 layers * 12 average slots * 2 heads * (2+2) * 2 = 384.
    // Two requests: (2000+384) * 2 = 4768.
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.flopsPerStep).toBe(4768)
  })

  it('reads weights plus average KV traffic for the batch', () => {
    // Weights: 2000 bytes. Mean KV per request: 2 layers * 12 slots * K/V 2 * 1 KV head * 2 dims * 2 bytes = 192.
    // Two requests: 2000 + 192*2 = 2384 bytes; final capacity is not average traffic.
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.bytesPerStep).toBe(2384)
  })

  it('timePerTokenS = max(flopsPerStep/tflops, bytesPerStep/bw)', () => {
    // Compute: 4768 / 1e12 seconds. Memory: 2384 / 1e9 seconds dominates.
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.timePerTokenS).toBeCloseTo(2384 / 1e9, 12)
    expect(d.regime).toBe('memory')
  })

  it('aggregateTokensPerS = concurrency / timePerTokenS', () => {
    const d = computeDecode(testInput, opPoint, memory)
    expect(d.aggregateTokensPerS).toBeCloseTo(2 / d.timePerTokenS, 6)
  })

  it('attention term caps at window for sliding attention', () => {
    // Positions 10..14 each attend 8 slots in each of 2 layers.
    // Attention: 2 * 8 * 16 = 256 per request. Total: (2000+256) * 2 = 4512.
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 8 }
    }
    const input = { ...testInput, model: slidingModel }
    const slidingMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, slidingMemory)
    expect(d.flopsPerStep).toBe(4512)
  })

  it('reads the expected union of routed experts across the batch', () => {
    // Four equal 250-parameter experts, one route per request, uniform independent routing.
    // Of 16 ordered pairs of routes, 4 touch one expert and 12 touch two.
    // Expected weight bytes: ((4*250 + 12*500)/16) * 2 = 875.
    // Mean KV: 192 bytes/request * 2. Total: 875+384 = 1259.
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        numSharedExperts: 0,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.bytesPerStep).toBe(1259)
  })

  it('flopsPerStep MLP term uses activeParams for MoE', () => {
    // Linear work still uses active parameters per token: 2*250 = 500.
    // Attention per request: 2 layers * 12 average slots * 16 = 384.
    // Two requests: (500+384)*2 = 1768.
    const moeModel = {
      ...testInput.model,
      architecture: {
        type: 'moe' as const,
        numExperts: 4,
        numExpertsActive: 1,
        numSharedExperts: 0,
        activeParamCount: 250
      }
    }
    const input = { ...testInput, model: moeModel }
    const moeMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, moeMemory)
    expect(d.flopsPerStep).toBe(1768)
  })

  it('uses absorbed MLA query-key and latent-value widths during decode', () => {
    // Absorbed MLA QK width is latent 10 + RoPE 2; AV aggregates latent width 10.
    // Attention per request: 2 layers * 12 slots * 2 heads * (12+10) * 2 = 2112.
    // Add linear 2000 and multiply by batch 2: 8224.
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const mlaMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, mlaMemory)
    expect(d.flopsPerStep).toBe(8224)
  })

  it('attention term uses hybrid formula in decode flopsPerStep', () => {
    // One sliding layer attends 5 slots at each position; global layer averages 12.
    // QK+AV: (5+12)*16 = 272 per request. Total: (2000+272)*2 = 4544.
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'hybrid' as const,
        slidingWindow: 5,
        numSlidingLayers: 1,
        numGlobalLayers: 1
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(4544)
  })

  it('attention term caps at topK for mla-dsa', () => {
    // DSA selects 4 available slots at each position in both layers.
    // Absorbed MLA: 2 layers * 4 slots * 2 heads * (12+10) * 2 = 704.
    // Total: (2000+704)*2 = 5408.
    const dsaModel = {
      ...testInput.model,
      attention: { type: 'mla-dsa' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2, topK: 4 }
    }
    const input = { ...testInput, model: dsaModel }
    const dsaMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, dsaMemory)
    expect(d.flopsPerStep).toBe(5408)
  })

  it('flopsPerStep and bytesPerStep for linear-mla-hybrid include KDA terms', () => {
    // One full MLA layer: 12 average slots * 2 heads * (QK width 6 + latent value width 5) * 2 = 528.
    // KDA updates: 2 * 1 layer * 2 heads * 2 squared = 16 per request.
    // Total FLOPs: (2000+528+16)*2 = 5088.
    // Mean cache: (latent 5 + RoPE 1)*2 bytes*12 slots + 16 state bytes = 160.
    // Read and write state: (160+16)*2 requests + 2000 weights = 2352 bytes.
    const hybridModel = {
      ...testInput.model,
      attention: {
        type: 'linear-mla-hybrid' as const,
        kvLoraRank: 5, qkRopeHeadDim: 1,
        qkNopeHeadDim: 1, vHeadDim: 1,
        numLinearLayers: 1, numFullLayers: 1,
        numLinearHeads: 2, linearHeadDim: 2
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(5088)
    expect(d.bytesPerStep).toBe(2352)
  })

  it('MTP doubles aggregateTokensPerS and halves timePerTokenS for numNextnLayers=1', () => {
    const mtpModel = { ...testInput.model, numNextnLayers: 1 }
    const input = { ...testInput, model: mtpModel }
    const mtpMemory = computeMemory(input)
    const dMtp = computeDecode(input, opPoint, mtpMemory)
    const dBase = computeDecode(testInput, opPoint, memory)
    // Per-pass FLOPs and bytes are unchanged
    expect(dMtp.flopsPerStep).toBe(dBase.flopsPerStep)
    expect(dMtp.bytesPerStep).toBe(dBase.bytesPerStep)
    // Effective per-token time halves; aggregate throughput doubles
    expect(dMtp.timePerTokenS).toBeCloseTo(dBase.timePerTokenS / 2, 12)
    expect(dMtp.aggregateTokensPerS).toBeCloseTo(dBase.aggregateTokensPerS * 2, 6)
  })

  it('flopsPerStep and bytesPerStep for csa-hca-hybrid include all three layer types', () => {
    // Positions 10..14: 3 local branches each attend 2 slots; CSA selects 3 slots.
    // HCA complete slots are 2,2,3,3,3, averaging 2.6.
    // Attention per request: (6+3+2.6)*16 = 185.6; FLOPs: (2000+185.6)*2 = 4371.2.
    // Persistent CSA slots are 5,5,6,6,7 (mean 5.8), not selected topK.
    // Shared KV vector has 2 dims * 2 bytes = 4 bytes per slot.
    // Mean traffic: 2000 weights + (6+5.8+2.6)*4*2 requests = 2115.2 bytes.
    const hybridModel: ModelArch = {
      ...testInput.model,
      layers: 3,
      attention: {
        type: 'csa-hca-hybrid',
        numSlidingLayers: 1, numCsaLayers: 1, numHcaLayers: 1,
        slidingWindow: 2,
        csaCompressionM: 2, csaTopK: 3,
        csaIndexerHeads: 2, csaIndexerHeadDim: 2,
        hcaCompressionM: 4
      }
    }
    const input = { ...testInput, model: hybridModel }
    const hybridMemory = computeMemory(input)
    const d = computeDecode(input, opPoint, hybridMemory)
    expect(d.flopsPerStep).toBe(4371.2)
    expect(d.bytesPerStep).toBe(2115.2)
  })
})
