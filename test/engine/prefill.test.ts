import { describe, it, expect } from 'vitest'
import { computePrefill } from '../../src/engine/prefill'
import { testInput } from '../fixtures'
import { computeMemory } from '../../src/engine/memory'
import type { ModelArch } from '../../src/engine/types'

describe('computePrefill', () => {
  const opPoint = testInput.accelerator.variants[0].operatingPoints[0]
  const memory = computeMemory(testInput)

  it('counts active linear work plus both causal attention matrices', () => {
    // Linear work: 2 * 1000 * 10 = 20000. Each layer has 1+...+10 = 55 causal pairs.
    // QK+AV: 2 layers * 55 pairs * 2 heads * (2+2) dimensions * 2 FLOPs = 1760.
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.flops).toBe(21760)
  })

  it('reads weights and activations for one prompt', () => {
    // One prompt reads 2000 weight bytes plus 10 * (4+8) * 2 * 2 = 480 activation bytes.
    // The two-request peak allocation is 960, but prefill models one request.
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.bytes).toBe(2480)
  })

  it('timeS = max(flops/tflops, bytes/bw)', () => {
    // Compute: 21760 / 1e12 seconds. Memory: 2480 / 1e9 seconds dominates.
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.timeS).toBeCloseTo(2480 / 1e9, 12)
    expect(p.regime).toBe('memory')
  })

  it('uses activation dtype to pick tflops', () => {
    // testInput uses fp16; opPoint.tflops.fp16 = 1
    const p = computePrefill(testInput, opPoint, memory)
    expect(p.timeS).toBeGreaterThan(0)
  })

  it('attention term caps at window for sliding attention', () => {
    // Visible pairs per layer: 1+2+3+4+5+5+5+5+5+5 = 40.
    // QK+AV: 2 layers * 40 pairs * 16 FLOPs = 1280, plus 20000 linear FLOPs.
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 5 }
    }
    const input = { ...testInput, model: slidingModel }
    const slidingMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, slidingMemory)
    expect(p.flops).toBe(21280)
  })

  it('FLOPs MLP term uses activeParams for MoE', () => {
    // Active linear work: 2 * 250 * 10 = 5000.
    // Attention: 2 layers * 55 causal pairs * 16 FLOPs = 1760.
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
    const p = computePrefill(input, opPoint, moeMemory)
    expect(p.flops).toBe(6760)
  })

  it('uses expanded MLA query-key and value widths during prefill', () => {
    // Expanded MLA QK width is no-PE 2 + RoPE 2; value width is 2.
    // 2 layers * 55 pairs * 2 heads * (4+2) * 2 FLOPs = 2640, plus 20000 linear FLOPs.
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const mlaMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, mlaMemory)
    expect(p.flops).toBe(22640)
  })

  it('attention term uses hybrid formula in prefill flops', () => {
    // One sliding layer has 40 pairs (1,2,3,4,5,5,5,5,5,5), global has 55.
    // Attention: (40+55) * 16 = 1520, plus 20000 linear FLOPs.
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
    const p = computePrefill(input, opPoint, hybridMemory)
    expect(p.flops).toBe(21520)
  })

  it('attention term caps at topK for mla-dsa', () => {
    // DSA pairs per layer: 1+2+3+3+3+3+3+3+3+3 = 27.
    // Expanded MLA: 2 layers * 27 * 2 heads * (4+2) * 2 = 1296; linear work 20000.
    const dsaModel = {
      ...testInput.model,
      attention: { type: 'mla-dsa' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2, topK: 3 }
    }
    const input = { ...testInput, model: dsaModel }
    const dsaMemory = computeMemory(input)
    const p = computePrefill(input, opPoint, dsaMemory)
    expect(p.flops).toBe(21296)
  })

  it('flops for linear-mla-hybrid includes KDA per-token term', () => {
    // One full MLA layer: 55 pairs * 2 heads * (QK width 2 + value width 1) * 2 = 660.
    // KDA updates: 2 * 1 layer * 2 heads * 2 squared * 10 tokens = 160.
    // Add linear work 20000 for total 20820.
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
    const p = computePrefill(input, opPoint, hybridMemory)
    expect(p.flops).toBe(20820)
  })

  it('flops for csa-hca-hybrid uses topK for CSA layer compute', () => {
    // Local pairs per layer: 1+2+2+2+2+2+2+2+2+2 = 19, across 3 layers.
    // CSA compressed pairs: 0+1+1+2+2+3+3+3+3+3 = 21.
    // HCA compressed pairs: 0+0+0+1+1+1+1+2+2+2 = 10.
    // Total attention: (3*19+21+10) * 16 = 1408; linear work 20000.
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
    const p = computePrefill(input, opPoint, hybridMemory)
    expect(p.flops).toBe(21408)
  })
})
