import { describe, it, expect } from 'vitest'
import { computeMemory, attendedSeqlenSummedOverLayers } from '../../src/engine/memory'
import { testInput } from '../fixtures'
import type { ModelArch } from '../../src/engine/types'

describe('computeMemory', () => {
  it('weights = paramCount × bytes(weight_dtype)', () => {
    // paramCount=1000, fp16=2 bytes → 2000 bytes
    const m = computeMemory(testInput)
    expect(m.weights).toBe(2000)
  })

  it('kvCachePerRequest = 2 × layers × kv_heads × head_dim × bytes(kv_dtype) × (prompt + output)', () => {
    // 2 × 2 × 1 × 2 × 2 (fp16) = 16 bytes per token
    // × (10 + 5) = 240 bytes per request
    const m = computeMemory(testInput)
    expect(m.kvCachePerRequest).toBe(240)
  })

  it('kvCacheTotal = kvCachePerRequest × concurrency', () => {
    // 240 × 2 = 480
    const m = computeMemory(testInput)
    expect(m.kvCacheTotal).toBe(480)
  })

  it('activationsPeak = concurrency × prompt × (hidden + intermediate) × bytes(act_dtype) × 2', () => {
    // 2 × 10 × (4 + 8) × 2 (fp16) × 2 = 960 bytes
    const m = computeMemory(testInput)
    expect(m.activationsPeak).toBe(960)
  })

  it('total = weights + kvCacheTotal + activationsPeak', () => {
    // 2000 + 480 + 960 = 3440
    const m = computeMemory(testInput)
    expect(m.total).toBe(3440)
  })

  it('hbmCapacityGB echoed from chosen variant', () => {
    const m = computeMemory(testInput)
    expect(m.hbmCapacityGB).toBe(1)
  })

  it('headroom = hbmCapacity_bytes − total, fits when ≥ 0', () => {
    // 1 GB = 1_073_741_824 bytes; headroom = 1_073_741_824 − 3440
    const m = computeMemory(testInput)
    expect(m.headroom).toBe(1_073_741_824 - 3440)
    expect(m.fits).toBe(true)
  })

  it('fits=false and negative headroom on OOM', () => {
    const bigModel = { ...testInput.model, paramCount: 10_000_000_000 }  // 10B params × 2B = 20GB
    const m = computeMemory({ ...testInput, model: bigModel })
    expect(m.fits).toBe(false)
    expect(m.headroom).toBeLessThan(0)
  })

  it('kvCachePerRequest caps at window for sliding attention', () => {
    // testModel uses full attention; build a sliding variant with window=8
    // (prompt+output=15, so should cap at 8 tokens instead of 15)
    const slidingModel = {
      ...testInput.model,
      attention: { type: 'sliding' as const, window: 8 }
    }
    const input = { ...testInput, model: slidingModel }
    const m = computeMemory(input)
    // 16 bytes per token × 8 (window) = 128 bytes per request
    expect(m.kvCachePerRequest).toBe(128)
    // × concurrency 2 = 256 bytes
    expect(m.kvCacheTotal).toBe(256)
  })

  it('kvCachePerRequest uses MLA formula for MLA models', () => {
    // testModel: layers=2, prompt+output=15.
    // MLA with kvLoraRank=10, rope=2: layers × (10+2) × 2 (fp16) = 48 bytes/token.
    // × 15 tokens = 720 bytes per request.
    // × concurrency 2 = 1440 bytes total.
    const mlaModel = {
      ...testInput.model,
      attention: { type: 'mla' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2 }
    }
    const input = { ...testInput, model: mlaModel }
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(720)
    expect(m.kvCacheTotal).toBe(1440)
  })

  it('kvCachePerRequest uses hybrid formula: numSliding × min(seq,W) + numGlobal × seq', () => {
    // testModel: layers=2, kvHeads=1, headDim=2, fp16 KV; prompt+output=15.
    // Hybrid with slidingWindow=5, numSlidingLayers=1, numGlobalLayers=1:
    //   per-layer KV bytes = 2 × 1 × 2 × 2 = 8
    //   attendedSeqlen = 1 × min(15, 5) + 1 × 15 = 5 + 15 = 20
    //   kvCachePerRequest = 8 × 20 = 160
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
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(160)
    // × concurrency 2 = 320 bytes total
    expect(m.kvCacheTotal).toBe(320)
  })

  it('kvCachePerRequest uses MLA formula for mla-dsa (DSA does not shrink KV)', () => {
    // testModel: layers=2, fp16 KV; prompt+output=15.
    // MLA-DSA with kvLoraRank=10, rope=2, topK=4:
    //   per-layer KV bytes = (10+2) × 2 = 24
    //   attendedSeqlen = 2 × 15 = 30  (KV unaffected by topK — every past token cached)
    //   kvCachePerRequest = 24 × 30 = 720
    const dsaModel = {
      ...testInput.model,
      attention: { type: 'mla-dsa' as const, kvLoraRank: 10, qkRopeHeadDim: 2, qkNopeHeadDim: 2, vHeadDim: 2, topK: 4 }
    }
    const input = { ...testInput, model: dsaModel }
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(720)
    // × concurrency 2 = 1440 bytes total
    expect(m.kvCacheTotal).toBe(1440)
  })

  it('kvCachePerRequest for linear-mla-hybrid = MLA kv + KDA state', () => {
    // testModel: layers=2, fp16; prompt+output=15.
    // linear-mla-hybrid with numLinear=1, numFull=1; MLA kvLoraRank=5, rope=1;
    // KDA: numLinearHeads=2, linearHeadDim=2.
    //   per-full-layer-per-token KV bytes = (5 + 1) × 2 = 12
    //   attendedSeqlen for kv (numFull × seq) = 1 × 15 = 15
    //   KDA state bytes = 1 × 2 × 2² × 2 = 16
    //   kvCachePerRequest = 12 × 15 + 16 = 196
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
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(196)
    // × concurrency 2 = 392 bytes total
    expect(m.kvCacheTotal).toBe(392)
  })

  it('exposes decodeActivationsPeak much smaller than activationsPeak (prefill activations)', () => {
    const m = computeMemory(testInput)
    expect(m.decodeActivationsPeak).toBeLessThan(m.activationsPeak)
    // decode activations are O(1 × hidden), prefill are O(promptTokens × hidden) — gap >> 10x in test config
    expect(m.activationsPeak / m.decodeActivationsPeak).toBeGreaterThan(2)
  })

  it("prefillSide.total equals weights + kvCacheTotal + activationsPeak (= today's total)", () => {
    const m = computeMemory(testInput)
    expect(m.prefillSide.total).toBe(m.weights + m.kvCacheTotal + m.activationsPeak)
    expect(m.prefillSide.total).toBe(m.total)
  })

  it('decodeSide.total uses decodeActivationsPeak instead of prefill activations', () => {
    const m = computeMemory(testInput)
    expect(m.decodeSide.total).toBe(m.weights + m.kvCacheTotal + m.decodeActivationsPeak)
    expect(m.decodeSide.total).toBeLessThan(m.prefillSide.total)
  })

  it('hbmCapacityGB per side defaults to prefill variant when decodeAccelerator absent', () => {
    const m = computeMemory(testInput)
    expect(m.prefillSide.hbmCapacityGB).toBe(testInput.accelerator.variants[0].hbmCapacityGB)
    expect(m.decodeSide.hbmCapacityGB).toBe(testInput.accelerator.variants[0].hbmCapacityGB)
  })

  it('per-side fits flags computed against their respective HBM capacities', () => {
    const m = computeMemory(testInput)
    const cap = testInput.accelerator.variants[0].hbmCapacityGB * 1024 * 1024 * 1024
    expect(m.prefillSide.fits).toBe(m.prefillSide.total <= cap)
    expect(m.decodeSide.fits).toBe(m.decodeSide.total <= cap)
  })

  it('backward-compat: total/fits/headroom/hbmCapacityGB mirror prefillSide', () => {
    const m = computeMemory(testInput)
    expect(m.total).toBe(m.prefillSide.total)
    expect(m.fits).toBe(m.prefillSide.fits)
    expect(m.headroom).toBe(m.prefillSide.headroom)
    expect(m.hbmCapacityGB).toBe(m.prefillSide.hbmCapacityGB)
  })

  it('kvCachePerRequest for csa-hca-hybrid sums sliding + CSA + HCA contributions', () => {
    // testModel base: prompt+output=15, fp16, concurrency=2.
    // csa-hca-hybrid with layers=3 (1 sliding + 1 CSA + 1 HCA):
    //   slidingWindow=2, csaCompressionM=2, csaTopK=3, hcaCompressionM=4
    // attendedSeqlen(forKv=true) =
    //   1 × min(15, 2) + 1 × (floor(15/2) + 2) + 1 × (floor(15/4) + 2) = 16
    // kvBytesPerTokenPerLayer = 2 dim × 2 bytes (fp16) = 4 (shared K/V)
    // kvCachePerRequest = 4 × 16 = 64
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
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(64)
    expect(m.kvCacheTotal).toBe(128)
  })

  it('msa-hybrid: full KV storage, but compute attention capped at topKBlocks × blockSize on sparse layers', () => {
    // testModel base: layers=2, kvHeads=1, headDim=2, fp16; prompt+output=15.
    // msa-hybrid with 1 full + 1 sparse, blockSize=2, topKBlocks=3 → cap 6 tokens.
    // Main KV: 2 × 15 × 8 = 240; sparse index keys: 1 × 15 × 2 × 2 = 60.
    const msaModel: ModelArch = {
      ...testInput.model,
      attention: {
        type: 'msa-hybrid',
        numFullLayers: 1, numSparseLayers: 1,
        blockSize: 2, topKBlocks: 3, indexHeadDim: 2
      }
    }
    const input = { ...testInput, model: msaModel }
    const m = computeMemory(input)
    expect(m.kvCachePerRequest).toBe(300)
    // Compute attention: 1 × 15 + 1 × min(15, 6) = 21
    expect(attendedSeqlenSummedOverLayers(msaModel, 15)).toBe(21)
    // Below the cap, sparse behaves as full: 1 × 4 + 1 × min(4, 6) = 8
    expect(attendedSeqlenSummedOverLayers(msaModel, 4)).toBe(8)
  })

  it('msa-hybrid throws when layer counts do not sum to model.layers', () => {
    const badModel: ModelArch = {
      ...testInput.model,
      attention: {
        type: 'msa-hybrid',
        numFullLayers: 1, numSparseLayers: 3,
        blockSize: 2, topKBlocks: 3, indexHeadDim: 2
      }
    }
    expect(() => attendedSeqlenSummedOverLayers(badModel, 15)).toThrow(/must sum/)
  })

  describe('perRank', () => {
    // buildSide only reads parallelism fields off multiDevice; system is unused.
    const md = (degrees: Record<string, number>) => ({
      system: {} as never,
      parallelism: Object.keys(degrees) as ('tp' | 'pp' | 'ep' | 'dp')[],
      parallelismDegrees: degrees
    })

    it.each([
      [{ tp: 4 }, 360, 720],
      [{ tp: 4, pp: 2 }, 180, 360],
      [{ tp: 4, pp: 2, dp: 2, ep: 2 }, 180, 180],
    ])('MSA replicates index keys across TP, divides across PP, and batches per DP replica: %j', (degrees, perRequest, total) => {
      // Main KV: 2 layers × 15 tokens × 2 K/V × 4 heads × 2 dim × 2B = 960.
      // Index: 1 sparse layer × 15 tokens × 4 dim × 2B = 120 (shared head).
      const model: ModelArch = {
        ...testInput.model, numKvHeads: 4,
        attention: { type: 'msa-hybrid', numFullLayers: 1, numSparseLayers: 1,
          blockSize: 2, topKBlocks: 3, indexHeadDim: 4 },
      }
      const m = computeMemory({ ...testInput, model,
        multiDevice: md({ tp: 2 }), decodeMultiDevice: md(degrees) })
      expect(m.kvCachePerRequest).toBe(1080)
      expect(m.prefillSide.perRank!.kvCachePerRequest).toBe(600)
      expect(m.decodeSide.perRank!.kvCachePerRequest).toBe(perRequest)
      expect(m.decodeSide.perRank!.kvCacheTotal).toBe(total)
    })

    it('exposes kvCacheTotal = per-rank kv per request × per-replica concurrency', () => {
      // numKvHeads=1 caps the KV shard at 1 under TP=2, so per-rank KV total
      // stays the full 240 × 2 = 480 while weights and activations halve.
      const m = computeMemory({ ...testInput, multiDevice: md({ tp: 2 }) })
      expect(m.perRank!.kvCacheTotal).toBe(480)
      expect(m.perRank!.weights).toBe(1000)
      expect(m.perRank!.activationsPeak).toBe(480)
    })

    it('per-rank components sum to the per-rank total (what the memory bar stacks)', () => {
      const m = computeMemory({ ...testInput, multiDevice: md({ tp: 2 }) })
      const pr = m.perRank!
      expect(pr.weights + pr.kvCacheTotal + pr.activationsPeak).toBe(pr.total)
    })
  })
})


it('DP activation capacity uses whole requests on the most-loaded replica', () => {
  const input = { ...testInput, workload: { ...testInput.workload, concurrency: 3 },
    multiDevice: { system: {} as never, parallelism: ['dp' as const], parallelismDegrees: { dp: 2 } },
  }
  const memory = computeMemory(input)
  // Three requests split 2+1: rank peak reserves two activation sets.
  expect(memory.prefillSide.perRank!.activations).toBe(960)
  expect(memory.decodeSide.perRank!.activations).toBe(96)
})
