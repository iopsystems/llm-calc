import { describe, it, expect } from 'vitest'
import { calculate } from '../../src/engine/calc'
import { defaultParallelism, perRankMemoryDivisors } from '../../src/engine/parallelism'
import { MODELS } from '../../src/data/models'
import { SYSTEMS } from '../../src/data/systems'
import { testInput } from '../fixtures'
import type { ModelArch, MultiAcceleratorSystem } from '../../src/engine/types'

const system: MultiAcceleratorSystem = {
  id: 'audit-8', name: 'Audit 8', vendor: 'test', releaseDate: '2026-01', formFactor: 'baseboard',
  accelerator: { id: 'test-accel', variantId: 'v', count: 8 }, interconnectId: 'unconstrained-test',
  aggregate: { totalHbmGB: 8, fabricBidirectionalTBs: 1 },
}
const dense: ModelArch = { ...testInput.model, numHeads: 8, numKvHeads: 8, paramCount: 1_000_000 }
const moe: ModelArch = { ...testInput.model,
  architecture: { type: 'moe', numExperts: 4, numExpertsActive: 1, numSharedExperts: 0, activeParamCount: 400 } }

describe('model math audit runtime regressions', () => {
  it('default MoE placement conserves all weights within the physical ranks', () => {
    const m = MODELS.find(m => m.id === 'deepseek-v3')!
    const hgx = SYSTEMS.find(s => s.id === 'hgx-h100-8')!
    const config = defaultParallelism(hgx, m)
    const divisors = perRankMemoryDivisors(config.parallelism, config.parallelismDegrees, m)
    expect(m.paramCount / divisors.weights * 8).toBeGreaterThanOrEqual(m.paramCount)
  })

  it('EP replicates the shared pool instead of sharding every parameter', () => {
    // P1000, A400, top1/4 => shared200 + routed800. EP2 stores200+400 each.
    const d = perRankMemoryDivisors(['ep'], { ep: 2 }, moe)
    expect(1000 / d.weights).toBeCloseTo(600)
  })

  it('rejects a mesh requiring 64 ranks on eight physical devices', () => {
    expect(() => calculate({ ...testInput, model: moe,
      multiDevice: { system, parallelism: ['tp', 'ep'], parallelismDegrees: { tp: 8, ep: 8 } } }))
      .toThrow(/physical|devices|ranks/i)
  })

  it('TP8 distributes dense weight reads and the eight KV heads', () => {
    const input = { ...testInput, model: dense, workload: { promptTokens: 10, outputTokens: 1, concurrency: 1 } }
    const r = calculate({ ...input,
      multiDevice: { system, parallelism: ['tp'], parallelismDegrees: { tp: 8 } } })
    // Main weights2e6 + KV2layers*2KV*8heads*2dim*2B*10context =2001280B.
    expect(r.perf.peak.decode.timePerTokenS).toBeCloseTo(250160 / 1e9, 12)
    // Prefill's one-prompt activation traffic is 10*(4+8)*2B*2=480B.
    expect(r.perf.peak.prefill.timeS).toBeCloseTo(250060 / 1e9, 12)
  })

  it('PP saves capacity but does not divide sequential single-request latency', () => {
    const input = { ...testInput, model: dense, workload: { promptTokens: 10, outputTokens: 1, concurrency: 1 } }
    const single = calculate(input)
    const pp = calculate({ ...input, multiDevice: { system, parallelism: ['pp'], parallelismDegrees: { pp: 2 } } })
    expect(pp.perf.peak.decode.timePerTokenS).toBe(single.perf.peak.decode.timePerTokenS)
    expect(pp.memory.perRank!.weights).toBe(single.memory.weights / 2)
  })

  it('DP partitions a full batch while retaining at least one request per active replica', () => {
    const input = { ...testInput, model: dense, workload: { promptTokens: 10, outputTokens: 1, concurrency: 1 } }
    const single = calculate(input)
    const md = { system, parallelism: ['dp'] as const, parallelismDegrees: { dp: 2 } }
    const dp = calculate({ ...input, workload: { ...input.workload, concurrency: 2 },
      multiDevice: { ...md, parallelism: ['dp'] } })
    expect(dp.perf.peak.decode.timePerTokenS).toBe(single.perf.peak.decode.timePerTokenS)
    expect(dp.perf.peak.decode.aggregateTokensPerS).toBe(single.perf.peak.decode.aggregateTokensPerS * 2)
  })

  it('batched MoE reads the expected union of routed experts', () => {
    // shared200 + routed800*(1-(3/4)^2)=550 params. KV fixture at context10=160/request.
    const r = calculate({ ...testInput, model: moe,
      workload: { promptTokens: 10, outputTokens: 1, concurrency: 2 } })
    expect(r.perf.peak.decode.bytesPerStep).toBe(1420)
  })

  it('aggregate MoE traffic sums uneven DP replica batches', () => {
    const r = calculate({ ...testInput, model: moe,
      workload: { promptTokens: 10, outputTokens: 1, concurrency: 3 },
      multiDevice: { system, parallelism: ['dp'], parallelismDegrees: { dp: 2 } } })
    // Replica batches 2+1: (550+400) params * 2B + 3 * 160B cache.
    expect(r.perf.peak.decode.bytesPerStep).toBe(2380)
    expect(r.perf.peak.decode.rankBytes).toBe(1420)
  })

  it('decode runtime reads mean sequence storage while capacity reserves the final sequence', () => {
    const r = calculate({ ...testInput, workload: { promptTokens: 1, outputTokens: 9, concurrency: 1 } })
    // Contexts1..9 average5. Main KV16B/token; final allocation10*16.
    expect(r.memory.kvCachePerRequest).toBe(160)
    expect(r.perf.peak.decode.bytesPerStep).toBe(2080)
  })
})
