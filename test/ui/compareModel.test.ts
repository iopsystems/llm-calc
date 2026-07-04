import { describe, it, expect } from 'vitest'
import { resolveCompareInput, computeCompareRow, resolveVaryingName, type ComparePivot } from '../../src/ui/compareModel'
import { ACCELERATORS, MODELS } from '../../src/data'
import { SYSTEMS } from '../../src/data/systems'

const fp16 = { weights: 'fp16', kv: 'fp16', activations: 'fp16' } as const
const wl = { promptTokens: 1024, outputTokens: 256, concurrency: 1 }

describe('resolveCompareInput', () => {
  it('resolves N-models-x-1-SKU: pivot=sku(accelerator), candidate=model', () => {
    const accel = ACCELERATORS[0]
    const model = MODELS[0]
    const pivot: ComparePivot = { kind: 'sku', id: accel.id }
    const input = resolveCompareInput(pivot, { varyingId: model.id, quant: fp16 }, wl)
    expect(input).not.toBeNull()
    expect(input!.accelerator.id).toBe(accel.id)
    expect(input!.acceleratorVariantId).toBe(accel.variants[0].id)
    expect(input!.model.id).toBe(model.id)
    expect(input!.multiDevice).toBeUndefined()
  })

  it('resolves N-SKUs-x-1-model: pivot=model, candidate=system → multiDevice set', () => {
    const system = SYSTEMS[0]
    const model = MODELS[0]
    const pivot: ComparePivot = { kind: 'model', id: model.id }
    const input = resolveCompareInput(pivot, { varyingId: system.id, quant: fp16 }, wl)
    expect(input).not.toBeNull()
    expect(input!.model.id).toBe(model.id)
    expect(input!.multiDevice?.system.id).toBe(system.id)
    expect(input!.acceleratorVariantId).toBe(system.accelerator.variantId)
  })

  it('returns null for an unknown varying id', () => {
    const pivot: ComparePivot = { kind: 'sku', id: ACCELERATORS[0].id }
    expect(resolveCompareInput(pivot, { varyingId: 'nope-not-a-model', quant: fp16 }, wl)).toBeNull()
  })

  it('returns null for an unknown pivot id', () => {
    const pivot: ComparePivot = { kind: 'sku', id: 'nope-not-a-sku' }
    expect(resolveCompareInput(pivot, { varyingId: MODELS[0].id, quant: fp16 }, wl)).toBeNull()
  })
})

describe('computeCompareRow', () => {
  it('produces an ok row with finite metrics for a valid candidate', () => {
    const row = computeCompareRow(
      { kind: 'sku', id: ACCELERATORS[0].id },
      { varyingId: MODELS[0].id, quant: fp16 }, wl,
    )
    expect(row.ok).toBe(true)
    if (row.ok) {
      expect(Number.isFinite(row.metrics.ttftMs)).toBe(true)
      expect(Number.isFinite(row.metrics.tpotMs)).toBe(true)
      expect(row.metrics.throughputTokS).toBeGreaterThan(0)
      expect(['compute', 'memory', 'comms']).toContain(row.metrics.regime)
      expect(row.name).toBe(MODELS[0].name)
    }
  })

  it('isolates errors: an unresolvable candidate becomes an error row, not a throw', () => {
    const row = computeCompareRow(
      { kind: 'sku', id: ACCELERATORS[0].id },
      { varyingId: 'nope-not-a-model', quant: fp16 }, wl,
    )
    expect(row.ok).toBe(false)
    if (!row.ok) expect(row.error).toMatch(/unknown/i)
  })

  it('isolates engine errors: an unsupported quant becomes an error row', () => {
    // fp4 activations: no datacenter accelerator lists fp4 TFLOPS → calculate throws.
    const row = computeCompareRow(
      { kind: 'sku', id: ACCELERATORS[0].id },
      { varyingId: MODELS[0].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp4' } }, wl,
    )
    expect(row.ok).toBe(false)
  })

  it('resolveVaryingName resolves the opposite dimension', () => {
    expect(resolveVaryingName({ kind: 'sku', id: ACCELERATORS[0].id }, MODELS[0].id)).toBe(MODELS[0].name)
    expect(resolveVaryingName({ kind: 'model', id: MODELS[0].id }, 'ghost-id')).toBe('ghost-id')
  })
})
