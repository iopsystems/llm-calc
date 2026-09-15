import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  comparePivot, compareCandidates, compareWorkload, compareResults, setComparePivotKind,
  seedCompareFromCalc, seedCompareOnFirstEntry, setCompareWorkloadField, acceleratorId, systemId, modelId,
} from '../../src/ui/stores'
import { defaultPivotId, firstVaryingId } from '../../src/ui/compareModel'
import { ACCELERATORS, MODELS } from '../../src/data'
import { route } from '../../src/ui/route'

describe('compare stores', () => {
  beforeEach(() => {
    comparePivot.set({ kind: 'sku', id: ACCELERATORS[0].id })
    compareCandidates.set([{ varyingId: MODELS[0].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } }])
    compareWorkload.set({ promptTokens: 1024, outputTokens: 256, concurrency: 1 })
  })

  it('compareResults maps each candidate to a row', () => {
    compareCandidates.update(cs => [...cs, { varyingId: MODELS[1].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } }])
    const rows = get(compareResults)
    expect(rows).toHaveLength(2)
    expect(rows[0].name).toBe(MODELS[0].name)
  })

  it('a bad candidate errors its own row without killing siblings', () => {
    compareCandidates.set([
      { varyingId: MODELS[0].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } },
      { varyingId: 'ghost',      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } },
    ])
    const rows = get(compareResults)
    expect(rows[0].ok).toBe(true)
    expect(rows[1].ok).toBe(false)
  })

  it('setComparePivotKind hard-clears candidates, reseeds, preserves workload', () => {
    compareWorkload.set({ promptTokens: 999, outputTokens: 111, concurrency: 3 })
    setComparePivotKind('model')
    expect(get(comparePivot)).toEqual({ kind: 'model', id: defaultPivotId('model') })
    const cs = get(compareCandidates)
    expect(cs).toHaveLength(1)
    expect(cs[0].varyingId).toBe(firstVaryingId('model'))
    expect(get(compareWorkload)).toEqual({ promptTokens: 999, outputTokens: 111, concurrency: 3 })
  })

  it('seedCompareFromCalc seeds the sku pivot + first candidate from calc stores', () => {
    acceleratorId.set(ACCELERATORS[1].id)
    systemId.set('')
    modelId.set(MODELS[1].id)
    seedCompareFromCalc()
    expect(get(comparePivot)).toEqual({ kind: 'sku', id: ACCELERATORS[1].id })
    const cs = get(compareCandidates)
    expect(cs).toHaveLength(1)
    expect(cs[0].varyingId).toBe(MODELS[1].id)
  })
})

describe('setCompareWorkloadField', () => {
  beforeEach(() => compareWorkload.set({ promptTokens: 1024, outputTokens: 256, concurrency: 1 }))

  it('accepts positive integers and k/m suffixes', () => {
    expect(setCompareWorkloadField('promptTokens', '4k')).toBe(true)
    expect(setCompareWorkloadField('concurrency', '8')).toBe(true)
    expect(get(compareWorkload)).toEqual({ promptTokens: 4096, outputTokens: 256, concurrency: 8 })
  })

  it('rejects negative, zero, empty, and non-numeric values without touching the store', () => {
    for (const bad of ['-1', '0', '', 'abc', '1e400']) {
      expect(setCompareWorkloadField('concurrency', bad)).toBe(false)
    }
    expect(get(compareWorkload)).toEqual({ promptTokens: 1024, outputTokens: 256, concurrency: 1 })
  })
})

describe('seedCompareOnFirstEntry', () => {
  it('seeds from the calc selection at the moment the compare tab is first entered', () => {
    route.set({ tab: 'calc' })
    acceleratorId.set(ACCELERATORS[0].id); systemId.set(''); modelId.set(MODELS[0].id)
    const stop = seedCompareOnFirstEntry()
    modelId.set(MODELS[2].id)
    route.set({ tab: 'compare' })
    expect(get(compareCandidates)[0].varyingId).toBe(MODELS[2].id)
    // Later re-entries leave a configured comparison alone.
    route.set({ tab: 'calc' })
    modelId.set(MODELS[3].id)
    route.set({ tab: 'compare' })
    expect(get(compareCandidates)[0].varyingId).toBe(MODELS[2].id)
    stop()
  })

  it('seeds immediately when the page loads on the compare tab', () => {
    route.set({ tab: 'compare' })
    modelId.set(MODELS[1].id)
    const stop = seedCompareOnFirstEntry()
    expect(get(compareCandidates)[0].varyingId).toBe(MODELS[1].id)
    stop()
  })
})
