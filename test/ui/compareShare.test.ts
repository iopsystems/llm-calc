import { describe, it, expect } from 'vitest'
import { encodeCompare, decodeCompare, type CompareState } from '../../src/ui/compareShare'
import { ACCELERATORS, MODELS } from '../../src/data'
import type { Quantization } from '../../src/engine/types'

const q = (w: string, kv: string, a: string) => ({ weights: w, kv, activations: a } as Quantization)

function sample(): CompareState {
  return {
    pivot: { kind: 'sku', id: ACCELERATORS[0].id },
    workload: { promptTokens: 1024, outputTokens: 256, concurrency: 4 },
    candidates: [
      { varyingId: MODELS[0].id, quant: q('fp16', 'fp16', 'fp16') },
      { varyingId: MODELS[1].id, quant: q('fp8', 'fp8', 'bf16') },
    ],
  }
}

describe('compare codec', () => {
  it('round-trips a full state', () => {
    const s = sample()
    const decoded = decodeCompare(encodeCompare(s))
    expect(decoded).toEqual(s)
  })

  it('encodes candidates as repeated c= keys', () => {
    const enc = encodeCompare(sample())
    expect(enc.match(/(^|&)c=/g)?.length).toBe(2)
    expect(enc).toContain('piv=sku%3A' + ACCELERATORS[0].id)  // ':' url-encoded
  })

  it('drops unknown candidate ids but keeps the good ones', () => {
    const s = sample()
    const enc = encodeCompare(s) + '&c=ghost-model~fp16.fp16.fp16'
    const decoded = decodeCompare(enc)
    expect(decoded!.candidates).toHaveLength(2)
  })

  it('drops candidates with an unknown dtype', () => {
    const enc = encodeCompare({ ...sample(), candidates: [] }) + '&c=' + MODELS[0].id + '~fp16.fp16.notadtype'
    expect(decodeCompare(enc)!.candidates).toHaveLength(0)
  })

  it('returns null when the pivot id is invalid', () => {
    expect(decodeCompare('piv=sku%3Aghost&pt=1&ot=1&cc=1')).toBeNull()
  })

  it('returns null when piv is missing entirely', () => {
    expect(decodeCompare('pt=1&ot=1&cc=1')).toBeNull()
  })
})
