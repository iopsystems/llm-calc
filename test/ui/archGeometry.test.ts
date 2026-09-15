import { describe, it, expect } from 'vitest'
import { MODELS } from '../../src/data'
import { lanesFor, fmtBytes } from '../../src/ui/archGeometry'

const byId = (id: string) => {
  const m = MODELS.find(x => x.id === id)
  if (!m) throw new Error(`no model ${id}`)
  return m
}

describe('fmtBytes', () => {
  it('formats B / KiB / MiB', () => {
    expect(fmtBytes(256)).toBe('256 B')
    expect(fmtBytes(4096)).toBe('4 KiB')
    expect(fmtBytes(1152)).toBe('1.1 KiB')
    expect(fmtBytes(2097152)).toBe('2 MiB')
  })
})

describe('lanesFor — KV-cached variants', () => {
  it('full: one lane, every layer, K+V rows, reads all', () => {
    const m = byId('llama-3.1-8b')
    const lanes = lanesFor(m)
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'full', color: 'full', count: 32,
      cache: { glyph: 'kv', bytesPerToken: 4096, fixedBytes: 0 },
      reach: { glyph: 'all' },
    })
    expect(lanes[0].cache.label).toBe('K + V · 8 KV heads × 128 · 4 KiB')
    expect(lanes[0].reach.label).toBe('all tokens')
  })

  it('sliding: window lane with the window in tokens', () => {
    const lanes = lanesFor(byId('mistral-7b-v0.1'))
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'window', color: 'window', count: 32,
      cache: { glyph: 'kv', bytesPerToken: 4096 },
      reach: { glyph: 'window', tokens: 4096, label: 'trailing 4096' },
    })
    expect(lanes[0].cache.label).toBe('K + V · 8 KV heads × 128 · 4 KiB, last 4096 tokens only')
  })

  it('hybrid: window lane then full lane, counts from config', () => {
    const lanes = lanesFor(byId('gpt-oss-120b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['window', 18], ['full', 18]])
    expect(lanes[0].reach).toMatchObject({ glyph: 'window', tokens: 128 })
    expect(lanes[1].reach).toMatchObject({ glyph: 'all' })
    expect(lanes[0].cache.bytesPerToken).toBe(2048)
  })

  it('partial: attending lane plus a pruned lane that caches nothing', () => {
    const lanes = lanesFor(byId('llama-3.3-nemotron-super-49b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['full', 49], ['pruned', 31]])
    expect(lanes[1]).toMatchObject({
      color: 'none',
      cache: { glyph: 'none', bytesPerToken: 0, fixedBytes: 0, label: 'nothing' },
      reach: { glyph: 'none', label: 'no attention' },
    })
  })

  it('lane counts sum to model.layers for these variants', () => {
    for (const id of ['llama-3.1-8b', 'mistral-7b-v0.1', 'gpt-oss-120b', 'llama-3.3-nemotron-super-49b']) {
      const m = byId(id)
      expect(lanesFor(m).reduce((n, l) => n + l.count, 0)).toBe(m.layers)
    }
  })
})
