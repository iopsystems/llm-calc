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

describe('lanesFor — latent and sparse variants', () => {
  it('mla: one latent lane, bytes = (kvLoraRank + qkRopeHeadDim) × 2', () => {
    const lanes = lanesFor(byId('deepseek-v3'))
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'mla', color: 'full', count: 61,
      cache: { glyph: 'latent', bytesPerToken: (512 + 64) * 2, fixedBytes: 0 },
      reach: { glyph: 'all' },
    })
    expect(lanes[0].cache.label).toBe('latent 512 + RoPE key 64 · 1.1 KiB')
  })

  it('mla-dsa: same cache as mla, reach is top-k tokens', () => {
    const lanes = lanesFor(byId('deepseek-v3.2'))
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'dsa', color: 'sparse', count: 61,
      cache: { glyph: 'latent', bytesPerToken: (512 + 64) * 2 },
      reach: { glyph: 'topk', tokens: 2048, label: 'top-2048 tokens' },
    })
  })

  it('msa-hybrid: full lane, then sparse lane whose cache adds the shared index key', () => {
    const lanes = lanesFor(byId('minimax-m3'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['full', 3], ['msa', 57]])
    expect(lanes[1]).toMatchObject({
      color: 'sparse',
      cache: { glyph: 'kvidx', bytesPerToken: 2048 + 128 * 2 },
      reach: { glyph: 'topk', tokens: 16 * 128, label: '16 blocks × 128 tokens' },
    })
    expect(lanes[1].cache.label).toBe('K + V 2 KiB + shared index key 128 · 256 B')
  })

  it('csa-hca-hybrid (V4-Flash): window, CSA ÷4 with top-k, HCA ÷128 compressed', () => {
    const lanes = lanesFor(byId('deepseek-v4-flash'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['window', 2], ['csa', 21], ['hca', 20]])
    expect(lanes[0].reach).toMatchObject({ glyph: 'window', tokens: 128 })
    expect(lanes[1]).toMatchObject({
      color: 'sparse',
      cache: { glyph: 'kvcomp', ratio: 4, bytesPerToken: 2048 / 4 },
      reach: { glyph: 'topk', tokens: 512 * 4, local: 128 },
    })
    expect(lanes[1].cache.label).toBe('K + V 2 KiB per 4 tokens · 512 B/token')
    expect(lanes[1].reach.label).toBe('top-512 of the 1 : 4 stream + 128 local')
    expect(lanes[2]).toMatchObject({
      cache: { glyph: 'kvcomp', ratio: 128, bytesPerToken: 2048 / 128 },
      reach: { glyph: 'compress', ratio: 128, local: 128, label: 'all of the 1 : 128 stream + 128 local' },
    })
  })

  it('csa-hca-hybrid (V4-Pro): no sliding lane when numSlidingLayers is 0', () => {
    const lanes = lanesFor(byId('deepseek-v4-pro'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['csa', 30], ['hca', 31]])
  })
})
