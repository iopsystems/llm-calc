import { describe, it, expect } from 'vitest'
import { MODELS } from '../../src/data'
import { lanesFor, fmtBytes, headsFor, captionFor, geometryModel } from '../../src/ui/archGeometry'
import { kvBytesPerTokenAtContext, fixedStateBytes, KV_REF_DTYPE } from '../../src/ui/catalogMetrics'
import { kvBytesPerTokenPerLayer } from '../../src/engine/memory'

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

describe('lanesFor — recurrent-state variants', () => {
  it('linear-mla-hybrid: KDA state lane then MLA latent lane', () => {
    const lanes = lanesFor(byId('kimi-linear'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['kda', 20], ['mla', 7]])
    expect(lanes[0]).toMatchObject({
      color: 'state',
      cache: { glyph: 'state', bytesPerToken: 0, fixedBytes: 32 * 128 * 128 * 2 },
      reach: { glyph: 'state', label: 'one state read' },
    })
    expect(lanes[0].cache.label).toBe('32 heads × 128² state · 1 MiB per layer')
    expect(lanes[1].cache).toMatchObject({ glyph: 'latent', bytesPerToken: (512 + 64) * 2 })
  })

  it('delta-hybrid: DeltaNet state lane then full-attention lane', () => {
    const lanes = lanesFor(byId('qwen3.5-397b-a17b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['delta', 45], ['full', 15]])
    expect(lanes[0].cache).toMatchObject({ glyph: 'state', fixedBytes: 64 * 128 * 128 * 2 })
    expect(lanes[0].cache.label).toBe('64 heads × 128² state · 2 MiB per layer')
    expect(lanes[1].cache).toMatchObject({ glyph: 'kv', bytesPerToken: 2048 })
  })

  it('mamba2-hybrid: Mamba state (fp32) lane, attention lane, FFN-only lane', () => {
    const lanes = lanesFor(byId('nemotron-3-nano-30b-a3b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['mamba', 23], ['full', 6], ['ffn', 23]])
    expect(lanes[0].cache).toMatchObject({ glyph: 'state', fixedBytes: 64 * 64 * 128 * 4 })
    expect(lanes[0].cache.label).toBe('64 heads × 64 × 128 fp32 state · 2 MiB per block')
    expect(lanes[2]).toMatchObject({ kind: 'ffn', color: 'none', cache: { glyph: 'none' }, reach: { glyph: 'none' } })
  })
})

describe('lanesFor — catalog-wide invariants', () => {
  it('handles every model, counts sum to layers, bytes are finite and non-negative', () => {
    for (const m of MODELS) {
      const lanes = lanesFor(m)
      expect(lanes.length, m.id).toBeGreaterThan(0)
      expect(lanes.reduce((n, l) => n + l.count, 0), m.id).toBe(m.layers)
      for (const l of lanes) {
        expect(l.count, `${m.id} ${l.kind}`).toBeGreaterThan(0)
        expect(Number.isFinite(l.cache.bytesPerToken) && l.cache.bytesPerToken >= 0, `${m.id} ${l.kind}`).toBe(true)
        expect(Number.isFinite(l.cache.fixedBytes) && l.cache.fixedBytes >= 0, `${m.id} ${l.kind}`).toBe(true)
      }
    }
  })
})

describe('headsFor', () => {
  it('GQA model: one gqa entry', () => {
    expect(headsFor(byId('llama-3.1-8b'))).toEqual([
      { kind: 'gqa', numHeads: 32, numKvHeads: 8, headDim: 128 },
    ])
  })
  it('MLA model: latent entry with full-KV width vs latent width', () => {
    expect(headsFor(byId('deepseek-v3'))).toEqual([
      { kind: 'latent', numHeads: 128, numKvHeads: 128, headDim: 192, fullKvWidth: 128 * 192 * 2, latentWidth: 512 + 64 },
    ])
  })
  it('delta-hybrid: state entry then gqa entry', () => {
    expect(headsFor(byId('qwen3.5-397b-a17b'))).toEqual([
      { kind: 'state', numHeads: 32, numKvHeads: 2, headDim: 256, state: { heads: 64, dim: 128, inner: 128 } },
      { kind: 'gqa', numHeads: 32, numKvHeads: 2, headDim: 256 },
    ])
  })
  it('mamba2-hybrid: state inner dim is the SSM state size', () => {
    expect(headsFor(byId('nemotron-3-nano-30b-a3b'))[0].state).toEqual({ heads: 64, dim: 64, inner: 128 })
  })
  it('linear-mla-hybrid: state entry then latent entry', () => {
    expect(headsFor(byId('kimi-linear')).map(h => h.kind)).toEqual(['state', 'latent'])
  })
})

describe('captionFor', () => {
  it('every caption ends with the counts-not-order sentence', () => {
    for (const t of ['full', 'mla-dsa', 'mamba2-hybrid', 'partial'] as const) {
      expect(captionFor(t).endsWith('Bar lengths are counts, not order.')).toBe(true)
    }
  })
  it('sparse variants state the whole-cache-streamed simplification', () => {
    for (const t of ['mla-dsa', 'msa-hybrid', 'csa-hca-hybrid'] as const) {
      expect(captionFor(t)).toContain('A decode step streams the whole cache; selection lowers attention FLOPs, not bytes.')
    }
    expect(captionFor('msa-hybrid')).toContain('not sharded by TP')
    expect(captionFor('mla')).not.toContain('streams the whole cache')
  })
})

describe('geometryModel', () => {
  it('dense FFN carries hidden and intermediate dims', () => {
    const g = geometryModel(byId('llama-3.1-8b'))
    expect(g.ffn).toEqual({ type: 'dense', hiddenDim: 4096, intermediateDim: 14336 })
    expect(g.schematicId).toBe('full')
    expect(g.layers).toBe(32)
    expect(g.maxContext).toBe(131072)
  })
  it('MoE FFN carries routed/active/shared and the active ratio', () => {
    const g = geometryModel(byId('deepseek-v3'))
    expect(g.ffn).toEqual({ type: 'moe', routed: 256, active: 8, shared: 1, activeRatio: 37e9 / 671e9 })
  })

  it('lane bytes agree with the engine at max context for every catalog model', () => {
    for (const m of MODELS) {
      const g = geometryModel(m)
      const S = m.maxContext
      const p = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
      let sum = 0
      let fixed = 0
      for (const l of g.lanes) {
        fixed += l.count * l.cache.fixedBytes
        if (l.cache.glyph === 'state' || l.cache.glyph === 'none') continue
        if (l.cache.glyph === 'kvcomp') {
          sum += l.count * (l.cache.bytesPerToken + p * Math.min(l.reach.local!, S) / S)
        } else if (l.reach.glyph === 'window') {
          sum += l.count * l.cache.bytesPerToken * Math.min(l.reach.tokens!, S) / S
        } else {
          sum += l.count * l.cache.bytesPerToken
        }
      }
      expect(sum, m.id).toBeCloseTo(kvBytesPerTokenAtContext(m, S), 6)
      expect(fixed, m.id).toBe(fixedStateBytes(m))
    }
  })
})
