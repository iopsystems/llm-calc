// Per-model geometry for the Info tab figure. Pure; every byte figure comes
// from src/engine/memory.ts so the drawing cannot disagree with the calc.
import type { ModelArch } from '../engine/types'
import { kvBytesPerTokenPerLayer } from '../engine/memory'
import { KV_REF_DTYPE } from './catalogMetrics'
import type { LaneColor, CacheGlyph, ReachGlyph } from './figure/types'

export type LaneKind =
  | 'full' | 'window' | 'mla' | 'dsa' | 'msa' | 'csa' | 'hca'
  | 'kda' | 'delta' | 'mamba' | 'ffn' | 'pruned'

export interface Lane {
  kind: LaneKind
  label: string
  color: LaneColor
  count: number
  cache: {
    glyph: CacheGlyph
    bytesPerToken: number    // per layer at KV_REF_DTYPE; compressed kinds already ÷ M
    fixedBytes: number       // per layer, 'state' only
    ratio?: number           // M for 'kvcomp'
    label: string
  }
  reach: {
    glyph: ReachGlyph
    tokens?: number          // window w, or top-k token count
    ratio?: number           // M for 'compress'
    local?: number           // local-window branch alongside a sparse/compressed read
    label: string
  }
}

const KIND_META: Record<LaneKind, { label: string; color: LaneColor }> = {
  full:   { label: 'full attention',   color: 'full' },
  window: { label: 'sliding window',   color: 'window' },
  mla:    { label: 'MLA',              color: 'full' },
  dsa:    { label: 'MLA + DSA top-k',  color: 'sparse' },
  msa:    { label: 'MSA block top-k',  color: 'sparse' },
  csa:    { label: 'CSA',              color: 'sparse' },
  hca:    { label: 'HCA',              color: 'sparse' },
  kda:    { label: 'KDA linear',       color: 'state' },
  delta:  { label: 'Gated DeltaNet',   color: 'state' },
  mamba:  { label: 'Mamba2',           color: 'state' },
  ffn:    { label: 'FFN only',         color: 'none' },
  pruned: { label: 'attention pruned', color: 'none' },
}

export function fmtBytes(b: number): string {
  const f = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1))
  if (b >= 1048576) return `${f(b / 1048576)} MiB`
  if (b >= 1024) return `${f(b / 1024)} KiB`
  return `${b} B`
}

function lane(kind: LaneKind, count: number, cache: Lane['cache'], reach: Lane['reach']): Lane {
  return { kind, count, cache, reach, ...KIND_META[kind] }
}

function kvCache(m: ModelArch, p: number, suffix = ''): Lane['cache'] {
  return {
    glyph: 'kv', bytesPerToken: p, fixedBytes: 0,
    label: `K + V · ${m.numKvHeads} KV heads × ${m.headDim} · ${fmtBytes(p)}${suffix}`,
  }
}

const NONE_CACHE: Lane['cache'] = { glyph: 'none', bytesPerToken: 0, fixedBytes: 0, label: 'nothing' }
const ALL_REACH: Lane['reach'] = { glyph: 'all', label: 'all tokens' }
const NONE_REACH: Lane['reach'] = { glyph: 'none', label: 'no attention' }
const windowReach = (w: number): Lane['reach'] => ({ glyph: 'window', tokens: w, label: `trailing ${w}` })

export function lanesFor(m: ModelArch): Lane[] {
  const att = m.attention
  const p = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
  switch (att.type) {
    case 'full':
      return [lane('full', m.layers, kvCache(m, p), ALL_REACH)]
    case 'sliding':
      return [lane('window', m.layers, kvCache(m, p, `, last ${att.window} tokens only`), windowReach(att.window))]
    case 'hybrid':
      return [
        lane('window', att.numSlidingLayers, kvCache(m, p), windowReach(att.slidingWindow)),
        lane('full', att.numGlobalLayers, kvCache(m, p), ALL_REACH),
      ]
    case 'partial':
      return [
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
        lane('pruned', m.layers - att.numFullLayers, NONE_CACHE, NONE_REACH),
      ]
    default:
      // Remaining variants land in Tasks 4 and 5; the never-check arrives with the last one.
      throw new Error(`lanesFor: variant not yet handled: ${att.type}`)
  }
}
