// Per-model geometry for the Info tab figure. Pure; every byte figure comes
// from src/engine/memory.ts so the drawing cannot disagree with the calc.
import type { ModelArch, AttentionConfig } from '../engine/types'
import { kvBytesPerTokenPerLayer } from '../engine/memory'
import { bytesOf } from '../engine/dtypes'
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

function latentCache(rank: number, rope: number, p: number): Lane['cache'] {
  return { glyph: 'latent', bytesPerToken: p, fixedBytes: 0, label: `latent ${rank} + RoPE key ${rope} · ${fmtBytes(p)}` }
}
const topkReach = (tokens: number, label: string, local?: number): Lane['reach'] =>
  ({ glyph: 'topk', tokens, label, ...(local === undefined ? {} : { local }) })

const STATE_REACH: Lane['reach'] = { glyph: 'state', label: 'one state read' }
function stateCache(fixedBytes: number, label: string): Lane['cache'] {
  return { glyph: 'state', bytesPerToken: 0, fixedBytes, label }
}

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
    case 'mla':
      return [lane('mla', m.layers, latentCache(att.kvLoraRank, att.qkRopeHeadDim, p), ALL_REACH)]
    case 'mla-dsa':
      return [lane('dsa', m.layers, latentCache(att.kvLoraRank, att.qkRopeHeadDim, p),
        topkReach(att.topK, `top-${att.topK} tokens`))]
    case 'msa-hybrid': {
      const idx = att.indexHeadDim * bytesOf(KV_REF_DTYPE)
      return [
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
        lane('msa', att.numSparseLayers, {
          glyph: 'kvidx', bytesPerToken: p + idx, fixedBytes: 0,
          label: `K + V ${fmtBytes(p)} + shared index key ${att.indexHeadDim} · ${fmtBytes(idx)}`,
        }, topkReach(att.topKBlocks * att.blockSize, `${att.topKBlocks} blocks × ${att.blockSize} tokens`)),
      ]
    }
    case 'csa-hca-hybrid': {
      const comp = (M: number): Lane['cache'] => ({
        glyph: 'kvcomp', ratio: M, bytesPerToken: p / M, fixedBytes: 0,
        label: `K + V ${fmtBytes(p)} per ${M} tokens · ${fmtBytes(p / M)}/token`,
      })
      const w = att.slidingWindow
      const lanes: Lane[] = []
      if (att.numSlidingLayers > 0) lanes.push(lane('window', att.numSlidingLayers, kvCache(m, p), windowReach(w)))
      lanes.push(lane('csa', att.numCsaLayers, comp(att.csaCompressionM),
        topkReach(att.csaTopK * att.csaCompressionM, `top-${att.csaTopK} of the 1 : ${att.csaCompressionM} stream + ${w} local`, w)))
      lanes.push(lane('hca', att.numHcaLayers, comp(att.hcaCompressionM), {
        glyph: 'compress', ratio: att.hcaCompressionM, local: w,
        label: `all of the 1 : ${att.hcaCompressionM} stream + ${w} local`,
      }))
      return lanes
    }
    case 'linear-mla-hybrid': {
      const per = att.numLinearHeads * att.linearHeadDim * att.linearHeadDim * bytesOf(KV_REF_DTYPE)
      return [
        lane('kda', att.numLinearLayers,
          stateCache(per, `${att.numLinearHeads} heads × ${att.linearHeadDim}² state · ${fmtBytes(per)} per layer`), STATE_REACH),
        lane('mla', att.numFullLayers, latentCache(att.kvLoraRank, att.qkRopeHeadDim, p), ALL_REACH),
      ]
    }
    case 'delta-hybrid': {
      const per = att.numDeltaNetHeads * att.deltaHeadDim * att.deltaHeadDim * bytesOf(KV_REF_DTYPE)
      return [
        lane('delta', att.numDeltaNetLayers,
          stateCache(per, `${att.numDeltaNetHeads} heads × ${att.deltaHeadDim}² state · ${fmtBytes(per)} per layer`), STATE_REACH),
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
      ]
    }
    case 'mamba2-hybrid': {
      // fp32 by config (mamba_ssm_cache_dtype), independent of the KV reference.
      const per = att.numMambaHeads * att.mambaHeadDim * att.ssmStateSize * 4
      return [
        lane('mamba', att.numMambaLayers,
          stateCache(per, `${att.numMambaHeads} heads × ${att.mambaHeadDim} × ${att.ssmStateSize} fp32 state · ${fmtBytes(per)} per block`), STATE_REACH),
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
        lane('ffn', att.numFfnLayers, NONE_CACHE, NONE_REACH),
      ]
    }
    default: {
      const _exhaustive: never = att
      throw new Error(`lanesFor: unhandled attention variant ${(_exhaustive as { type: string }).type}`)
    }
  }
}

export interface HeadGeometry {
  kind: 'gqa' | 'latent' | 'state'
  numHeads: number
  numKvHeads: number
  headDim: number
  fullKvWidth?: number       // 'latent': what per-head K+V would be, in elements
  latentWidth?: number       // 'latent': kvLoraRank + qkRopeHeadDim
  state?: { heads: number; dim: number; inner: number }   // 'state': elements = heads × dim × inner
}

export function headsFor(m: ModelArch): HeadGeometry[] {
  const base = { numHeads: m.numHeads, numKvHeads: m.numKvHeads, headDim: m.headDim }
  const gqa: HeadGeometry = { kind: 'gqa', ...base }
  const latent = (rank: number, rope: number): HeadGeometry =>
    ({ kind: 'latent', ...base, fullKvWidth: m.numHeads * m.headDim * 2, latentWidth: rank + rope })
  const state = (heads: number, dim: number, inner: number): HeadGeometry =>
    ({ kind: 'state', ...base, state: { heads, dim, inner } })
  const att = m.attention
  switch (att.type) {
    case 'full': case 'sliding': case 'hybrid': case 'partial':
    case 'msa-hybrid': case 'csa-hca-hybrid':
      return [gqa]
    case 'mla': case 'mla-dsa':
      return [latent(att.kvLoraRank, att.qkRopeHeadDim)]
    case 'linear-mla-hybrid':
      return [state(att.numLinearHeads, att.linearHeadDim, att.linearHeadDim), latent(att.kvLoraRank, att.qkRopeHeadDim)]
    case 'delta-hybrid':
      return [state(att.numDeltaNetHeads, att.deltaHeadDim, att.deltaHeadDim), gqa]
    case 'mamba2-hybrid':
      return [state(att.numMambaHeads, att.mambaHeadDim, att.ssmStateSize), gqa]
    default: {
      const _exhaustive: never = att
      throw new Error(`headsFor: unhandled ${(_exhaustive as { type: string }).type}`)
    }
  }
}

const COUNTS_NOTE = 'Bar lengths are counts, not order.'
const SPARSE_NOTE = 'A decode step streams the whole cache; selection lowers attention FLOPs, not bytes. The index branch that selects is not costed.'

export function captionFor(type: AttentionConfig['type']): string {
  switch (type) {
    case 'mla-dsa': case 'csa-hca-hybrid':
      return `${SPARSE_NOTE} ${COUNTS_NOTE}`
    case 'msa-hybrid':
      return `${SPARSE_NOTE} The shared index key is cached alongside K and V and is not sharded by TP. ${COUNTS_NOTE}`
    case 'mamba2-hybrid':
      return `Attention, Mamba2 and FFN are separate entries in the block count. SSM state is held in fp32. ${COUNTS_NOTE}`
    case 'partial':
      return `NAS pruning removed attention from the grey blocks; their FFN widths vary and are absorbed by the parameter count. ${COUNTS_NOTE}`
    case 'full': case 'sliding': case 'hybrid': case 'mla':
    case 'linear-mla-hybrid': case 'delta-hybrid':
      return COUNTS_NOTE
    default: {
      const _exhaustive: never = type
      throw new Error(`captionFor: unhandled ${_exhaustive as string}`)
    }
  }
}

export interface GeometryModel {
  layers: number
  maxContext: number
  lanes: Lane[]
  heads: HeadGeometry[]
  ffn:
    | { type: 'dense'; hiddenDim: number; intermediateDim: number }
    | { type: 'moe'; routed: number; active: number; shared: number; activeRatio: number }
  caption: string
  schematicId: AttentionConfig['type']
}

export function geometryModel(m: ModelArch): GeometryModel {
  const a = m.architecture
  return {
    layers: m.layers,
    maxContext: m.maxContext,
    lanes: lanesFor(m),
    heads: headsFor(m),
    ffn: a.type === 'moe'
      ? { type: 'moe', routed: a.numExperts, active: a.numExpertsActive, shared: a.numSharedExperts, activeRatio: a.activeParamCount / m.paramCount }
      : { type: 'dense', hiddenDim: m.hiddenDim, intermediateDim: m.intermediateDim },
    caption: captionFor(m.attention.type),
    schematicId: m.attention.type,
  }
}
