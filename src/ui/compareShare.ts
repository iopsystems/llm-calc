// URL-hash codec for the Compare tab. Kept separate from share.ts (which owns
// the calc/sim payload) so each file stays single-responsibility. Payload form:
//   piv=<kind>:<id>&pt=..&ot=..&cc=..&c=<varyingId>~<w>.<kv>.<a>&c=...
// Slug-based (order-independent, survives catalog reordering); unknown ids and
// dtypes are silently dropped, an invalid pivot yields null (nothing to render).
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import type { Dtype, Quantization, Workload } from '../engine/types'
import type { ComparePivot, ComparePivotKind, CompareCandidate } from './compareModel'

export interface CompareState {
  pivot: ComparePivot
  candidates: CompareCandidate[]
  workload: Workload
}

const DTYPES: readonly Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']
const isDtype = (s: string): s is Dtype => (DTYPES as readonly string[]).includes(s)

const skuExists = (id: string) => !!ACCELERATORS.find(a => a.id === id) || !!SYSTEMS.find(s => s.id === id)
const modelExists = (id: string) => !!MODELS.find(m => m.id === id)

// A pivot of kind 'sku' varies models (and vice-versa): validate against the
// opposite catalog.
const varyingExists = (kind: ComparePivotKind, id: string) => kind === 'sku' ? modelExists(id) : skuExists(id)
const pivotExists   = (kind: ComparePivotKind, id: string) => kind === 'sku' ? skuExists(id) : modelExists(id)

function parsePos(raw: string | null, fallback: number): number {
  if (raw === null) return fallback
  const n = parseInt(raw, 10)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

function parseQuant(raw: string | undefined): Quantization | null {
  if (!raw) return null
  const [w, kv, a] = raw.split('.')
  if (w && kv && a && isDtype(w) && isDtype(kv) && isDtype(a)) return { weights: w, kv, activations: a }
  return null
}

export function encodeCompare(state: CompareState): string {
  const p = new URLSearchParams()
  p.set('piv', `${state.pivot.kind}:${state.pivot.id}`)
  p.set('pt', String(state.workload.promptTokens))
  p.set('ot', String(state.workload.outputTokens))
  p.set('cc', String(state.workload.concurrency))
  for (const c of state.candidates) {
    p.append('c', `${c.varyingId}~${c.quant.weights}.${c.quant.kv}.${c.quant.activations}`)
  }
  return p.toString()
}

export function decodeCompare(payload: string): CompareState | null {
  const params = new URLSearchParams(payload)
  const pivRaw = params.get('piv')
  if (!pivRaw) return null
  const sep = pivRaw.indexOf(':')
  if (sep < 0) return null
  const kind = pivRaw.slice(0, sep)
  const id = pivRaw.slice(sep + 1)
  if (kind !== 'sku' && kind !== 'model') return null
  if (!pivotExists(kind, id)) return null

  const workload: Workload = {
    promptTokens: parsePos(params.get('pt'), 2048),
    outputTokens: parsePos(params.get('ot'), 512),
    concurrency:  parsePos(params.get('cc'), 1),
  }

  const candidates: CompareCandidate[] = []
  for (const raw of params.getAll('c')) {
    const tilde = raw.indexOf('~')
    const varyingId = tilde < 0 ? raw : raw.slice(0, tilde)
    if (!varyingExists(kind, varyingId)) continue
    const quant = parseQuant(tilde < 0 ? undefined : raw.slice(tilde + 1))
    if (!quant) continue
    candidates.push({ varyingId, quant })
  }

  return { pivot: { kind, id }, candidates, workload }
}

import { get } from 'svelte/store'
import { comparePivot, compareCandidates, compareWorkload } from './stores'
import { parseRoute } from './route'

function readStoreCompareState(): CompareState {
  return { pivot: get(comparePivot), candidates: get(compareCandidates), workload: get(compareWorkload) }
}

function applyCompareState(s: CompareState): void {
  comparePivot.set(s.pivot)
  compareCandidates.set(s.candidates)
  compareWorkload.set(s.workload)
}

// Read the compare payload from the URL on load, iff the hash targets the
// compare tab. No-op otherwise (calc/sim links are handled by share.ts).
export function readCompareUrlIntoStores(): void {
  if (typeof window === 'undefined') return
  const h = window.location.hash.replace(/^#/, '')
  if (!h.startsWith('compare?')) return
  const decoded = decodeCompare(h.slice('compare?'.length))
  if (decoded) applyCompareState(decoded)
}

// Mirror the compare stores back to the hash while on the compare tab. Mirrors
// share.ts.startUrlSync structure (hold `ready` until all subs wired).
export function startCompareUrlSync(): () => void {
  if (typeof window === 'undefined') return () => {}
  let ready = false
  const write = () => {
    if (!ready) return
    if (parseRoute(window.location.hash).tab !== 'compare') return
    const encoded = encodeCompare(readStoreCompareState())
    const next = `${window.location.pathname}${window.location.search}#compare?${encoded}`
    window.history.replaceState(window.history.state, '', next)
  }
  const unsubs = [comparePivot.subscribe(write), compareCandidates.subscribe(write), compareWorkload.subscribe(write)]
  ready = true
  write()
  return () => unsubs.forEach(u => u())
}
