// Pure domain logic for the Compare tab. No Svelte imports — importable from
// tests and the store layer alike. Resolves a (pivot, candidate) tuple into a
// CalcInput, mirroring the accelerator-vs-system resolution the `input` derived
// store does, then runs the shared engine per candidate with error isolation.
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { defaultParallelism } from '../engine/parallelism'
import { calculate } from '../engine'
import type { CalcInput, MultiDeviceConfig, Quantization, Workload, CalcResult, PerfTier } from '../engine/types'

export type ComparePivotKind = 'sku' | 'model'
export interface ComparePivot { kind: ComparePivotKind; id: string }
export interface CompareCandidate { varyingId: string; quant: Quantization }

// Build a CalcInput from a concrete (modelId, skuId) pair. skuId may name an
// accelerator OR a system; a system id wins and wires multiDevice with default
// parallelism (same precedence as share.ts / the `input` store). A bare
// accelerator resolves to its first variant (v1 limitation — no per-variant
// SKU compare yet). Returns null if any id is unknown.
function buildInput(modelId: string, skuId: string, quant: Quantization, workload: Workload): CalcInput | null {
  const model = MODELS.find(m => m.id === modelId)
  if (!model) return null

  const system = SYSTEMS.find(s => s.id === skuId)
  if (system) {
    const accelerator = ACCELERATORS.find(a => a.id === system.accelerator.id)
    if (!accelerator) return null
    const pc = defaultParallelism(system, model)
    const multiDevice: MultiDeviceConfig = {
      system, parallelism: pc.parallelism, parallelismDegrees: pc.parallelismDegrees,
    }
    return { accelerator, acceleratorVariantId: system.accelerator.variantId, model, quant, workload, multiDevice }
  }

  const accelerator = ACCELERATORS.find(a => a.id === skuId)
  if (!accelerator) return null
  return { accelerator, acceleratorVariantId: accelerator.variants[0].id, model, quant, workload }
}

export function resolveCompareInput(
  pivot: ComparePivot, candidate: CompareCandidate, workload: Workload,
): CalcInput | null {
  return pivot.kind === 'sku'
    ? buildInput(candidate.varyingId, pivot.id, candidate.quant, workload)   // pivot = sku, varying = model
    : buildInput(pivot.id, candidate.varyingId, candidate.quant, workload)   // pivot = model, varying = sku
}

export interface CompareMetrics {
  ttftMs: number
  tpotMs: number
  throughputTokS: number
  kvTotalGB: number
  fits: boolean
  regime: 'compute' | 'memory' | 'comms'
}

export type CompareRow =
  | { ok: true;  name: string; candidate: CompareCandidate; metrics: CompareMetrics }
  | { ok: false; name: string; candidate: CompareCandidate; error: string }

// The comparison reports the peak (theoretical) operating point — the same tier
// the roofline panel treats as "Theoretical". Fall back to the first available
// point for any accelerator that has no 'peak' id.
export function pickPerfTier(result: CalcResult): PerfTier | null {
  return result.perf['peak'] ?? Object.values(result.perf)[0] ?? null
}

// Display name of the *varying* dimension (the opposite of the pivot kind).
export function resolveVaryingName(pivot: ComparePivot, varyingId: string): string {
  if (pivot.kind === 'sku') {
    return MODELS.find(m => m.id === varyingId)?.name ?? varyingId
  }
  return SYSTEMS.find(s => s.id === varyingId)?.name
    ?? ACCELERATORS.find(a => a.id === varyingId)?.name
    ?? varyingId
}

export function computeCompareRow(
  pivot: ComparePivot, candidate: CompareCandidate, workload: Workload,
): CompareRow {
  const name = resolveVaryingName(pivot, candidate.varyingId)
  const input = resolveCompareInput(pivot, candidate, workload)
  if (!input) return { ok: false, name, candidate, error: 'unknown model or accelerator' }
  try {
    const result = calculate(input)
    const perf = pickPerfTier(result)
    if (!perf) return { ok: false, name, candidate, error: 'no operating point' }
    return {
      ok: true, name, candidate,
      metrics: {
        ttftMs: perf.ttftS * 1000,
        tpotMs: perf.decode.timePerTokenS * 1000,
        throughputTokS: perf.decode.aggregateTokensPerS,
        kvTotalGB: result.memory.kvCacheTotal / 1e9,
        fits: result.memory.fits,
        regime: perf.decode.regime,
      },
    }
  } catch (err) {
    return { ok: false, name, candidate, error: (err as Error).message }
  }
}
