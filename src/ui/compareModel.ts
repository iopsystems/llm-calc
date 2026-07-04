// Pure domain logic for the Compare tab. No Svelte imports — importable from
// tests and the store layer alike. Resolves a (pivot, candidate) tuple into a
// CalcInput, mirroring the accelerator-vs-system resolution the `input` derived
// store does, then runs the shared engine per candidate with error isolation.
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { defaultParallelism } from '../engine/parallelism'
import type { CalcInput, MultiDeviceConfig, Quantization, Workload } from '../engine/types'

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
