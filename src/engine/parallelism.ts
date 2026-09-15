import type { Dtype, ModelArch, MultiAcceleratorSystem, MultiDeviceConfig, ParallelismMode } from './types'
import { bytesOf } from './dtypes'

// Effective shared/routed pools inferred from catalog total and per-token active
// counts, assuming equally sized routed experts. Bounds absorb rounded card totals.
export function parameterPools(model: ModelArch): { shared: number; routed: number; fraction: number } {
  if (model.architecture.type !== 'moe') return { shared: model.paramCount, routed: 0, fraction: 0 }
  const a = model.architecture
  const fraction = Math.min(1, a.numExpertsActive / a.numExperts)
  const shared = fraction >= 1 ? model.paramCount
    : Math.min(model.paramCount, Math.max(0, (a.activeParamCount - fraction * model.paramCount) / (1 - fraction)))
  return { shared, routed: model.paramCount - shared, fraction }
}

// EP runtime assumes ideal balance (average routed work per rank).
// Small batches can be slower when selected experts concentrate on one rank.
export function activeParametersOnRank(model: ModelArch, tp = 1, ep = 1): number {
  const { shared, routed, fraction } = parameterPools(model)
  return (shared + routed * fraction / ep) / tp
}

// Independent uniform top-k routing: expected union of experts across a batch.
// Shared parameters are read once per replica; EP distributes only routed experts.
export function expectedWeightParameters(model: ModelArch, batch: number, tp = 1, ep = 1): number {
  if (batch <= 0) return 0
  const { shared, routed, fraction } = parameterPools(model)
  const coverage = fraction >= 1 ? 1 : -Math.expm1(batch * Math.log1p(-fraction))
  return (shared + routed * coverage / ep) / tp
}

export function parallelDegrees(config?: MultiDeviceConfig): { tp: number; pp: number; ep: number; dp: number } {
  const degree = (id: ParallelismMode['id']) => config?.parallelism.includes(id) ? (config.parallelismDegrees[id] ?? 1) : 1
  return { tp: degree('tp'), pp: degree('pp'), ep: degree('ep'), dp: degree('dp') }
}

export function validateParallelism(config: MultiDeviceConfig | undefined, model: ModelArch): void {
  if (!config) return
  const { tp, pp, ep, dp } = parallelDegrees(config)
  if (![tp, pp, ep, dp].every(n => Number.isInteger(n) && n >= 1)) {
    throw new Error('Parallelism degrees must be positive integer ranks')
  }
  if (tp * pp * ep * dp > config.system.accelerator.count) {
    throw new Error(`Parallelism requires ${tp * pp * ep * dp} ranks but the system has ${config.system.accelerator.count} physical devices`)
  }
  if (ep > 1 && model.architecture.type !== 'moe') throw new Error('Expert parallelism requires an MoE model')
}

export interface RankDivisors {
  weights: number
  kv: number
  activations: number
  replicas: number
}

export function perRankMemoryDivisors(
  parallelism: ParallelismMode['id'][],
  degrees: Partial<Record<ParallelismMode['id'], number>>,
  model: ModelArch
): RankDivisors {
  const tp = parallelism.includes('tp') ? (degrees.tp ?? 1) : 1
  const pp = parallelism.includes('pp') ? (degrees.pp ?? 1) : 1
  const ep = parallelism.includes('ep') ? (degrees.ep ?? 1) : 1
  const dp = parallelism.includes('dp') ? (degrees.dp ?? 1) : 1

  // TP and EP are independent physical mesh axes. Shared weights replicate
  // across EP ranks; routed weights alone receive the EP divisor.
  const { shared, routed } = parameterPools(model)
  const weightsDivisor = model.paramCount > 0
    ? model.paramCount / ((shared + routed / ep) / (tp * pp)) : tp * pp

  // KV cache: TP shards heads (capped at numKvHeads), PP per-stage, EP/DP replicated.
  // MLA-family exception: the cache is one shared compressed latent per token
  // (kv_lora_rank + rope dims), not per-head slices — every TP rank needs the
  // full latent, which is why MLA deployments serve attention data-parallel.
  // PP still divides (each stage caches only its own layers' latents).
  const att = model.attention.type
  const mlaKv = att === 'mla' || att === 'mla-dsa' || att === 'linear-mla-hybrid' || att === 'csa-hca-hybrid'
  const kvShard = mlaKv ? 1 : Math.min(tp, model.numKvHeads)
  const kvDivisor = kvShard * pp

  // Activations: TP shards them; PP/EP/DP don't (per-stage forward, replicated).
  const activationsDivisor = tp

  return {
    weights: weightsDivisor,
    kv: kvDivisor,
    activations: activationsDivisor,
    replicas: dp
  }
}

export function commsBytesPerStep(
  parallelism: ParallelismMode['id'][],
  degrees: Partial<Record<ParallelismMode['id'], number>>,
  model: ModelArch,
  B: number,
  activationDtype: Dtype
): number {
  const tp = parallelism.includes('tp') ? (degrees.tp ?? 1) : 1
  const pp = parallelism.includes('pp') ? (degrees.pp ?? 1) : 1
  const ep = parallelism.includes('ep') ? (degrees.ep ?? 1) : 1

  const d = model.hiddenDim
  const L = model.layers
  const bytes = bytesOf(activationDtype)
  let total = 0

  // TP: two all-reduces per layer (ring algorithm): 2 × (N-1)/N × B × d × bytes each
  if (tp > 1) {
    total += 2 * L * 2 * ((tp - 1) / tp) * B * d * bytes
  }
  // PP: (N-1) point-to-point sends per forward pass
  if (pp > 1) {
    total += (pp - 1) * B * d * bytes
  }
  // EP: all-to-all per MoE layer (forward gather + scatter); zero for dense
  if (ep > 1 && model.architecture.type === 'moe') {
    total += 2 * L * (1 - 1 / ep) * B * d * bytes
  }

  return total
}

export interface ParallelismConfig {
  parallelism: ParallelismMode['id'][]
  parallelismDegrees: Partial<Record<ParallelismMode['id'], number>>
}

export function defaultParallelism(
  system: MultiAcceleratorSystem,
  model: ModelArch
): ParallelismConfig {
  const N = system.accelerator.count
  // Use a valid TP×PP mesh by default. EP is an explicit independent axis.
  let tp = Math.min(N, 8, model.numHeads)
  while (N % tp !== 0 || model.numHeads % tp !== 0) tp--
  const pp = N / tp

  const parallelism: ParallelismMode['id'][] = ['tp']
  const degrees: Partial<Record<ParallelismMode['id'], number>> = { tp }

  if (pp > 1) {
    parallelism.push('pp')
    degrees.pp = pp
  }
  return { parallelism, parallelismDegrees: degrees }
}
