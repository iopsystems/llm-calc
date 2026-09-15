import type {
  CalcInput, Dtype, AcceleratorVariant, MemoryResult, MemorySide,
  MultiDeviceConfig, ModelArch, Workload
} from './types'
import { bytesOf } from './dtypes'
import { perRankMemoryDivisors } from './parallelism'

const BYTES_PER_GB = 1024 ** 3

export function activeParams(model: ModelArch): number {
  return model.architecture.type === 'moe'
    ? model.architecture.activeParamCount
    : model.paramCount
}

export function kvBytesPerTokenPerLayer(model: ModelArch, kvDtype: Dtype): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'linear-mla-hybrid') {
    return (att.kvLoraRank + att.qkRopeHeadDim) * bytesOf(kvDtype)
  }
  if (att.type === 'delta-hybrid') {
    // DeltaNet layers: no KV cache. Only Gated Attention layers store KV.
    return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
  }
  if (att.type === 'csa-hca-hybrid') {
    // V4 caches one shared vector used as both K and V.
    return model.headDim * bytesOf(kvDtype)
  }
  return 2 * model.numKvHeads * model.headDim * bytesOf(kvDtype)
}

export function attentionDim(model: ModelArch): number {
  const att = model.attention
  if (att.type === 'mla' || att.type === 'mla-dsa') return att.kvLoraRank + att.qkRopeHeadDim
  if (att.type === 'linear-mla-hybrid') return att.kvLoraRank + att.qkRopeHeadDim
  if (att.type === 'delta-hybrid') return model.numHeads * model.headDim
  if (att.type === 'csa-hca-hybrid') return model.numHeads * model.headDim
  return model.numHeads * model.headDim
}

// forKv=true: KV storage (mla-dsa caches all tokens, topK only limits compute attention).
export function attendedSeqlenSummedOverLayers(model: ModelArch, seqlen: number, forKv = false): number {
  const att = model.attention
  if (att.type === 'hybrid') {
    if (att.numSlidingLayers + att.numGlobalLayers !== model.layers) {
      throw new Error(
        `hybrid layer counts must sum to model.layers: ` +
        `${att.numSlidingLayers} + ${att.numGlobalLayers} ≠ ${model.layers}`
      )
    }
    return att.numSlidingLayers * Math.min(seqlen, att.slidingWindow)
         + att.numGlobalLayers * seqlen
  }
  if (att.type === 'linear-mla-hybrid') {
    if (att.numLinearLayers + att.numFullLayers !== model.layers) {
      throw new Error(
        `linear-mla-hybrid layer counts must sum to model.layers: ` +
        `${att.numLinearLayers} + ${att.numFullLayers} ≠ ${model.layers}`
      )
    }
    return att.numFullLayers * seqlen
  }
  if (att.type === 'delta-hybrid') {
    if (att.numDeltaNetLayers + att.numFullLayers !== model.layers) {
      throw new Error(
        `delta-hybrid layer counts must sum to model.layers: ` +
        `${att.numDeltaNetLayers} + ${att.numFullLayers} ≠ ${model.layers}`
      )
    }
    // DeltaNet layers: no attention over sequence (constant-time state update).
    // Gated Attention layers: full sequence attention.
    return att.numFullLayers * seqlen
  }
  if (att.type === 'csa-hca-hybrid') {
    if (att.numSlidingLayers + att.numCsaLayers + att.numHcaLayers !== model.layers) {
      throw new Error(
        `csa-hca-hybrid layer counts must sum to model.layers: ` +
        `${att.numSlidingLayers} + ${att.numCsaLayers} + ${att.numHcaLayers} ≠ ${model.layers}`
      )
    }
    const compressed = Math.floor(seqlen / att.csaCompressionM)
    const csaCount = forKv ? compressed : Math.min(compressed, att.csaTopK)
    const window = Math.min(seqlen, att.slidingWindow)
    return att.numSlidingLayers * window
         + att.numCsaLayers * (csaCount + window)
         + att.numHcaLayers * (Math.floor(seqlen / att.hcaCompressionM) + window)
  }
  if (att.type === 'mamba2-hybrid') {
    if (att.numMambaLayers + att.numFullLayers + att.numFfnLayers !== model.layers) {
      throw new Error(
        `mamba2-hybrid block counts must sum to model.layers: ` +
        `${att.numMambaLayers} + ${att.numFullLayers} + ${att.numFfnLayers} ≠ ${model.layers}`
      )
    }
    // Mamba2 blocks: constant-time state update; FFN blocks: no attention.
    return att.numFullLayers * seqlen
  }
  if (att.type === 'partial') {
    if (att.numFullLayers > model.layers) {
      throw new Error(
        `partial numFullLayers must not exceed model.layers: ` +
        `${att.numFullLayers} > ${model.layers}`
      )
    }
    // NAS-pruned blocks have no attention at all.
    return att.numFullLayers * seqlen
  }
  if (att.type === 'msa-hybrid') {
    if (att.numFullLayers + att.numSparseLayers !== model.layers) {
      throw new Error(
        `msa-hybrid layer counts must sum to model.layers: ` +
        `${att.numFullLayers} + ${att.numSparseLayers} ≠ ${model.layers}`
      )
    }
    // Sparse layers cache full KV; top-k block selection only caps compute.
    const sparseCount = forKv ? seqlen : Math.min(seqlen, att.topKBlocks * att.blockSize)
    return att.numFullLayers * seqlen + att.numSparseLayers * sparseCount
  }
  if (att.type === 'mla-dsa') return model.layers * (forKv ? seqlen : Math.min(seqlen, att.topK))
  const perLayer = att.type === 'sliding' ? Math.min(seqlen, att.window) : seqlen
  return model.layers * perLayer
}

// Constant per-request state bytes from linear-attention layers. Zero for non-linear models.
export function linearAttentionStateBytes(model: ModelArch, kvDtype: Dtype): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim * bytesOf(kvDtype)
}

// FLOPs per token from linear-attention layers (constant in seqlen). Zero for non-linear models.
export function linearAttentionFlopsPerToken(model: ModelArch): number {
  if (model.attention.type !== 'linear-mla-hybrid') return 0
  const a = model.attention
  return 2 * a.numLinearLayers * a.numLinearHeads * a.linearHeadDim * a.linearHeadDim
}

// Constant per-request state bytes from DeltaNet (Gated DeltaNet) layers. Zero for non-DeltaNet models.
export function deltaStateBytes(model: ModelArch, _kvDtype: Dtype): number {
  const att = model.attention
  if (att.type !== 'delta-hybrid') return 0
  // Qwen configs pin recurrent state to FP32, independently of KV precision.
  return att.numDeltaNetLayers * att.numDeltaNetHeads * att.deltaHeadDim * att.deltaHeadDim * 4
}

// Constant per-request Mamba2 SSM state bytes. Zero for non-Mamba models.
// Cached in fp32 regardless of the user's KV quant — NemotronH configs pin
// mamba_ssm_cache_dtype: float32 — so this takes no dtype parameter.
// The depthwise-conv state ((expand·hidden + 2·groups·state) × (kernel−1)
// elements) is ~1000× smaller and omitted.
export function mambaStateBytes(model: ModelArch): number {
  const att = model.attention
  if (att.type !== 'mamba2-hybrid') return 0
  return att.numMambaLayers * att.numMambaHeads * att.mambaHeadDim * att.ssmStateSize * 4
}

// FLOPs per token from Mamba2 blocks (constant in seqlen). Zero for non-Mamba models.
export function mambaFlopsPerToken(model: ModelArch): number {
  const att = model.attention
  if (att.type !== 'mamba2-hybrid') return 0
  // Per-block SSM scan ≈ 2 ops per state element (update + readout), matching
  // the convention used for linear/DeltaNet variants above.
  return 2 * att.numMambaLayers * att.numMambaHeads * att.mambaHeadDim * att.ssmStateSize
}

// FLOPs per token from DeltaNet layers (constant in seqlen). Zero for non-DeltaNet models.
export function deltaAttentionFlopsPerToken(model: ModelArch): number {
  const att = model.attention
  if (att.type !== 'delta-hybrid') return 0
  // Per-layer DeltaNet FLOPs ≈ 2 × numDeltaNetHeads × deltaHeadDim²
  return 2 * att.numDeltaNetLayers * att.numDeltaNetHeads * att.deltaHeadDim * att.deltaHeadDim
}

function findVariant(input: CalcInput): AcceleratorVariant {
  const v = input.accelerator.variants.find(v => v.id === input.acceleratorVariantId)
  if (!v) throw new Error(`Variant ${input.acceleratorVariantId} not in ${input.accelerator.id}`)
  return v
}

export interface CacheBytes {
  // Total persistent bytes per request, including recurrent state.
  bytes: number
  recurrentBytes: number
}

// Sum floor(s / block) for integer contexts s = 0..end, in constant time.
function compressedSlotSum(end: number, block: number): number {
  if (end < 0) return 0
  const complete = Math.floor(end / block)
  const remainder = end - complete * block
  return block * complete * (complete - 1) / 2 + complete * (remainder + 1)
}

function cacheBytesOverContexts(
  model: ModelArch, kvDtype: Dtype, start: number, count: number, tp: number, pp: number,
): CacheBytes {
  const att = model.attention
  const mean = start + (count - 1) / 2
  const windowMean = (window: number) => {
    const uncapped = Math.max(0, Math.min(count, Math.ceil(window - start)))
    return (uncapped * (start + (uncapped - 1) / 2) + (count - uncapped) * window) / count
  }
  const compressedMean = (block: number) =>
    (compressedSlotSum(start + count - 1, block) - compressedSlotSum(start - 1, block)) / count

  // The common helper also validates hybrid layer counts. For uncapped caches,
  // storage is linear in context and can be evaluated directly at its mean.
  let slots = attendedSeqlenSummedOverLayers(model, mean, true)
  if (att.type === 'sliding') slots = model.layers * windowMean(att.window)
  if (att.type === 'hybrid') {
    slots = att.numSlidingLayers * windowMean(att.slidingWindow) + att.numGlobalLayers * mean
  }
  if (att.type === 'csa-hca-hybrid') {
    slots = model.layers * windowMean(att.slidingWindow)
      + att.numCsaLayers * compressedMean(att.csaCompressionM)
      + att.numHcaLayers * compressedMean(att.hcaCompressionM)
  }

  // Shared compressed vectors are replicated across TP. Ordinary KV tensors
  // shard across KV heads; recurrent matrices shard across their own heads.
  const sharedKv = att.type === 'mla' || att.type === 'mla-dsa'
    || att.type === 'linear-mla-hybrid' || att.type === 'csa-hca-hybrid'
  const kvShard = sharedKv ? 1 : Math.min(tp, model.numKvHeads)
  let recurrentBytes = 0
  if (att.type === 'linear-mla-hybrid') {
    recurrentBytes = linearAttentionStateBytes(model, kvDtype) / Math.min(tp, att.numLinearHeads) / pp
  } else if (att.type === 'delta-hybrid') {
    recurrentBytes = deltaStateBytes(model, kvDtype) / Math.min(tp, att.numDeltaNetHeads) / pp
  } else if (att.type === 'mamba2-hybrid') {
    recurrentBytes = mambaStateBytes(model) / Math.min(tp, att.numMambaHeads) / pp
  }
  // MSA index keys are shared across main KV groups; only PP partitions them.
  const indexBytes = att.type === 'msa-hybrid'
    ? att.numSparseLayers * mean * att.indexHeadDim * bytesOf(kvDtype) / pp : 0
  return {
    bytes: kvBytesPerTokenPerLayer(model, kvDtype) * slots / kvShard / pp + indexBytes + recurrentBytes,
    recurrentBytes,
  }
}

/** Persistent cache capacity at a sequence length, optionally on one TP/PP rank. */
export function cacheBytesAtSequence(
  model: ModelArch, kvDtype: Dtype, seqlen: number, tp = 1, pp = 1,
): CacheBytes {
  return cacheBytesOverContexts(model, kvDtype, seqlen, 1, tp, pp)
}

/** Mean stored cache over contexts prompt..prompt+max(output,1)-1.
 * Sparse traffic retains the full-cache-read approximation. Compressor/indexer
 * scratch space and non-MSA index state are not included. PP uses uniform layers.
 */
export function averageDecodeCacheBytes(
  model: ModelArch, kvDtype: Dtype, prompt: number, output: number, tp = 1, pp = 1,
): CacheBytes {
  return cacheBytesOverContexts(model, kvDtype, prompt, Math.max(output, 1), tp, pp)
}

export function computeMemory(input: CalcInput): MemoryResult {
  const { model, quant, workload } = input
  const prefillVariant = findVariant(input)
  const seqlen = workload.promptTokens + workload.outputTokens

  const weights = model.paramCount * bytesOf(quant.weights)
  const kvCachePerRequest = cacheBytesAtSequence(model, quant.kv, seqlen).bytes
  const kvCacheTotal = kvCachePerRequest * workload.concurrency

  // Prefill activations: one big batched pass, scales with promptTokens × hidden.
  const activationsPeak =
    workload.concurrency * workload.promptTokens *
    (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2

  // Decode activations: single-token forward pass per layer; orders of magnitude smaller.
  const decodeActivationsPeak =
    workload.concurrency * 1 *
    (model.hiddenDim + model.intermediateDim) * bytesOf(quant.activations) * 2

  // Resolve decode-side variant; falls back to prefill when asymmetric fields absent.
  const decodeAccelerator = input.decodeAccelerator ?? input.accelerator
  const decodeVariantId = input.decodeAcceleratorVariantId ?? input.acceleratorVariantId
  const decodeVariant =
    decodeAccelerator.variants.find(v => v.id === decodeVariantId) ?? prefillVariant

  const prefillSide = buildSide(
    weights, kvCacheTotal, activationsPeak,
    prefillVariant.hbmCapacityGB,
    input.multiDevice, model, workload, quant.kv
  )
  const decodeSide = buildSide(
    weights, kvCacheTotal, decodeActivationsPeak,
    decodeVariant.hbmCapacityGB,
    input.decodeMultiDevice ?? input.multiDevice, model, workload, quant.kv
  )

  return {
    weights,
    kvCachePerRequest,
    kvCacheTotal,
    activationsPeak,
    decodeActivationsPeak,
    prefillSide,
    decodeSide,
    // Backward-compat: mirror prefillSide
    total: prefillSide.total,
    hbmCapacityGB: prefillSide.hbmCapacityGB,
    headroom: prefillSide.headroom,
    fits: prefillSide.fits,
    ...(prefillSide.perRank && {
      perRank: {
        weights: prefillSide.perRank.weights,
        kvCachePerRequest: prefillSide.perRank.kvCachePerRequest,
        kvCacheTotal: prefillSide.perRank.kvCacheTotal,
        activationsPeak: prefillSide.perRank.activations,
        total: prefillSide.perRank.total,
        headroom: prefillSide.perRank.headroom,
        fits: prefillSide.perRank.fits,
      }
    })
  }
}

function buildSide(
  weights: number,
  kvCacheTotal: number,
  activations: number,
  hbmCapacityGB: number,
  multiDevice: MultiDeviceConfig | undefined,
  model: ModelArch,
  workload: Workload,
  kvDtype: Dtype,
): MemorySide {
  const total = weights + kvCacheTotal + activations
  const hbmCapacityBytes = hbmCapacityGB * BYTES_PER_GB
  const headroom = hbmCapacityBytes - total
  const fits = headroom >= 0

  let perRank: MemorySide['perRank'] = undefined
  if (multiDevice) {
    const divisors = perRankMemoryDivisors(
      multiDevice.parallelism,
      multiDevice.parallelismDegrees,
      model
    )
    const rankWeights = weights / divisors.weights
    const perReplicaConcurrency = Math.ceil(workload.concurrency / divisors.replicas)
    const pp = multiDevice.parallelism.includes('pp') ? (multiDevice.parallelismDegrees.pp ?? 1) : 1
    const tp = multiDevice.parallelism.includes('tp') ? (multiDevice.parallelismDegrees.tp ?? 1) : 1
    const rankKvPerRequest = cacheBytesAtSequence(
      model, kvDtype, workload.promptTokens + workload.outputTokens, tp, pp
    ).bytes
    const rankKvTotal = rankKvPerRequest * perReplicaConcurrency
    const rankActivations = activations / Math.max(1, workload.concurrency)
      * perReplicaConcurrency / divisors.activations
    const rankTotal = rankWeights + rankKvTotal + rankActivations
    const rankHeadroom = hbmCapacityBytes - rankTotal
    perRank = {
      weights: rankWeights,
      kvCachePerRequest: rankKvPerRequest,
      kvCacheTotal: rankKvTotal,
      activations: rankActivations,
      total: rankTotal,
      headroom: rankHeadroom,
      fits: rankHeadroom >= 0,
    }
  }

  return {
    weights,
    activations,
    kvCache: kvCacheTotal,
    total,
    hbmCapacityGB,
    headroom,
    fits,
    ...(perRank && { perRank }),
  }
}
