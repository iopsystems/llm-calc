import type { ModelArch } from './types'

// Sum min(floor(position / compression), cap) for positions 1..end.
// Complete compression groups are visible; an incomplete group creates no slot.
// The closed form keeps million-token contexts independent of context length.
function slotPrefix(end: number, compression = 1, cap = Infinity): number {
  if (end <= 0 || cap === 0) return 0
  const uncappedEnd = Math.min(end, cap * compression - 1)
  const groups = Math.floor(uncappedEnd / compression)
  const sum = compression * groups * (groups - 1) / 2
    + groups * (uncappedEnd - groups * compression + 1)
  return sum + (end > uncappedEnd ? (end - uncappedEnd) * cap : 0)
}

// Total query/key pairs, summed over attention-bearing layers and query
// positions. Recurrent layers contribute no pairs; their updates stay in the
// runtime callers. Sparse index scoring and compressor work remain omitted.
function pairs(model: ModelArch, first: number, last: number): number {
  const a = model.attention
  const slots = (compression = 1, cap = Infinity) =>
    slotPrefix(last, compression, cap) - slotPrefix(first - 1, compression, cap)
  switch (a.type) {
    case 'full':
    case 'mla':
      return model.layers * slots()
    case 'sliding':
      return model.layers * slots(1, a.window)
    case 'hybrid':
      return a.numSlidingLayers * slots(1, a.slidingWindow) + a.numGlobalLayers * slots()
    case 'mla-dsa':
      return model.layers * slots(1, a.topK)
    case 'linear-mla-hybrid':
    case 'delta-hybrid':
    case 'mamba2-hybrid':
    case 'partial':
      return a.numFullLayers * slots()
    case 'msa-hybrid':
      return a.numFullLayers * slots() + a.numSparseLayers * slots(1, a.topKBlocks * a.blockSize)
    case 'csa-hca-hybrid':
      // Every layer has the local branch; compressed branches are additional
      // slots. Shared KV storage does not eliminate either QK or AV work.
      return (a.numSlidingLayers + a.numCsaLayers + a.numHcaLayers) * slots(1, a.slidingWindow)
        + a.numCsaLayers * slots(a.csaCompressionM, a.csaTopK)
        + a.numHcaLayers * slots(a.hcaCompressionM)
  }
}

function flopsPerPair(model: ModelArch, phase: 'prefill' | 'decode'): number {
  const a = model.attention
  if (a.type === 'mla' || a.type === 'mla-dsa' || a.type === 'linear-mla-hybrid') {
    // Expanded prefill: QK uses no-PE + RoPE, AV uses the value width.
    // Absorbed decode: QK uses latent + RoPE, AV aggregates the latent.
    const width = phase === 'prefill'
      ? a.qkNopeHeadDim + a.qkRopeHeadDim + a.vHeadDim
      : 2 * a.kvLoraRank + a.qkRopeHeadDim
    return 2 * model.numHeads * width
  }
  // GQA shares cached KV heads, but every query head computes both products.
  return 4 * model.numHeads * model.headDim
}

/** Per-request QK + AV FLOPs for causal prefill, positions 1..prompt. */
export function prefillAttentionFlops(model: ModelArch, prompt: number): number {
  return flopsPerPair(model, 'prefill') * pairs(model, 1, prompt)
}

/** Mean per-request QK + AV FLOPs at positions prompt..prompt+max(output,1)-1. */
export function averageDecodeAttentionFlops(model: ModelArch, prompt: number, output: number): number {
  const steps = Math.max(output, 1)
  return flopsPerPair(model, 'decode') * pairs(model, prompt, prompt + steps - 1) / steps
}
