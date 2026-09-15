import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier, MultiDeviceConfig } from './types'
import { roofline } from './roofline'
import { activeParams, linearAttentionFlopsPerToken, deltaAttentionFlopsPerToken, mambaFlopsPerToken } from './memory'
import { commsBytesPerStep, activeParametersOnRank, expectedWeightParameters, parallelDegrees, validateParallelism } from './parallelism'
import { prefillAttentionFlops } from './attention'
import { bytesOf } from './dtypes'
import { INTERCONNECTS } from '../data/interconnects'

export function computePrefill(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  memory: MemoryResult,
  multiDeviceOverride?: MultiDeviceConfig,
): PerfTier['prefill'] {
  const { model, quant, workload } = input
  const p = workload.promptTokens
  const multiDevice = multiDeviceOverride ?? input.multiDevice

  validateParallelism(multiDevice, model)
  const { tp, ep } = parallelDegrees(multiDevice)
  const attentionFlops = prefillAttentionFlops(model, p)
  const recurrentFlops = p * (linearAttentionFlopsPerToken(model)
    + deltaAttentionFlopsPerToken(model) + mambaFlopsPerToken(model))
  // Prefill models one prompt. Concurrency is the decode batch size; memory
  // capacity separately reserves activations for concurrent prompts.
  const flops = 2 * activeParams(model) * p + attentionFlops + recurrentFlops
  const activations = memory.activationsPeak / Math.max(1, workload.concurrency)
  const bytes = expectedWeightParameters(model, p) * bytesOf(quant.weights) + activations
  // PP stages execute sequentially for request latency; divide storage by PP,
  // not the work along a request's full forward path.
  const rankFlops = 2 * activeParametersOnRank(model, tp, ep) * p + (attentionFlops + recurrentFlops) / tp
  const rankBytes = expectedWeightParameters(model, p, tp, ep) * bytesOf(quant.weights) + activations / tp

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  let commsBytes: number | undefined = undefined
  let interconnectBwGBs: number | undefined = undefined
  if (multiDevice) {
    const B = workload.promptTokens
    commsBytes = commsBytesPerStep(
      multiDevice.parallelism,
      multiDevice.parallelismDegrees,
      model,
      B,
      quant.activations
    )
    const ic = INTERCONNECTS.find(i => i.id === multiDevice.system.interconnectId)
    if (ic) interconnectBwGBs = ic.perDirectionGBs ?? ic.perGpuBandwidthGBs / 2
  }

  const { timeS, regime } = roofline({
    flops: rankFlops, bytes: rankBytes, tflops, bwGBs: opPoint.hbmBandwidthGBs,
    commsBytes, interconnectBwGBs
  })
  return { flops, bytes, rankFlops, rankBytes, timeS, regime, ...(commsBytes !== undefined && { commsBytes }) }
}
