import type { CalcInput, AcceleratorOperatingPoint, MemoryResult, PerfTier, MultiDeviceConfig } from './types'
import { roofline } from './roofline'
import {
  activeParams, averageDecodeCacheBytes, linearAttentionFlopsPerToken,
  deltaAttentionFlopsPerToken, mambaFlopsPerToken
} from './memory'
import { averageDecodeAttentionFlops } from './attention'
import { bytesOf } from './dtypes'
import { commsBytesPerStep, activeParametersOnRank, expectedWeightParameters, parallelDegrees, validateParallelism } from './parallelism'
import { INTERCONNECTS } from '../data/interconnects'

export function computeDecode(
  input: CalcInput,
  opPoint: AcceleratorOperatingPoint,
  _memory: MemoryResult,
  multiDeviceOverride?: MultiDeviceConfig | null,
): PerfTier['decode'] {
  const { model, quant, workload } = input
  const multiDevice = multiDeviceOverride === undefined
    ? input.decodeMultiDevice ?? input.multiDevice : multiDeviceOverride ?? undefined
  validateParallelism(multiDevice, model)
  const { tp, ep, dp } = parallelDegrees(multiDevice)
  const batch = workload.concurrency
  const replicaBatch = Math.ceil(batch / dp)
  const attentionFlops = averageDecodeAttentionFlops(model, workload.promptTokens, workload.outputTokens)
  const recurrentFlops = linearAttentionFlopsPerToken(model)
    + deltaAttentionFlopsPerToken(model) + mambaFlopsPerToken(model)
  const flopsPerStep = (2 * activeParams(model) + attentionFlops + recurrentFlops) * batch
  const cache = averageDecodeCacheBytes(model, quant.kv, workload.promptTokens, workload.outputTokens)
  // Aggregate logical work remains available for inspection. Runtime uses the
  // most-loaded DP replica with ideal EP balance; each replica loads its own weights.
  const smallerBatch = Math.floor(batch / dp)
  const largerReplicas = batch % dp
  const weightParams = largerReplicas * expectedWeightParameters(model, smallerBatch + 1)
    + (dp - largerReplicas) * expectedWeightParameters(model, smallerBatch)
  const bytesPerStep = weightParams * bytesOf(quant.weights)
    + (cache.bytes + cache.recurrentBytes) * batch
  const rankCache = averageDecodeCacheBytes(model, quant.kv, workload.promptTokens, workload.outputTokens, tp)
  const rankFlops = (2 * activeParametersOnRank(model, tp, ep)
    + (attentionFlops + recurrentFlops) / tp) * replicaBatch
  const rankBytes = expectedWeightParameters(model, replicaBatch, tp, ep) * bytesOf(quant.weights)
    + (rankCache.bytes + rankCache.recurrentBytes) * replicaBatch

  const tflops = opPoint.tflops[quant.activations]
  if (tflops === undefined) {
    throw new Error(`Operating point ${opPoint.id} lacks tflops for ${quant.activations}`)
  }

  let commsBytes: number | undefined = undefined
  let interconnectBwGBs: number | undefined = undefined
  if (multiDevice) {
    const B = replicaBatch  // decode: one token per request in this DP replica
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
    flops: rankFlops, bytes: rankBytes,
    tflops, bwGBs: opPoint.hbmBandwidthGBs,
    commsBytes, interconnectBwGBs
  })

  const mtpFactor = 1 + model.numNextnLayers
  return {
    rankFlops, rankBytes,
    flopsPerStep,
    bytesPerStep,
    timePerTokenS: timeS / mtpFactor,
    regime,
    aggregateTokensPerS: workload.concurrency * mtpFactor / timeS,
    ...(commsBytes !== undefined && { commsBytes })
  }
}
