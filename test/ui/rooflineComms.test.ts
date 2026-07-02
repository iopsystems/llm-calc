import { describe, it, expect } from 'vitest'
import { commsCeilings } from '../../src/ui/rooflineComms'

// Values from the mi355x-8 / glm-5.2 / tp8.ep8 / 204800-prompt repro that
// exposed the original bug (single interconnect line drawn in HBM-AI space
// with the wrong byte denominator, so comms-bound markers never touched it).
const IC_BW = 537.6e9 // xgmi-mi350 per-direction, bytes/s

const PREFILL = { phase: 'prefill' as const, flops: 1.642e16, hbmBytes: 1.808e12, commsBytes: 2.061e13 }
const DECODE  = { phase: 'decode'  as const, flops: 1.604e12, hbmBytes: 4.573e11, commsBytes: 1.006e8 }

describe('commsCeilings', () => {
  it('computes one ceiling per phase with comms traffic', () => {
    const out = commsCeilings([PREFILL, DECODE], IC_BW)
    expect(out).toHaveLength(2)
    expect(out[0].phase).toBe('prefill')
    expect(out[1].phase).toBe('decode')
  })

  it('scales the interconnect slope by the phase comms/HBM byte ratio', () => {
    const [pre] = commsCeilings([PREFILL], IC_BW)
    expect(pre.ratio).toBeCloseTo(2.061e13 / 1.808e12, 3) // ≈ 11.4
    expect(pre.slopeBytesPerS).toBeCloseTo(IC_BW / pre.ratio, 0)
  })

  it('places a comms-bound marker exactly on its phase ceiling', () => {
    // Marker: x = flops/hbmBytes, y = flops/commsTime. On the ceiling,
    // y must equal x × slope — that's the property the old single line broke.
    const [pre] = commsCeilings([PREFILL], IC_BW)
    const ai = PREFILL.flops / PREFILL.hbmBytes
    const commsTimeS = PREFILL.commsBytes / IC_BW
    const markerPerf = PREFILL.flops / commsTimeS
    expect(ai * pre.slopeBytesPerS).toBeCloseTo(markerPerf, -6)
  })

  it('labels each line with the phase and the comms-to-HBM-bytes multiplier', () => {
    const [pre, dec] = commsCeilings([PREFILL, DECODE], IC_BW)
    expect(pre.label).toBe('prefill: comms = 11.4× HBM bytes')
    expect(dec.label).toBe('decode: comms = 0.00022× HBM bytes')
  })

  it('skips phases without comms traffic (single-device or zero volume)', () => {
    const noComms = { phase: 'prefill' as const, flops: 1e15, hbmBytes: 1e12, commsBytes: undefined }
    const zero = { phase: 'decode' as const, flops: 1e12, hbmBytes: 1e11, commsBytes: 0 }
    expect(commsCeilings([noComms, zero], IC_BW)).toHaveLength(0)
  })
})
