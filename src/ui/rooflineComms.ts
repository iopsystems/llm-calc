// Per-phase interconnect ceilings for the roofline chart.
//
// The roofline's x-axis is HBM arithmetic intensity (FLOPs ÷ HBM bytes), but
// comms time divides by a different denominator (collective-traffic bytes).
// A raw y = x × fabric_BW line therefore never touches a comms-bound marker —
// it implicitly assumes comms bytes == HBM bytes. Rescaling the slope by each
// phase's actual comms/HBM byte ratio puts the ceiling in the marker's own
// coordinate space: a comms-bound marker sits exactly on its phase's line,
// the same way memory-/compute-bound markers sit on the HBM roof.

export interface PhaseTraffic {
  phase: 'prefill' | 'decode'
  flops: number
  hbmBytes: number
  commsBytes?: number
}

export interface CommsCeiling {
  phase: 'prefill' | 'decode'
  // commsBytes ÷ hbmBytes — how much more (or less) traffic the phase pushes
  // over the fabric than over HBM. Surfaced in the line label.
  ratio: number
  // Ceiling in HBM-AI space: perf = AI × slope. Equals fabric BW ÷ ratio.
  slopeBytesPerS: number
  label: string
}

// 3 significant figures, trailing zeros stripped ("11.4", "0.00022", "1").
function fmtRatio(v: number): string {
  return parseFloat(v.toPrecision(3)).toString()
}

export function commsCeilings(
  phases: PhaseTraffic[],
  interconnectPerDirectionBs: number,
): CommsCeiling[] {
  const out: CommsCeiling[] = []
  for (const p of phases) {
    if (!p.commsBytes || p.commsBytes <= 0 || p.hbmBytes <= 0) continue
    const ratio = p.commsBytes / p.hbmBytes
    out.push({
      phase: p.phase,
      ratio,
      slopeBytesPerS: interconnectPerDirectionBs / ratio,
      label: `${p.phase}: comms = ${fmtRatio(ratio)}× HBM bytes`,
    })
  }
  return out
}
