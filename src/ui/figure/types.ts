// Glyph vocabulary shared by the per-model geometry figure and the
// per-variant schematics. Colour encodes what a layer reads at decode.
export type LaneColor = 'full' | 'window' | 'sparse' | 'state' | 'none'
export type CacheGlyph = 'kv' | 'latent' | 'kvidx' | 'kvcomp' | 'state' | 'none'
export type ReachGlyph = 'all' | 'window' | 'topk' | 'compress' | 'state' | 'none'

export const KIND_COLORS: Record<LaneColor, string> = {
  full: '#2f5fd0',
  window: '#3f9cc2',
  sparse: '#7a4fc9',
  state: '#1f9a6e',
  none: '#b4bac8',
}
export const EXPERT_COLOR = '#e07a2a'
