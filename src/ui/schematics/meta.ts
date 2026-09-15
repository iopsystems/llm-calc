// Registry of educational architecture schematics, keyed by the engine's
// discriminants so a new variant is a type error until it has a drawing.
// No Svelte imports here: components.ts maps ids to components.
import type { AttentionConfig, ArchitectureConfig, ModelArch } from '../../engine/types'
import { SOURCES } from '../../data/sources'

export type SchematicId = AttentionConfig['type'] | ArchitectureConfig['type']
export type SchematicGroup = 'sequence' | 'window' | 'sparse' | 'state' | 'ffn'
export type Growth = 'unbounded' | 'bounded' | 'constant' | 'none'

export interface SchematicMeta {
  id: SchematicId
  title: string
  group: SchematicGroup
  summary: string
  leaves: string       // what one token leaves behind, per layer of this kind
  reads: string        // what a decode step reads back
  growth: Growth       // how memory scales with context
  sources: (keyof typeof SOURCES)[]
}

export const ATTENTION_SCHEMATICS: Record<AttentionConfig['type'], SchematicMeta> = {
  'full': {
    id: 'full', title: 'Full attention (MHA / GQA)', group: 'sequence',
    summary: 'Every layer projects the token to query heads and to a smaller set of KV heads, appends one K and one V row per KV head to the cache, and attends over every cached token. Grouped-query attention shares each KV head across several query heads, which shrinks the cache without changing what is read.',
    leaves: 'K and V rows, one per KV head', reads: 'every cached token', growth: 'unbounded',
    sources: ['arxiv-2305-13245'],
  },
  'sliding': {
    id: 'sliding', title: 'Sliding-window attention', group: 'window',
    summary: 'The same block as full attention, but the cache is a ring of w slots. A token older than w is dropped, so the cache stops growing and each layer sees only the trailing window; depth stacks windows into a wider receptive field.',
    leaves: 'K and V rows, kept for w tokens', reads: 'the trailing w tokens', growth: 'bounded',
    sources: ['arxiv-2310-06825'],
  },
  'hybrid': {
    id: 'hybrid', title: 'Sliding / global hybrid', group: 'window',
    summary: 'Sliding layers are interleaved with a few global layers. The sliding caches are bounded; the global caches grow with context and are what let the model see the whole prompt.',
    leaves: 'K and V rows; bounded on sliding layers, growing on global layers', reads: 'window on most layers, everything on global layers', growth: 'unbounded',
    sources: ['arxiv-2503-19786'],
  },
  'partial': {
    id: 'partial', title: 'Partial attention (NAS-pruned)', group: 'sequence',
    summary: 'Neural architecture search removed the attention path from some blocks, leaving only their FFN. The surviving blocks are ordinary full attention with the parent model’s head geometry.',
    leaves: 'K and V rows on attending blocks; nothing on pruned blocks', reads: 'every cached token, on attending blocks', growth: 'unbounded',
    sources: ['arxiv-2411-19146'],
  },
  'mla': {
    id: 'mla', title: 'Multi-head latent attention (MLA)', group: 'sequence',
    summary: 'The token is down-projected to one latent vector plus a small RoPE key, and only those are cached. At use, the latent is up-projected into all heads’ keys and values. The cache row is the latent, not the heads, so it is tens of times smaller than per-head K and V at the same head count.',
    leaves: 'one latent vector and one RoPE key', reads: 'every cached token', growth: 'unbounded',
    sources: ['arxiv-2405-04434'],
  },
  'mla-dsa': {
    id: 'mla-dsa', title: 'MLA with DeepSeek sparse attention (DSA)', group: 'sparse',
    summary: 'The MLA block gains an indexer branch that scores every cached token and keeps the top k. The main attention reads only those k. The cache is unchanged; what stops growing with context is the attention compute.',
    leaves: 'one latent vector and one RoPE key', reads: 'the top-k scored tokens', growth: 'unbounded',
    sources: ['deepseek-v3-2-exp', 'arxiv-2405-04434'],
  },
  'msa-hybrid': {
    id: 'msa-hybrid', title: 'MiniMax sparse attention (MSA)', group: 'sparse',
    summary: 'A GQA block plus one index-key head, shared by every GQA group, cached per token. Blocks of the sequence are scored per group and the top-k blocks are attended. A few layers stay full attention.',
    leaves: 'K and V rows plus one shared index key', reads: 'the top-k scored blocks', growth: 'unbounded',
    sources: ['arxiv-2606-13392'],
  },
  'csa-hca-hybrid': {
    id: 'csa-hca-hybrid', title: 'Compressed sparse and heavily compressed attention (CSA / HCA)', group: 'sparse',
    summary: 'Two block types store the past at reduced resolution. CSA compresses every M tokens into one cache entry and attends a top-k of those; HCA compresses 128 : 1 and attends all of them. Both add a short local-window branch so recent tokens are seen at full resolution.',
    leaves: 'one compressed entry per M tokens, plus a local window', reads: 'top-k compressed entries (CSA) or all of them (HCA), plus the local window', growth: 'unbounded',
    sources: ['deepseek-v4-report'],
  },
  'linear-mla-hybrid': {
    id: 'linear-mla-hybrid', title: 'Kimi delta attention with MLA (linear / MLA hybrid)', group: 'state',
    summary: 'Most layers are KDA: a fixed state matrix per head is updated by a gated delta rule and read by the query. Nothing is stored per token. Every fourth layer is MLA and keeps an ordinary growing latent cache.',
    leaves: 'nothing on KDA layers; a latent on MLA layers', reads: 'one state matrix on KDA layers; everything on MLA layers', growth: 'unbounded',
    sources: ['arxiv-2510-26692'],
  },
  'delta-hybrid': {
    id: 'delta-hybrid', title: 'Gated DeltaNet with gated attention', group: 'state',
    summary: 'Three of every four layers are Gated DeltaNet: a per-head state matrix updated by a gated delta rule. The remaining layers are gated attention with partial RoPE and a small KV cache.',
    leaves: 'nothing on DeltaNet layers; K and V rows on attention layers', reads: 'one state matrix, or every cached token', growth: 'unbounded',
    sources: ['arxiv-2412-06464'],
  },
  'mamba2-hybrid': {
    id: 'mamba2-hybrid', title: 'Mamba2 hybrid (NemotronH)', group: 'state',
    summary: 'Mamba2 blocks run a convolution then a selective state-space scan against a per-head state held in fp32. Attention blocks and FFN-only blocks are separate entries in the stack, not paired with the Mamba2 blocks.',
    leaves: 'nothing on Mamba2 blocks; K and V rows on attention blocks', reads: 'one SSM state, or every cached token', growth: 'unbounded',
    sources: ['arxiv-2405-21060', 'arxiv-2504-03624'],
  },
}

export const FFN_SCHEMATICS: Record<ArchitectureConfig['type'], SchematicMeta> = {
  'dense': {
    id: 'dense', title: 'Dense FFN', group: 'ffn',
    summary: 'Up, gate and down projections. Every token pays the full intermediate width, and a decode step streams all of it from memory.',
    leaves: 'nothing', reads: 'the whole FFN weight', growth: 'none',
    sources: ['arxiv-2305-13245'],
  },
  'moe': {
    id: 'moe', title: 'Mixture of experts', group: 'ffn',
    summary: 'A router picks k of N routed experts per token; shared experts run for every token. Only the active experts’ weights are streamed per decode step, so the active parameter count, not the total, sets the weight-side cost.',
    leaves: 'nothing', reads: 'k routed experts plus the shared experts', growth: 'none',
    sources: ['arxiv-2101-03961', 'arxiv-2401-06066'],
  },
}

export const SCHEMATIC_GROUPS: { id: SchematicGroup; title: string; ids: SchematicId[] }[] = [
  { id: 'sequence', title: 'Every layer keeps every token', ids: ['full', 'partial', 'mla'] },
  { id: 'window', title: 'Windows cap the cache', ids: ['sliding', 'hybrid'] },
  { id: 'sparse', title: 'Keep everything, read a selection', ids: ['mla-dsa', 'msa-hybrid', 'csa-hca-hybrid'] },
  { id: 'state', title: 'Fixed state instead of a cache', ids: ['linear-mla-hybrid', 'delta-hybrid', 'mamba2-hybrid'] },
  { id: 'ffn', title: 'Feed-forward', ids: ['dense', 'moe'] },
]

const ALL: Record<string, SchematicMeta> = { ...ATTENTION_SCHEMATICS, ...FFN_SCHEMATICS }

export function schematicFor(id: string): SchematicMeta | undefined {
  return ALL[id]
}

export function modelsUsing(id: SchematicId, models: ModelArch[]): ModelArch[] {
  return models.filter(m => m.attention.type === id || m.architecture.type === id)
}
