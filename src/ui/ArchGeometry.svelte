<!-- Per-model geometry figure: five stacked panels drawn to scale from a
     GeometryModel. No math here; archGeometry.ts owns the numbers. -->
<script lang="ts">
  import type { GeometryModel } from './archGeometry'
  import { fmtBytes } from './archGeometry'
  import { KIND_COLORS, EXPERT_COLOR } from './figure/types'
  import CacheGlyph from './figure/CacheGlyph.svelte'
  import ReachGlyph from './figure/ReachGlyph.svelte'

  export let geometry: GeometryModel

  const W = 640, LX = 14, BX = 110, BW = W - BX - 14
  const CACHE_SCALE = 8192   // bytes per full bar width; longer bars clip with the value printed
  const ROW = 34             // two lines per lane: label, then glyph + bar
  const BAR_X = BX + 48      // clear of the 40px cache glyph
  const BAR_W = BW - 48

  $: g = geometry
  $: S = g.maxContext
  $: endLabel = S >= 1048576 ? `${S / 1048576}M` : `${Math.round(S / 1024)}K`
  // Panel origins. Heights depend on lane count and head-geometry entries.
  $: y1 = 0
  $: y2 = y1 + 44
  $: y3 = y2 + 24 + g.heads.length * 40
  $: y4 = y3 + 20 + g.lanes.length * ROW
  $: y5 = y4 + 20 + g.lanes.length * 36
  $: H = y5 + 70
  $: label = `${g.layers} blocks: ${g.lanes.map(l => `${l.count} ${l.label}`).join(', ')}. ` +
    (g.ffn.type === 'moe' ? `${g.ffn.routed} routed experts, ${g.ffn.active} active, ${g.ffn.shared} shared.` : 'Dense FFN.')

  function fracOf(l: GeometryModel['lanes'][number]): number {
    return l.reach.tokens !== undefined ? Math.min(1, l.reach.tokens / S) : 0
  }
  function localOf(l: GeometryModel['lanes'][number]): number {
    return l.reach.local !== undefined ? Math.min(1, l.reach.local / S) : 0
  }
  function stackX(i: number): number {
    return BX + g.lanes.slice(0, i).reduce((acc, l) => acc + BW * l.count / g.layers, 0)
  }
</script>

<svg viewBox="0 0 {W} {H}" role="img" aria-label={label}>
  <!-- Panel 1: block stack -->
  <text x={LX} y={y1 + 12} class="lbl">BLOCKS</text>
  <text x={BX + BW} y={y1 + 12} class="lbl" text-anchor="end">{g.layers} total · counts, not order</text>
  {#each g.lanes as l, i}
    {@const w = BW * l.count / g.layers}
    <rect x={stackX(i)} y={y1 + 18} width={Math.max(w - 1.5, 1)} height="14" rx="1" fill={KIND_COLORS[l.color]} />
    {#if w > 26}<text x={stackX(i) + 4} y={y1 + 29} font-size="10.5" fill="#fff" font-weight="600">{l.count}</text>{/if}
  {/each}

  <!-- Panel 2: head geometry -->
  <text x={LX} y={y2 + 12} class="lbl">HEADS</text>
  {#each g.heads as h, i}
    {@const hy = y2 + 18 + i * 40}
    {#if h.kind === 'gqa'}
      {@const groups = Math.min(h.numKvHeads, 8)}
      {@const perGroup = Math.min(h.numHeads / h.numKvHeads, 16)}
      {#each Array.from({ length: groups }) as _, gi}
        {@const gx = BX + gi * 58}
        <rect x={gx} y={hy} width="10" height="10" fill={KIND_COLORS.full} />
        {#each Array.from({ length: perGroup }) as _, qi}
          <rect x={gx + 13 + (qi % 8) * 5} y={hy + Math.floor(qi / 8) * 5} width="4" height="4" fill={KIND_COLORS.full} opacity="0.45" />
        {/each}
      {/each}
      <text x={BX} y={hy + 26} class="txt">
        {h.numHeads} query heads on {h.numKvHeads} KV heads · {h.numHeads / h.numKvHeads} : 1 · head dim {h.headDim}{#if h.numKvHeads > 8}{' '}· first 8 groups drawn{/if}
      </text>
    {:else if h.kind === 'latent'}
      {@const scale = 300 / (h.fullKvWidth ?? 1)}
      <rect x={BX} y={hy} width={(h.fullKvWidth ?? 0) * scale} height="8" fill={KIND_COLORS.full} opacity="0.3" />
      <rect x={BX} y={hy + 11} width={Math.max(3, (h.latentWidth ?? 0) * scale)} height="8" fill={KIND_COLORS.full} />
      <text x={BX} y={hy + 32} class="txt">
        per-head K+V would be {h.fullKvWidth} elements; the cached latent is {h.latentWidth} · {((h.fullKvWidth ?? 1) / (h.latentWidth ?? 1)).toFixed(0)}× smaller
      </text>
    {:else if h.state}
      {@const rw = Math.min(120, h.state.dim)}
      {@const rh = Math.min(24, h.state.inner * 24 / Math.max(h.state.inner, h.state.dim))}
      <rect x={BX} y={hy} width={rw} height={rh} fill="none" stroke={KIND_COLORS.state} stroke-width="1.5" />
      <text x={BX + rw + 8} y={hy + 12} class="txt">× {h.state.heads} heads</text>
      <text x={BX} y={hy + rh + 12} class="txt">
        state {h.state.heads} × {h.state.dim} × {h.state.inner} = {(h.state.heads * h.state.dim * h.state.inner).toLocaleString()} elements per layer
      </text>
    {/if}
  {/each}

  <!-- Panel 3: cache row per token -->
  <text x={LX} y={y3 + 12} class="lbl">PER TOKEN</text>
  <text x={BX + BW} y={y3 + 12} class="lbl" text-anchor="end">bar = {fmtBytes(CACHE_SCALE)} per layer</text>
  {#each g.lanes as l, i}
    {@const ry = y3 + 18 + i * ROW}
    {@const bw = Math.min(BAR_W, BAR_W * l.cache.bytesPerToken / CACHE_SCALE)}
    <rect x={BX - 8} y={ry + 2} width="5" height="12" fill={KIND_COLORS[l.color]} />
    <text x={BX} y={ry + 10} class="txt">{l.label} × {l.count}: {l.cache.label}</text>
    <CacheGlyph glyph={l.cache.glyph} color={KIND_COLORS[l.color]} x={BX} y={ry + 14} w={40} ratio={l.cache.ratio} />
    {#if l.cache.glyph !== 'state' && l.cache.glyph !== 'none'}
      <rect x={BAR_X} y={ry + 17} width={Math.max(2, bw)} height="10" fill={KIND_COLORS[l.color]} />
    {/if}
  {/each}

  <!-- Panel 4: reach at trained context -->
  <text x={LX} y={y4 + 12} class="lbl">DECODE READ</text>
  {#each g.lanes as l, i}
    {@const ry = y4 + 18 + i * 36}
    <text x={BX} y={ry - 2} class="txt">{l.label}: {l.reach.label}</text>
    <ReachGlyph glyph={l.reach.glyph} color={KIND_COLORS[l.color]} x={BX} y={ry + 2} w={BW} frac={fracOf(l)} local={localOf(l)} {endLabel} />
  {/each}

  <!-- Panel 5: FFN -->
  <text x={LX} y={y5 + 12} class="lbl">FFN</text>
  {#if g.ffn.type === 'dense'}
    {@const scale = 300 / Math.max(g.ffn.hiddenDim, g.ffn.intermediateDim)}
    <rect x={BX} y={y5 + 18} width={g.ffn.hiddenDim * scale} height="8" fill={EXPERT_COLOR} opacity="0.45" />
    <rect x={BX} y={y5 + 29} width={g.ffn.intermediateDim * scale} height="8" fill={EXPERT_COLOR} />
    <text x={BX} y={y5 + 50} class="txt">dense · hidden {g.ffn.hiddenDim} → intermediate {g.ffn.intermediateDim} · every token pays the full width</text>
  {:else}
    {@const aw = Math.max(4, BW * g.ffn.active / g.ffn.routed)}
    <rect x={BX} y={y5 + 18} width={BW} height="16" rx="2" fill="none" stroke={EXPERT_COLOR} stroke-width="1.5" />
    <rect x={BX + 1} y={y5 + 19} width={aw} height="14" rx="1" fill={EXPERT_COLOR} />
    {#each Array.from({ length: g.ffn.shared }) as _, si}
      <rect x={BX + si * 22} y={y5 + 38} width="18" height="8" rx="1" fill={EXPERT_COLOR} opacity="0.55" />
    {/each}
    <text x={BX} y={y5 + 60} class="txt">
      {g.ffn.routed} routed experts · {g.ffn.active} active{#if g.ffn.shared}{' '}+ {g.ffn.shared} shared, always on{/if} · {(g.ffn.activeRatio * 100).toFixed(1)}% of parameters active per token
    </text>
  {/if}
</svg>

<style>
  svg { display: block; width: 100%; min-width: 520px; height: auto; font-family: inherit; }
  .lbl { font-size: 10.5px; letter-spacing: 0.06em; fill: #666; }
  .txt { font-size: 10.5px; fill: #222; }
</style>
