<svelte:options namespace="svg" />
<script lang="ts">
  import type { ReachGlyph } from './types'
  export let glyph: ReachGlyph
  export let color: string
  export let x: number
  export let y: number
  export let w: number
  export let frac = 0
  export let local = 0
  export let endLabel: string
  // Scattered block positions for top-k: deterministic so the figure is stable.
  const SPOTS = [0.08, 0.27, 0.46, 0.68, 0.9]
  $: localW = local > 0 ? Math.max(3, w * local) : 0
</script>

<line x1={x} y1={y + 6} x2={x + w} y2={y + 6} stroke="#d6d9e2" stroke-width="1" />
{#if glyph === 'all'}
  <rect {x} y={y + 2} width={w} height="8" fill={color} />
{:else if glyph === 'window'}
  {@const ww = Math.max(4, w * frac)}
  <rect x={x + w - ww} y={y + 2} width={ww} height="8" fill={color} />
{:else if glyph === 'topk'}
  {@const bw = Math.max(6, w * frac) / SPOTS.length}
  {#each SPOTS as p}
    <rect x={x + p * (w - bw)} y={y + 2} width={Math.max(2, bw)} height="8" fill={color} />
  {/each}
  {#if localW > 0}<rect x={x + w - localW} y={y + 2} width={localW} height="8" fill={color} opacity="0.6" />{/if}
{:else if glyph === 'compress'}
  <rect {x} y={y + 4} width={w} height="4" fill={color} opacity="0.45" />
  {#if localW > 0}<rect x={x + w - localW} y={y + 2} width={localW} height="8" fill={color} opacity="0.6" />{/if}
{:else if glyph === 'state'}
  <circle cx={x + w - 5} cy={y + 6} r="4.5" fill={color} />
{:else}
  <line x1={x} y1={y + 6} x2={x + w} y2={y + 6} stroke={color} stroke-width="1.5" stroke-dasharray="2 3" />
{/if}
<text {x} y={y + 22} font-size="9.5" fill="#777">tok 0</text>
<text x={x + w} y={y + 22} font-size="9.5" fill="#777" text-anchor="end">{endLabel}</text>
