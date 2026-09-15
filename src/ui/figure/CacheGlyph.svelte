<svelte:options namespace="svg" />
<script lang="ts">
  import type { CacheGlyph } from './types'
  export let glyph: CacheGlyph
  export let color: string
  export let x: number
  export let y: number
  export let w = 64
  export let ratio: number | undefined = undefined
</script>

{#if glyph === 'kv' || glyph === 'kvidx' || glyph === 'kvcomp'}
  {@const kw = glyph === 'kvidx' ? w - 10 : w}
  <rect {x} {y} width={kw} height="7" rx="1" fill={color} />
  <rect {x} y={y + 9} width={kw} height="7" rx="1" fill={color} opacity="0.55" />
  {#if glyph === 'kvidx'}
    <rect x={x + kw + 3} {y} width="7" height="16" rx="1" fill={color} opacity="0.35" />
  {/if}
  {#if glyph === 'kvcomp'}
    <text x={x + w / 2} y={y + 12.5} font-size="10" text-anchor="middle" fill="#fff" font-weight="600">÷{ratio}</text>
  {/if}
{:else if glyph === 'latent'}
  <rect {x} y={y + 3} width={w * 0.68} height="10" rx="1" fill={color} />
  <rect x={x + w * 0.68 + 3} y={y + 3} width={w * 0.32 - 3} height="10" rx="1" fill={color} opacity="0.45" />
{:else if glyph === 'state'}
  <rect {x} {y} width="16" height="16" rx="1" fill="none" stroke={color} stroke-width="1.5" />
  <path d="M{x} {y + 8}h16M{x + 8} {y}v16" stroke={color} stroke-width="1" />
{:else}
  <rect {x} y={y + 2} width={w} height="12" rx="1" fill="none" stroke={color} stroke-width="1.5" stroke-dasharray="3 2" />
{/if}
