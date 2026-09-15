<!-- Hosts one schematic (top 190px) and its class-level footprint strip. -->
<script lang="ts">
  import type { Component } from 'svelte'
  import type { SchematicMeta } from './meta'
  import { KIND_COLORS } from '../figure/types'
  export let meta: SchematicMeta
  export let component: Component

  const GROWTH_LABEL = {
    unbounded: 'grows with context', bounded: 'bounded by the window',
    constant: 'constant per request', none: 'no per-request memory',
  } as const
  $: growthColor = meta.growth === 'unbounded' ? KIND_COLORS.full
    : meta.growth === 'bounded' ? KIND_COLORS.window
    : meta.growth === 'constant' ? KIND_COLORS.state : KIND_COLORS.none
</script>

<svg viewBox="0 0 640 262" role="img" aria-label="{meta.title}: leaves {meta.leaves}; reads {meta.reads}; memory {GROWTH_LABEL[meta.growth]}.">
  <defs>
    <marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0L8 4L0 8z" fill="#222" />
    </marker>
  </defs>
  <svelte:component this={component} />
  <line x1="0" y1="196" x2="640" y2="196" stroke="#ddd" />
  <text x="8" y="212" class="lbl">LEAVES PER TOKEN</text>
  <text x="8" y="228" class="txt">{meta.leaves}</text>
  <text x="240" y="212" class="lbl">READS AT DECODE</text>
  <text x="240" y="228" class="txt">{meta.reads}</text>
  <text x="480" y="212" class="lbl">MEMORY</text>
  <rect x="480" y="219" width="10" height="10" fill={growthColor} />
  <text x="495" y="228" class="txt">{GROWTH_LABEL[meta.growth]}</text>
</svg>

<style>
  svg { display: block; width: 100%; min-width: 520px; height: auto; font-family: inherit; }
  .lbl { font-size: 10px; letter-spacing: 0.06em; fill: #666; }
  .txt { font-size: 10.5px; fill: #222; }
</style>
