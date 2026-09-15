<svelte:options namespace="svg" />
<script lang="ts">
  import { KIND_COLORS, type LaneColor } from '../../figure/types'
  export let x: number
  export let y: number
  export let w: number
  export let h: number
  export let label: string
  export let color: LaneColor
  export let dashed = false
  export let sub = ''
  $: c = KIND_COLORS[color]
</script>
<rect {x} {y} width={w} height={h} rx="3" fill={c} fill-opacity="0.16" stroke={c} stroke-width="1.5" stroke-dasharray={dashed ? '4 3' : undefined} />
<text x={x + w / 2} y={y + h / 2 + (sub ? -2 : 4)} font-size="11" text-anchor="middle" fill="#222" font-weight="600">{label}</text>
{#if sub}<text x={x + w / 2} y={y + h / 2 + 11} font-size="9.5" text-anchor="middle" fill="#555">{sub}</text>{/if}
