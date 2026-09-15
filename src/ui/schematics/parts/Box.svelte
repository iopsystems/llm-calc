<svelte:options namespace="svg" />
<script lang="ts">
  export let x: number
  export let y: number
  export let w: number
  export let h: number
  export let label: string
  export let sub = ''
</script>
<rect {x} {y} width={w} height={h} rx="3" fill="#fff" stroke="#222" stroke-width="1.2" />
<text x={x + w / 2} y={y + h / 2 + (sub ? -2 : 4)} font-size="11" text-anchor="middle" fill="#222" font-weight="600">{label}</text>
{#if sub}<text x={x + w / 2} y={y + h / 2 + 11} font-size="9.5" text-anchor="middle" fill="#555">{sub}</text>{/if}
