<svelte:options namespace="svg" />
<script lang="ts">
  export let x1: number
  export let y1: number
  export let x2: number
  export let y2: number
  export let label = ''
  $: mx = (x1 + x2) / 2
  $: my = (y1 + y2) / 2
</script>
<line {x1} {y1} {x2} {y2} stroke="#222" stroke-width="1.2" marker-end="url(#arr)" />
{#if label}<text x={mx} y={my - 5} font-size="9.5" text-anchor="middle" fill="#444">{label}</text>{/if}
