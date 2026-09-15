<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { EXPERT_COLOR } from '../figure/types'
  // Eight drawn experts stand in for N; two lit ones stand in for k.
  const ACTIVE = new Set([1, 5])
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={95} x2={96} y2={95} />
<Box x={96} y={78} w={80} h={34} label="router" sub="top-k of N" />
{#each Array.from({ length: 8 }) as _, i}
  {@const ex = 220 + i * 46}
  <Arrow x1={176} y1={95} x2={ex} y2={74} />
  <rect x={ex} y={44} width="38" height="30" rx="3"
    fill={ACTIVE.has(i) ? EXPERT_COLOR : '#fff'} stroke={EXPERT_COLOR} stroke-width="1.5" />
  <text x={ex + 19} y={63} font-size="10" text-anchor="middle" fill={ACTIVE.has(i) ? '#fff' : '#555'} font-weight="600">E{i + 1}</text>
{/each}
<text x="580" y="63" font-size="10" fill="#555">… E_N</text>
<rect x="220" y="120" width="170" height="30" rx="3" fill={EXPERT_COLOR} fill-opacity="0.55" stroke={EXPERT_COLOR} stroke-width="1.5" />
<text x="305" y="139" font-size="10.5" text-anchor="middle" fill="#222" font-weight="600">shared expert(s) · always on</text>
<Arrow x1={176} y1={100} x2={220} y2={135} />
<Arrow x1={266} y1={74} x2={470} y2={140} label="" />
<Arrow x1={450} y1={74} x2={490} y2={130} />
<Arrow x1={390} y1={135} x2={470} y2={150} />
<Box x={470} y={130} w={90} h={34} label="Σ weighted" />
<text x="8" y="186" font-size="10" fill="#555">A decode step streams k routed experts plus the shared ones, not all N. Active parameters, not total, set the weight-side cost.</text>
