<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={88} x2={96} y2={50} />
<Arrow x1={52} y1={96} x2={96} y2={96} />
<Arrow x1={52} y1={104} x2={96} y2={146} />
<Box x={96} y={32} w={110} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={78} w={110} h={36} label="W_dkv" sub="down-project" />
<Box x={96} y={128} w={110} h={36} label="indexer" sub="lightning, few heads" />
<Arrow x1={206} y1={96} x2={246} y2={96} label="cache" />
<Mem x={246} y={78} w={140} h={36} label="latent + k_rope" sub="every token, unchanged from MLA" color="full" />
<Arrow x1={316} y1={114} x2={316} y2={140} label="score all" />
<Arrow x1={206} y1={146} x2={266} y2={146} />
<Box x={266} y={128} w={100} h={36} label="scores" sub="one per cached token" />
<Arrow x1={366} y1={146} x2={412} y2={146} label="top-k" />
<Mem x={412} y={128} w={100} h={36} label="k tokens" sub="selected set" color="sparse" />
<Arrow x1={512} y1={146} x2={548} y2={110} label="read k" />
<Arrow x1={206} y1={50} x2={548} y2={60} />
<Box x={548} y={40} w={84} h={72} label="attention" sub="k tokens only" />
<text x="8" y="186" font-size="10" fill="#555">Cache size is MLA's. Attention FLOPs stop growing past k. The calculator does not cost the indexer.</text>
