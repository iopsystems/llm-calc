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
<Box x={96} y={32} w={110} h={36} label="W_q" sub="H heads in G groups" />
<Box x={96} y={78} w={110} h={36} label="W_k, W_v" sub="G KV heads" />
<Box x={96} y={128} w={110} h={36} label="index key" sub="1 head, shared by groups" />
<Arrow x1={206} y1={96} x2={246} y2={96} label="append" />
<Mem x={246} y={78} w={140} h={36} label="KV cache" sub="every token, full" color="full" />
<Arrow x1={206} y1={146} x2={246} y2={146} label="append" />
<Mem x={246} y={128} w={140} h={36} label="index keys" sub="one per token, not TP-sharded" color="sparse" />
<Arrow x1={386} y1={146} x2={420} y2={146} label="per group" />
<Box x={420} y={128} w={100} h={36} label="block scores" sub="B tokens per block" />
<Arrow x1={470} y1={128} x2={470} y2={114} label="top-k blocks" />
<Mem x={420} y={78} w={100} h={36} label="k × B tokens" color="sparse" />
<Arrow x1={520} y1={96} x2={548} y2={84} label="read" />
<Arrow x1={206} y1={50} x2={548} y2={60} />
<Box x={548} y={40} w={84} h={60} label="attention" sub="selected blocks" />
<text x="8" y="186" font-size="10" fill="#555">A few layers stay full attention. Sparse layers keep the whole KV cache; selection caps compute only.</text>
