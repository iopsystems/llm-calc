<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden" />
<Arrow x1={52} y1={88} x2={96} y2={58} label="" />
<Arrow x1={52} y1={104} x2={96} y2={128} label="" />
<Box x={96} y={40} w={118} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={110} w={118} h={36} label="W_k, W_v" sub="H_kv KV heads (H_kv ≤ H)" />
<Arrow x1={214} y1={128} x2={250} y2={128} label="append" />
<Mem x={250} y={100} w={160} h={56} label="KV cache" sub="one K row + one V row per KV head, per token" color="full" />
<Arrow x1={214} y1={58} x2={440} y2={58} label="H/H_kv query heads share each KV head" />
<Arrow x1={410} y1={128} x2={440} y2={90} label="read all" />
<Box x={440} y={40} w={150} h={60} label="attention" sub="softmax(q·K) V over every token" />
<Arrow x1={590} y1={70} x2={620} y2={70} label="" />
<text x="612" y="60" font-size="10" fill="#444">W_o</text>
<text x="8" y="176" font-size="10" fill="#555">GQA: fewer KV heads than query heads shrinks the cache row; the read is still the whole sequence.</text>
