<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden" />
<Arrow x1={52} y1={88} x2={96} y2={58} />
<Arrow x1={52} y1={104} x2={96} y2={128} />
<Box x={96} y={40} w={118} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={110} w={118} h={36} label="W_k, W_v" sub="H_kv KV heads" />
<Arrow x1={214} y1={128} x2={250} y2={128} label="append" />
<Mem x={250} y={100} w={160} h={56} label="ring of w slots" sub="K and V for the last w tokens" color="window" />
<Mem x={430} y={116} w={80} h={24} label="t − w" sub="" color="none" dashed />
<Arrow x1={410} y1={128} x2={430} y2={128} label="evict" />
<Arrow x1={214} y1={58} x2={440} y2={58} />
<Arrow x1={330} y1={100} x2={440} y2={80} label="read the window" />
<Box x={440} y={40} w={150} h={50} label="attention" sub="over the trailing w tokens only" />
<text x="8" y="176" font-size="10" fill="#555">Each layer sees w tokens; L layers stack to a receptive field of about L × w. The cache never exceeds w rows per layer.</text>
