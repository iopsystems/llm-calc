<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden" />
<Arrow x1={52} y1={88} x2={96} y2={50} />
<Arrow x1={52} y1={104} x2={96} y2={128} />
<Box x={96} y={32} w={110} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={110} w={110} h={36} label="W_dkv" sub="down-project" />
<Arrow x1={206} y1={120} x2={246} y2={110} label="cache" />
<Arrow x1={206} y1={136} x2={246} y2={150} label="cache" />
<Mem x={246} y={92} w={130} h={30} label="latent c_t" sub="rank r" color="full" />
<Mem x={246} y={136} w={130} h={30} label="k_rope" sub="d_rope, shared by heads" color="full" />
<Arrow x1={376} y1={107} x2={412} y2={107} label="expand" />
<Box x={412} y={90} w={100} h={36} label="W_uk, W_uv" sub="→ H heads" />
<Arrow x1={512} y1={107} x2={536} y2={80} />
<Arrow x1={206} y1={50} x2={536} y2={50} />
<Arrow x1={376} y1={151} x2={536} y2={90} label="RoPE part" />
<Box x={536} y={36} w={96} h={60} label="attention" sub="all tokens, H heads" />
<text x="8" y="176" font-size="10" fill="#555">The cache row is r + d_rope elements per token, not 2 × H × d. Heads are recovered from the latent at read time.</text>
