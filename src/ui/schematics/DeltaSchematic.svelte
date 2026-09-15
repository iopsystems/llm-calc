<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { KIND_COLORS } from '../figure/types'
  const PATTERN = ['state', 'state', 'state', 'full'] as const
</script>
<text x="8" y="20" font-size="10.5" fill="#666" letter-spacing="0.06em">GATED DELTANET BLOCK</text>
<Box x={8} y={40} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={57} x2={90} y2={57} />
<Box x={90} y={40} w={120} h={34} label="W_q W_k W_v" sub="+ decay gate, β" />
<Arrow x1={210} y1={57} x2={250} y2={57} label="delta rule" />
<Mem x={250} y={30} w={170} h={54} label="state S per head" sub="d × d, fixed size" color="state" />
<Arrow x1={420} y1={57} x2={460} y2={57} label="o = q · S" />
<Box x={460} y={40} w={80} h={34} label="out" />
<text x="8" y="146" font-size="10.5" fill="#666" letter-spacing="0.06em">STACK</text>
{#each PATTERN as k, i}
  <rect x={60 + i * 50} y={134} width="46" height="18" rx="2" fill={KIND_COLORS[k]} />
  <text x={83 + i * 50} y={147} font-size="9.5" text-anchor="middle" fill="#fff" font-weight="600">{k === 'full' ? 'attn' : 'ΔNet'}</text>
{/each}
<Mem x={280} y={130} w={240} h={26} label="gated attention KV cache · partial RoPE · grows" color="full" />
<text x="8" y="186" font-size="10" fill="#555">Three of four layers hold a fixed state; the attention layers keep a small GQA cache that grows with context.</text>
