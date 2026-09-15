<svelte:options namespace="svg" />
<script lang="ts">
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { KIND_COLORS } from '../figure/types'
  // The paper's fixed ratio (e.g. 5 sliding : 1 global for Gemma 3); the catalog carries counts, not order.
  const PATTERN = ['window', 'window', 'window', 'window', 'window', 'full'] as const
</script>
<text x="8" y="22" font-size="10.5" fill="#666" letter-spacing="0.06em">LAYER STACK (one repeat of the paper's pattern)</text>
{#each PATTERN as k, i}
  <rect x={8 + i * 44} y={30} width="40" height="18" rx="2" fill={KIND_COLORS[k]} />
  <text x={28 + i * 44} y={43} font-size="9.5" text-anchor="middle" fill="#fff" font-weight="600">{k === 'full' ? 'global' : 'sliding'}</text>
{/each}
<Arrow x1={120} y1={48} x2={120} y2={98} label="" />
<Arrow x1={228} y1={48} x2={400} y2={98} label="" />
<Mem x={30} y={100} w={180} h={56} label="sliding caches" sub="w rows per layer · bounded" color="window" />
<Mem x={320} y={100} w={220} h={56} label="global caches" sub="one row per token per layer · grows" color="full" />
<text x="8" y="176" font-size="10" fill="#555">The global layers carry the whole prompt; the sliding layers add a fixed cost. Which layers are global is fixed by the paper, not stored in the catalog.</text>
