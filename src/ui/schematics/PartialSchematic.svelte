<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { KIND_COLORS } from '../figure/types'
</script>
<text x="8" y="22" font-size="10.5" fill="#666" letter-spacing="0.06em">TWO BLOCK SHAPES IN ONE STACK</text>
<!-- attending block -->
<Box x={8} y={40} w={70} h={30} label="x" />
<Arrow x1={78} y1={55} x2={110} y2={55} />
<Box x={110} y={40} w={90} h={30} label="attention" />
<Arrow x1={155} y1={70} x2={155} y2={96} label="append" />
<Mem x={110} y={96} w={90} h={30} label="KV cache" color="full" />
<Arrow x1={200} y1={55} x2={232} y2={55} />
<Box x={232} y={40} w={70} h={30} label="FFN" />
<text x="8" y="146" font-size="10" fill="#444">attending block: full GQA, parent geometry</text>
<!-- pruned block -->
<Box x={340} y={40} w={70} h={30} label="x" />
<Arrow x1={410} y1={55} x2={442} y2={55} />
<rect x="442" y="40" width="90" height="30" rx="3" fill="none" stroke={KIND_COLORS.none} stroke-width="1.5" stroke-dasharray="4 3" />
<text x="487" y="59" font-size="10" text-anchor="middle" fill="#888">attention removed</text>
<Arrow x1={532} y1={55} x2={564} y2={55} />
<Box x={564} y={40} w={70} h={30} label="FFN" sub="width varies" />
<text x="340" y="146" font-size="10" fill="#444">pruned block: no attention, no cache, no read</text>
<text x="8" y="176" font-size="10" fill="#555">Search decides per block; the catalog records how many blocks kept attention.</text>
