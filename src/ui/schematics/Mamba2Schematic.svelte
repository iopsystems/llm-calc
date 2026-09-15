<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { KIND_COLORS } from '../figure/types'
</script>
<text x="8" y="20" font-size="10.5" fill="#666" letter-spacing="0.06em">MAMBA2 BLOCK</text>
<Box x={8} y={40} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={57} x2={84} y2={57} />
<Box x={84} y={40} w={70} h={34} label="in-proj" />
<Arrow x1={154} y1={57} x2={186} y2={57} />
<Box x={186} y={40} w={70} h={34} label="conv1d" />
<Arrow x1={256} y1={57} x2={288} y2={57} />
<Box x={288} y={40} w={90} h={34} label="SSM scan" sub="selective" />
<Arrow x1={333} y1={74} x2={333} y2={98} label="read, update" />
<Mem x={258} y={98} w={150} h={34} label="state per head" sub="d × N, held in fp32" color="state" />
<Arrow x1={378} y1={57} x2={410} y2={57} />
<Box x={410} y={40} w={70} h={34} label="out-proj" />
<text x="8" y="158" font-size="10.5" fill="#666" letter-spacing="0.06em">STACK: three block kinds, counted separately</text>
<rect x="270" y="146" width="40" height="16" rx="2" fill={KIND_COLORS.state} /><text x="290" y="158" font-size="9.5" text-anchor="middle" fill="#fff" font-weight="600">M</text>
<rect x="314" y="146" width="40" height="16" rx="2" fill={KIND_COLORS.full} /><text x="334" y="158" font-size="9.5" text-anchor="middle" fill="#fff" font-weight="600">attn</text>
<rect x="358" y="146" width="40" height="16" rx="2" fill={KIND_COLORS.none} /><text x="378" y="158" font-size="9.5" text-anchor="middle" fill="#fff" font-weight="600">FFN</text>
<text x="8" y="186" font-size="10" fill="#555">Attention blocks keep a GQA cache; FFN blocks keep nothing. The layer count in the config counts all three kinds.</text>
