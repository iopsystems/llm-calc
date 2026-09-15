<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<text x="8" y="20" font-size="10.5" fill="#666" letter-spacing="0.06em">CSA BLOCK</text>
<Box x={8} y={30} w={60} h={28} label="tokens" />
<Arrow x1={68} y1={44} x2={104} y2={44} label="" />
<Box x={104} y={30} w={72} h={28} label="÷ M" sub="compress" />
<Arrow x1={176} y1={44} x2={212} y2={44} label="cache" />
<Mem x={212} y={30} w={130} h={28} label="1 entry per M tokens" color="sparse" />
<Arrow x1={342} y1={44} x2={378} y2={44} label="indexer" />
<Mem x={378} y={30} w={90} h={28} label="top-k entries" color="sparse" />
<Arrow x1={468} y1={44} x2={520} y2={44} />
<Mem x={212} y={66} w={130} h={22} label="local window w" color="window" />
<Arrow x1={342} y1={77} x2={520} y2={56} />
<Box x={520} y={30} w={112} h={40} label="attention" sub="k entries + window" />

<text x="8" y="112" font-size="10.5" fill="#666" letter-spacing="0.06em">HCA BLOCK</text>
<Box x={8} y={122} w={60} h={28} label="tokens" />
<Arrow x1={68} y1={136} x2={104} y2={136} />
<Box x={104} y={122} w={72} h={28} label="÷ 128" sub="compress" />
<Arrow x1={176} y1={136} x2={212} y2={136} label="cache" />
<Mem x={212} y={122} w={130} h={28} label="1 entry per 128 tokens" color="sparse" />
<Arrow x1={342} y1={136} x2={520} y2={136} label="read all entries" />
<Mem x={212} y={158} w={130} h={22} label="local window w" color="window" />
<Arrow x1={342} y1={169} x2={520} y2={148} />
<Box x={520} y={122} w={112} h={40} label="attention" sub="all entries + window" />
<text x="8" y="192" font-size="10" fill="#555">Cache grows at 1/M and 1/128 of a token per token. The compressor and indexer are not costed.</text>
