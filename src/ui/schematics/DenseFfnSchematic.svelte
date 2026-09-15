<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { EXPERT_COLOR } from '../figure/types'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden d" />
<Arrow x1={52} y1={88} x2={110} y2={58} />
<Arrow x1={52} y1={104} x2={110} y2={128} />
<rect x="110" y="40" width="140" height="36" rx="3" fill={EXPERT_COLOR} fill-opacity="0.2" stroke={EXPERT_COLOR} stroke-width="1.5" />
<text x="180" y="62" font-size="11" text-anchor="middle" fill="#222" font-weight="600">W_up · d → d_ff</text>
<rect x="110" y="110" width="140" height="36" rx="3" fill={EXPERT_COLOR} fill-opacity="0.2" stroke={EXPERT_COLOR} stroke-width="1.5" />
<text x="180" y="132" font-size="11" text-anchor="middle" fill="#222" font-weight="600">W_gate · d → d_ff</text>
<Arrow x1={250} y1={58} x2={300} y2={84} />
<Arrow x1={250} y1={128} x2={300} y2={102} />
<Box x={300} y={76} w={70} h={34} label="act ⊙" />
<Arrow x1={370} y1={93} x2={420} y2={93} />
<rect x="420" y="75" width="140" height="36" rx="3" fill={EXPERT_COLOR} fill-opacity="0.2" stroke={EXPERT_COLOR} stroke-width="1.5" />
<text x="490" y="97" font-size="11" text-anchor="middle" fill="#222" font-weight="600">W_down · d_ff → d</text>
<text x="8" y="176" font-size="10" fill="#555">Every token multiplies through all three matrices; a decode step streams all of them. Nothing is cached.</text>
