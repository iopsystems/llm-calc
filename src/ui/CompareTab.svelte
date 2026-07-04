<script lang="ts">
  import { ACCELERATORS, MODELS } from '../data'
  import { SYSTEMS } from '../data/systems'
  import { comparePivot, compareCandidates, compareWorkload, setComparePivotKind } from './stores'
  import { firstVaryingId, seededQuantFor } from './compareModel'
  import CompareTable from './CompareTable.svelte'

  // Options for the varying dimension (what each candidate picks from) and the
  // pivot dimension (the single locked selector), keyed off the current pivot.
  const varyingOptions = $derived(
    $comparePivot.kind === 'sku'
      ? MODELS.map(m => ({ id: m.id, name: m.name }))
      : [...SYSTEMS.map(s => ({ id: s.id, name: s.name })), ...ACCELERATORS.map(a => ({ id: a.id, name: a.name }))]
  )
  const pivotOptions = $derived(
    $comparePivot.kind === 'sku'
      ? [...ACCELERATORS.map(a => ({ id: a.id, name: a.name })), ...SYSTEMS.map(s => ({ id: s.id, name: s.name }))]
      : MODELS.map(m => ({ id: m.id, name: m.name }))
  )

  function addCandidate() {
    const id = firstVaryingId($comparePivot.kind)
    const quant = $comparePivot.kind === 'sku' ? seededQuantFor(id) : seededQuantFor($comparePivot.id)
    compareCandidates.update(cs => [...cs, { varyingId: id, quant }])
  }
  function removeCandidate(i: number) {
    compareCandidates.update(cs => cs.filter((_, j) => j !== i))
  }
</script>

<section class="controls">
  <div class="row">
    <span class="lbl">Compare</span>
    <label><input type="radio" checked={$comparePivot.kind === 'sku'} onchange={() => setComparePivotKind('sku')} /> models on one accelerator</label>
    <label><input type="radio" checked={$comparePivot.kind === 'model'} onchange={() => setComparePivotKind('model')} /> accelerators for one model</label>
  </div>

  <div class="row">
    <span class="lbl">{$comparePivot.kind === 'sku' ? 'Accelerator' : 'Model'} (fixed)</span>
    <select value={$comparePivot.id} onchange={e => comparePivot.update(p => ({ ...p, id: (e.currentTarget as HTMLSelectElement).value }))}>
      {#each pivotOptions as o}<option value={o.id}>{o.name}</option>{/each}
    </select>
  </div>

  <div class="row workload">
    <label>Prompt <input type="number" min="1" value={$compareWorkload.promptTokens} onchange={e => compareWorkload.update(w => ({ ...w, promptTokens: +(e.currentTarget as HTMLInputElement).value }))} /></label>
    <label>Output <input type="number" min="1" value={$compareWorkload.outputTokens} onchange={e => compareWorkload.update(w => ({ ...w, outputTokens: +(e.currentTarget as HTMLInputElement).value }))} /></label>
    <label>Concurrency <input type="number" min="1" value={$compareWorkload.concurrency} onchange={e => compareWorkload.update(w => ({ ...w, concurrency: +(e.currentTarget as HTMLInputElement).value }))} /></label>
  </div>

  <div class="candidates">
    <span class="lbl">Candidates ({$comparePivot.kind === 'sku' ? 'models' : 'accelerators'})</span>
    {#each $compareCandidates as c, i}
      <div class="cand">
        <select value={c.varyingId} onchange={e => compareCandidates.update(cs => cs.map((x, j) => j === i ? { ...x, varyingId: (e.currentTarget as HTMLSelectElement).value } : x))}>
          {#each varyingOptions as o}<option value={o.id}>{o.name}</option>{/each}
        </select>
        <span class="quant">{c.quant.weights} · kv {c.quant.kv} · act {c.quant.activations}</span>
        <button type="button" class="rm" onclick={() => removeCandidate(i)} disabled={$compareCandidates.length <= 1}>✕</button>
      </div>
    {/each}
    <button type="button" class="add" onclick={addCandidate}>+ add candidate</button>
  </div>
</section>

<CompareTable />

<style>
  .controls { display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 1.25rem; }
  .row, .workload, .cand { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
  .lbl { font-weight: 600; font-size: 0.85rem; color: #444; min-width: 7rem; }
  .workload input { width: 6rem; }
  .candidates { display: flex; flex-direction: column; gap: 0.4rem; }
  .quant { font-size: 0.8rem; color: #777; }
  .rm { border: none; background: none; color: #999; cursor: pointer; }
  .rm:disabled { opacity: 0.3; cursor: default; }
  .add { align-self: flex-start; font: inherit; font-size: 0.85rem; padding: 0.3rem 0.7rem; border: 1px solid #c8c8c8; border-radius: 0.3rem; background: #fff; cursor: pointer; }
  select, input { font: inherit; font-size: 0.85rem; padding: 0.25rem 0.4rem; }
</style>
