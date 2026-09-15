<script lang="ts">
  import { compareResults } from './stores'
  import type { CompareRow } from './compareModel'

  type MetricKey = 'ttftMs' | 'tpotMs' | 'throughputTokS' | 'kvTotalGB'
  const COLUMNS: { key: MetricKey; label: string; lowerIsBetter: boolean; digits: number }[] = [
    { key: 'ttftMs',        label: 'TTFT (ms)',      lowerIsBetter: true,  digits: 1 },
    { key: 'tpotMs',        label: 'TPOT (ms)',      lowerIsBetter: true,  digits: 2 },
    { key: 'throughputTokS', label: 'Tput (tok/s)',  lowerIsBetter: false, digits: 0 },
    { key: 'kvTotalGB',     label: 'KV (GB)',        lowerIsBetter: true,  digits: 2 },
  ]

  let sortKey: MetricKey = $state('throughputTokS')
  let sortAsc = $state(false)

  function setSort(k: MetricKey) {
    if (sortKey === k) { sortAsc = !sortAsc } else { sortKey = k; sortAsc = COLUMNS.find(c => c.key === k)!.lowerIsBetter }
  }

  // Best value per metric column, across ok rows only. Used to highlight winners.
  const best = $derived.by(() => {
    const out = {} as Record<MetricKey, number | undefined>
    for (const col of COLUMNS) {
      const vals = $compareResults.filter((r): r is Extract<CompareRow, { ok: true }> => r.ok).map(r => r.metrics[col.key])
      out[col.key] = vals.length ? (col.lowerIsBetter ? Math.min(...vals) : Math.max(...vals)) : undefined
    }
    return out
  })

  const sorted = $derived.by(() => {
    const rows = [...$compareResults]
    rows.sort((a, b) => {
      // Error rows sink to the bottom regardless of direction.
      if (!a.ok && !b.ok) return 0
      if (!a.ok) return 1
      if (!b.ok) return -1
      const d = a.metrics[sortKey] - b.metrics[sortKey]
      return sortAsc ? d : -d
    })
    return rows
  })
</script>

<table class="cmp">
  <thead>
    <tr>
      <th class="name">Candidate</th>
      {#each COLUMNS as col}
        <th class="num" class:sorted={sortKey === col.key}>
          <button type="button" onclick={() => setSort(col.key)}>
            {col.label}{sortKey === col.key ? (sortAsc ? ' ▲' : ' ▼') : ''}
          </button>
        </th>
      {/each}
      <th class="regime">Decode regime</th>
    </tr>
  </thead>
  <tbody>
    {#each sorted as row}
      <tr class:err={!row.ok}>
        <td class="name">{row.name}</td>
        {#if row.ok}
          {#each COLUMNS as col}
            <td class="num" class:win={best[col.key] === row.metrics[col.key]} class:oom={col.key === 'kvTotalGB' && !row.metrics.fits}>
              {row.metrics[col.key].toFixed(col.digits)}
            </td>
          {/each}
          <td class="regime">{row.metrics.regime}{row.metrics.fits ? '' : ' · OOM'}</td>
        {:else}
          <td class="num err-msg" colspan={COLUMNS.length + 1}>⚠ {row.error}</td>
        {/if}
      </tr>
    {/each}
  </tbody>
</table>

<style>
  .cmp { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #e2e2e2; text-align: left; }
  th.num, td.num, th.regime { text-align: right; }
  td.name, th.name { font-weight: 600; }
  th button { font: inherit; font-weight: 600; background: none; border: none; cursor: pointer; color: #333; padding: 0; }
  th.sorted button { color: #111; }
  td.win { background: #e5f5e5; font-weight: 600; }
  td.oom { color: #8a1f1f; }
  tr.err td.err-msg { color: #8a1f1f; text-align: left; }
</style>
