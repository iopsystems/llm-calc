# Multi-candidate comparison — design

**Date:** 2026-07-03
**Roadmap:** item #5 (multi-candidate comparison / model-comparison view)
**Status:** approved, pre-implementation

## Problem

The calculator answers "how does *this* (accelerator, model, quant, workload) tuple perform." It can't answer the comparative question users actually ask: *which* of these models is fastest on my SKU, or *which* accelerator is best for my model. Today that means manually flipping one selector and eyeballing numbers across reloads.

Ship a **Compare** tab that runs the engine for N candidates against a shared workload and lays the computed results out side-by-side.

## Decisions (locked)

- **Compares computed calc results**, not static catalog specs. TTFT / TPOT / throughput / KV footprint / roofline regime per candidate — the calculator's real value, not a spec-sheet reprint.
- **Both axes, one pivot fixed.** Either N models × 1 fixed SKU, or N SKUs × 1 fixed model. A pivot-axis toggle selects which dimension is locked.
- **Single-point metrics table for v1.** One shared workload → one row per candidate. Under-load curves (overlaid N-sweep) are an explicit follow-up, not v1.
- **Per-candidate quant.** Each candidate carries its own `Quantization` — models compared "as actually deployed" (e.g. one at fp8, another at fp16). Workload and concurrency are shared across all candidates; quant is not.
- **Candidate model = explicit pivot + varying candidates** (Approach A). State is `{ pivot, candidates[], workload }`; each candidate carries only the field that varies plus its quant. Rejected the homogeneous-full-tuple alternative (Approach B) as more freedom than the single-pivot constraint needs — bulkier encoding, ambiguous "which field is fixed."

## State / stores (`src/ui/stores.ts`)

New writables, independent of the existing single-selection calc/sim stores. Do **not** overload `modelId` into a list — the `input` / `nMaxCalc` / all sim derivations assume scalar selection.

```ts
type ComparePivot = { kind: 'sku' | 'model'; id: string }   // sku id = accelerator id OR system id
type CompareCandidate = { varyingId: string; quant: Quantization }

comparePivot:      writable<ComparePivot>
compareCandidates: writable<CompareCandidate[]>
compareWorkload:   writable<Workload>   // shared prompt/output/concurrency (single fixed concurrency)
```

- `pivot.kind === 'sku'` → each `varyingId` is a **model** id (N models × 1 SKU).
- `pivot.kind === 'model'` → each `varyingId` is an **accelerator/system** id (N SKUs × 1 model).
- The pivot `id` uses the same accel-vs-system ambiguity resolution as the existing `input` derived store (a non-empty system id resolves the accelerator from the `MultiAcceleratorSystem`).

Derived `compareResults: Readable<CompareRow[]>` — see engine reuse.

**Pivot-axis toggle behavior:** switching `pivot.kind` **hard-clears** `candidates` (a model-list can't remap to a SKU-list). `compareWorkload` is preserved; one default candidate is seeded on the new axis so the table is never empty.

## Engine reuse (no engine changes)

Pure reuse of the existing `calc()` path — the engine is untouched.

For each candidate, assemble a `CalcInput` from `{ pivot, candidate.varyingId, candidate.quant, compareWorkload }`, reusing the accelerator-from-system resolution the `input` derived store already does. Surface per row: TTFT, TPOT, throughput (tok/s), KV footprint, memory-fit flag, roofline regime (compute / memory / comms-bound).

**Error isolation** — the one bit of new engine-adjacent logic, unit-tested directly:

```ts
type CompareRow =
  | { candidate: CompareCandidate; result: CalcResult }
  | { candidate: CompareCandidate; error: string }
```

Each candidate runs in its own `try { calc(...) } catch`. A quant a given model doesn't support (the per-candidate-quant caveat) becomes an error row, never kills the table.

## URL codec (`src/ui/share.ts`, `src/ui/route.ts`)

New `'compare'` route branch, query-payload style (like calc/sim, unlike info's path-style), slug-based, list via repeated `c=` keys:

```
#compare?piv=sku:h100-sxm&pt=1024&ot=512&cc=1&c=llama-3.3-70b~w8.kv16.a16&c=glm-5~w8.kv8.a16
```

- `piv=<kind>:<id>` — the fixed pivot.
- `pt` / `ot` / `cc` — shared workload prompt / output / concurrency.
- repeated `c=<varyingId>~<w>.<kv>.<a>` — one per candidate, reusing the existing `w.kv.a` quant sub-codec form (mirrors the parallelism dot-join codec).
- `encodeCompare` / `decodeCompare` parallel to `encodeState` / `decodeState`. Reuse the existing **validate-and-drop-unknown** id pattern: stale candidate ids (catalog reordered/removed across versions) are silently dropped, consistent with today's behavior. A dropped candidate never errors the decode.
- Wire `'compare'` into `route.ts` (`Route` union + `parseRoute` / `serializeRoute`), `tabPayloadFromHash` (share.ts:371), and the `startUrlSync` tab gate (share.ts:405).

## Components + layout (`src/ui/`)

New tab at the insertion point `TabBar.svelte:4` already reserves.

- **`CompareTab.svelte`** — top controls: pivot-axis toggle (SKU-fixed ↔ model-fixed), the fixed-pivot selector, shared workload inputs (prompt / output / concurrency), and an "add candidate" chip-adder. Holds no state the stores don't.
- **Results table** — **candidates as rows, metrics as columns** (sort by any metric column is natural; scales to many SKUs without column cramping). Sortable; best value per metric highlighted; infeasible (memory-overflow) and error rows clearly marked. A new dense presentational component built from `compareResults` — it does **not** reuse `ModelSpecSheet` / `SkuSpecSheet` (those are static-catalog cards, wrong for computed results).
- **App wiring:** `Route` union + `App.svelte:52` `{:else if $route.tab === 'compare'}` branch.

**Seeding:** the Compare tab seeds its pivot + first candidate from the current calc selection, so a "compare this against…" flow is one click from the Calculator tab.

## Testing (TDD)

Write the failing test first for each unit; engine untouched so no `test/engine/` changes.

- `test/ui/compareCodec.test.ts` — encode/decode round-trip; repeated-key list; stale-id drop; quant sub-codec; both pivot kinds.
- `test/ui/compareResults.test.ts` — derived mapping: N candidates → N rows; error isolation (unsupported quant → error row, siblings unaffected); pivot resolution for both axes including system-backed SKUs.
- `test/ui/compareStores.test.ts` — pivot-axis toggle hard-clears candidates, preserves workload, seeds one default.

## Scope guards (YAGNI)

Explicitly **out** of v1, each a named follow-up:

- Under-load / overlaid load curves (reuse `computeNMax` / `loadCurve`) — the roadmap "curves later" item.
- Per-candidate workload (workload stays shared).
- Cost / $-per-token metrics.
- Any relaxation of the single-pivot constraint (no free-form full-tuple candidates).

## Files touched

- `src/ui/stores.ts` — new compare stores + `compareResults` derived.
- `src/ui/share.ts` — `encodeCompare` / `decodeCompare` + tab gates.
- `src/ui/route.ts` — `'compare'` route branch.
- `src/ui/CompareTab.svelte` — new.
- results-table component — new.
- `src/ui/TabBar.svelte`, `src/ui/App.svelte` — tab wiring.
- `test/ui/compare*.test.ts` — new.
