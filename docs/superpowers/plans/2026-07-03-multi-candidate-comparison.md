# Multi-candidate Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **Compare** tab that runs the engine for N candidates against a shared workload and lays the computed results out side-by-side — either N models × 1 fixed SKU, or N SKUs × 1 fixed model.

**Architecture:** A pure, Svelte-free domain module (`compareModel.ts`) resolves a `(pivot, candidate, workload)` tuple into a `CalcInput`, runs the existing `calculate()` per candidate with per-row error isolation, and extracts display metrics. A pure codec module (`compareShare.ts`) round-trips the compare state through the URL hash. Three new Svelte stores hold the state; a derived store maps them to result rows. Two thin Svelte components render controls + a sortable table. The engine is untouched.

**Tech Stack:** TypeScript, Svelte 5, Vitest (node env, no DOM), Vite.

## Global Constraints

- **TDD, engine untouched.** Every pure module gets a failing test first. No `src/engine/` or `test/engine/` changes.
- **Svelte components are not unit-tested** (repo convention: `test/ui/` tests pure logic only, node env, no DOM). Components are verified via `npm run check` (svelte-check) + `npm run dev`. Keep all testable logic in the pure modules.
- **Data discipline:** no new data entries; this feature only consumes `ACCELERATORS` / `MODELS` / `SYSTEMS`.
- **Never overload `modelId` / calc stores into a list** — the `input`, `nMaxCalc`, and all sim derivations assume scalar selection. Compare state is fully separate.
- Run `npm run check` and `npm test` before considering any task done.
- Commit after every task. **Do not add a `Co-Authored-By` footer** (repo owner's standing rule).

## File Structure

- Create `src/ui/compareModel.ts` — pure domain types + input resolution + row computation. No Svelte imports.
- Create `src/ui/compareShare.ts` — pure URL codec + startup/sync wiring for compare state.
- Create `src/ui/CompareTable.svelte` — presentational sortable results table.
- Create `src/ui/CompareTab.svelte` — controls (pivot toggle, pivot selector, workload inputs, add/remove candidate) + seeding.
- Modify `src/ui/stores.ts` — add compare writables + `compareResults` derived.
- Modify `src/ui/route.ts` — add `'compare'` to the `Route` union + parse/serialize.
- Modify `src/ui/TabBar.svelte` — add the Compare tab entry.
- Modify `src/ui/App.svelte` — add the `{:else if $route.tab === 'compare'}` branch.
- Modify `src/main.ts` — wire compare URL read + sync at startup.
- Create `test/ui/compareModel.test.ts`, `test/ui/compareShare.test.ts`, `test/ui/compareStores.test.ts`.

**Note vs spec:** the spec listed the codec as living in `share.ts`. `share.ts` is already ~450 lines; putting the compare codec in its own focused `compareShare.ts` keeps both files single-responsibility. `share.ts`/`route.ts` are still touched only for the tab-gate wiring (route union + startup).

**Known v1 limitation (documented, not a bug):** a bare accelerator SKU resolves to its **first variant** (`variants[0]`). Comparing two form-factors of the *same* accelerator (e.g. H100 SXM vs H100 PCIe) is out of scope for v1 — comparing *different* accelerators/systems is the primary use. Named follow-up.

---

### Task 1: Compare domain types + input resolution

**Files:**
- Create: `src/ui/compareModel.ts`
- Test: `test/ui/compareModel.test.ts`

**Interfaces:**
- Consumes: `ACCELERATORS`, `MODELS` from `../data`; `SYSTEMS` from `../data/systems`; `defaultParallelism` from `../engine/parallelism`; types from `../engine/types`.
- Produces:
  - `type ComparePivotKind = 'sku' | 'model'`
  - `interface ComparePivot { kind: ComparePivotKind; id: string }`
  - `interface CompareCandidate { varyingId: string; quant: Quantization }`
  - `function resolveCompareInput(pivot: ComparePivot, candidate: CompareCandidate, workload: Workload): CalcInput | null`

- [ ] **Step 1: Write the failing test**

Create `test/ui/compareModel.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { resolveCompareInput, type ComparePivot } from '../../src/ui/compareModel'
import { ACCELERATORS, MODELS } from '../../src/data'
import { SYSTEMS } from '../../src/data/systems'

const fp16 = { weights: 'fp16', kv: 'fp16', activations: 'fp16' } as const
const wl = { promptTokens: 1024, outputTokens: 256, concurrency: 1 }

describe('resolveCompareInput', () => {
  it('resolves N-models-x-1-SKU: pivot=sku(accelerator), candidate=model', () => {
    const accel = ACCELERATORS[0]
    const model = MODELS[0]
    const pivot: ComparePivot = { kind: 'sku', id: accel.id }
    const input = resolveCompareInput(pivot, { varyingId: model.id, quant: fp16 }, wl)
    expect(input).not.toBeNull()
    expect(input!.accelerator.id).toBe(accel.id)
    expect(input!.acceleratorVariantId).toBe(accel.variants[0].id)
    expect(input!.model.id).toBe(model.id)
    expect(input!.multiDevice).toBeUndefined()
  })

  it('resolves N-SKUs-x-1-model: pivot=model, candidate=system → multiDevice set', () => {
    const system = SYSTEMS[0]
    const model = MODELS[0]
    const pivot: ComparePivot = { kind: 'model', id: model.id }
    const input = resolveCompareInput(pivot, { varyingId: system.id, quant: fp16 }, wl)
    expect(input).not.toBeNull()
    expect(input!.model.id).toBe(model.id)
    expect(input!.multiDevice?.system.id).toBe(system.id)
    expect(input!.acceleratorVariantId).toBe(system.accelerator.variantId)
  })

  it('returns null for an unknown varying id', () => {
    const pivot: ComparePivot = { kind: 'sku', id: ACCELERATORS[0].id }
    expect(resolveCompareInput(pivot, { varyingId: 'nope-not-a-model', quant: fp16 }, wl)).toBeNull()
  })

  it('returns null for an unknown pivot id', () => {
    const pivot: ComparePivot = { kind: 'sku', id: 'nope-not-a-sku' }
    expect(resolveCompareInput(pivot, { varyingId: MODELS[0].id, quant: fp16 }, wl)).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- compareModel`
Expected: FAIL — `compareModel.ts` doesn't exist / `resolveCompareInput is not a function`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ui/compareModel.ts`:

```ts
// Pure domain logic for the Compare tab. No Svelte imports — importable from
// tests and the store layer alike. Resolves a (pivot, candidate) tuple into a
// CalcInput, mirroring the accelerator-vs-system resolution the `input` derived
// store does, then runs the shared engine per candidate with error isolation.
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import { defaultParallelism } from '../engine/parallelism'
import type { CalcInput, MultiDeviceConfig, Quantization, Workload } from '../engine/types'

export type ComparePivotKind = 'sku' | 'model'
export interface ComparePivot { kind: ComparePivotKind; id: string }
export interface CompareCandidate { varyingId: string; quant: Quantization }

// Build a CalcInput from a concrete (modelId, skuId) pair. skuId may name an
// accelerator OR a system; a system id wins and wires multiDevice with default
// parallelism (same precedence as share.ts / the `input` store). A bare
// accelerator resolves to its first variant (v1 limitation — no per-variant
// SKU compare yet). Returns null if any id is unknown.
function buildInput(modelId: string, skuId: string, quant: Quantization, workload: Workload): CalcInput | null {
  const model = MODELS.find(m => m.id === modelId)
  if (!model) return null

  const system = SYSTEMS.find(s => s.id === skuId)
  if (system) {
    const accelerator = ACCELERATORS.find(a => a.id === system.accelerator.id)
    if (!accelerator) return null
    const pc = defaultParallelism(system, model)
    const multiDevice: MultiDeviceConfig = {
      system, parallelism: pc.parallelism, parallelismDegrees: pc.parallelismDegrees,
    }
    return { accelerator, acceleratorVariantId: system.accelerator.variantId, model, quant, workload, multiDevice }
  }

  const accelerator = ACCELERATORS.find(a => a.id === skuId)
  if (!accelerator) return null
  return { accelerator, acceleratorVariantId: accelerator.variants[0].id, model, quant, workload }
}

export function resolveCompareInput(
  pivot: ComparePivot, candidate: CompareCandidate, workload: Workload,
): CalcInput | null {
  return pivot.kind === 'sku'
    ? buildInput(candidate.varyingId, pivot.id, candidate.quant, workload)   // pivot = sku, varying = model
    : buildInput(pivot.id, candidate.varyingId, candidate.quant, workload)   // pivot = model, varying = sku
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- compareModel`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ui/compareModel.ts test/ui/compareModel.test.ts
git commit -m "feat(compare): resolve pivot+candidate into a CalcInput"
```

---

### Task 2: Row computation — metrics extraction + error isolation + names

**Files:**
- Modify: `src/ui/compareModel.ts`
- Test: `test/ui/compareModel.test.ts`

**Interfaces:**
- Consumes: `calculate` from `../engine`; `resolveCompareInput` (Task 1); `CalcResult` / `PerfTier` from `../engine/types`.
- Produces:
  - `interface CompareMetrics { ttftMs; tpotMs; throughputTokS; kvTotalGB; fits; regime }`
  - `type CompareRow` (discriminated `ok: true`/`false`, both carrying `name` + `candidate`)
  - `function pickPerfTier(result: CalcResult): PerfTier | null`
  - `function resolveVaryingName(pivot: ComparePivot, varyingId: string): string`
  - `function computeCompareRow(pivot: ComparePivot, candidate: CompareCandidate, workload: Workload): CompareRow`

- [ ] **Step 1: Write the failing test**

Append to `test/ui/compareModel.test.ts`:

```ts
import { computeCompareRow, resolveVaryingName } from '../../src/ui/compareModel'

describe('computeCompareRow', () => {
  it('produces an ok row with finite metrics for a valid candidate', () => {
    const row = computeCompareRow(
      { kind: 'sku', id: ACCELERATORS[0].id },
      { varyingId: MODELS[0].id, quant: fp16 }, wl,
    )
    expect(row.ok).toBe(true)
    if (row.ok) {
      expect(Number.isFinite(row.metrics.ttftMs)).toBe(true)
      expect(Number.isFinite(row.metrics.tpotMs)).toBe(true)
      expect(row.metrics.throughputTokS).toBeGreaterThan(0)
      expect(['compute', 'memory', 'comms']).toContain(row.metrics.regime)
      expect(row.name).toBe(MODELS[0].name)
    }
  })

  it('isolates errors: an unresolvable candidate becomes an error row, not a throw', () => {
    const row = computeCompareRow(
      { kind: 'sku', id: ACCELERATORS[0].id },
      { varyingId: 'nope-not-a-model', quant: fp16 }, wl,
    )
    expect(row.ok).toBe(false)
    if (!row.ok) expect(row.error).toMatch(/unknown/i)
  })

  it('isolates engine errors: an unsupported quant becomes an error row', () => {
    // fp4 activations: no datacenter accelerator lists fp4 TFLOPS → calculate throws.
    const row = computeCompareRow(
      { kind: 'sku', id: ACCELERATORS[0].id },
      { varyingId: MODELS[0].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp4' } }, wl,
    )
    expect(row.ok).toBe(false)
  })

  it('resolveVaryingName resolves the opposite dimension', () => {
    expect(resolveVaryingName({ kind: 'sku', id: ACCELERATORS[0].id }, MODELS[0].id)).toBe(MODELS[0].name)
    expect(resolveVaryingName({ kind: 'model', id: MODELS[0].id }, 'ghost-id')).toBe('ghost-id')
  })
})
```

> If `MODELS[0]` happens to natively support fp4 activations and the third test's `calculate` does not throw, swap the quant to another dtype absent from the SKU's `tflops` map (inspect `ACCELERATORS[0].variants[0].operatingPoints[0].tflops`). The test's intent is "an engine throw becomes an error row."

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- compareModel`
Expected: FAIL — `computeCompareRow is not a function`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/ui/compareModel.ts`:

```ts
import { calculate } from '../engine'
import type { CalcResult, PerfTier } from '../engine/types'

export interface CompareMetrics {
  ttftMs: number
  tpotMs: number
  throughputTokS: number
  kvTotalGB: number
  fits: boolean
  regime: 'compute' | 'memory' | 'comms'
}

export type CompareRow =
  | { ok: true;  name: string; candidate: CompareCandidate; metrics: CompareMetrics }
  | { ok: false; name: string; candidate: CompareCandidate; error: string }

// The comparison reports the peak (theoretical) operating point — the same tier
// the roofline panel treats as "Theoretical". Fall back to the first available
// point for any accelerator that has no 'peak' id.
export function pickPerfTier(result: CalcResult): PerfTier | null {
  return result.perf['peak'] ?? Object.values(result.perf)[0] ?? null
}

// Display name of the *varying* dimension (the opposite of the pivot kind).
export function resolveVaryingName(pivot: ComparePivot, varyingId: string): string {
  if (pivot.kind === 'sku') {
    return MODELS.find(m => m.id === varyingId)?.name ?? varyingId
  }
  return SYSTEMS.find(s => s.id === varyingId)?.name
    ?? ACCELERATORS.find(a => a.id === varyingId)?.name
    ?? varyingId
}

export function computeCompareRow(
  pivot: ComparePivot, candidate: CompareCandidate, workload: Workload,
): CompareRow {
  const name = resolveVaryingName(pivot, candidate.varyingId)
  const input = resolveCompareInput(pivot, candidate, workload)
  if (!input) return { ok: false, name, candidate, error: 'unknown model or accelerator' }
  try {
    const result = calculate(input)
    const perf = pickPerfTier(result)
    if (!perf) return { ok: false, name, candidate, error: 'no operating point' }
    return {
      ok: true, name, candidate,
      metrics: {
        ttftMs: perf.ttftS * 1000,
        tpotMs: perf.decode.timePerTokenS * 1000,
        throughputTokS: perf.decode.aggregateTokensPerS,
        kvTotalGB: result.memory.kvCacheTotal / 1e9,
        fits: result.memory.fits,
        regime: perf.decode.regime,
      },
    }
  } catch (err) {
    return { ok: false, name, candidate, error: (err as Error).message }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- compareModel`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/ui/compareModel.ts test/ui/compareModel.test.ts
git commit -m "feat(compare): compute result rows with metric extraction + error isolation"
```

---

### Task 3: Seeding + pivot-axis reshape helpers

**Files:**
- Modify: `src/ui/compareModel.ts`
- Test: `test/ui/compareModel.test.ts`

**Interfaces:**
- Consumes: `ACCELERATORS`, `MODELS` (already imported).
- Produces:
  - `function defaultPivotId(kind: ComparePivotKind): string` — first entry of the pivot's own catalog.
  - `function firstVaryingId(kind: ComparePivotKind): string` — first entry of the varying catalog.
  - `function seededQuantFor(modelId: string): Quantization` — model's native weights/activations, kv=fp16. Falls back to fp16/fp16/fp16 for an unknown model id.

- [ ] **Step 1: Write the failing test**

Append to `test/ui/compareModel.test.ts`:

```ts
import { defaultPivotId, firstVaryingId, seededQuantFor } from '../../src/ui/compareModel'

describe('seeding helpers', () => {
  it('defaultPivotId picks the pivot catalog; firstVaryingId picks the opposite', () => {
    expect(defaultPivotId('sku')).toBe(ACCELERATORS[0].id)
    expect(firstVaryingId('sku')).toBe(MODELS[0].id)
    expect(defaultPivotId('model')).toBe(MODELS[0].id)
    expect(firstVaryingId('model')).toBe(ACCELERATORS[0].id)
  })

  it('seededQuantFor uses the model native dtype for weights, fp16 kv', () => {
    const q = seededQuantFor(MODELS[0].id)
    expect(q.weights).toBe(MODELS[0].nativeDtype)
    expect(q.kv).toBe('fp16')
  })

  it('seededQuantFor falls back to fp16 for an unknown model', () => {
    expect(seededQuantFor('ghost')).toEqual({ weights: 'fp16', kv: 'fp16', activations: 'fp16' })
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- compareModel`
Expected: FAIL — `defaultPivotId is not a function`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/ui/compareModel.ts`:

```ts
import type { Dtype } from '../engine/types'

export function defaultPivotId(kind: ComparePivotKind): string {
  return kind === 'sku' ? ACCELERATORS[0].id : MODELS[0].id
}

export function firstVaryingId(kind: ComparePivotKind): string {
  return kind === 'sku' ? MODELS[0].id : ACCELERATORS[0].id
}

// 4-bit ship formats (int4/fp4) run their matmuls in bf16 after in-kernel
// dequant — no datacenter chip exposes 4-bit tensor cores — so seed activations
// to bf16 there, mirroring stores.defaultActivationsFor. Kept as a local copy to
// keep this module Svelte/stores-free (stores imports this module).
function activationsFor(native: Dtype): Dtype {
  return native === 'int4' || native === 'fp4' ? 'bf16' : native
}

export function seededQuantFor(modelId: string): Quantization {
  const m = MODELS.find(x => x.id === modelId)
  if (!m) return { weights: 'fp16', kv: 'fp16', activations: 'fp16' }
  return { weights: m.nativeDtype, kv: 'fp16', activations: activationsFor(m.nativeDtype) }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- compareModel`
Expected: PASS (11 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/ui/compareModel.ts test/ui/compareModel.test.ts
git commit -m "feat(compare): pivot-default + varying-seed helpers"
```

---

### Task 4: URL codec (`compareShare.ts`)

**Files:**
- Create: `src/ui/compareShare.ts`
- Test: `test/ui/compareShare.test.ts`

**Interfaces:**
- Consumes: `ACCELERATORS`, `MODELS`, `SYSTEMS`; types + `ComparePivot`, `CompareCandidate`, `ComparePivotKind` from `./compareModel`; `Dtype`, `Quantization`, `Workload` from `../engine/types`.
- Produces:
  - `interface CompareState { pivot: ComparePivot; candidates: CompareCandidate[]; workload: Workload }`
  - `function encodeCompare(state: CompareState): string`
  - `function decodeCompare(payload: string): CompareState | null`

- [ ] **Step 1: Write the failing test**

Create `test/ui/compareShare.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { encodeCompare, decodeCompare, type CompareState } from '../../src/ui/compareShare'
import { ACCELERATORS, MODELS } from '../../src/data'

const q = (w: string, kv: string, a: string) => ({ weights: w, kv, activations: a }) as const

function sample(): CompareState {
  return {
    pivot: { kind: 'sku', id: ACCELERATORS[0].id },
    workload: { promptTokens: 1024, outputTokens: 256, concurrency: 4 },
    candidates: [
      { varyingId: MODELS[0].id, quant: q('fp16', 'fp16', 'fp16') },
      { varyingId: MODELS[1].id, quant: q('fp8', 'fp8', 'bf16') },
    ],
  }
}

describe('compare codec', () => {
  it('round-trips a full state', () => {
    const s = sample()
    const decoded = decodeCompare(encodeCompare(s))
    expect(decoded).toEqual(s)
  })

  it('encodes candidates as repeated c= keys', () => {
    const enc = encodeCompare(sample())
    expect(enc.match(/(^|&)c=/g)?.length).toBe(2)
    expect(enc).toContain('piv=sku%3A' + ACCELERATORS[0].id)  // ':' url-encoded
  })

  it('drops unknown candidate ids but keeps the good ones', () => {
    const s = sample()
    const enc = encodeCompare(s) + '&c=ghost-model~fp16.fp16.fp16'
    const decoded = decodeCompare(enc)
    expect(decoded!.candidates).toHaveLength(2)
  })

  it('drops candidates with an unknown dtype', () => {
    const enc = encodeCompare({ ...sample(), candidates: [] }) + '&c=' + MODELS[0].id + '~fp16.fp16.notadtype'
    expect(decodeCompare(enc)!.candidates).toHaveLength(0)
  })

  it('returns null when the pivot id is invalid', () => {
    expect(decodeCompare('piv=sku%3Aghost&pt=1&ot=1&cc=1')).toBeNull()
  })

  it('returns null when piv is missing entirely', () => {
    expect(decodeCompare('pt=1&ot=1&cc=1')).toBeNull()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- compareShare`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

Create `src/ui/compareShare.ts`:

```ts
// URL-hash codec for the Compare tab. Kept separate from share.ts (which owns
// the calc/sim payload) so each file stays single-responsibility. Payload form:
//   piv=<kind>:<id>&pt=..&ot=..&cc=..&c=<varyingId>~<w>.<kv>.<a>&c=...
// Slug-based (order-independent, survives catalog reordering); unknown ids and
// dtypes are silently dropped, an invalid pivot yields null (nothing to render).
import { ACCELERATORS, MODELS } from '../data'
import { SYSTEMS } from '../data/systems'
import type { Dtype, Quantization, Workload } from '../engine/types'
import type { ComparePivot, ComparePivotKind, CompareCandidate } from './compareModel'

export interface CompareState {
  pivot: ComparePivot
  candidates: CompareCandidate[]
  workload: Workload
}

const DTYPES: readonly Dtype[] = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int8', 'int4']
const isDtype = (s: string): s is Dtype => (DTYPES as readonly string[]).includes(s)

const skuExists = (id: string) => !!ACCELERATORS.find(a => a.id === id) || !!SYSTEMS.find(s => s.id === id)
const modelExists = (id: string) => !!MODELS.find(m => m.id === id)

// A pivot of kind 'sku' varies models (and vice-versa): validate against the
// opposite catalog.
const varyingExists = (kind: ComparePivotKind, id: string) => kind === 'sku' ? modelExists(id) : skuExists(id)
const pivotExists   = (kind: ComparePivotKind, id: string) => kind === 'sku' ? skuExists(id) : modelExists(id)

function parsePos(raw: string | null, fallback: number): number {
  if (raw === null) return fallback
  const n = parseInt(raw, 10)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

function parseQuant(raw: string | undefined): Quantization | null {
  if (!raw) return null
  const [w, kv, a] = raw.split('.')
  if (w && kv && a && isDtype(w) && isDtype(kv) && isDtype(a)) return { weights: w, kv, activations: a }
  return null
}

export function encodeCompare(state: CompareState): string {
  const p = new URLSearchParams()
  p.set('piv', `${state.pivot.kind}:${state.pivot.id}`)
  p.set('pt', String(state.workload.promptTokens))
  p.set('ot', String(state.workload.outputTokens))
  p.set('cc', String(state.workload.concurrency))
  for (const c of state.candidates) {
    p.append('c', `${c.varyingId}~${c.quant.weights}.${c.quant.kv}.${c.quant.activations}`)
  }
  return p.toString()
}

export function decodeCompare(payload: string): CompareState | null {
  const params = new URLSearchParams(payload)
  const pivRaw = params.get('piv')
  if (!pivRaw) return null
  const sep = pivRaw.indexOf(':')
  if (sep < 0) return null
  const kind = pivRaw.slice(0, sep)
  const id = pivRaw.slice(sep + 1)
  if (kind !== 'sku' && kind !== 'model') return null
  if (!pivotExists(kind, id)) return null

  const workload: Workload = {
    promptTokens: parsePos(params.get('pt'), 2048),
    outputTokens: parsePos(params.get('ot'), 512),
    concurrency:  parsePos(params.get('cc'), 1),
  }

  const candidates: CompareCandidate[] = []
  for (const raw of params.getAll('c')) {
    const tilde = raw.indexOf('~')
    const varyingId = tilde < 0 ? raw : raw.slice(0, tilde)
    if (!varyingExists(kind, varyingId)) continue
    const quant = parseQuant(tilde < 0 ? undefined : raw.slice(tilde + 1))
    if (!quant) continue
    candidates.push({ varyingId, quant })
  }

  return { pivot: { kind, id }, candidates, workload }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- compareShare`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ui/compareShare.ts test/ui/compareShare.test.ts
git commit -m "feat(compare): URL-hash codec for compare state"
```

---

### Task 5: Compare stores + `compareResults` derived

**Files:**
- Modify: `src/ui/stores.ts`
- Test: `test/ui/compareStores.test.ts`

**Interfaces:**
- Consumes: `computeCompareRow`, `defaultPivotId`, `firstVaryingId`, `seededQuantFor`, types from `./compareModel`.
- Produces (exported from `stores.ts`):
  - `comparePivot: writable<ComparePivot>`
  - `compareCandidates: writable<CompareCandidate[]>`
  - `compareWorkload: writable<Workload>`
  - `compareResults: Readable<CompareRow[]>`
  - `function setComparePivotKind(kind: ComparePivotKind): void` — flips the axis: hard-clears candidates, preserves `compareWorkload`, reseeds pivot id + one default candidate.
  - `function seedCompareFromCalc(): void` — seeds the SKU pivot from the current calc accelerator/system selection and the first candidate from the current model (the "compare this against…" one-click entry).

- [ ] **Step 1: Write the failing test**

Create `test/ui/compareStores.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import {
  comparePivot, compareCandidates, compareWorkload, compareResults, setComparePivotKind,
  seedCompareFromCalc, acceleratorId, systemId, modelId,
} from '../../src/ui/stores'
import { defaultPivotId, firstVaryingId } from '../../src/ui/compareModel'
import { ACCELERATORS, MODELS } from '../../src/data'

describe('compare stores', () => {
  beforeEach(() => {
    comparePivot.set({ kind: 'sku', id: ACCELERATORS[0].id })
    compareCandidates.set([{ varyingId: MODELS[0].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } }])
    compareWorkload.set({ promptTokens: 1024, outputTokens: 256, concurrency: 1 })
  })

  it('compareResults maps each candidate to a row', () => {
    compareCandidates.update(cs => [...cs, { varyingId: MODELS[1].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } }])
    const rows = get(compareResults)
    expect(rows).toHaveLength(2)
    expect(rows[0].name).toBe(MODELS[0].name)
  })

  it('a bad candidate errors its own row without killing siblings', () => {
    compareCandidates.set([
      { varyingId: MODELS[0].id, quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } },
      { varyingId: 'ghost',      quant: { weights: 'fp16', kv: 'fp16', activations: 'fp16' } },
    ])
    const rows = get(compareResults)
    expect(rows[0].ok).toBe(true)
    expect(rows[1].ok).toBe(false)
  })

  it('setComparePivotKind hard-clears candidates, reseeds, preserves workload', () => {
    compareWorkload.set({ promptTokens: 999, outputTokens: 111, concurrency: 3 })
    setComparePivotKind('model')
    expect(get(comparePivot)).toEqual({ kind: 'model', id: defaultPivotId('model') })
    const cs = get(compareCandidates)
    expect(cs).toHaveLength(1)
    expect(cs[0].varyingId).toBe(firstVaryingId('model'))
    expect(get(compareWorkload)).toEqual({ promptTokens: 999, outputTokens: 111, concurrency: 3 })
  })

  it('seedCompareFromCalc seeds the sku pivot + first candidate from calc stores', () => {
    acceleratorId.set(ACCELERATORS[1].id)
    systemId.set('')
    modelId.set(MODELS[1].id)
    seedCompareFromCalc()
    expect(get(comparePivot)).toEqual({ kind: 'sku', id: ACCELERATORS[1].id })
    const cs = get(compareCandidates)
    expect(cs).toHaveLength(1)
    expect(cs[0].varyingId).toBe(MODELS[1].id)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- compareStores`
Expected: FAIL — exports not found.

- [ ] **Step 3: Write minimal implementation**

Add `get` to the existing `svelte/store` import at the top of `src/ui/stores.ts`:

```ts
import { writable, derived, get, type Readable } from 'svelte/store'
```

Add to `src/ui/stores.ts` imports (the compare types + helpers):

```ts
import {
  computeCompareRow, defaultPivotId, firstVaryingId, seededQuantFor,
  type ComparePivot, type ComparePivotKind, type CompareCandidate, type CompareRow,
} from './compareModel'
```

Append at the end of `src/ui/stores.ts`:

```ts
// --- Compare tab ---
// Fully independent of the single-selection calc/sim stores (the calc `input`
// derivation assumes scalar model/sku; compare holds lists). Pivot = the locked
// dimension; each candidate carries the varying id + its own quant.
export const comparePivot = writable<ComparePivot>({ kind: 'sku', id: defaultPivotId('sku') })
export const compareCandidates = writable<CompareCandidate[]>([
  { varyingId: firstVaryingId('sku'), quant: seededQuantFor(firstVaryingId('sku')) },
])
export const compareWorkload = writable<Workload>({ promptTokens: 2048, outputTokens: 512, concurrency: 1 })

export const compareResults: Readable<CompareRow[]> = derived(
  [comparePivot, compareCandidates, compareWorkload],
  ([$pivot, $candidates, $workload]) =>
    $candidates.map(c => computeCompareRow($pivot, c, $workload)),
)

// Flip the pivot axis. A model-list can't remain valid as a sku-list, so
// candidates are hard-cleared and reseeded with one default on the new axis;
// the shared workload is preserved.
export function setComparePivotKind(kind: ComparePivotKind): void {
  comparePivot.set({ kind, id: defaultPivotId(kind) })
  const seedId = firstVaryingId(kind)
  const seedQuant = kind === 'sku' ? seededQuantFor(seedId) : seededQuantFor(defaultPivotId(kind))
  compareCandidates.set([{ varyingId: seedId, quant: seedQuant }])
}

// Seed the compare view from the current calc selection so "compare this
// against…" is one click. Always uses the SKU pivot (fixed = current
// accelerator or system, candidate = current model). Called at startup only
// when the URL carried no compare payload, so a shared compare link always wins.
export function seedCompareFromCalc(): void {
  const sku = get(systemId) || get(acceleratorId)
  const model = get(modelId)
  comparePivot.set({ kind: 'sku', id: sku })
  compareCandidates.set([{ varyingId: model, quant: seededQuantFor(model) }])
}
```

> When `kind === 'model'` the varying dimension is a SKU (no native dtype of its own), so seed the candidate quant from the *pivot model's* native dtype. When `kind === 'sku'` the varying dimension is a model, so seed from that model.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- compareStores`
Expected: PASS (3 tests).

- [ ] **Step 5: Run full check + suite**

Run: `npm run check && npm test`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/ui/stores.ts test/ui/compareStores.test.ts
git commit -m "feat(compare): stores + compareResults derived + pivot-axis reshape"
```

---

### Task 6: Route + tab wiring

**Files:**
- Modify: `src/ui/route.ts`
- Test: `test/ui/route.test.ts` (create if absent)

**Interfaces:**
- Produces: `Route` union gains `{ tab: 'compare' }`; `parseRoute` / `serializeRoute` handle `#compare` and `#compare?<payload>`.

- [ ] **Step 1: Write the failing test**

Create (or append to) `test/ui/route.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { parseRoute, serializeRoute } from '../../src/ui/route'

describe('compare route', () => {
  it('parses #compare and #compare?<payload>', () => {
    expect(parseRoute('#compare')).toEqual({ tab: 'compare' })
    expect(parseRoute('#compare?piv=sku:h100')).toEqual({ tab: 'compare' })
  })
  it('serializes the compare tab with an optional payload', () => {
    expect(serializeRoute({ tab: 'compare' })).toBe('#compare')
    expect(serializeRoute({ tab: 'compare' }, 'piv=sku:h100')).toBe('#compare?piv=sku:h100')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- route`
Expected: FAIL — `'compare'` not assignable / wrong serialization.

- [ ] **Step 3: Write minimal implementation**

In `src/ui/route.ts`, extend the union (line 6-10):

```ts
export type Route =
  | { tab: 'calc' }
  | { tab: 'sim' }
  | { tab: 'compare' }
  | { tab: 'info' }
  | { tab: 'info'; detail: { kind: 'model' | 'sku'; id: string } }
```

In `parseRoute`, add after the `sim` line (line 17):

```ts
  if (h === 'compare' || h.startsWith('compare?')) return { tab: 'compare' }
```

In `serializeRoute`, add after the `sim` line (line 30):

```ts
  if (route.tab === 'compare') return payload ? `#compare?${payload}` : '#compare'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- route`
Expected: PASS.

- [ ] **Step 5: Add the tab button**

In `src/ui/TabBar.svelte`, add to the `tabs` array (after the `sim` entry, line 7):

```ts
    { id: 'compare' as const, label: 'Compare' },
```

- [ ] **Step 6: Commit**

```bash
git add src/ui/route.ts src/ui/TabBar.svelte test/ui/route.test.ts
git commit -m "feat(compare): compare route + tab button"
```

---

### Task 7: Results table component (`CompareTable.svelte`)

**Files:**
- Create: `src/ui/CompareTable.svelte`

**Interfaces:**
- Consumes: `compareResults` store; `CompareRow` type.
- Produces: a presentational, sortable table — candidates as **rows**, metrics as **columns**; best value per metric highlighted; error rows marked; infeasible (`fits === false`) flagged.

No unit test (Svelte component; repo convention). Verified via `npm run check` + `npm run dev`.

- [ ] **Step 1: Create the component**

Create `src/ui/CompareTable.svelte`:

```svelte
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
```

- [ ] **Step 2: Type-check**

Run: `npm run check`
Expected: no errors for `CompareTable.svelte`.

- [ ] **Step 3: Commit**

```bash
git add src/ui/CompareTable.svelte
git commit -m "feat(compare): sortable results table with winner highlighting"
```

---

### Task 8: Compare tab controls + App wiring + URL sync

**Files:**
- Create: `src/ui/CompareTab.svelte`
- Modify: `src/ui/App.svelte`, `src/ui/compareShare.ts`, `src/main.ts`

**Interfaces:**
- Consumes: compare stores + `setComparePivotKind` + `seedCompareFromCalc`; `ACCELERATORS`, `MODELS`, `SYSTEMS`; `firstVaryingId`, `seededQuantFor` from `./compareModel`; `CompareTable`.
- Produces:
  - `CompareTab.svelte` — pivot-axis toggle, pivot-id selector, shared workload inputs, per-candidate list (add/remove + per-candidate quant), embeds `CompareTable`.
  - `compareShare.ts` gains `readCompareUrlIntoStores()` + `startCompareUrlSync()`.
  - `App.svelte` renders `<CompareTab />` on the compare tab.
  - `main.ts` calls the two sync functions at startup.

- [ ] **Step 1: Add URL sync to `compareShare.ts`**

Append to `src/ui/compareShare.ts`:

```ts
import { get } from 'svelte/store'
import { comparePivot, compareCandidates, compareWorkload } from './stores'
import { parseRoute } from './route'

function readStoreCompareState(): CompareState {
  return { pivot: get(comparePivot), candidates: get(compareCandidates), workload: get(compareWorkload) }
}

function applyCompareState(s: CompareState): void {
  comparePivot.set(s.pivot)
  compareCandidates.set(s.candidates)
  compareWorkload.set(s.workload)
}

// Read the compare payload from the URL on load, iff the hash targets the
// compare tab. No-op otherwise (calc/sim links are handled by share.ts).
export function readCompareUrlIntoStores(): void {
  if (typeof window === 'undefined') return
  const h = window.location.hash.replace(/^#/, '')
  if (!h.startsWith('compare?')) return
  const decoded = decodeCompare(h.slice('compare?'.length))
  if (decoded) applyCompareState(decoded)
}

// Mirror the compare stores back to the hash while on the compare tab. Mirrors
// share.ts.startUrlSync structure (hold `ready` until all subs wired).
export function startCompareUrlSync(): () => void {
  if (typeof window === 'undefined') return () => {}
  let ready = false
  const write = () => {
    if (!ready) return
    if (parseRoute(window.location.hash).tab !== 'compare') return
    const encoded = encodeCompare(readStoreCompareState())
    const next = `${window.location.pathname}${window.location.search}#compare?${encoded}`
    window.history.replaceState(window.history.state, '', next)
  }
  const unsubs = [comparePivot.subscribe(write), compareCandidates.subscribe(write), compareWorkload.subscribe(write)]
  ready = true
  write()
  return () => unsubs.forEach(u => u())
}
```

- [ ] **Step 2: Wire startup in `main.ts`**

Modify `src/main.ts`:

```ts
import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'
import { readCompareUrlIntoStores, startCompareUrlSync } from './ui/compareShare'
import { initRouteSync } from './ui/route'
import { initNativeDtypeSync, seedCompareFromCalc } from './ui/stores'

readUrlIntoStores()
// Seed compare from the calc selection unless the URL already carries a compare
// payload (a shared compare link must win over the seed).
const hasCompareUrl = typeof window !== 'undefined'
  && window.location.hash.replace(/^#/, '').startsWith('compare?')
readCompareUrlIntoStores()
if (!hasCompareUrl) seedCompareFromCalc()
initNativeDtypeSync()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
startCompareUrlSync()
initRouteSync()
export default app
```

- [ ] **Step 3: Create `CompareTab.svelte`**

Create `src/ui/CompareTab.svelte`:

```svelte
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
```

> Per-candidate quant is displayed read-only in v1 (seeded from the model's native dtype). A per-candidate quant editor is a small follow-up; the store shape already carries `quant` per candidate, so it's additive. This keeps v1 scope honest while satisfying the "per-candidate quant" decision (each candidate resolves at its own seeded precision).

- [ ] **Step 4: Wire into `App.svelte`**

In `src/ui/App.svelte`, add the import (after the `Simulator` import, line 9):

```ts
  import CompareTab from './CompareTab.svelte'
```

Add the branch (after the `sim` branch, line 55):

```svelte
  {:else if $route.tab === 'compare'}
    <CompareTab />
```

- [ ] **Step 5: Type-check + full suite**

Run: `npm run check && npm test`
Expected: PASS, no type errors.

- [ ] **Step 6: Manual verification in the app**

Run: `npm run dev`, then in the browser:
1. Click the **Compare** tab. Confirm it defaults to "models on one accelerator" with one candidate and a populated table.
2. Add 2-3 more model candidates; confirm rows appear with metrics and the best value per column is highlighted green.
3. Sort by each column header; confirm ascending/descending toggles and error rows sink to the bottom.
4. Toggle to "accelerators for one model"; confirm candidates hard-clear to a single reseeded accelerator candidate and the workload is preserved.
5. Pick a candidate/quant combo that OOMs or errors (e.g. a huge model on a small SKU); confirm the row shows OOM / an error message without blanking the table.
6. Copy the URL, open in a fresh tab; confirm the compare state restores.

- [ ] **Step 7: Commit**

```bash
git add src/ui/CompareTab.svelte src/ui/compareShare.ts src/main.ts src/ui/App.svelte
git commit -m "feat(compare): compare tab controls, URL sync, app wiring"
```

---

## Post-implementation

- [ ] Run `npm run check && npm test` one final time — all green.
- [ ] Update the roadmap memory: mark item #5 done (or drop it) in `project_calc_roadmap.md`, and note the named follow-ups spawned here (per-candidate quant *editor*, per-variant SKU compare, under-load curves).
- [ ] Offer to open a PR (the `pr` skill) — do not push without the owner's go-ahead.

## Named follow-ups (out of v1 scope)

- **Under-load / overlaid load curves** (reuse `computeNMax` / `loadCurve`) — the roadmap "curves later" item.
- **Per-candidate quant editor** — store already carries per-candidate quant; add an inline editor.
- **Per-variant SKU compare** — today a bare accelerator resolves to `variants[0]`; add variant selection to compare SXM vs PCIe.
- **Cost / $-per-token column** — depends on the cloud-availability database.
