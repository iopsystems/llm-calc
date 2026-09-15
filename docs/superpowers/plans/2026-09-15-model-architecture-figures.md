# Model Architecture Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-variant architecture schematics (educational, hand-drawn, cited) and per-model geometry figures (computed from `ModelArch`) to the Info tab, and fix the spec sheet's overcounted KV-per-token figure.

**Architecture:** Two pure, tested TypeScript modules produce data objects (`archGeometry.ts` for per-model geometry, `schematics/meta.ts` for the schematic registry); dumb Svelte components draw them as inline SVG. Geometry is derived from `ModelArch` through an exhaustive switch cross-checked against the engine; schematics are keyed by the `AttentionConfig` / `ArchitectureConfig` discriminants so a new variant fails `npm run check` until both are extended. A new `#info/arch/<id>` route and an Architectures sub-section in the Info tab host the schematics; the model spec sheet hosts the geometry figure and links to its schematic.

**Tech Stack:** TypeScript, Svelte 5 in legacy mode (`export let`, `$:`), Vitest (node env, no DOM), inline SVG, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-model-architecture-figure-design.md`

## Global Constraints

- TDD: write the failing test, run it red, implement, run green. Never implement before a failing test exists.
- Comments say *why*, terse. No comments that restate the code.
- Before claiming any task done: `npm run check` and `npm test` both pass.
- **Commits require the user's go-ahead.** The user's standing rule is "do not commit or push git changes for me without asking for permission first." At the start of execution, ask once whether per-task commits are authorized for this plan. If yes, commit at each commit step. If no, stop at each commit step with the changes staged and report. Never push.
- Commit messages: conventional prefix (`feat:`, `fix:`, `test:`, `docs:`), no `Co-Authored-By` footer (user preference).
- Svelte components use legacy syntax matching the codebase: `<script lang="ts">`, `export let`, `$:` reactivity, `on:click`. SVG-fragment components declare `<svelte:options namespace="svg" />` as their first line.
- Fixed KV reference dtype for every displayed byte figure: `fp16` (`KV_REF_DTYPE`, exported from `catalogMetrics.ts` in Task 2). Never hard-code `2` for bytes; call `bytesOf(KV_REF_DTYPE)`.
- Every byte figure comes from an engine function in `src/engine/memory.ts`. Figure modules never multiply head dims themselves.
- Kind colours, fixed: full `#2f5fd0`, window `#3f9cc2`, sparse `#7a4fc9`, state `#1f9a6e`, none `#b4bac8`, experts `#e07a2a`. Light theme only.
- Every `<svg>` root has `role="img"` and an `aria-label`.
- Source URLs are verified by fetching before they are written into `src/data/sources.ts`. Never write a URL from memory.
- Bar lengths in block-stack panels are counts. Never draw an interleave order the data does not carry.

---

## File map

| Path | Responsibility |
|---|---|
| `src/ui/route.ts` (modify) | Add `'arch'` detail kind and `#info/arch/<id>` parse/serialize. |
| `src/ui/catalogMetrics.ts` (modify) | Export `KV_REF_DTYPE`; fix `kvBytesPerToken`; add `fixedStateBytes`, `attentionReachLayers`. |
| `src/ui/ModelSpecSheet.svelte` (modify) | New metric rows; embed `ArchGeometry`; link to schematic. |
| `src/ui/figure/types.ts` (create) | `LaneColor`, `CacheGlyph`, `ReachGlyph`, `KIND_COLORS`, `EXPERT_COLOR`. Shared by both figure types. |
| `src/ui/figure/CacheGlyph.svelte`, `ReachGlyph.svelte` (create) | SVG-fragment glyph components used by geometry panels and schematic footprint strips. |
| `src/ui/archGeometry.ts` (create) | Pure. `geometryModel(model)`: lanes, head geometry, FFN, caption. Exhaustive over `AttentionConfig`. |
| `src/ui/ArchGeometry.svelte` (create) | Draws a `GeometryModel` as five stacked SVG panels. |
| `src/ui/schematics/meta.ts` (create) | Pure registry: `SchematicMeta` per discriminant, groups, `schematicFor`, `modelsUsing`. No Svelte imports. |
| `src/ui/schematics/components.ts` (create) | `SCHEMATIC_COMPONENTS: Record<SchematicId, Component>`. |
| `src/ui/schematics/parts/{Box,Mem,Arrow}.svelte` (create) | SVG primitives for hand-drawn schematics. |
| `src/ui/schematics/SchematicFrame.svelte` (create) | Root `<svg>` with marker defs, hosts one schematic, draws the footprint strip. |
| `src/ui/schematics/<Variant>Schematic.svelte` ×13 (create) | One block diagram each. |
| `src/ui/ArchPage.svelte` (create) | Schematic page: frame, summary, sources, models using it. |
| `src/ui/InfoPanel.svelte` (modify) | Third sub-section "Architectures"; routes `arch` detail to `ArchPage`. |
| `src/data/sources.ts` (modify) | Paper entries for schematic citations. |
| `docs/architecture-figures.md` (create) | Grammar reference. |
| `.claude/skills/adding-a-model/SKILL.md` (modify) | `## Figures` section. |
| `.claude/hooks/check-skill-sync.mjs` (modify) | `extractSkillFigureVariants` + check. |
| Tests | `test/ui/route.test.ts`, `test/ui/catalogMetrics.test.ts`, `test/ui/archGeometry.test.ts`, `test/ui/archSchematics.test.ts`, `test/check-skill-sync.test.ts`. |

---

### Task 1: `arch` route detail kind

**Files:**
- Modify: `src/ui/route.ts`
- Test: `test/ui/route.test.ts`

**Interfaces:**
- Produces: `Route` now includes `{ tab: 'info'; detail: { kind: 'model' | 'sku' | 'arch'; id: string } }`. Hash form `#info/arch/<id>`.

- [ ] **Step 1: Write the failing tests**

Append to the `parseRoute` describe block in `test/ui/route.test.ts`:

```ts
  it('info arch detail', () => {
    expect(parseRoute('#info/arch/mla'))
      .toEqual({ tab: 'info', detail: { kind: 'arch', id: 'mla' } })
  })
  it('info arch detail with hyphenated id', () => {
    expect(parseRoute('#info/arch/csa-hca-hybrid'))
      .toEqual({ tab: 'info', detail: { kind: 'arch', id: 'csa-hca-hybrid' } })
  })
```

Append to the `serializeRoute` describe block:

```ts
  it('info arch detail round-trips', () => {
    const r: Route = { tab: 'info', detail: { kind: 'arch', id: 'delta-hybrid' } }
    expect(serializeRoute(r)).toBe('#info/arch/delta-hybrid')
    expect(parseRoute(serializeRoute(r))).toEqual(r)
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/ui/route.test.ts`
Expected: the two parse tests FAIL (`#info/arch/...` currently falls back to `{ tab: 'calc' }`); the serialize test fails on the round-trip assertion.

- [ ] **Step 3: Implement**

In `src/ui/route.ts`, change the `Route` union member and the parser:

```ts
export type Route =
  | { tab: 'calc' }
  | { tab: 'sim' }
  | { tab: 'compare' }
  | { tab: 'info' }
  | { tab: 'info'; detail: { kind: 'model' | 'sku' | 'arch'; id: string } }
```

```ts
  const m = h.match(/^info\/(model|sku|arch)\/(.+)$/)
  if (m) return { tab: 'info', detail: { kind: m[1] as 'model' | 'sku' | 'arch', id: m[2] } }
```

`serializeRoute` already emits `#info/${kind}/${id}`; no change.

- [ ] **Step 4: Run the tests and the type check**

Run: `npx vitest run test/ui/route.test.ts && npm run check`
Expected: PASS. `npm run check` will report `InfoPanel.svelte` still compiles (it compares `kind === 'model'` and treats anything else as SKUs; Task 13 fixes that mapping).

- [ ] **Step 5: Commit**

```bash
git add src/ui/route.ts test/ui/route.test.ts
git commit -m "feat(route): add info/arch detail kind"
```

---

### Task 2: Fix `kvBytesPerToken` and add state / reach metrics

**Files:**
- Modify: `src/ui/catalogMetrics.ts`
- Modify: `src/ui/ModelSpecSheet.svelte` (rows only; the figure comes in Task 8)
- Test: `test/ui/catalogMetrics.test.ts`

**Interfaces:**
- Produces:
  - `export const KV_REF_DTYPE = 'fp16' as const`
  - `export function kvBytesPerTokenAtContext(m: ModelArch, seqlen: number): number`
  - `export function fixedStateBytes(m: ModelArch): number`
  - `export function attentionReachLayers(m: ModelArch, seqlen: number): number`
  - `ModelMetrics` gains `fixedStateBytes: number` and `attentionReachLayers: number`; `kvBytesPerToken` is now evaluated at `m.maxContext`.

- [ ] **Step 1: Write the failing tests**

Add to the `modelMetrics` describe block in `test/ui/catalogMetrics.test.ts`:

```ts
  it('hybrid model: KV per token counts only caching layers (Qwen3.5-397B → 15 × 2 KiB)', () => {
    const m = MODELS.find(x => x.id === 'qwen3.5-397b-a17b')!
    const r = modelMetrics(m)
    // 45 DeltaNet layers cache nothing; 15 gated-attention layers × 2 KV heads × 256 × 2 B.
    expect(r.kvBytesPerToken).toBe(15 * 2048)
  })
  it('MSA model: KV per token includes the per-sparse-layer index key', () => {
    const m = MODELS.find(x => x.id === 'minimax-m3')!
    // 60 layers × 2 KiB main cache + 57 sparse layers × 128 × 2 B index key.
    expect(modelMetrics(m).kvBytesPerToken).toBe(60 * 2048 + 57 * 128 * 2)
  })
  it('sliding model: KV per token at max context is the window share', () => {
    const m = MODELS.find(x => x.id === 'mistral-7b-v0.1')!
    // 32 layers × 4 KiB × (4096 window / 32768 max context).
    expect(modelMetrics(m).kvBytesPerToken).toBeCloseTo(32 * 4096 * 4096 / 32768, 6)
  })
  it('fixedStateBytes: Mamba2 state is fp32 regardless of KV reference', () => {
    const m = MODELS.find(x => x.id === 'nemotron-3-nano-30b-a3b')!
    expect(modelMetrics(m).fixedStateBytes).toBe(23 * 64 * 64 * 128 * 4)
  })
  it('fixedStateBytes: DeltaNet state at fp16', () => {
    const m = MODELS.find(x => x.id === 'qwen3.5-397b-a17b')!
    expect(modelMetrics(m).fixedStateBytes).toBe(45 * 64 * 128 * 128 * 2)
  })
  it('fixedStateBytes is zero for a full-attention model', () => {
    const m = MODELS.find(x => x.id === 'llama-3.3-70b')!
    expect(modelMetrics(m).fixedStateBytes).toBe(0)
  })
  it('attentionReachLayers: MLA reaches every layer, DSA reaches topK/S of each', () => {
    const v3 = MODELS.find(x => x.id === 'deepseek-v3')!
    const v32 = MODELS.find(x => x.id === 'deepseek-v3.2')!
    expect(modelMetrics(v3).attentionReachLayers).toBe(61)
    expect(modelMetrics(v32).attentionReachLayers).toBeCloseTo(2048 * 61 / 163840, 6)
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/ui/catalogMetrics.test.ts`
Expected: the Qwen3.5, MSA and sliding cases FAIL on `kvBytesPerToken` (current value multiplies by all layers); the `fixedStateBytes` and `attentionReachLayers` cases FAIL with `undefined`.

- [ ] **Step 3: Implement**

Replace the top of `src/ui/catalogMetrics.ts` through `modelMetrics` with:

```ts
// Derived static metrics for spec sheets. Pure; reuses engine helpers so the
// numbers can't drift from the calculator. fp16 is the fixed KV reference.
import type {
  ModelArch, AcceleratorSpec, MultiAcceleratorSystem,
} from '../engine/types'
import {
  kvBytesPerTokenPerLayer, activeParams, attendedSeqlenSummedOverLayers,
  linearAttentionStateBytes, deltaStateBytes, mambaStateBytes,
} from '../engine/memory'
import { bytesOf } from '../engine/dtypes'
import { SOURCES } from '../data/sources'

export const KV_REF_DTYPE = 'fp16' as const

// Marginal cache bytes one token adds at sequence length `seqlen`, summed
// over layers with the engine's own storage rule (windows contribute w/S of
// a layer, compressed streams 1/M, recurrent state nothing). MSA keeps a
// separate index-key row per sparse layer that the KV formula doesn't cover.
export function kvBytesPerTokenAtContext(m: ModelArch, seqlen: number): number {
  const perLayer = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
  const indexKey = m.attention.type === 'msa-hybrid'
    ? m.attention.numSparseLayers * m.attention.indexHeadDim * bytesOf(KV_REF_DTYPE)
    : 0
  return perLayer * attendedSeqlenSummedOverLayers(m, seqlen, true) / seqlen + indexKey
}

// Per-request bytes that exist before the first token: recurrent state.
export function fixedStateBytes(m: ModelArch): number {
  return linearAttentionStateBytes(m, KV_REF_DTYPE)
    + deltaStateBytes(m, KV_REF_DTYPE)
    + mambaStateBytes(m)
}

// How much of the sequence attention math touches per decode step, in
// full-layer equivalents. Distinguishes DSA (topK/S per layer) from MLA.
export function attentionReachLayers(m: ModelArch, seqlen: number): number {
  return attendedSeqlenSummedOverLayers(m, seqlen, false) / seqlen
}
```

Keep `attentionLabel` and `mtpLabel` unchanged. Update the interface and `modelMetrics`:

```ts
export interface ModelMetrics {
  kvBytesPerTokenPerLayer: number
  kvBytesPerToken: number        // at m.maxContext
  fixedStateBytes: number
  attentionReachLayers: number   // at m.maxContext
  gqaRatio: number
  attentionLabel: string
  mtpLabel: string
  moeActiveRatio?: number
}

export function modelMetrics(m: ModelArch): ModelMetrics {
  const perLayer = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
  const out: ModelMetrics = {
    kvBytesPerTokenPerLayer: perLayer,
    kvBytesPerToken: kvBytesPerTokenAtContext(m, m.maxContext),
    fixedStateBytes: fixedStateBytes(m),
    attentionReachLayers: attentionReachLayers(m, m.maxContext),
    gqaRatio: m.numHeads / m.numKvHeads,
    attentionLabel: attentionLabel(m),
    mtpLabel: mtpLabel(m),
  }
  if (m.architecture.type === 'moe') {
    out.moeActiveRatio = activeParams(m) / m.paramCount
  }
  return out
}
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run test/ui/catalogMetrics.test.ts`
Expected: PASS, including the pre-existing Llama 3.3 70B case (full attention: `attendedSeqlen / S` is exactly `layers`).

- [ ] **Step 5: Update the spec sheet rows**

In `src/ui/ModelSpecSheet.svelte`, replace the "Derived memory" `<dl>` with:

```svelte
  <h3>Derived memory <span class="ref">(fp16 KV reference)</span></h3>
  <dl>
    <dt>KV / token / layer</dt><dd>{kb(m.kvBytesPerTokenPerLayer)}</dd>
    <dt>KV / token (model, at max context)</dt><dd>{kb(m.kvBytesPerToken)}</dd>
    {#if m.fixedStateBytes > 0}
      <dt>Fixed state / request</dt><dd>{mb(m.fixedStateBytes)}</dd>
    {/if}
    <dt>Attention reach</dt>
    <dd>{m.attentionReachLayers.toFixed(m.attentionReachLayers < 10 ? 2 : 0)} × S of {model.layers} blocks</dd>
  </dl>
```

Add next to `kb`:

```ts
  function mb(bytes: number): string {
    return `${(bytes / 1048576).toFixed(1)} MiB`
  }
```

- [ ] **Step 6: Type check and full test run**

Run: `npm run check && npm test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ui/catalogMetrics.ts src/ui/ModelSpecSheet.svelte test/ui/catalogMetrics.test.ts
git commit -m "fix(info): count only caching layers in KV/token; add fixed-state and reach rows"
```

---

### Task 3: Geometry lanes for KV-cached variants (`full`, `sliding`, `hybrid`, `partial`)

**Files:**
- Create: `src/ui/figure/types.ts`
- Create: `src/ui/archGeometry.ts`
- Test: `test/ui/archGeometry.test.ts`

**Interfaces:**
- Produces (`src/ui/figure/types.ts`):
  ```ts
  export type LaneColor = 'full' | 'window' | 'sparse' | 'state' | 'none'
  export type CacheGlyph = 'kv' | 'latent' | 'kvidx' | 'kvcomp' | 'state' | 'none'
  export type ReachGlyph = 'all' | 'window' | 'topk' | 'compress' | 'state' | 'none'
  export const KIND_COLORS: Record<LaneColor, string>
  export const EXPERT_COLOR: string
  ```
- Produces (`src/ui/archGeometry.ts`): `LaneKind`, `Lane`, `lanesFor(m: ModelArch): Lane[]`, `fmtBytes(b: number): string`. Later tasks add cases and `geometryModel`.

- [ ] **Step 1: Write the failing tests**

Create `test/ui/archGeometry.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { MODELS } from '../../src/data'
import { lanesFor, fmtBytes } from '../../src/ui/archGeometry'

const byId = (id: string) => {
  const m = MODELS.find(x => x.id === id)
  if (!m) throw new Error(`no model ${id}`)
  return m
}

describe('fmtBytes', () => {
  it('formats B / KiB / MiB', () => {
    expect(fmtBytes(256)).toBe('256 B')
    expect(fmtBytes(4096)).toBe('4 KiB')
    expect(fmtBytes(1152)).toBe('1.1 KiB')
    expect(fmtBytes(2097152)).toBe('2 MiB')
  })
})

describe('lanesFor — KV-cached variants', () => {
  it('full: one lane, every layer, K+V rows, reads all', () => {
    const m = byId('llama-3.1-8b')
    const lanes = lanesFor(m)
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'full', color: 'full', count: 32,
      cache: { glyph: 'kv', bytesPerToken: 4096, fixedBytes: 0 },
      reach: { glyph: 'all' },
    })
    expect(lanes[0].cache.label).toBe('K + V · 8 KV heads × 128 · 4 KiB')
    expect(lanes[0].reach.label).toBe('all tokens')
  })

  it('sliding: window lane with the window in tokens', () => {
    const lanes = lanesFor(byId('mistral-7b-v0.1'))
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'window', color: 'window', count: 32,
      cache: { glyph: 'kv', bytesPerToken: 4096 },
      reach: { glyph: 'window', tokens: 4096, label: 'trailing 4096' },
    })
    expect(lanes[0].cache.label).toBe('K + V · 8 KV heads × 128 · 4 KiB, last 4096 tokens only')
  })

  it('hybrid: window lane then full lane, counts from config', () => {
    const lanes = lanesFor(byId('gpt-oss-120b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['window', 18], ['full', 18]])
    expect(lanes[0].reach).toMatchObject({ glyph: 'window', tokens: 128 })
    expect(lanes[1].reach).toMatchObject({ glyph: 'all' })
    expect(lanes[0].cache.bytesPerToken).toBe(2048)
  })

  it('partial: attending lane plus a pruned lane that caches nothing', () => {
    const lanes = lanesFor(byId('llama-3.3-nemotron-super-49b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['full', 49], ['pruned', 31]])
    expect(lanes[1]).toMatchObject({
      color: 'none',
      cache: { glyph: 'none', bytesPerToken: 0, fixedBytes: 0, label: 'nothing' },
      reach: { glyph: 'none', label: 'no attention' },
    })
  })

  it('lane counts sum to model.layers for these variants', () => {
    for (const id of ['llama-3.1-8b', 'mistral-7b-v0.1', 'gpt-oss-120b', 'llama-3.3-nemotron-super-49b']) {
      const m = byId(id)
      expect(lanesFor(m).reduce((n, l) => n + l.count, 0)).toBe(m.layers)
    }
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/ui/archGeometry.test.ts`
Expected: FAIL, module `../../src/ui/archGeometry` not found.

- [ ] **Step 3: Create the shared figure types**

Create `src/ui/figure/types.ts`:

```ts
// Glyph vocabulary shared by the per-model geometry figure and the
// per-variant schematics. Colour encodes what a layer reads at decode.
export type LaneColor = 'full' | 'window' | 'sparse' | 'state' | 'none'
export type CacheGlyph = 'kv' | 'latent' | 'kvidx' | 'kvcomp' | 'state' | 'none'
export type ReachGlyph = 'all' | 'window' | 'topk' | 'compress' | 'state' | 'none'

export const KIND_COLORS: Record<LaneColor, string> = {
  full: '#2f5fd0',
  window: '#3f9cc2',
  sparse: '#7a4fc9',
  state: '#1f9a6e',
  none: '#b4bac8',
}
export const EXPERT_COLOR = '#e07a2a'
```

- [ ] **Step 4: Create the geometry module with the four KV-cached cases**

Create `src/ui/archGeometry.ts`:

```ts
// Per-model geometry for the Info tab figure. Pure; every byte figure comes
// from src/engine/memory.ts so the drawing cannot disagree with the calc.
import type { ModelArch } from '../engine/types'
import { kvBytesPerTokenPerLayer } from '../engine/memory'
import { KV_REF_DTYPE } from './catalogMetrics'
import type { LaneColor, CacheGlyph, ReachGlyph } from './figure/types'

export type LaneKind =
  | 'full' | 'window' | 'mla' | 'dsa' | 'msa' | 'csa' | 'hca'
  | 'kda' | 'delta' | 'mamba' | 'ffn' | 'pruned'

export interface Lane {
  kind: LaneKind
  label: string
  color: LaneColor
  count: number
  cache: {
    glyph: CacheGlyph
    bytesPerToken: number    // per layer at KV_REF_DTYPE; compressed kinds already ÷ M
    fixedBytes: number       // per layer, 'state' only
    ratio?: number           // M for 'kvcomp'
    label: string
  }
  reach: {
    glyph: ReachGlyph
    tokens?: number          // window w, or top-k token count
    ratio?: number           // M for 'compress'
    local?: number           // local-window branch alongside a sparse/compressed read
    label: string
  }
}

const KIND_META: Record<LaneKind, { label: string; color: LaneColor }> = {
  full:   { label: 'full attention',   color: 'full' },
  window: { label: 'sliding window',   color: 'window' },
  mla:    { label: 'MLA',              color: 'full' },
  dsa:    { label: 'MLA + DSA top-k',  color: 'sparse' },
  msa:    { label: 'MSA block top-k',  color: 'sparse' },
  csa:    { label: 'CSA',              color: 'sparse' },
  hca:    { label: 'HCA',              color: 'sparse' },
  kda:    { label: 'KDA linear',       color: 'state' },
  delta:  { label: 'Gated DeltaNet',   color: 'state' },
  mamba:  { label: 'Mamba2',           color: 'state' },
  ffn:    { label: 'FFN only',         color: 'none' },
  pruned: { label: 'attention pruned', color: 'none' },
}

export function fmtBytes(b: number): string {
  const f = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1))
  if (b >= 1048576) return `${f(b / 1048576)} MiB`
  if (b >= 1024) return `${f(b / 1024)} KiB`
  return `${b} B`
}

function lane(kind: LaneKind, count: number, cache: Lane['cache'], reach: Lane['reach']): Lane {
  return { kind, count, cache, reach, ...KIND_META[kind] }
}

function kvCache(m: ModelArch, p: number, suffix = ''): Lane['cache'] {
  return {
    glyph: 'kv', bytesPerToken: p, fixedBytes: 0,
    label: `K + V · ${m.numKvHeads} KV heads × ${m.headDim} · ${fmtBytes(p)}${suffix}`,
  }
}

const NONE_CACHE: Lane['cache'] = { glyph: 'none', bytesPerToken: 0, fixedBytes: 0, label: 'nothing' }
const ALL_REACH: Lane['reach'] = { glyph: 'all', label: 'all tokens' }
const NONE_REACH: Lane['reach'] = { glyph: 'none', label: 'no attention' }
const windowReach = (w: number): Lane['reach'] => ({ glyph: 'window', tokens: w, label: `trailing ${w}` })

export function lanesFor(m: ModelArch): Lane[] {
  const att = m.attention
  const p = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
  switch (att.type) {
    case 'full':
      return [lane('full', m.layers, kvCache(m, p), ALL_REACH)]
    case 'sliding':
      return [lane('window', m.layers, kvCache(m, p, `, last ${att.window} tokens only`), windowReach(att.window))]
    case 'hybrid':
      return [
        lane('window', att.numSlidingLayers, kvCache(m, p), windowReach(att.slidingWindow)),
        lane('full', att.numGlobalLayers, kvCache(m, p), ALL_REACH),
      ]
    case 'partial':
      return [
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
        lane('pruned', m.layers - att.numFullLayers, NONE_CACHE, NONE_REACH),
      ]
    default:
      // Remaining variants land in Tasks 4 and 5; the never-check arrives with the last one.
      throw new Error(`lanesFor: variant not yet handled: ${att.type}`)
  }
}
```

- [ ] **Step 5: Run the tests**

Run: `npx vitest run test/ui/archGeometry.test.ts && npm run check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ui/figure/types.ts src/ui/archGeometry.ts test/ui/archGeometry.test.ts
git commit -m "feat(info): geometry lanes for full/sliding/hybrid/partial attention"
```

---

### Task 4: Geometry lanes for latent and sparse variants (`mla`, `mla-dsa`, `msa-hybrid`, `csa-hca-hybrid`)

**Files:**
- Modify: `src/ui/archGeometry.ts`
- Test: `test/ui/archGeometry.test.ts`

**Interfaces:**
- Consumes: `lane`, `kvCache`, `ALL_REACH`, `windowReach` from Task 3.
- Produces: `lanesFor` handles the four variants above.

- [ ] **Step 1: Write the failing tests**

Append to `test/ui/archGeometry.test.ts`:

```ts
describe('lanesFor — latent and sparse variants', () => {
  it('mla: one latent lane, bytes = (kvLoraRank + qkRopeHeadDim) × 2', () => {
    const lanes = lanesFor(byId('deepseek-v3'))
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'mla', color: 'full', count: 61,
      cache: { glyph: 'latent', bytesPerToken: (512 + 64) * 2, fixedBytes: 0 },
      reach: { glyph: 'all' },
    })
    expect(lanes[0].cache.label).toBe('latent 512 + RoPE key 64 · 1.1 KiB')
  })

  it('mla-dsa: same cache as mla, reach is top-k tokens', () => {
    const lanes = lanesFor(byId('deepseek-v3.2'))
    expect(lanes).toHaveLength(1)
    expect(lanes[0]).toMatchObject({
      kind: 'dsa', color: 'sparse', count: 61,
      cache: { glyph: 'latent', bytesPerToken: (512 + 64) * 2 },
      reach: { glyph: 'topk', tokens: 2048, label: 'top-2048 tokens' },
    })
  })

  it('msa-hybrid: full lane, then sparse lane whose cache adds the shared index key', () => {
    const lanes = lanesFor(byId('minimax-m3'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['full', 3], ['msa', 57]])
    expect(lanes[1]).toMatchObject({
      color: 'sparse',
      cache: { glyph: 'kvidx', bytesPerToken: 2048 + 128 * 2 },
      reach: { glyph: 'topk', tokens: 16 * 128, label: '16 blocks × 128 tokens' },
    })
    expect(lanes[1].cache.label).toBe('K + V 2 KiB + shared index key 128 · 256 B')
  })

  it('csa-hca-hybrid (V4-Flash): window, CSA ÷4 with top-k, HCA ÷128 compressed', () => {
    const lanes = lanesFor(byId('deepseek-v4-flash'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['window', 2], ['csa', 21], ['hca', 20]])
    expect(lanes[0].reach).toMatchObject({ glyph: 'window', tokens: 128 })
    expect(lanes[1]).toMatchObject({
      color: 'sparse',
      cache: { glyph: 'kvcomp', ratio: 4, bytesPerToken: 2048 / 4 },
      reach: { glyph: 'topk', tokens: 512 * 4, local: 128 },
    })
    expect(lanes[1].cache.label).toBe('K + V 2 KiB per 4 tokens · 512 B/token')
    expect(lanes[1].reach.label).toBe('top-512 of the 1 : 4 stream + 128 local')
    expect(lanes[2]).toMatchObject({
      cache: { glyph: 'kvcomp', ratio: 128, bytesPerToken: 2048 / 128 },
      reach: { glyph: 'compress', ratio: 128, local: 128, label: 'all of the 1 : 128 stream + 128 local' },
    })
  })

  it('csa-hca-hybrid (V4-Pro): no sliding lane when numSlidingLayers is 0', () => {
    const lanes = lanesFor(byId('deepseek-v4-pro'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['csa', 30], ['hca', 31]])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/ui/archGeometry.test.ts`
Expected: the five new cases FAIL with `lanesFor: variant not yet handled`.

- [ ] **Step 3: Implement the four cases**

In `src/ui/archGeometry.ts`, add `bytesOf` to the imports:

```ts
import { bytesOf } from '../engine/dtypes'
```

Add helpers after `windowReach`:

```ts
function latentCache(rank: number, rope: number, p: number): Lane['cache'] {
  return { glyph: 'latent', bytesPerToken: p, fixedBytes: 0, label: `latent ${rank} + RoPE key ${rope} · ${fmtBytes(p)}` }
}
const topkReach = (tokens: number, label: string, local?: number): Lane['reach'] =>
  ({ glyph: 'topk', tokens, label, ...(local === undefined ? {} : { local }) })
```

Add the cases before `default:`:

```ts
    case 'mla':
      return [lane('mla', m.layers, latentCache(att.kvLoraRank, att.qkRopeHeadDim, p), ALL_REACH)]
    case 'mla-dsa':
      return [lane('dsa', m.layers, latentCache(att.kvLoraRank, att.qkRopeHeadDim, p),
        topkReach(att.topK, `top-${att.topK} tokens`))]
    case 'msa-hybrid': {
      const idx = att.indexHeadDim * bytesOf(KV_REF_DTYPE)
      return [
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
        lane('msa', att.numSparseLayers, {
          glyph: 'kvidx', bytesPerToken: p + idx, fixedBytes: 0,
          label: `K + V ${fmtBytes(p)} + shared index key ${att.indexHeadDim} · ${fmtBytes(idx)}`,
        }, topkReach(att.topKBlocks * att.blockSize, `${att.topKBlocks} blocks × ${att.blockSize} tokens`)),
      ]
    }
    case 'csa-hca-hybrid': {
      const comp = (M: number): Lane['cache'] => ({
        glyph: 'kvcomp', ratio: M, bytesPerToken: p / M, fixedBytes: 0,
        label: `K + V ${fmtBytes(p)} per ${M} tokens · ${fmtBytes(p / M)}/token`,
      })
      const w = att.slidingWindow
      const lanes: Lane[] = []
      if (att.numSlidingLayers > 0) lanes.push(lane('window', att.numSlidingLayers, kvCache(m, p), windowReach(w)))
      lanes.push(lane('csa', att.numCsaLayers, comp(att.csaCompressionM),
        topkReach(att.csaTopK * att.csaCompressionM, `top-${att.csaTopK} of the 1 : ${att.csaCompressionM} stream + ${w} local`, w)))
      lanes.push(lane('hca', att.numHcaLayers, comp(att.hcaCompressionM), {
        glyph: 'compress', ratio: att.hcaCompressionM, local: w,
        label: `all of the 1 : ${att.hcaCompressionM} stream + ${w} local`,
      }))
      return lanes
    }
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run test/ui/archGeometry.test.ts && npm run check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ui/archGeometry.ts test/ui/archGeometry.test.ts
git commit -m "feat(info): geometry lanes for MLA, DSA, MSA and CSA/HCA attention"
```

---

### Task 5: Geometry lanes for state variants and the exhaustiveness check

**Files:**
- Modify: `src/ui/archGeometry.ts`
- Test: `test/ui/archGeometry.test.ts`

**Interfaces:**
- Produces: `lanesFor` handles `linear-mla-hybrid`, `delta-hybrid`, `mamba2-hybrid`; `default` becomes a `never` check.

- [ ] **Step 1: Write the failing tests**

Append to `test/ui/archGeometry.test.ts`:

```ts
describe('lanesFor — recurrent-state variants', () => {
  it('linear-mla-hybrid: KDA state lane then MLA latent lane', () => {
    const lanes = lanesFor(byId('kimi-linear'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['kda', 20], ['mla', 7]])
    expect(lanes[0]).toMatchObject({
      color: 'state',
      cache: { glyph: 'state', bytesPerToken: 0, fixedBytes: 32 * 128 * 128 * 2 },
      reach: { glyph: 'state', label: 'one state read' },
    })
    expect(lanes[0].cache.label).toBe('32 heads × 128² state · 1 MiB per layer')
    expect(lanes[1].cache).toMatchObject({ glyph: 'latent', bytesPerToken: (512 + 64) * 2 })
  })

  it('delta-hybrid: DeltaNet state lane then full-attention lane', () => {
    const lanes = lanesFor(byId('qwen3.5-397b-a17b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['delta', 45], ['full', 15]])
    expect(lanes[0].cache).toMatchObject({ glyph: 'state', fixedBytes: 64 * 128 * 128 * 2 })
    expect(lanes[0].cache.label).toBe('64 heads × 128² state · 2 MiB per layer')
    expect(lanes[1].cache).toMatchObject({ glyph: 'kv', bytesPerToken: 2048 })
  })

  it('mamba2-hybrid: Mamba state (fp32) lane, attention lane, FFN-only lane', () => {
    const lanes = lanesFor(byId('nemotron-3-nano-30b-a3b'))
    expect(lanes.map(l => [l.kind, l.count])).toEqual([['mamba', 23], ['full', 6], ['ffn', 23]])
    expect(lanes[0].cache).toMatchObject({ glyph: 'state', fixedBytes: 64 * 64 * 128 * 4 })
    expect(lanes[0].cache.label).toBe('64 heads × 64 × 128 fp32 state · 2 MiB per block')
    expect(lanes[2]).toMatchObject({ kind: 'ffn', color: 'none', cache: { glyph: 'none' }, reach: { glyph: 'none' } })
  })
})

describe('lanesFor — catalog-wide invariants', () => {
  it('handles every model, counts sum to layers, bytes are finite and non-negative', () => {
    for (const m of MODELS) {
      const lanes = lanesFor(m)
      expect(lanes.length, m.id).toBeGreaterThan(0)
      expect(lanes.reduce((n, l) => n + l.count, 0), m.id).toBe(m.layers)
      for (const l of lanes) {
        expect(l.count, `${m.id} ${l.kind}`).toBeGreaterThan(0)
        expect(Number.isFinite(l.cache.bytesPerToken) && l.cache.bytesPerToken >= 0, `${m.id} ${l.kind}`).toBe(true)
        expect(Number.isFinite(l.cache.fixedBytes) && l.cache.fixedBytes >= 0, `${m.id} ${l.kind}`).toBe(true)
      }
    }
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/ui/archGeometry.test.ts`
Expected: the three variant cases and the catalog-wide case FAIL with `variant not yet handled`.

- [ ] **Step 3: Implement the three cases and the never-check**

In `src/ui/archGeometry.ts`, add a helper after `topkReach`:

```ts
const STATE_REACH: Lane['reach'] = { glyph: 'state', label: 'one state read' }
function stateCache(fixedBytes: number, label: string): Lane['cache'] {
  return { glyph: 'state', bytesPerToken: 0, fixedBytes, label }
}
```

Add the cases and replace the `default` branch:

```ts
    case 'linear-mla-hybrid': {
      const per = att.numLinearHeads * att.linearHeadDim * att.linearHeadDim * bytesOf(KV_REF_DTYPE)
      return [
        lane('kda', att.numLinearLayers,
          stateCache(per, `${att.numLinearHeads} heads × ${att.linearHeadDim}² state · ${fmtBytes(per)} per layer`), STATE_REACH),
        lane('mla', att.numFullLayers, latentCache(att.kvLoraRank, att.qkRopeHeadDim, p), ALL_REACH),
      ]
    }
    case 'delta-hybrid': {
      const per = att.numDeltaNetHeads * att.deltaHeadDim * att.deltaHeadDim * bytesOf(KV_REF_DTYPE)
      return [
        lane('delta', att.numDeltaNetLayers,
          stateCache(per, `${att.numDeltaNetHeads} heads × ${att.deltaHeadDim}² state · ${fmtBytes(per)} per layer`), STATE_REACH),
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
      ]
    }
    case 'mamba2-hybrid': {
      // fp32 by config (mamba_ssm_cache_dtype), independent of the KV reference.
      const per = att.numMambaHeads * att.mambaHeadDim * att.ssmStateSize * 4
      return [
        lane('mamba', att.numMambaLayers,
          stateCache(per, `${att.numMambaHeads} heads × ${att.mambaHeadDim} × ${att.ssmStateSize} fp32 state · ${fmtBytes(per)} per block`), STATE_REACH),
        lane('full', att.numFullLayers, kvCache(m, p), ALL_REACH),
        lane('ffn', att.numFfnLayers, NONE_CACHE, NONE_REACH),
      ]
    }
    default: {
      const _exhaustive: never = att
      throw new Error(`lanesFor: unhandled attention variant ${(_exhaustive as { type: string }).type}`)
    }
```

The per-layer state bytes above are the per-layer factor of `linearAttentionStateBytes` / `deltaStateBytes` / `mambaStateBytes`; Task 6's cross-check ties them back to those functions so the two cannot drift.

- [ ] **Step 4: Run the tests and type check**

Run: `npx vitest run test/ui/archGeometry.test.ts && npm run check`
Expected: PASS. If `npm run check` reports the `never` assignment failing, a variant is missing a case; add it rather than loosening the check.

- [ ] **Step 5: Commit**

```bash
git add src/ui/archGeometry.ts test/ui/archGeometry.test.ts
git commit -m "feat(info): geometry lanes for KDA, DeltaNet and Mamba2 hybrids; exhaustive over AttentionConfig"
```

---

### Task 6: Head geometry, FFN, caption, `geometryModel`, engine cross-check

**Files:**
- Modify: `src/ui/archGeometry.ts`
- Test: `test/ui/archGeometry.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export interface HeadGeometry {
    kind: 'gqa' | 'latent' | 'state'
    numHeads: number; numKvHeads: number; headDim: number
    fullKvWidth?: number; latentWidth?: number
    state?: { heads: number; dim: number; inner: number }
  }
  export interface GeometryModel {
    layers: number; maxContext: number
    lanes: Lane[]; heads: HeadGeometry[]
    ffn: { type: 'dense'; hiddenDim: number; intermediateDim: number }
       | { type: 'moe'; routed: number; active: number; shared: number; activeRatio: number }
    caption: string
    schematicId: AttentionConfig['type']
  }
  export function headsFor(m: ModelArch): HeadGeometry[]
  export function captionFor(type: AttentionConfig['type']): string
  export function geometryModel(m: ModelArch): GeometryModel
  ```

- [ ] **Step 1: Write the failing tests**

Append to `test/ui/archGeometry.test.ts`. First replace the module import line with these three:

```ts
import { lanesFor, fmtBytes, headsFor, captionFor, geometryModel } from '../../src/ui/archGeometry'
import { kvBytesPerTokenAtContext, fixedStateBytes, KV_REF_DTYPE } from '../../src/ui/catalogMetrics'
import { kvBytesPerTokenPerLayer } from '../../src/engine/memory'
```

Then append:

```ts
describe('headsFor', () => {
  it('GQA model: one gqa entry', () => {
    expect(headsFor(byId('llama-3.1-8b'))).toEqual([
      { kind: 'gqa', numHeads: 32, numKvHeads: 8, headDim: 128 },
    ])
  })
  it('MLA model: latent entry with full-KV width vs latent width', () => {
    expect(headsFor(byId('deepseek-v3'))).toEqual([
      { kind: 'latent', numHeads: 128, numKvHeads: 128, headDim: 192, fullKvWidth: 128 * 192 * 2, latentWidth: 512 + 64 },
    ])
  })
  it('delta-hybrid: state entry then gqa entry', () => {
    expect(headsFor(byId('qwen3.5-397b-a17b'))).toEqual([
      { kind: 'state', numHeads: 32, numKvHeads: 2, headDim: 256, state: { heads: 64, dim: 128, inner: 128 } },
      { kind: 'gqa', numHeads: 32, numKvHeads: 2, headDim: 256 },
    ])
  })
  it('mamba2-hybrid: state inner dim is the SSM state size', () => {
    expect(headsFor(byId('nemotron-3-nano-30b-a3b'))[0].state).toEqual({ heads: 64, dim: 64, inner: 128 })
  })
  it('linear-mla-hybrid: state entry then latent entry', () => {
    expect(headsFor(byId('kimi-linear')).map(h => h.kind)).toEqual(['state', 'latent'])
  })
})

describe('captionFor', () => {
  it('every caption ends with the counts-not-order sentence', () => {
    for (const t of ['full', 'mla-dsa', 'mamba2-hybrid', 'partial'] as const) {
      expect(captionFor(t).endsWith('Bar lengths are counts, not order.')).toBe(true)
    }
  })
  it('sparse variants state the whole-cache-streamed simplification', () => {
    for (const t of ['mla-dsa', 'msa-hybrid', 'csa-hca-hybrid'] as const) {
      expect(captionFor(t)).toContain('A decode step streams the whole cache; selection lowers attention FLOPs, not bytes.')
    }
    expect(captionFor('msa-hybrid')).toContain('not sharded by TP')
    expect(captionFor('mla')).not.toContain('streams the whole cache')
  })
})

describe('geometryModel', () => {
  it('dense FFN carries hidden and intermediate dims', () => {
    const g = geometryModel(byId('llama-3.1-8b'))
    expect(g.ffn).toEqual({ type: 'dense', hiddenDim: 4096, intermediateDim: 14336 })
    expect(g.schematicId).toBe('full')
    expect(g.layers).toBe(32)
    expect(g.maxContext).toBe(131072)
  })
  it('MoE FFN carries routed/active/shared and the active ratio', () => {
    const g = geometryModel(byId('deepseek-v3'))
    expect(g.ffn).toEqual({ type: 'moe', routed: 256, active: 8, shared: 1, activeRatio: 37e9 / 671e9 })
  })

  it('lane bytes agree with the engine at max context for every catalog model', () => {
    for (const m of MODELS) {
      const g = geometryModel(m)
      const S = m.maxContext
      const p = kvBytesPerTokenPerLayer(m, KV_REF_DTYPE)
      let sum = 0
      let fixed = 0
      for (const l of g.lanes) {
        fixed += l.count * l.cache.fixedBytes
        if (l.cache.glyph === 'state' || l.cache.glyph === 'none') continue
        if (l.cache.glyph === 'kvcomp') {
          sum += l.count * (l.cache.bytesPerToken + p * Math.min(l.reach.local!, S) / S)
        } else if (l.reach.glyph === 'window') {
          sum += l.count * l.cache.bytesPerToken * Math.min(l.reach.tokens!, S) / S
        } else {
          sum += l.count * l.cache.bytesPerToken
        }
      }
      expect(sum, m.id).toBeCloseTo(kvBytesPerTokenAtContext(m, S), 6)
      expect(fixed, m.id).toBe(fixedStateBytes(m))
    }
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/ui/archGeometry.test.ts`
Expected: FAIL, `headsFor` / `captionFor` / `geometryModel` are not exported.

- [ ] **Step 3: Implement**

In `src/ui/archGeometry.ts`, extend the type import to `import type { ModelArch, AttentionConfig } from '../engine/types'` and append:

```ts
export interface HeadGeometry {
  kind: 'gqa' | 'latent' | 'state'
  numHeads: number
  numKvHeads: number
  headDim: number
  fullKvWidth?: number       // 'latent': what per-head K+V would be, in elements
  latentWidth?: number       // 'latent': kvLoraRank + qkRopeHeadDim
  state?: { heads: number; dim: number; inner: number }   // 'state': elements = heads × dim × inner
}

export function headsFor(m: ModelArch): HeadGeometry[] {
  const base = { numHeads: m.numHeads, numKvHeads: m.numKvHeads, headDim: m.headDim }
  const gqa: HeadGeometry = { kind: 'gqa', ...base }
  const latent = (rank: number, rope: number): HeadGeometry =>
    ({ kind: 'latent', ...base, fullKvWidth: m.numHeads * m.headDim * 2, latentWidth: rank + rope })
  const state = (heads: number, dim: number, inner: number): HeadGeometry =>
    ({ kind: 'state', ...base, state: { heads, dim, inner } })
  const att = m.attention
  switch (att.type) {
    case 'full': case 'sliding': case 'hybrid': case 'partial':
    case 'msa-hybrid': case 'csa-hca-hybrid':
      return [gqa]
    case 'mla': case 'mla-dsa':
      return [latent(att.kvLoraRank, att.qkRopeHeadDim)]
    case 'linear-mla-hybrid':
      return [state(att.numLinearHeads, att.linearHeadDim, att.linearHeadDim), latent(att.kvLoraRank, att.qkRopeHeadDim)]
    case 'delta-hybrid':
      return [state(att.numDeltaNetHeads, att.deltaHeadDim, att.deltaHeadDim), gqa]
    case 'mamba2-hybrid':
      return [state(att.numMambaHeads, att.mambaHeadDim, att.ssmStateSize), gqa]
    default: {
      const _exhaustive: never = att
      throw new Error(`headsFor: unhandled ${(_exhaustive as { type: string }).type}`)
    }
  }
}

const COUNTS_NOTE = 'Bar lengths are counts, not order.'
const SPARSE_NOTE = 'A decode step streams the whole cache; selection lowers attention FLOPs, not bytes. The index branch that selects is not costed.'

export function captionFor(type: AttentionConfig['type']): string {
  switch (type) {
    case 'mla-dsa': case 'csa-hca-hybrid':
      return `${SPARSE_NOTE} ${COUNTS_NOTE}`
    case 'msa-hybrid':
      return `${SPARSE_NOTE} The shared index key is cached alongside K and V and is not sharded by TP. ${COUNTS_NOTE}`
    case 'mamba2-hybrid':
      return `Attention, Mamba2 and FFN are separate entries in the block count. SSM state is held in fp32. ${COUNTS_NOTE}`
    case 'partial':
      return `NAS pruning removed attention from the grey blocks; their FFN widths vary and are absorbed by the parameter count. ${COUNTS_NOTE}`
    case 'full': case 'sliding': case 'hybrid': case 'mla':
    case 'linear-mla-hybrid': case 'delta-hybrid':
      return COUNTS_NOTE
    default: {
      const _exhaustive: never = type
      throw new Error(`captionFor: unhandled ${_exhaustive as string}`)
    }
  }
}

export interface GeometryModel {
  layers: number
  maxContext: number
  lanes: Lane[]
  heads: HeadGeometry[]
  ffn:
    | { type: 'dense'; hiddenDim: number; intermediateDim: number }
    | { type: 'moe'; routed: number; active: number; shared: number; activeRatio: number }
  caption: string
  schematicId: AttentionConfig['type']
}

export function geometryModel(m: ModelArch): GeometryModel {
  const a = m.architecture
  return {
    layers: m.layers,
    maxContext: m.maxContext,
    lanes: lanesFor(m),
    heads: headsFor(m),
    ffn: a.type === 'moe'
      ? { type: 'moe', routed: a.numExperts, active: a.numExpertsActive, shared: a.numSharedExperts, activeRatio: a.activeParamCount / m.paramCount }
      : { type: 'dense', hiddenDim: m.hiddenDim, intermediateDim: m.intermediateDim },
    caption: captionFor(m.attention.type),
    schematicId: m.attention.type,
  }
}
```

- [ ] **Step 4: Run the tests and type check**

Run: `npx vitest run test/ui/archGeometry.test.ts && npm run check`
Expected: PASS. If the cross-check fails for one model, the lane table in Tasks 3–5 disagrees with the engine for that variant; fix the lane, never the test.

- [ ] **Step 5: Commit**

```bash
git add src/ui/archGeometry.ts test/ui/archGeometry.test.ts
git commit -m "feat(info): geometryModel with head geometry, FFN, captions and engine cross-check"
```

---

### Task 7: Glyph components

**Files:**
- Create: `src/ui/figure/CacheGlyph.svelte`
- Create: `src/ui/figure/ReachGlyph.svelte`

**Interfaces:**
- Produces: two SVG-fragment components.
  - `CacheGlyph` props: `glyph: CacheGlyph`, `color: string`, `x: number`, `y: number`, `w = 64`, `ratio: number | undefined`.
  - `ReachGlyph` props: `glyph: ReachGlyph`, `color: string`, `x: number`, `y: number`, `w: number`, `frac = 0` (share of the sequence: w/S or k/S), `local = 0` (share for the local sliver), `endLabel: string`.
- No vitest; the check is `npm run check` plus rendering in Task 8.

- [ ] **Step 1: Create `CacheGlyph.svelte`**

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import type { CacheGlyph } from './types'
  export let glyph: CacheGlyph
  export let color: string
  export let x: number
  export let y: number
  export let w = 64
  export let ratio: number | undefined = undefined
</script>

{#if glyph === 'kv' || glyph === 'kvidx' || glyph === 'kvcomp'}
  {@const kw = glyph === 'kvidx' ? w - 10 : w}
  <rect {x} {y} width={kw} height="7" rx="1" fill={color} />
  <rect {x} y={y + 9} width={kw} height="7" rx="1" fill={color} opacity="0.55" />
  {#if glyph === 'kvidx'}
    <rect x={x + kw + 3} {y} width="7" height="16" rx="1" fill={color} opacity="0.35" />
  {/if}
  {#if glyph === 'kvcomp'}
    <text x={x + w / 2} y={y + 12.5} font-size="10" text-anchor="middle" fill="#fff" font-weight="600">÷{ratio}</text>
  {/if}
{:else if glyph === 'latent'}
  <rect {x} y={y + 3} width={w * 0.68} height="10" rx="1" fill={color} />
  <rect x={x + w * 0.68 + 3} y={y + 3} width={w * 0.32 - 3} height="10" rx="1" fill={color} opacity="0.45" />
{:else if glyph === 'state'}
  <rect {x} {y} width="16" height="16" rx="1" fill="none" stroke={color} stroke-width="1.5" />
  <path d="M{x} {y + 8}h16M{x + 8} {y}v16" stroke={color} stroke-width="1" />
{:else}
  <rect {x} y={y + 2} width={w} height="12" rx="1" fill="none" stroke={color} stroke-width="1.5" stroke-dasharray="3 2" />
{/if}
```

- [ ] **Step 2: Create `ReachGlyph.svelte`**

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import type { ReachGlyph } from './types'
  export let glyph: ReachGlyph
  export let color: string
  export let x: number
  export let y: number
  export let w: number
  export let frac = 0
  export let local = 0
  export let endLabel: string
  // Scattered block positions for top-k: deterministic so the figure is stable.
  const SPOTS = [0.08, 0.27, 0.46, 0.68, 0.9]
  $: localW = local > 0 ? Math.max(3, w * local) : 0
</script>

<line x1={x} y1={y + 6} x2={x + w} y2={y + 6} stroke="#d6d9e2" stroke-width="1" />
{#if glyph === 'all'}
  <rect {x} y={y + 2} width={w} height="8" fill={color} />
{:else if glyph === 'window'}
  {@const ww = Math.max(4, w * frac)}
  <rect x={x + w - ww} y={y + 2} width={ww} height="8" fill={color} />
{:else if glyph === 'topk'}
  {@const bw = Math.max(6, w * frac) / SPOTS.length}
  {#each SPOTS as p}
    <rect x={x + p * (w - bw)} y={y + 2} width={Math.max(2, bw)} height="8" fill={color} />
  {/each}
  {#if localW > 0}<rect x={x + w - localW} y={y + 2} width={localW} height="8" fill={color} opacity="0.6" />{/if}
{:else if glyph === 'compress'}
  <rect {x} y={y + 4} width={w} height="4" fill={color} opacity="0.45" />
  {#if localW > 0}<rect x={x + w - localW} y={y + 2} width={localW} height="8" fill={color} opacity="0.6" />{/if}
{:else if glyph === 'state'}
  <circle cx={x + w - 5} cy={y + 6} r="4.5" fill={color} />
{:else}
  <line x1={x} y1={y + 6} x2={x + w} y2={y + 6} stroke={color} stroke-width="1.5" stroke-dasharray="2 3" />
{/if}
<text {x} y={y + 22} font-size="9.5" fill="#777">tok 0</text>
<text x={x + w} y={y + 22} font-size="9.5" fill="#777" text-anchor="end">{endLabel}</text>
```

- [ ] **Step 3: Type check**

Run: `npm run check`
Expected: PASS (unused components are still compiled by svelte-check).

- [ ] **Step 4: Commit**

```bash
git add src/ui/figure/CacheGlyph.svelte src/ui/figure/ReachGlyph.svelte
git commit -m "feat(info): cache and reach glyph components"
```

---

### Task 8: `ArchGeometry.svelte` and spec-sheet embed

**Files:**
- Create: `src/ui/ArchGeometry.svelte`
- Modify: `src/ui/ModelSpecSheet.svelte`

**Interfaces:**
- Consumes: `GeometryModel`, `fmtBytes` (Task 6); `KIND_COLORS`, `EXPERT_COLOR` (Task 3); glyph components (Task 7); `modelMetrics(m).attentionLabel` (existing).
- Produces: `ArchGeometry` with prop `geometry: GeometryModel`.

- [ ] **Step 1: Create `src/ui/ArchGeometry.svelte`**

```svelte
<!-- Per-model geometry figure: five stacked panels drawn to scale from a
     GeometryModel. No math here; archGeometry.ts owns the numbers. -->
<script lang="ts">
  import type { GeometryModel } from './archGeometry'
  import { fmtBytes } from './archGeometry'
  import { KIND_COLORS, EXPERT_COLOR } from './figure/types'
  import CacheGlyph from './figure/CacheGlyph.svelte'
  import ReachGlyph from './figure/ReachGlyph.svelte'

  export let geometry: GeometryModel

  const W = 640, LX = 14, BX = 110, BW = W - BX - 14
  const CACHE_SCALE = 8192   // bytes per full bar width; longer bars clip with the value printed
  const ROW = 24

  $: g = geometry
  $: S = g.maxContext
  $: endLabel = S >= 1048576 ? `${S / 1048576}M` : `${Math.round(S / 1024)}K`
  // Panel origins. Heights depend on lane count and head-geometry entries.
  $: y1 = 0
  $: y2 = y1 + 44
  $: y3 = y2 + 24 + g.heads.length * 40
  $: y4 = y3 + 20 + g.lanes.length * ROW
  $: y5 = y4 + 20 + g.lanes.length * 36
  $: H = y5 + 70
  $: label = `${g.layers} blocks: ${g.lanes.map(l => `${l.count} ${l.label}`).join(', ')}. ` +
    (g.ffn.type === 'moe' ? `${g.ffn.routed} routed experts, ${g.ffn.active} active, ${g.ffn.shared} shared.` : 'Dense FFN.')

  function fracOf(l: GeometryModel['lanes'][number]): number {
    return l.reach.tokens !== undefined ? Math.min(1, l.reach.tokens / S) : 0
  }
  function localOf(l: GeometryModel['lanes'][number]): number {
    return l.reach.local !== undefined ? Math.min(1, l.reach.local / S) : 0
  }
  function stackX(i: number): number {
    return BX + g.lanes.slice(0, i).reduce((acc, l) => acc + BW * l.count / g.layers, 0)
  }
</script>

<svg viewBox="0 0 {W} {H}" role="img" aria-label={label}>
  <!-- Panel 1: block stack -->
  <text x={LX} y={y1 + 12} class="lbl">BLOCKS</text>
  <text x={BX + BW} y={y1 + 12} class="lbl" text-anchor="end">{g.layers} total · counts, not order</text>
  {#each g.lanes as l, i}
    {@const w = BW * l.count / g.layers}
    <rect x={stackX(i)} y={y1 + 18} width={Math.max(w - 1.5, 1)} height="14" rx="1" fill={KIND_COLORS[l.color]} />
    {#if w > 26}<text x={stackX(i) + 4} y={y1 + 29} font-size="10.5" fill="#fff" font-weight="600">{l.count}</text>{/if}
  {/each}

  <!-- Panel 2: head geometry -->
  <text x={LX} y={y2 + 12} class="lbl">HEADS</text>
  {#each g.heads as h, i}
    {@const hy = y2 + 18 + i * 40}
    {#if h.kind === 'gqa'}
      {@const groups = Math.min(h.numKvHeads, 8)}
      {@const perGroup = Math.min(h.numHeads / h.numKvHeads, 16)}
      {#each Array.from({ length: groups }) as _, gi}
        {@const gx = BX + gi * 58}
        <rect x={gx} y={hy} width="10" height="10" fill={KIND_COLORS.full} />
        {#each Array.from({ length: perGroup }) as _, qi}
          <rect x={gx + 13 + (qi % 8) * 5} y={hy + Math.floor(qi / 8) * 5} width="4" height="4" fill={KIND_COLORS.full} opacity="0.45" />
        {/each}
      {/each}
      <text x={BX} y={hy + 26} class="txt">
        {h.numHeads} query heads on {h.numKvHeads} KV heads · {h.numHeads / h.numKvHeads} : 1 · head dim {h.headDim}{#if h.numKvHeads > 8} · first 8 groups drawn{/if}
      </text>
    {:else if h.kind === 'latent'}
      {@const scale = 300 / (h.fullKvWidth ?? 1)}
      <rect x={BX} y={hy} width={(h.fullKvWidth ?? 0) * scale} height="8" fill={KIND_COLORS.full} opacity="0.3" />
      <rect x={BX} y={hy + 11} width={Math.max(3, (h.latentWidth ?? 0) * scale)} height="8" fill={KIND_COLORS.full} />
      <text x={BX} y={hy + 32} class="txt">
        per-head K+V would be {h.fullKvWidth} elements; the cached latent is {h.latentWidth} · {((h.fullKvWidth ?? 1) / (h.latentWidth ?? 1)).toFixed(0)}× smaller
      </text>
    {:else if h.state}
      {@const rw = Math.min(120, h.state.dim)}
      {@const rh = Math.min(40, h.state.inner * 40 / Math.max(h.state.inner, h.state.dim))}
      <rect x={BX} y={hy} width={rw} height={rh} fill="none" stroke={KIND_COLORS.state} stroke-width="1.5" />
      <text x={BX + rw + 8} y={hy + 12} class="txt">× {h.state.heads} heads</text>
      <text x={BX} y={hy + rh + 14} class="txt">
        state {h.state.heads} × {h.state.dim} × {h.state.inner} = {(h.state.heads * h.state.dim * h.state.inner).toLocaleString()} elements per layer
      </text>
    {/if}
  {/each}

  <!-- Panel 3: cache row per token -->
  <text x={LX} y={y3 + 12} class="lbl">PER TOKEN</text>
  <text x={BX + BW} y={y3 + 12} class="lbl" text-anchor="end">bar = {fmtBytes(CACHE_SCALE)} per layer</text>
  {#each g.lanes as l, i}
    {@const ry = y3 + 18 + i * ROW}
    {@const bw = Math.min(BW, BW * l.cache.bytesPerToken / CACHE_SCALE)}
    <rect x={BX - 8} y={ry + 2} width="5" height="12" fill={KIND_COLORS[l.color]} />
    {#if l.cache.glyph === 'state' || l.cache.glyph === 'none'}
      <CacheGlyph glyph={l.cache.glyph} color={KIND_COLORS[l.color]} x={BX} y={ry} w={40} />
    {:else}
      <rect x={BX} y={ry + 3} width={Math.max(2, bw)} height="10" fill={KIND_COLORS[l.color]} />
    {/if}
    <text x={BX + Math.max(48, bw) + 8} y={ry + 12} class="txt">{l.label} × {l.count}: {l.cache.label}</text>
  {/each}

  <!-- Panel 4: reach at trained context -->
  <text x={LX} y={y4 + 12} class="lbl">READS AT DECODE</text>
  {#each g.lanes as l, i}
    {@const ry = y4 + 18 + i * 36}
    <text x={BX} y={ry - 2} class="txt">{l.label}: {l.reach.label}</text>
    <ReachGlyph glyph={l.reach.glyph} color={KIND_COLORS[l.color]} x={BX} y={ry + 2} w={BW} frac={fracOf(l)} local={localOf(l)} {endLabel} />
  {/each}

  <!-- Panel 5: FFN -->
  <text x={LX} y={y5 + 12} class="lbl">FFN</text>
  {#if g.ffn.type === 'dense'}
    {@const scale = 300 / Math.max(g.ffn.hiddenDim, g.ffn.intermediateDim)}
    <rect x={BX} y={y5 + 18} width={g.ffn.hiddenDim * scale} height="8" fill={EXPERT_COLOR} opacity="0.45" />
    <rect x={BX} y={y5 + 29} width={g.ffn.intermediateDim * scale} height="8" fill={EXPERT_COLOR} />
    <text x={BX} y={y5 + 50} class="txt">dense · hidden {g.ffn.hiddenDim} → intermediate {g.ffn.intermediateDim} · every token pays the full width</text>
  {:else}
    {@const aw = Math.max(4, BW * g.ffn.active / g.ffn.routed)}
    <rect x={BX} y={y5 + 18} width={BW} height="16" rx="2" fill="none" stroke={EXPERT_COLOR} stroke-width="1.5" />
    <rect x={BX + 1} y={y5 + 19} width={aw} height="14" rx="1" fill={EXPERT_COLOR} />
    {#each Array.from({ length: g.ffn.shared }) as _, si}
      <rect x={BX + si * 22} y={y5 + 38} width="18" height="8" rx="1" fill={EXPERT_COLOR} opacity="0.55" />
    {/each}
    <text x={BX} y={y5 + 60} class="txt">
      {g.ffn.routed} routed experts · {g.ffn.active} active{#if g.ffn.shared} + {g.ffn.shared} shared, always on{/if} · {(g.ffn.activeRatio * 100).toFixed(1)}% of parameters active per token
    </text>
  {/if}
</svg>

<style>
  svg { display: block; width: 100%; min-width: 520px; height: auto; font-family: inherit; }
  .lbl { font-size: 10.5px; letter-spacing: 0.06em; fill: #666; }
  .txt { font-size: 10.5px; fill: #222; }
</style>
```

- [ ] **Step 2: Embed in the spec sheet**

In `src/ui/ModelSpecSheet.svelte` script block add:

```ts
  import ArchGeometry from './ArchGeometry.svelte'
  import { geometryModel } from './archGeometry'
  $: geo = geometryModel(model)
```

Insert after the Design `<dl>` (before the Scale rule):

```svelte
  <div class="rule"></div>
  <h3>Architecture <span class="ref">(fp16 KV reference, reach at max context)</span></h3>
  <div class="figure"><ArchGeometry geometry={geo} /></div>
  <p class="caption">
    {geo.caption}
    <a href={`#info/arch/${geo.schematicId}`}>How {m.attentionLabel} works</a>
  </p>
```

Add to the style block:

```css
  .figure { overflow-x: auto; margin: 0.4rem 0; }
  .caption { margin: 0.2rem 0 0; font-size: 0.85rem; color: #555; }
  .caption a { color: #1a4f8a; }
```

- [ ] **Step 3: Type check, tests, and look at it**

Run: `npm run check && npm test`
Expected: PASS.

Run: `npm run dev`, open `#info/model/deepseek-v3.2`, `#info/model/qwen3.5-397b-a17b`, `#info/model/nemotron-3-nano-30b-a3b`, `#info/model/deepseek-v4-flash`, `#info/model/llama-3.1-8b`. Check: no NaN, no clipped labels, lanes sum matches the "total" text, the latent bar is visibly shorter than the full-KV bar on DeepSeek, the MoE active slice is visible on every MoE model. Fix layout issues in the component before committing; do not touch `archGeometry.ts` for layout.

- [ ] **Step 4: Commit**

```bash
git add src/ui/ArchGeometry.svelte src/ui/ModelSpecSheet.svelte
git commit -m "feat(info): per-model architecture geometry figure on the spec sheet"
```

---

### Task 9: Schematic registry, sources, and tests

**Files:**
- Modify: `src/data/sources.ts`
- Create: `src/ui/schematics/meta.ts`
- Test: `test/ui/archSchematics.test.ts`

**Interfaces:**
- Produces (`src/ui/schematics/meta.ts`):
  ```ts
  export type SchematicId = AttentionConfig['type'] | ArchitectureConfig['type']
  export type SchematicGroup = 'sequence' | 'window' | 'sparse' | 'state' | 'ffn'
  export type Growth = 'unbounded' | 'bounded' | 'constant' | 'none'
  export interface SchematicMeta {
    id: SchematicId; title: string; group: SchematicGroup; summary: string
    leaves: string; reads: string; growth: Growth
    sources: (keyof typeof SOURCES)[]
  }
  export const ATTENTION_SCHEMATICS: Record<AttentionConfig['type'], SchematicMeta>
  export const FFN_SCHEMATICS: Record<ArchitectureConfig['type'], SchematicMeta>
  export const SCHEMATIC_GROUPS: { id: SchematicGroup; title: string; ids: SchematicId[] }[]
  export function schematicFor(id: string): SchematicMeta | undefined
  export function modelsUsing(id: SchematicId, models: ModelArch[]): ModelArch[]
  ```

- [ ] **Step 1: Verify and add the paper sources**

For each row below, fetch the URL (WebFetch) and confirm the page title matches the title column before writing the entry. If a fetch fails or the title differs, search by title (WebSearch), use the URL found, and record what changed in the commit message. Do not write an entry you did not fetch. If a paper cannot be found at all, drop that key from the schematic's `sources` and cite the model's HuggingFace model card README instead, with its fetched URL.

| Key | Title to confirm | Candidate URL |
|---|---|---|
| `arxiv-2305-13245` | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | https://arxiv.org/abs/2305.13245 |
| `arxiv-2310-06825` | Mistral 7B | https://arxiv.org/abs/2310.06825 |
| `arxiv-2503-19786` | Gemma 3 Technical Report | https://arxiv.org/abs/2503.19786 |
| `arxiv-2411-19146` | Puzzle: Distillation-Based NAS for Inference-Optimized LLMs | https://arxiv.org/abs/2411.19146 |
| `arxiv-2405-04434` | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | https://arxiv.org/abs/2405.04434 |
| `deepseek-v3-2-exp` | DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention | search: official deepseek-ai GitHub release or arXiv listing |
| `arxiv-2606-13392` | MiniMax M3 / MiniMax Sparse Attention (the id is cited in `src/data/models.ts` next to the M3 entry) | https://arxiv.org/abs/2606.13392 |
| `deepseek-v4-report` | DeepSeek-V4 technical report (CSA / HCA) | search: official deepseek-ai release; fall back to the V4-Flash model card |
| `arxiv-2510-26692` | Kimi Linear: An Expressive, Efficient Attention Architecture | https://arxiv.org/abs/2510.26692 |
| `arxiv-2412-06464` | Gated Delta Networks: Improving Mamba2 with Delta Rule | https://arxiv.org/abs/2412.06464 |
| `arxiv-2405-21060` | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | https://arxiv.org/abs/2405.21060 |
| `arxiv-2504-03624` | Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models | https://arxiv.org/abs/2504.03624 |
| `arxiv-2101-03906` | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | https://arxiv.org/abs/2101.03906 |
| `arxiv-2401-06066` | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | https://arxiv.org/abs/2401.06066 |

Add each confirmed entry to the `SOURCES` object in `src/data/sources.ts` in the same `{ title, url }` shape as the existing entries, under a comment line `// Architecture papers (schematic citations)`.

- [ ] **Step 2: Write the failing tests**

Create `test/ui/archSchematics.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { MODELS } from '../../src/data'
import { SOURCES } from '../../src/data/sources'
import {
  ATTENTION_SCHEMATICS, FFN_SCHEMATICS, SCHEMATIC_GROUPS, schematicFor, modelsUsing,
} from '../../src/ui/schematics/meta'
// @ts-expect-error — plain .mjs, no .d.ts
import { extractDiscriminants } from '../../.claude/hooks/check-skill-sync.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const typesSrc = readFileSync(resolve(__dirname, '../../src/engine/types.ts'), 'utf8')

describe('schematic registry', () => {
  it('has one attention schematic per AttentionConfig discriminant', () => {
    const inTypes = (extractDiscriminants(typesSrc, 'AttentionConfig') as string[]).sort()
    expect(Object.keys(ATTENTION_SCHEMATICS).sort()).toEqual(inTypes)
  })
  it('has one FFN schematic per ArchitectureConfig discriminant', () => {
    const inTypes = (extractDiscriminants(typesSrc, 'ArchitectureConfig') as string[]).sort()
    expect(Object.keys(FFN_SCHEMATICS).sort()).toEqual(inTypes)
  })
  it('every schematic cites at least one source that resolves in SOURCES', () => {
    for (const s of [...Object.values(ATTENTION_SCHEMATICS), ...Object.values(FFN_SCHEMATICS)]) {
      expect(s.sources.length, s.id).toBeGreaterThan(0)
      for (const k of s.sources) expect(SOURCES[k], `${s.id} → ${k}`).toBeDefined()
    }
  })
  it('groups cover every schematic exactly once', () => {
    const all = SCHEMATIC_GROUPS.flatMap(g => g.ids).sort()
    const expected = [...Object.keys(ATTENTION_SCHEMATICS), ...Object.keys(FFN_SCHEMATICS)].sort()
    expect(all).toEqual(expected)
  })
  it('schematicFor resolves ids and rejects unknowns', () => {
    expect(schematicFor('mla')?.id).toBe('mla')
    expect(schematicFor('moe')?.id).toBe('moe')
    expect(schematicFor('bogus')).toBeUndefined()
  })
  it('modelsUsing lists the catalog models on a variant', () => {
    const dsa = modelsUsing('mla-dsa', MODELS).map(m => m.id)
    expect(dsa).toContain('deepseek-v3.2')
    expect(dsa).toContain('glm-5')
    expect(dsa).not.toContain('deepseek-v3')
    const moe = modelsUsing('moe', MODELS).map(m => m.id)
    expect(moe).toContain('deepseek-v3')
    expect(moe).not.toContain('llama-3.1-8b')
  })
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `npx vitest run test/ui/archSchematics.test.ts`
Expected: FAIL, module not found.

- [ ] **Step 4: Create `src/ui/schematics/meta.ts`**

Use only source keys that Step 1 actually added. Where a key below was dropped in Step 1, substitute the model-card key you added.

```ts
// Registry of educational architecture schematics, keyed by the engine's
// discriminants so a new variant is a type error until it has a drawing.
// No Svelte imports here: components.ts maps ids to components.
import type { AttentionConfig, ArchitectureConfig, ModelArch } from '../../engine/types'
import { SOURCES } from '../../data/sources'

export type SchematicId = AttentionConfig['type'] | ArchitectureConfig['type']
export type SchematicGroup = 'sequence' | 'window' | 'sparse' | 'state' | 'ffn'
export type Growth = 'unbounded' | 'bounded' | 'constant' | 'none'

export interface SchematicMeta {
  id: SchematicId
  title: string
  group: SchematicGroup
  summary: string
  leaves: string       // what one token leaves behind, per layer of this kind
  reads: string        // what a decode step reads back
  growth: Growth       // how memory scales with context
  sources: (keyof typeof SOURCES)[]
}

export const ATTENTION_SCHEMATICS: Record<AttentionConfig['type'], SchematicMeta> = {
  'full': {
    id: 'full', title: 'Full attention (MHA / GQA)', group: 'sequence',
    summary: 'Every layer projects the token to query heads and to a smaller set of KV heads, appends one K and one V row per KV head to the cache, and attends over every cached token. Grouped-query attention shares each KV head across several query heads, which shrinks the cache without changing what is read.',
    leaves: 'K and V rows, one per KV head', reads: 'every cached token', growth: 'unbounded',
    sources: ['arxiv-2305-13245'],
  },
  'sliding': {
    id: 'sliding', title: 'Sliding-window attention', group: 'window',
    summary: 'The same block as full attention, but the cache is a ring of w slots. A token older than w is dropped, so the cache stops growing and each layer sees only the trailing window; depth stacks windows into a wider receptive field.',
    leaves: 'K and V rows, kept for w tokens', reads: 'the trailing w tokens', growth: 'bounded',
    sources: ['arxiv-2310-06825'],
  },
  'hybrid': {
    id: 'hybrid', title: 'Sliding / global hybrid', group: 'window',
    summary: 'Sliding layers are interleaved with a few global layers. The sliding caches are bounded; the global caches grow with context and are what let the model see the whole prompt.',
    leaves: 'K and V rows; bounded on sliding layers, growing on global layers', reads: 'window on most layers, everything on global layers', growth: 'unbounded',
    sources: ['arxiv-2503-19786'],
  },
  'partial': {
    id: 'partial', title: 'Partial attention (NAS-pruned)', group: 'sequence',
    summary: 'Neural architecture search removed the attention path from some blocks, leaving only their FFN. The surviving blocks are ordinary full attention with the parent model’s head geometry.',
    leaves: 'K and V rows on attending blocks; nothing on pruned blocks', reads: 'every cached token, on attending blocks', growth: 'unbounded',
    sources: ['arxiv-2411-19146'],
  },
  'mla': {
    id: 'mla', title: 'Multi-head latent attention (MLA)', group: 'sequence',
    summary: 'The token is down-projected to one latent vector plus a small RoPE key, and only those are cached. At use, the latent is up-projected into all heads’ keys and values. The cache row is the latent, not the heads, so it is tens of times smaller than per-head K and V at the same head count.',
    leaves: 'one latent vector and one RoPE key', reads: 'every cached token', growth: 'unbounded',
    sources: ['arxiv-2405-04434'],
  },
  'mla-dsa': {
    id: 'mla-dsa', title: 'MLA with DeepSeek sparse attention (DSA)', group: 'sparse',
    summary: 'The MLA block gains an indexer branch that scores every cached token and keeps the top k. The main attention reads only those k. The cache is unchanged; what stops growing with context is the attention compute.',
    leaves: 'one latent vector and one RoPE key', reads: 'the top-k scored tokens', growth: 'unbounded',
    sources: ['deepseek-v3-2-exp', 'arxiv-2405-04434'],
  },
  'msa-hybrid': {
    id: 'msa-hybrid', title: 'MiniMax sparse attention (MSA)', group: 'sparse',
    summary: 'A GQA block plus one index-key head, shared by every GQA group, cached per token. Blocks of the sequence are scored per group and the top-k blocks are attended. A few layers stay full attention.',
    leaves: 'K and V rows plus one shared index key', reads: 'the top-k scored blocks', growth: 'unbounded',
    sources: ['arxiv-2606-13392'],
  },
  'csa-hca-hybrid': {
    id: 'csa-hca-hybrid', title: 'Compressed sparse and heavily compressed attention (CSA / HCA)', group: 'sparse',
    summary: 'Two block types store the past at reduced resolution. CSA compresses every M tokens into one cache entry and attends a top-k of those; HCA compresses 128 : 1 and attends all of them. Both add a short local-window branch so recent tokens are seen at full resolution.',
    leaves: 'one compressed entry per M tokens, plus a local window', reads: 'top-k compressed entries (CSA) or all of them (HCA), plus the local window', growth: 'unbounded',
    sources: ['deepseek-v4-report'],
  },
  'linear-mla-hybrid': {
    id: 'linear-mla-hybrid', title: 'Kimi delta attention with MLA (linear / MLA hybrid)', group: 'state',
    summary: 'Most layers are KDA: a fixed state matrix per head is updated by a gated delta rule and read by the query. Nothing is stored per token. Every fourth layer is MLA and keeps an ordinary growing latent cache.',
    leaves: 'nothing on KDA layers; a latent on MLA layers', reads: 'one state matrix on KDA layers; everything on MLA layers', growth: 'unbounded',
    sources: ['arxiv-2510-26692'],
  },
  'delta-hybrid': {
    id: 'delta-hybrid', title: 'Gated DeltaNet with gated attention', group: 'state',
    summary: 'Three of every four layers are Gated DeltaNet: a per-head state matrix updated by a gated delta rule. The remaining layers are gated attention with partial RoPE and a small KV cache.',
    leaves: 'nothing on DeltaNet layers; K and V rows on attention layers', reads: 'one state matrix, or every cached token', growth: 'unbounded',
    sources: ['arxiv-2412-06464'],
  },
  'mamba2-hybrid': {
    id: 'mamba2-hybrid', title: 'Mamba2 hybrid (NemotronH)', group: 'state',
    summary: 'Mamba2 blocks run a convolution then a selective state-space scan against a per-head state held in fp32. Attention blocks and FFN-only blocks are separate entries in the stack, not paired with the Mamba2 blocks.',
    leaves: 'nothing on Mamba2 blocks; K and V rows on attention blocks', reads: 'one SSM state, or every cached token', growth: 'unbounded',
    sources: ['arxiv-2405-21060', 'arxiv-2504-03624'],
  },
}

export const FFN_SCHEMATICS: Record<ArchitectureConfig['type'], SchematicMeta> = {
  'dense': {
    id: 'dense', title: 'Dense FFN', group: 'ffn',
    summary: 'Up, gate and down projections. Every token pays the full intermediate width, and a decode step streams all of it from memory.',
    leaves: 'nothing', reads: 'the whole FFN weight', growth: 'none',
    sources: ['arxiv-2305-13245'],
  },
  'moe': {
    id: 'moe', title: 'Mixture of experts', group: 'ffn',
    summary: 'A router picks k of N routed experts per token; shared experts run for every token. Only the active experts’ weights are streamed per decode step, so the active parameter count, not the total, sets the weight-side cost.',
    leaves: 'nothing', reads: 'k routed experts plus the shared experts', growth: 'none',
    sources: ['arxiv-2101-03906', 'arxiv-2401-06066'],
  },
}

export const SCHEMATIC_GROUPS: { id: SchematicGroup; title: string; ids: SchematicId[] }[] = [
  { id: 'sequence', title: 'Every layer keeps every token', ids: ['full', 'partial', 'mla'] },
  { id: 'window', title: 'Windows cap the cache', ids: ['sliding', 'hybrid'] },
  { id: 'sparse', title: 'Keep everything, read a selection', ids: ['mla-dsa', 'msa-hybrid', 'csa-hca-hybrid'] },
  { id: 'state', title: 'Fixed state instead of a cache', ids: ['linear-mla-hybrid', 'delta-hybrid', 'mamba2-hybrid'] },
  { id: 'ffn', title: 'Feed-forward', ids: ['dense', 'moe'] },
]

const ALL: Record<string, SchematicMeta> = { ...ATTENTION_SCHEMATICS, ...FFN_SCHEMATICS }

export function schematicFor(id: string): SchematicMeta | undefined {
  return ALL[id]
}

export function modelsUsing(id: SchematicId, models: ModelArch[]): ModelArch[] {
  return models.filter(m => m.attention.type === id || m.architecture.type === id)
}
```

- [ ] **Step 5: Run the tests and type check**

Run: `npx vitest run test/ui/archSchematics.test.ts && npm run check`
Expected: PASS. A `sources` key that Step 1 did not add is a type error here; fix the meta, not the registry.

- [ ] **Step 6: Commit**

```bash
git add src/data/sources.ts src/ui/schematics/meta.ts test/ui/archSchematics.test.ts
git commit -m "feat(info): schematic registry keyed by attention/architecture discriminants, with cited sources"
```

---

### Task 10: Schematic frame, drawing parts, and the KV-cached schematics

**Files:**
- Create: `src/ui/schematics/parts/Box.svelte`, `Mem.svelte`, `Arrow.svelte`
- Create: `src/ui/schematics/SchematicFrame.svelte`
- Create: `src/ui/schematics/FullSchematic.svelte`, `SlidingSchematic.svelte`, `HybridSchematic.svelte`, `PartialSchematic.svelte`
- Create: `src/ui/schematics/components.ts` (partial map; completed in Task 12)

**Interfaces:**
- Consumes: `SchematicMeta`, `SchematicId` (Task 9); `KIND_COLORS` (Task 3).
- Produces:
  - `Box` props: `x, y, w, h: number`, `label: string`, `sub = ''`.
  - `Mem` props: `x, y, w, h: number`, `label: string`, `color: LaneColor`, `dashed = false`, `sub = ''`.
  - `Arrow` props: `x1, y1, x2, y2: number`, `label = ''`.
  - `SchematicFrame` props: `meta: SchematicMeta`, `component: Component`. Renders `<svg viewBox="0 0 640 262">` with `<marker id="arr">`, the schematic in the top 190px, and the footprint strip beneath.
  - `SCHEMATIC_COMPONENTS: Partial<Record<SchematicId, Component>>` until Task 12 makes it `Record`.

- [ ] **Step 1: Create the parts**

`src/ui/schematics/parts/Box.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  export let x: number
  export let y: number
  export let w: number
  export let h: number
  export let label: string
  export let sub = ''
</script>
<rect {x} {y} width={w} height={h} rx="3" fill="#fff" stroke="#222" stroke-width="1.2" />
<text x={x + w / 2} y={y + h / 2 + (sub ? -2 : 4)} font-size="11" text-anchor="middle" fill="#222" font-weight="600">{label}</text>
{#if sub}<text x={x + w / 2} y={y + h / 2 + 11} font-size="9.5" text-anchor="middle" fill="#555">{sub}</text>{/if}
```

`src/ui/schematics/parts/Mem.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import { KIND_COLORS, type LaneColor } from '../../figure/types'
  export let x: number
  export let y: number
  export let w: number
  export let h: number
  export let label: string
  export let color: LaneColor
  export let dashed = false
  export let sub = ''
  $: c = KIND_COLORS[color]
</script>
<rect {x} {y} width={w} height={h} rx="3" fill={c} fill-opacity="0.16" stroke={c} stroke-width="1.5" stroke-dasharray={dashed ? '4 3' : undefined} />
<text x={x + w / 2} y={y + h / 2 + (sub ? -2 : 4)} font-size="11" text-anchor="middle" fill="#222" font-weight="600">{label}</text>
{#if sub}<text x={x + w / 2} y={y + h / 2 + 11} font-size="9.5" text-anchor="middle" fill="#555">{sub}</text>{/if}
```

`src/ui/schematics/parts/Arrow.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  export let x1: number
  export let y1: number
  export let x2: number
  export let y2: number
  export let label = ''
  $: mx = (x1 + x2) / 2
  $: my = (y1 + y2) / 2
</script>
<line {x1} {y1} {x2} {y2} stroke="#222" stroke-width="1.2" marker-end="url(#arr)" />
{#if label}<text x={mx} y={my - 5} font-size="9.5" text-anchor="middle" fill="#444">{label}</text>{/if}
```

- [ ] **Step 2: Create the frame**

`src/ui/schematics/SchematicFrame.svelte`:

```svelte
<!-- Hosts one schematic (top 190px) and its class-level footprint strip. -->
<script lang="ts">
  import type { Component } from 'svelte'
  import type { SchematicMeta } from './meta'
  import { KIND_COLORS } from '../figure/types'
  export let meta: SchematicMeta
  export let component: Component

  const GROWTH_LABEL = {
    unbounded: 'grows with context', bounded: 'bounded by the window',
    constant: 'constant per request', none: 'no per-request memory',
  } as const
  $: growthColor = meta.growth === 'unbounded' ? KIND_COLORS.full
    : meta.growth === 'bounded' ? KIND_COLORS.window
    : meta.growth === 'constant' ? KIND_COLORS.state : KIND_COLORS.none
</script>

<svg viewBox="0 0 640 262" role="img" aria-label="{meta.title}: leaves {meta.leaves}; reads {meta.reads}; memory {GROWTH_LABEL[meta.growth]}.">
  <defs>
    <marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0L8 4L0 8z" fill="#222" />
    </marker>
  </defs>
  <svelte:component this={component} />
  <line x1="0" y1="196" x2="640" y2="196" stroke="#ddd" />
  <text x="0" y="212" class="lbl">LEAVES PER TOKEN</text>
  <text x="0" y="228" class="txt">{meta.leaves}</text>
  <text x="240" y="212" class="lbl">READS AT DECODE</text>
  <text x="240" y="228" class="txt">{meta.reads}</text>
  <text x="480" y="212" class="lbl">MEMORY</text>
  <rect x="480" y="219" width="10" height="10" fill={growthColor} />
  <text x="495" y="228" class="txt">{GROWTH_LABEL[meta.growth]}</text>
</svg>

<style>
  svg { display: block; width: 100%; min-width: 520px; height: auto; font-family: inherit; }
  .lbl { font-size: 10px; letter-spacing: 0.06em; fill: #666; }
  .txt { font-size: 10.5px; fill: #222; }
</style>
```

- [ ] **Step 3: Draw the four KV-cached schematics**

`src/ui/schematics/FullSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden" />
<Arrow x1={52} y1={88} x2={96} y2={58} label="" />
<Arrow x1={52} y1={104} x2={96} y2={128} label="" />
<Box x={96} y={40} w={118} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={110} w={118} h={36} label="W_k, W_v" sub="H_kv KV heads (H_kv ≤ H)" />
<Arrow x1={214} y1={128} x2={250} y2={128} label="append" />
<Mem x={250} y={100} w={160} h={56} label="KV cache" sub="one K row + one V row per KV head, per token" color="full" />
<Arrow x1={214} y1={58} x2={440} y2={58} label="H/H_kv query heads share each KV head" />
<Arrow x1={410} y1={128} x2={440} y2={90} label="read all" />
<Box x={440} y={40} w={150} h={60} label="attention" sub="softmax(q·K) V over every token" />
<Arrow x1={590} y1={70} x2={620} y2={70} label="" />
<text x="612" y="60" font-size="10" fill="#444">W_o</text>
<text x="8" y="176" font-size="10" fill="#555">GQA: fewer KV heads than query heads shrinks the cache row; the read is still the whole sequence.</text>
```

`src/ui/schematics/SlidingSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden" />
<Arrow x1={52} y1={88} x2={96} y2={58} />
<Arrow x1={52} y1={104} x2={96} y2={128} />
<Box x={96} y={40} w={118} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={110} w={118} h={36} label="W_k, W_v" sub="H_kv KV heads" />
<Arrow x1={214} y1={128} x2={250} y2={128} label="append" />
<Mem x={250} y={100} w={160} h={56} label="ring of w slots" sub="K and V for the last w tokens" color="window" />
<Mem x={430} y={116} w={80} h={24} label="t − w" sub="" color="none" dashed />
<Arrow x1={410} y1={128} x2={430} y2={128} label="evict" />
<Arrow x1={214} y1={58} x2={440} y2={58} />
<Arrow x1={330} y1={100} x2={440} y2={80} label="read the window" />
<Box x={440} y={40} w={150} h={50} label="attention" sub="over the trailing w tokens only" />
<text x="8" y="176" font-size="10" fill="#555">Each layer sees w tokens; L layers stack to a receptive field of about L × w. The cache never exceeds w rows per layer.</text>
```

`src/ui/schematics/HybridSchematic.svelte`:

```svelte
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
```

`src/ui/schematics/PartialSchematic.svelte`:

```svelte
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
```

- [ ] **Step 4: Create the partial component map**

`src/ui/schematics/components.ts`:

```ts
import type { Component } from 'svelte'
import type { SchematicId } from './meta'
import FullSchematic from './FullSchematic.svelte'
import SlidingSchematic from './SlidingSchematic.svelte'
import HybridSchematic from './HybridSchematic.svelte'
import PartialSchematic from './PartialSchematic.svelte'

// Partial until Task 12 adds the remaining drawings and tightens this to Record.
export const SCHEMATIC_COMPONENTS: Partial<Record<SchematicId, Component>> = {
  'full': FullSchematic,
  'sliding': SlidingSchematic,
  'hybrid': HybridSchematic,
  'partial': PartialSchematic,
}
```

- [ ] **Step 5: Type check**

Run: `npm run check`
Expected: PASS. If svelte-check rejects `Component` for legacy components, change the map's value type to `typeof FullSchematic` (all schematics share the same empty-props signature) and keep going; note the change in the commit body.

- [ ] **Step 6: Commit**

```bash
git add src/ui/schematics
git commit -m "feat(info): schematic frame, drawing parts, and full/sliding/hybrid/partial schematics"
```

---

### Task 11: Latent and sparse schematics

**Files:**
- Create: `src/ui/schematics/MlaSchematic.svelte`, `MlaDsaSchematic.svelte`, `MsaSchematic.svelte`, `CsaHcaSchematic.svelte`
- Modify: `src/ui/schematics/components.ts`

**Interfaces:**
- Consumes: parts from Task 10.

- [ ] **Step 1: Draw the four schematics**

`src/ui/schematics/MlaSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" sub="hidden" />
<Arrow x1={52} y1={88} x2={96} y2={50} />
<Arrow x1={52} y1={104} x2={96} y2={128} />
<Box x={96} y={32} w={110} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={110} w={110} h={36} label="W_dkv" sub="down-project" />
<Arrow x1={206} y1={120} x2={246} y2={110} label="cache" />
<Arrow x1={206} y1={136} x2={246} y2={150} label="cache" />
<Mem x={246} y={92} w={130} h={30} label="latent c_t" sub="rank r" color="full" />
<Mem x={246} y={136} w={130} h={30} label="k_rope" sub="d_rope, shared by heads" color="full" />
<Arrow x1={376} y1={107} x2={412} y2={107} label="expand" />
<Box x={412} y={90} w={100} h={36} label="W_uk, W_uv" sub="→ H heads" />
<Arrow x1={512} y1={107} x2={536} y2={80} />
<Arrow x1={206} y1={50} x2={536} y2={50} />
<Arrow x1={376} y1={151} x2={536} y2={90} label="RoPE part" />
<Box x={536} y={36} w={96} h={60} label="attention" sub="all tokens, H heads" />
<text x="8" y="176" font-size="10" fill="#555">The cache row is r + d_rope elements per token, not 2 × H × d. Heads are recovered from the latent at read time.</text>
```

`src/ui/schematics/MlaDsaSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={88} x2={96} y2={50} />
<Arrow x1={52} y1={96} x2={96} y2={96} />
<Arrow x1={52} y1={104} x2={96} y2={146} />
<Box x={96} y={32} w={110} h={36} label="W_q" sub="H query heads" />
<Box x={96} y={78} w={110} h={36} label="W_dkv" sub="down-project" />
<Box x={96} y={128} w={110} h={36} label="indexer" sub="lightning, few heads" />
<Arrow x1={206} y1={96} x2={246} y2={96} label="cache" />
<Mem x={246} y={78} w={140} h={36} label="latent + k_rope" sub="every token, unchanged from MLA" color="full" />
<Arrow x1={316} y1={114} x2={316} y2={140} label="score all" />
<Arrow x1={206} y1={146} x2={266} y2={146} />
<Box x={266} y={128} w={100} h={36} label="scores" sub="one per cached token" />
<Arrow x1={366} y1={146} x2={412} y2={146} label="top-k" />
<Mem x={412} y={128} w={100} h={36} label="k tokens" sub="selected set" color="sparse" />
<Arrow x1={512} y1={146} x2={548} y2={110} label="read k" />
<Arrow x1={206} y1={50} x2={548} y2={60} />
<Box x={548} y={40} w={84} h={72} label="attention" sub="k tokens only" />
<text x="8" y="186" font-size="10" fill="#555">Cache size is MLA's. Attention FLOPs stop growing past k. The calculator does not cost the indexer.</text>
```

`src/ui/schematics/MsaSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={88} x2={96} y2={50} />
<Arrow x1={52} y1={96} x2={96} y2={96} />
<Arrow x1={52} y1={104} x2={96} y2={146} />
<Box x={96} y={32} w={110} h={36} label="W_q" sub="H heads in G groups" />
<Box x={96} y={78} w={110} h={36} label="W_k, W_v" sub="G KV heads" />
<Box x={96} y={128} w={110} h={36} label="index key" sub="1 head, shared by groups" />
<Arrow x1={206} y1={96} x2={246} y2={96} label="append" />
<Mem x={246} y={78} w={140} h={36} label="KV cache" sub="every token, full" color="full" />
<Arrow x1={206} y1={146} x2={246} y2={146} label="append" />
<Mem x={246} y={128} w={140} h={36} label="index keys" sub="one per token, not TP-sharded" color="sparse" />
<Arrow x1={386} y1={146} x2={420} y2={146} label="per group" />
<Box x={420} y={128} w={100} h={36} label="block scores" sub="B tokens per block" />
<Arrow x1={470} y1={128} x2={470} y2={114} label="top-k blocks" />
<Mem x={420} y={78} w={100} h={36} label="k × B tokens" color="sparse" />
<Arrow x1={520} y1={96} x2={548} y2={84} label="read" />
<Arrow x1={206} y1={50} x2={548} y2={60} />
<Box x={548} y={40} w={84} h={60} label="attention" sub="selected blocks" />
<text x="8" y="186" font-size="10" fill="#555">A few layers stay full attention. Sparse layers keep the whole KV cache; selection caps compute only.</text>
```

`src/ui/schematics/CsaHcaSchematic.svelte`:

```svelte
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
```

- [ ] **Step 2: Register them**

In `src/ui/schematics/components.ts` add imports and entries:

```ts
import MlaSchematic from './MlaSchematic.svelte'
import MlaDsaSchematic from './MlaDsaSchematic.svelte'
import MsaSchematic from './MsaSchematic.svelte'
import CsaHcaSchematic from './CsaHcaSchematic.svelte'
```

```ts
  'mla': MlaSchematic,
  'mla-dsa': MlaDsaSchematic,
  'msa-hybrid': MsaSchematic,
  'csa-hca-hybrid': CsaHcaSchematic,
```

- [ ] **Step 3: Type check**

Run: `npm run check`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ui/schematics
git commit -m "feat(info): MLA, DSA, MSA and CSA/HCA schematics"
```

---

### Task 12: State-space and FFN schematics; complete the component map

**Files:**
- Create: `src/ui/schematics/LinearMlaSchematic.svelte`, `DeltaSchematic.svelte`, `Mamba2Schematic.svelte`, `DenseFfnSchematic.svelte`, `MoeSchematic.svelte`
- Modify: `src/ui/schematics/components.ts` (tighten to `Record`)

- [ ] **Step 1: Draw the five schematics**

`src/ui/schematics/LinearMlaSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Mem from './parts/Mem.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { KIND_COLORS } from '../figure/types'
  const PATTERN = ['state', 'state', 'state', 'full'] as const
</script>
<text x="8" y="20" font-size="10.5" fill="#666" letter-spacing="0.06em">KDA BLOCK</text>
<Box x={8} y={40} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={57} x2={90} y2={57} />
<Box x={90} y={40} w={120} h={34} label="W_q W_k W_v" sub="+ gates α, β" />
<Arrow x1={210} y1={57} x2={250} y2={57} label="update" />
<Mem x={250} y={30} w={170} h={54} label="state S per head" sub="d × d, fixed size" color="state" />
<Arrow x1={335} y1={84} x2={335} y2={112} label="S ← α S + β (v − S k) kᵀ" />
<Arrow x1={420} y1={57} x2={460} y2={57} label="o = q · S" />
<Box x={460} y={40} w={80} h={34} label="out" />
<text x="8" y="146" font-size="10.5" fill="#666" letter-spacing="0.06em">STACK</text>
{#each PATTERN as k, i}
  <rect x={60 + i * 50} y={134} width="46" height="18" rx="2" fill={KIND_COLORS[k]} />
  <text x={83 + i * 50} y={147} font-size="9.5" text-anchor="middle" fill="#fff" font-weight="600">{k === 'full' ? 'MLA' : 'KDA'}</text>
{/each}
<Mem x={280} y={130} w={200} h={26} label="MLA latent cache · grows" color="full" />
<text x="8" y="186" font-size="10" fill="#555">KDA stores nothing per token: the state is read and written once per step. Every fourth layer is MLA and keeps a growing latent cache.</text>
```

`src/ui/schematics/DeltaSchematic.svelte`:

```svelte
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
```

`src/ui/schematics/Mamba2Schematic.svelte`:

```svelte
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
```

`src/ui/schematics/DenseFfnSchematic.svelte`:

```svelte
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
```

`src/ui/schematics/MoeSchematic.svelte`:

```svelte
<svelte:options namespace="svg" />
<script lang="ts">
  import Box from './parts/Box.svelte'
  import Arrow from './parts/Arrow.svelte'
  import { EXPERT_COLOR, KIND_COLORS } from '../figure/types'
  // Eight drawn experts stand in for N; two lit ones stand in for k.
  const ACTIVE = new Set([1, 5])
</script>
<Box x={8} y={78} w={44} h={34} label="x_t" />
<Arrow x1={52} y1={95} x2={96} y2={95} />
<Box x={96} y={78} w={80} h={34} label="router" sub="top-k of N" />
{#each Array.from({ length: 8 }) as _, i}
  {@const ex = 220 + i * 46}
  <Arrow x1={176} y1={95} x2={ex} y2={ACTIVE.has(i) ? 66 : 74} label="" />
  <rect x={ex} y={44} width="38" height="30" rx="3"
    fill={ACTIVE.has(i) ? EXPERT_COLOR : '#fff'} stroke={EXPERT_COLOR} stroke-width="1.5" />
  <text x={ex + 19} y={63} font-size="10" text-anchor="middle" fill={ACTIVE.has(i) ? '#fff' : '#555'} font-weight="600">E{i + 1}</text>
{/each}
<text x="580" y="63" font-size="10" fill="#555">… E_N</text>
<rect x="220" y="120" width="120" height="30" rx="3" fill={EXPERT_COLOR} fill-opacity="0.55" stroke={EXPERT_COLOR} stroke-width="1.5" />
<text x="280" y="139" font-size="10.5" text-anchor="middle" fill="#222" font-weight="600">shared expert(s) · always on</text>
<Arrow x1={176} y1={100} x2={220} y2={135} />
<Arrow x1={266} y1={74} x2={470} y2={140} label="" />
<Arrow x1={450} y1={74} x2={480} y2={140} label="" />
<Arrow x1={340} y1={135} x2={470} y2={150} label="" />
<Box x={470} y={130} w={90} h={34} label="Σ weighted" />
<text x="8" y="186" font-size="10" fill="#555">A decode step streams k routed experts plus the shared ones, not all N. Active parameters, not total, set the weight-side cost.</text>
```

- [ ] **Step 2: Complete the component map**

Replace `src/ui/schematics/components.ts` with:

```ts
import type { Component } from 'svelte'
import type { SchematicId } from './meta'
import FullSchematic from './FullSchematic.svelte'
import SlidingSchematic from './SlidingSchematic.svelte'
import HybridSchematic from './HybridSchematic.svelte'
import PartialSchematic from './PartialSchematic.svelte'
import MlaSchematic from './MlaSchematic.svelte'
import MlaDsaSchematic from './MlaDsaSchematic.svelte'
import MsaSchematic from './MsaSchematic.svelte'
import CsaHcaSchematic from './CsaHcaSchematic.svelte'
import LinearMlaSchematic from './LinearMlaSchematic.svelte'
import DeltaSchematic from './DeltaSchematic.svelte'
import Mamba2Schematic from './Mamba2Schematic.svelte'
import DenseFfnSchematic from './DenseFfnSchematic.svelte'
import MoeSchematic from './MoeSchematic.svelte'

// Record, not Partial: a new discriminant without a drawing fails `npm run check`.
export const SCHEMATIC_COMPONENTS: Record<SchematicId, Component> = {
  'full': FullSchematic,
  'sliding': SlidingSchematic,
  'hybrid': HybridSchematic,
  'partial': PartialSchematic,
  'mla': MlaSchematic,
  'mla-dsa': MlaDsaSchematic,
  'msa-hybrid': MsaSchematic,
  'csa-hca-hybrid': CsaHcaSchematic,
  'linear-mla-hybrid': LinearMlaSchematic,
  'delta-hybrid': DeltaSchematic,
  'mamba2-hybrid': Mamba2Schematic,
  'dense': DenseFfnSchematic,
  'moe': MoeSchematic,
}
```

- [ ] **Step 3: Type check**

Run: `npm run check`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ui/schematics
git commit -m "feat(info): KDA, DeltaNet, Mamba2, dense and MoE schematics; complete the component map"
```

---

### Task 13: Architectures sub-section and `ArchPage`

**Files:**
- Create: `src/ui/ArchPage.svelte`
- Modify: `src/ui/InfoPanel.svelte`

**Interfaces:**
- Consumes: `schematicFor`, `modelsUsing`, `SCHEMATIC_GROUPS`, `ATTENTION_SCHEMATICS`, `FFN_SCHEMATICS` (Task 9); `SCHEMATIC_COMPONENTS` (Task 12); `SchematicFrame` (Task 10); `navigate`, `Route` with `'arch'` (Task 1).
- Produces: `ArchPage` with prop `meta: SchematicMeta`.

- [ ] **Step 1: Create `src/ui/ArchPage.svelte`**

```svelte
<!-- One architecture schematic: drawing, footprint strip, prose, sources,
     and the catalog models built on it. -->
<script lang="ts">
  import type { SchematicMeta } from './schematics/meta'
  import { modelsUsing } from './schematics/meta'
  import { SCHEMATIC_COMPONENTS } from './schematics/components'
  import SchematicFrame from './schematics/SchematicFrame.svelte'
  import { MODELS } from '../data'
  import { SOURCES } from '../data/sources'
  import { navigate } from './route'
  export let meta: SchematicMeta
  $: users = modelsUsing(meta.id, MODELS)
</script>

<article class="arch">
  <h2><slot />{meta.title}</h2>
  <div class="rule-thick"></div>
  <div class="figure"><SchematicFrame {meta} component={SCHEMATIC_COMPONENTS[meta.id]} /></div>
  <p class="summary">{meta.summary}</p>

  <div class="rule"></div>
  <h3>Sources</h3>
  <ul class="sources">
    {#each meta.sources as k}
      <li><a href={SOURCES[k].url} target="_blank" rel="noopener">{SOURCES[k].title}</a></li>
    {/each}
  </ul>

  <div class="rule"></div>
  <h3>Models in the catalog using this</h3>
  {#if users.length === 0}
    <p class="none">None yet.</p>
  {:else}
    <ul class="users">
      {#each users as m}
        <li><button class="entry" on:click={() => navigate({ tab: 'info', detail: { kind: 'model', id: m.id } })}>{m.name}</button></li>
      {/each}
    </ul>
  {/if}
</article>

<style>
  /* Same nutrition-label frame as the spec sheets. */
  .arch { max-width: 700px; border: 2px solid #111; border-radius: 4px; padding: 0.9rem 1.1rem; background: #fff; }
  h2 { margin: 0 0 0.4rem; font-size: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }
  h3 { margin: 0.6rem 0 0.4rem; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #333; }
  .rule-thick { border-bottom: 6px solid #111; margin: 0.3rem 0 0.6rem; }
  .rule { border-bottom: 1px solid #111; margin: 0.7rem 0 0; }
  .figure { overflow-x: auto; }
  .summary { margin: 0.5rem 0 0; max-width: 66ch; line-height: 1.45; }
  .none { color: #777; margin: 0; }
  ul { list-style: none; margin: 0; padding: 0; }
  .sources li { font-size: 0.9rem; }
  .sources a { color: #1a4f8a; }
  .users { columns: 220px; column-gap: 1.5rem; }
  .entry { font: inherit; font-size: 0.95rem; background: none; border: none; padding: 0.2rem 0.3rem; cursor: pointer; color: #1a4f8a; border-radius: 0.25rem; }
  .entry:hover { background: #eef2f7; }
</style>
```

- [ ] **Step 2: Wire the sub-section into `InfoPanel.svelte`**

Script block: extend imports and state.

```ts
  import ArchPage from './ArchPage.svelte'
  import { SCHEMATIC_GROUPS, schematicFor } from './schematics/meta'
```

```ts
  let section: 'models' | 'skus' | 'archs' = 'models'
```

Replace the `effSection` line and `selectSection` signature:

```ts
  $: activeArch = routeDetail?.kind === 'arch' ? schematicFor(routeDetail.id) : undefined
  // A detail route forces its section; otherwise the manual toggle wins.
  $: effSection = routeDetail
    ? (routeDetail.kind === 'model' ? 'models' : routeDetail.kind === 'sku' ? 'skus' : 'archs')
    : section

  function selectSection(s: 'models' | 'skus' | 'archs') {
    section = s
    navigate({ tab: 'info' })  // clear any detail so the toggle takes effect
  }
```

Markup: add the third button after the SKUs button.

```svelte
    <button class:active={effSection === 'archs'} on:click={() => selectSection('archs')}>Architectures</button>
```

Change `{:else}` (the SKUs branch) to `{:else if effSection === 'skus'}` and append a new branch before `{/if}`:

```svelte
  {:else}
    {#if activeArch}
      <div class="cardwrap">
        {#if cardOpen}
          <ArchPage meta={activeArch}>
            <button class="cardtoggle" title="Collapse" aria-label="Collapse"
              aria-expanded="true" on:click={() => cardOpen = false}>−</button>
          </ArchPage>
        {:else}
          <div class="collapsed">
            <button class="cardtoggle" title="Expand" aria-label="Expand"
              aria-expanded="false" on:click={() => cardOpen = true}>+</button>
            {activeArch.title}
          </div>
        {/if}
      </div>
    {/if}
    <div class="groups">
      {#each SCHEMATIC_GROUPS as g}
        <div class="group">
          <h3>{g.title}</h3>
          <ul>
            {#each g.ids as id}
              {@const s = schematicFor(id)}
              {#if s}
                <li>
                  <button class="entry" class:pinned={id === activeArch?.id}
                    on:click={() => navigate({ tab: 'info', detail: { kind: 'arch', id } })}>
                    {s.title}
                  </button>
                </li>
              {/if}
            {/each}
          </ul>
        </div>
      {/each}
    </div>
```

- [ ] **Step 3: Type check, tests, and look at it**

Run: `npm run check && npm test`
Expected: PASS.

Run: `npm run dev`. Open `#info`, click Architectures, click through all thirteen entries. For each: the drawing renders inside the frame with no overlapping text, arrows point at boxes, the footprint strip reads correctly, sources link out, and "Models in the catalog using this" lists the expected models (DSA → DeepSeek-V3.2 and the GLM-5 line; MoE → every MoE model). Open `#info/model/deepseek-v3.2` and follow the "How … works" link; it must land on the DSA page. Open `#info/arch/bogus`; it must show the Architectures list with no card. Fix overlaps by editing the schematic component coordinates, then commit.

- [ ] **Step 4: Commit**

```bash
git add src/ui/ArchPage.svelte src/ui/InfoPanel.svelte
git commit -m "feat(info): Architectures sub-section with per-variant schematic pages"
```

---

### Task 14: Grammar doc, skill section, and skill-sync check

**Files:**
- Create: `docs/architecture-figures.md`
- Modify: `.claude/skills/adding-a-model/SKILL.md`
- Modify: `.claude/hooks/check-skill-sync.mjs`
- Test: `test/check-skill-sync.test.ts`

**Interfaces:**
- Produces: `extractSkillFigureVariants(skillMd): string[]` in the hook; a `## Figures` section in the skill listing every `AttentionConfig` discriminant as `- **\`name\`**`.

- [ ] **Step 1: Write the failing tests**

Add `extractSkillFigureVariants` to the import list in `test/check-skill-sync.test.ts` and append:

```ts
describe('extractSkillFigureVariants', () => {
  it('parses bullet list of **`name`** items in the Figures section', () => {
    const md = `
## Figures

- **\`full\`** — lanes: full × layers · FullSchematic
- **\`mla-dsa\`** — lanes: dsa × layers · MlaDsaSchematic

## Process
`
    expect(extractSkillFigureVariants(md)).toEqual(['full', 'mla-dsa'])
  })
  it('returns [] when the section is absent', () => {
    expect(extractSkillFigureVariants('## Other\n- **`full`**')).toEqual([])
  })
})
```

And inside the existing `integration: current files are in sync` describe block:

```ts
  it('AttentionConfig discriminants — every one has a Figures entry in SKILL.md', () => {
    const inTypes = new Set<string>(extractDiscriminants(typesSrc, 'AttentionConfig'))
    const inSkill = new Set<string>(extractSkillFigureVariants(skillMd))
    expect([...inTypes].sort()).toEqual([...inSkill].sort())
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run test/check-skill-sync.test.ts`
Expected: FAIL, `extractSkillFigureVariants` is not exported.

- [ ] **Step 3: Extend the hook**

In `.claude/hooks/check-skill-sync.mjs`, after `extractSkillAttentionVariants`:

```js
export function extractSkillFigureVariants(skillMd) {
  const section = findSection(skillMd, '## Figures')
  return Array.from(section.matchAll(/-\s+\*\*`([a-z][\w-]*)`\*\*/g)).map(m => m[1])
}
```

In `runCheck`, after the AttentionConfig block:

```js
  const figSkill = new Set(extractSkillFigureVariants(skillMd))
  const figMissing = [...attnTypes].filter(x => !figSkill.has(x))
  const figExtra = [...figSkill].filter(x => !attnTypes.has(x))
  if (figMissing.length || figExtra.length) {
    failures.push({ label: 'Figures section (AttentionConfig discriminants)', missing: figMissing, extra: figExtra })
  }
```

- [ ] **Step 4: Add the `## Figures` section to the skill**

In `.claude/skills/adding-a-model/SKILL.md`, insert before `## Process`:

```markdown
## Figures

Every attention variant has two drawings in the Info tab, and a new variant needs both before its first model entry lands. The grammar is in [`docs/architecture-figures.md`](../../../docs/architecture-figures.md).

- A **schematic** (educational, hand-drawn, cited): add a `SchematicMeta` in `src/ui/schematics/meta.ts` and a `<Name>Schematic.svelte` in `src/ui/schematics/`, registered in `components.ts`. Both are `Record`s keyed by the discriminant, so `npm run check` fails until they exist. Cite the paper through `src/data/sources.ts`; fetch the URL before writing it.
- A **geometry** case: add the variant to `lanesFor`, `headsFor` and `captionFor` in `src/ui/archGeometry.ts` (each has a `never` fallthrough) and a test in `test/ui/archGeometry.test.ts` asserting exact lanes for a real catalog entry. The catalog-wide test cross-checks lane bytes against the engine; if it fails, the lane is wrong, not the engine.

Lanes per variant (kind × count field):

- **`full`** — full × layers · FullSchematic
- **`sliding`** — window × layers · SlidingSchematic
- **`hybrid`** — window × numSlidingLayers; full × numGlobalLayers · HybridSchematic
- **`partial`** — full × numFullLayers; pruned × (layers − numFullLayers) · PartialSchematic
- **`mla`** — mla × layers · MlaSchematic
- **`mla-dsa`** — dsa × layers · MlaDsaSchematic
- **`msa-hybrid`** — full × numFullLayers; msa × numSparseLayers · MsaSchematic
- **`csa-hca-hybrid`** — window × numSlidingLayers (omitted when 0); csa × numCsaLayers; hca × numHcaLayers · CsaHcaSchematic
- **`linear-mla-hybrid`** — kda × numLinearLayers; mla × numFullLayers · LinearMlaSchematic
- **`delta-hybrid`** — delta × numDeltaNetLayers; full × numFullLayers · DeltaSchematic
- **`mamba2-hybrid`** — mamba × numMambaLayers; full × numFullLayers; ffn × numFfnLayers · Mamba2Schematic
```

Also add to the "Attention variant — when to use which" closing paragraph, after "Brainstorm with the user first; new attention is a meaningful design change, not a data update.":

```markdown
A new variant also needs its two figures; see [Figures](#figures).
```

- [ ] **Step 5: Write the grammar doc**

Create `docs/architecture-figures.md`:

```markdown
# Architecture figures

Two figure types live in the Info tab. They share a glyph grammar so a reader who has learned one can read the other.

**Schematics** (`src/ui/schematics/`) are educational: one per attention variant and one per FFN type, hand-drawn, cited, keyed by the engine's discriminants. They show the mechanism a class of models shares, with no model-specific numbers.

**Geometry figures** (`src/ui/archGeometry.ts`, `ArchGeometry.svelte`) are per model, computed from `ModelArch`, and cross-checked against `src/engine/memory.ts`. They draw one model's numbers to scale inside the shape its schematic established.

## Colour: what a layer reads at decode

| Role | Hex | Meaning | Kinds |
|---|---|---|---|
| full | `#2f5fd0` | whole sequence | full attention, MLA |
| window | `#3f9cc2` | trailing window | sliding |
| sparse | `#7a4fc9` | a selection or a compression | DSA, MSA, CSA, HCA |
| state | `#1f9a6e` | fixed recurrent state | KDA, DeltaNet, Mamba2 |
| none | `#b4bac8` | no attention | FFN-only, pruned |

Experts use `#e07a2a`. Light theme only; the spec sheets are black on white by design.

## Cache glyphs: what one token leaves behind, per layer

| Glyph | Drawing | Meaning |
|---|---|---|
| `kv` | two stacked rows | K and V rows, one per KV head |
| `latent` | one wide, one narrow rect | one latent vector plus a RoPE key (MLA) |
| `kvidx` | `kv` plus a thin third rect | K, V and one shared index key (MSA) |
| `kvcomp` | `kv` stamped ÷M | one entry per M tokens (CSA, HCA) |
| `state` | square matrix outline | nothing per token; a fixed state matrix instead |
| `none` | dashed empty rect | nothing |

## Reach glyphs: how far a decode step reads back

Drawn over a ruler from token 0 to S (the model's trained context on geometry figures).

| Glyph | Drawing | Meaning |
|---|---|---|
| `all` | full-width bar | every cached token |
| `window` | bar at the right end, width w/S | trailing window |
| `topk` | five scattered blocks totalling k/S, plus a local sliver if the variant has one | top-k selection |
| `compress` | thin full-width bar | every token, at 1 : M resolution |
| `state` | one dot at the right end | one state read, independent of length |
| `none` | dashed ruler | no attention |

## Geometry panels

1. **Blocks**: proportional bar of block counts by kind. Counts, never order.
2. **Heads**: query heads grouped onto KV heads (GQA ratio as group size); for MLA, full per-head K+V width against the latent to one scale; for state variants, the state matrix dims.
3. **Per token**: one bar per lane, bytes per token per layer, 8 KiB = full width, longer bars clip with the value printed.
4. **Reads at decode**: one ruler per lane at the model's max context.
5. **FFN**: hidden → intermediate widths (dense) or routed pool with the active slice and shared experts (MoE), with the active-parameter ratio.

## What the engine does not count

The figures show what the calculator models, with its simplifications:

- A decode step streams the whole cache from HBM even on DSA, MSA, CSA and HCA layers. Selection and compression lower attention FLOPs; only compression lowers bytes.
- The index or scoring branch that selects tokens or blocks is not costed.
- MSA's shared index-key cache is counted; DSA's indexer cache is not.
- Mamba2 state is fp32 regardless of the KV reference dtype, because the NemotronH configs pin it.

## Adding a variant

See the `## Figures` section of `.claude/skills/adding-a-model/SKILL.md`. The type system, the catalog-wide test in `test/ui/archGeometry.test.ts`, and the skill-sync hook each fail until the new variant has both figures.
```

- [ ] **Step 6: Run the hook, the tests, and the type check**

Run: `npm run check:skill-sync && npm test && npm run check`
Expected: the hook prints `✓ adding-a-model SKILL.md in sync with types.ts`; all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add docs/architecture-figures.md .claude/skills/adding-a-model/SKILL.md .claude/hooks/check-skill-sync.mjs test/check-skill-sync.test.ts
git commit -m "docs(info): architecture figure grammar; Figures section in adding-a-model with sync check"
```

---

## Final verification

- [ ] `npm run check` passes.
- [ ] `npm test` passes.
- [ ] `npm run check:skill-sync` passes.
- [ ] `npm run build` passes.
- [ ] In the dev server: every model's spec sheet renders its geometry figure without NaN or clipped text; every Architectures entry renders; the spec-sheet link lands on the right schematic; `#info/arch/bogus` shows the list.
- [ ] Spec coverage: schematics (§1) Tasks 9–13; geometry (§2) Tasks 3–8; shared conventions (§3) Tasks 3, 7; metrics fix (§4) Task 2; drift guards (§5) Tasks 5, 6, 9, 12, 14; skill and docs (§6) Task 14; testing (§7) every task; route Task 1.
