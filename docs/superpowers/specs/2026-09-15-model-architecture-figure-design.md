# Model Architecture Figures — Design

**Status:** Draft for review (brainstorm 2026-09-14/15, revised 2026-09-15 to split into two figure types)
**Scope:** Info tab. Two figure types, one route addition, one metrics fix. No engine math changes.
**Prototype:** the "What a token leaves behind" gallery artifact (2026-09-14): twelve exemplar cards, hand-transcribed from the model entries. It mixed the two figure types this spec separates.

## Goal

Two kinds of figure, with different jobs and different sources of truth.

**Architecture schematics** are educational. One per attention variant and one per FFN type, showing the mechanism that class of model shares: where the token's data flows inside one block, what gets written to memory, what a decode step reads back, and how those grow with context. No model-specific numbers. A schematic represents every model that uses the variant, and the page lists them.

**Geometry figures** are per model. Derived from a `ModelArch` entry, they draw that model's numbers to scale inside the shape its schematic established: how many blocks of each kind, how query heads group onto KV heads, how wide the cache row is per token, how far a decode step reaches at the trained context, how the expert pool is populated.

A reader lands on a model's spec sheet, sees its geometry, and follows one link to the schematic that explains the shape. A reader lands on a schematic and sees which catalog models are built on it.

## Non-goals

- **Interleave order.** `AttentionConfig` carries per-kind counts, not the layer pattern. Geometry figures show counts and say so. Schematics may show a representative alternation only where the paper defines a fixed ratio, labelled as the paper's, not the catalog's.
- **Costs the engine does not model.** No index-branch scoring cost for DSA/MSA/CSA, no DSA indexer cache. Schematics draw the index branch because it is the mechanism; a caption states that the calculator does not cost it.
- **Training-time structure, vision towers, tokenizers, MTP heads.** Decode-path only, matching the engine.
- **SKU topology figures** (deferred in the Info page spec, still deferred).
- **Compare-tab strips, catalog mini-bars, CLI export.** Follow-ons.
- **Interactivity, animation, a diagram library.** Static inline SVG.

## 1. Architecture schematics

### 1.1 Set

Keyed by discriminant. A `Record<AttentionConfig['type'], Schematic>` plus a `Record<ArchitectureConfig['type'], Schematic>` so a new variant without a schematic is a type error.

Attention: `full`, `sliding`, `hybrid`, `partial`, `mla`, `mla-dsa`, `msa-hybrid`, `csa-hca-hybrid`, `linear-mla-hybrid`, `delta-hybrid`, `mamba2-hybrid`. FFN: `dense`, `moe`.

### 1.2 What each schematic must show

Every schematic is one block's decode path, left to right: input hidden state, the projections, the memory the block reads and writes, the attention or state operation, the output. A footprint strip beneath states three class-level facts in words and glyphs: what one token leaves behind, what a decode step reads, and whether memory grows with context (unbounded / bounded / constant). The mechanism each one hinges on:

| Variant | The thing the drawing must make visible |
|---|---|
| `full` | Query heads grouped onto fewer KV heads (GQA); one K and one V row appended per KV head per token; attention over every cached token. |
| `sliding` | Same block with the cache drawn as a ring of w slots; the token falling off the end. |
| `hybrid` | A stack alternating sliding and global blocks; the two caches side by side, one bounded, one not. |
| `partial` | A stack where some blocks have the attention path removed and only the FFN remains. |
| `mla` | Down-projection to one latent vector plus one small RoPE key, both cached; up-projection re-expanding the latent into all heads at use. The cache row is the latent, not the heads. |
| `mla-dsa` | MLA block plus the indexer branch: scores every cached token, keeps the top k, main attention reads only those. Cache unchanged. |
| `msa-hybrid` | GQA block plus one shared index-key head cached per token; block scores per GQA group; attention over the selected blocks. Some blocks are plain full attention. |
| `csa-hca-hybrid` | Two block types. CSA: every M tokens compressed to one cache entry, an indexer picks the top k of those, plus a local window branch. HCA: 128 : 1 compression, attention over all compressed entries, plus the local branch. |
| `linear-mla-hybrid` | KDA block: a fixed state matrix per head, updated by a delta rule with gating, read by the query. No cache. MLA blocks at a fixed ratio. |
| `delta-hybrid` | Gated DeltaNet block with the same state-matrix shape; gated attention blocks with partial RoPE at a fixed ratio. |
| `mamba2-hybrid` | Mamba2 block: convolution, then the SSM scan against a per-head state held in fp32. Attention blocks and FFN-only blocks are separate entries in the stack. |
| `dense` | Up, gate and down projections; every token pays the full width. |
| `moe` | Router picks k of N routed experts; shared experts run for every token; the active slice is what a decode step streams. |

### 1.3 Authoring and provenance

Schematics are hand-authored SVG, one Svelte component each, because the set is fixed and small and none of them depends on catalog data. Each carries a `sources` list of keys into `src/data/sources.ts`, pointing at the paper or technical report the mechanism is drawn from. Paper entries are added to the registry as needed. The existing design specs under `docs/superpowers/specs/` (MLA, DSA, linear attention, hybrid, sliding, V4 attention, MoE, shared experts) already name these papers; the schematic cites the paper, not the spec.

Drawing rules follow the artifact-diagramming conventions: native shapes and text, `currentColor` for structure, the five kind colours from section 3.3 for memory, labelled arrows (`append`, `score`, `select top-k`, `expand`, `update`), no explanatory sentences inside the drawing.

### 1.4 Placement and route

- `InfoPanel.svelte` gains a third sub-section: **Models | SKUs | Architectures**.
- The Architectures list groups variants by the footprint class they belong to (whole sequence / windowed / compressed or selected / recurrent state), the same grouping the prototype used, then lists FFN types.
- Each entry opens a page: the schematic, its footprint strip, a short prose paragraph, its sources, and **Models using this** derived from `MODELS` at render time.
- `route.ts` extends the detail kind: `{ kind: 'model' | 'sku' | 'arch'; id }`, hash `#info/arch/<discriminant>`. Unknown discriminant falls back to the Architectures list, matching the existing unknown-id behaviour.

## 2. Geometry figures

### 2.1 Module boundaries

Same split the Info tab already uses: a pure tested module produces a data object, a presentational component draws it.

- **`src/ui/archGeometry.ts`** — pure. `geometryModel(model: ModelArch): GeometryModel`. Exhaustive `switch` on `model.attention.type` with a `never` fallthrough (same idiom as `attentionLabel` in `catalogMetrics.ts`). Bytes via `kvBytesPerTokenPerLayer`, `linearAttentionStateBytes`, `deltaStateBytes`, `mambaStateBytes` from `src/engine/memory.ts`, at the spec sheet's fixed fp16 reference (`KV_REF_DTYPE`, to be exported from `catalogMetrics.ts` rather than duplicated).
- **`src/ui/ArchGeometry.svelte`** — prop `geometry: GeometryModel`. Draws inline SVG. No math, no imports from `engine/`.
- **`src/ui/ModelSpecSheet.svelte`** — new "Architecture" section between Design and Scale: the geometry figure, its caption, and a link "How `<variant label>` works" to `#info/arch/<type>`.

### 2.2 Panels

Five panels, stacked. Each draws a number to scale; the scale is stated on the panel.

1. **Block stack.** Proportional bar of block counts by kind, coloured by kind, total printed. Caption: counts, not order.
2. **Head geometry.** Query heads as small squares, grouped onto their KV head, so the GQA ratio is visible as group size; head dim printed. For MLA variants, two bars to one scale: the full per-head KV width (numHeads × headDim × 2) against the latent (kvLoraRank + qkRopeHeadDim), so the compression is a visible ratio. For state variants, the state matrix as a rectangle with its dims (heads × dim × dim, or heads × dim × stateSize), to a scale shared with the KV bars.
3. **Cache row per token.** One bar per lane, length ∝ bytes per token per layer, on a fixed catalog-wide scale (8 KiB = full width; longer bars clip with the value printed). Index-key bytes drawn as a stub on MSA lanes; compressed lanes show the per-token share after ÷M.
4. **Reach at trained context.** One ruler per lane from token 0 to `maxContext`. Whole sequence, window (w/S, floor 4px), top-k (scattered blocks summing to k/S, floor 6px), compressed (thin full-width bar), state (one dot), none (dashed).
5. **FFN geometry.** Dense: hidden → intermediate as two widths to one scale. MoE: the routed pool as a bar with the active slice lit (k/N, floor 4px), shared experts as separate blocks, and the active/total parameter ratio printed.

### 2.3 Data contract

```typescript
export type LaneKind =
  | 'full' | 'window' | 'mla' | 'dsa' | 'msa' | 'csa' | 'hca'
  | 'kda' | 'delta' | 'mamba' | 'ffn' | 'pruned'
export type LaneColor = 'full' | 'window' | 'sparse' | 'state' | 'none'
export type CacheGlyph = 'kv' | 'latent' | 'kvidx' | 'kvcomp' | 'state' | 'none'
export type ReachGlyph = 'all' | 'window' | 'topk' | 'compress' | 'state' | 'none'

export interface Lane {
  kind: LaneKind
  label: string
  color: LaneColor
  count: number
  cache: {
    glyph: CacheGlyph
    bytesPerToken: number    // per layer, at KV_REF_DTYPE; compressed kinds already divided by M
    fixedBytes: number       // per layer, 'state' only; 0 otherwise
    ratio?: number           // M for 'kvcomp'
    label: string
  }
  reach: {
    glyph: ReachGlyph
    tokens?: number          // window w, or top-k token count
    ratio?: number           // M for 'compress'
    label: string
  }
}

export interface HeadGeometry {
  kind: 'gqa' | 'latent' | 'state'
  numHeads: number
  numKvHeads: number
  headDim: number
  fullKvWidth?: number       // 'latent': numHeads × headDim × 2
  latentWidth?: number       // 'latent': kvLoraRank + qkRopeHeadDim
  state?: { heads: number; dim: number; inner: number }   // 'state': heads × dim × inner elements
}

export interface GeometryModel {
  layers: number
  maxContext: number
  lanes: Lane[]              // counts sum to layers
  heads: HeadGeometry[]      // one per distinct attention kind present
  ffn:
    | { type: 'dense'; hiddenDim: number; intermediateDim: number }
    | { type: 'moe'; routed: number; active: number; shared: number; activeRatio: number }
  caption: string
  schematicId: AttentionConfig['type']
}
```

### 2.4 Variant → lanes mapping

Exact, from the `AttentionConfig` union. `p` = `kvBytesPerTokenPerLayer(model, KV_REF_DTYPE)`.

| `attention.type` | Lanes (kind × count) | Cache | Reach |
|---|---|---|---|
| `full` | full × layers | `kv`, p | `all` |
| `sliding` | window × layers | `kv`, p, label "last w tokens only" | `window`, w |
| `hybrid` | window × numSlidingLayers; full × numGlobalLayers | `kv`, p | `window`, w; `all` |
| `partial` | full × numFullLayers; pruned × (layers − numFullLayers) | `kv`, p; `none` | `all`; `none` |
| `mla` | mla × layers | `latent`, p | `all` |
| `mla-dsa` | dsa × layers | `latent`, p | `topk`, topK |
| `msa-hybrid` | full × numFullLayers; msa × numSparseLayers | `kv`, p; `kvidx`, p + indexHeadDim × bytesOf(ref) | `all`; `topk`, topKBlocks × blockSize |
| `csa-hca-hybrid` | window × numSlidingLayers; csa × numCsaLayers; hca × numHcaLayers | `kv`, p; `kvcomp` M=csaCompressionM, p/M; `kvcomp` M=hcaCompressionM, p/M | `window`, w; `topk`, csaTopK × csaCompressionM; `compress`, M |
| `linear-mla-hybrid` | kda × numLinearLayers; mla × numFullLayers | `state`, fixed = numLinearHeads × linearHeadDim² × bytesOf(ref); `latent`, p | `state`; `all` |
| `delta-hybrid` | delta × numDeltaNetLayers; full × numFullLayers | `state`, fixed = numDeltaNetHeads × deltaHeadDim² × bytesOf(ref); `kv`, p | `state`; `all` |
| `mamba2-hybrid` | mamba × numMambaLayers; full × numFullLayers; ffn × numFfnLayers | `state`, fixed = numMambaHeads × mambaHeadDim × ssmStateSize × 4 (fp32, pinned by config); `kv`, p; `none` | `state`; `all`; `none` |

Lanes with count 0 are not emitted (V4-Pro has no sliding layers). Head geometry: `gqa` for kv-caching kinds, `latent` for MLA kinds, `state` for recurrent kinds; one entry per distinct kind present.

Captions, verbatim in the module so they are testable:

- `mla-dsa`, `msa-hybrid`, `csa-hca-hybrid`: "A decode step streams the whole cache; selection lowers attention FLOPs, not bytes. The index branch that selects is not costed."
- `msa-hybrid` adds: "The shared index key is cached alongside K and V and is not sharded by TP."
- `mamba2-hybrid`: "Attention, Mamba2 and FFN are separate entries in the block count. SSM state is held in fp32."
- `partial`: "NAS pruning removed attention from the grey blocks; their FFN widths vary and are absorbed by the parameter count."
- All: "Bar lengths are counts, not order."

## 3. Shared conventions

### 3.1 Kind colours

Fixed for both figure types. Colour encodes what the layer reads at decode.

| Role | Meaning | Kinds |
|---|---|---|
| `full` | whole sequence | full attention, MLA |
| `window` | trailing window | sliding |
| `sparse` | selection or compression | DSA, MSA, CSA, HCA |
| `state` | fixed recurrent state | KDA, DeltaNet, Mamba2 |
| `none` | no attention | FFN-only, pruned |

Light palette from the prototype (`#2f5fd0`, `#3f9cc2`, `#7a4fc9`, `#1f9a6e`, `#b4bac8`; experts `#e07a2a`). The spec sheet is black-on-white by design and the app has no theme tokens, so no dark set in v1. Declared once as CSS custom properties in a shared stylesheet both components import.

### 3.2 Glyphs

Cache glyphs (`kv`, `latent`, `kvidx`, `kvcomp`, `state`, `none`) and reach glyphs (`all`, `window`, `topk`, `compress`, `state`, `none`) are drawn identically in the schematic footprint strip and the geometry panels, from one small SVG helper module so they cannot diverge. Definitions as in the prototype legend.

### 3.3 Accessibility and layout

- Every `<svg>` has `role="img"` and an `aria-label` stating the figure's claim in words.
- `viewBox` width 640; height by panel. Figure containers get `overflow-x: auto` and the SVG `min-width: 520px`, so phone widths scroll the figure rather than crush the labels.
- Text 10.5–12px at the drawn scale; sentences go in captions.

## 4. Metrics fix and additions

`modelMetrics(m).kvBytesPerToken` is currently `perLayer × m.layers`. That is right only for `full`, `mla`, `mla-dsa` and, at short context, `sliding`. For every hybrid it counts non-caching layers: Qwen3.5-397B shows 120 KiB where the engine stores 30 KiB; Nemotron-H 56B shows 472 KiB against 40 KiB. The spec sheet has been publishing the wrong number since the hybrids landed.

Fix, using the engine's own accounting:

```
kvBytesPerToken = perLayer × attendedSeqlenSummedOverLayers(m, S, true) / S
                + (msa ? numSparseLayers × indexHeadDim × bytesOf(ref) : 0)
where S = m.maxContext
```

This is the marginal cache cost per token at the trained ceiling. The spec sheet row is relabelled "KV / token (model, at max context)".

Additions to `ModelMetrics`:

- `fixedStateBytes` — sum of the three state functions at the reference dtype. New row "Fixed state / request", shown only when non-zero.
- `attentionReachLayers` — `attendedSeqlenSummedOverLayers(m, S, false) / S`, in full-layer equivalents at S = maxContext. New row "Attention reach", rendered as "n.n × S of L blocks". Without it DSA and MLA look identical on the sheet.

`geometryModel` derives its lane bytes from the per-variant table, not from these aggregates. A test asserts the two agree for every catalog model, applying the engine's storage rule per glyph with S = maxContext and w the lane's window:

| Lane | Contribution |
|---|---|
| `kv` / `latent` / `kvidx` with reach `all` or `topk` | count × bytesPerToken |
| `kv` with reach `window` | count × bytesPerToken × min(w, S) / S |
| `kvcomp` (CSA, HCA) | count × (bytesPerToken + p × min(w, S) / S), the second term being the local branch |
| `state`, `none` | 0 |

The sum must equal `kvBytesPerToken` within floating-point rounding for every entry in `MODELS`.

## 5. Drift guards

1. **Type system.** The `never` fallthrough in `geometryModel`, and the `Record` keyed by discriminant for schematics, make a new variant a compile error in `npm run check` until both are extended.
2. **Catalog-wide test.** For every entry in `MODELS`: `geometryModel` does not throw, lane counts sum to `layers`, bytes are finite and non-negative, the lane-sum cross-check holds, and `schematicId` resolves to a schematic whose `sources` resolve in the registry.
3. **Skill sync.** `.claude/hooks/check-skill-sync.mjs` gains a check that the `## Figures` section of the adding-a-model skill names every `AttentionConfig` discriminant, using the same extractor pattern as `extractSkillAttentionVariants`.

## 6. Skill and docs changes

- **`docs/architecture-figures.md`** — the grammar: kind colours, glyphs, the schematic checklist from 1.2, the geometry panels from 2.2, and the "what the engine does not count" list.
- **`.claude/skills/adding-a-model/SKILL.md`** — new `## Figures` section: one bullet per attention variant naming its lanes and its schematic component, and the procedure step: "a new variant needs a schematic with sources, a `geometryModel` case, and tests in `test/ui/archGeometry.test.ts` before the model entry lands." Linked from "Attention variant — when to use which".

## 7. Testing

TDD throughout; red before green.

- **`test/ui/archGeometry.test.ts`** — one case per attention variant using a real catalog entry, asserting exact lanes, head geometry, FFN and caption. Exemplars: Llama 3.1 8B, Mistral 7B, gpt-oss-120b, Llama-3.3-Nemotron-Super 49B, DeepSeek-V3, DeepSeek-V3.2, MiniMax M3, DeepSeek-V4-Flash and V4-Pro (zero-sliding case), Kimi-Linear-48B-A3B, Qwen3.5-397B-A17B, Nemotron 3 Nano 30B-A3B. Plus the catalog-wide invariants from section 5.
- **`test/ui/archSchematics.test.ts`** — every discriminant has a schematic; every schematic's `sources` resolve in `SOURCES`; the models-using-this derivation returns the right ids for two variants.
- **`test/ui/catalogMetrics.test.ts`** — `kvBytesPerToken` expectations updated; a hybrid case (Qwen3.5-397B → 30 KiB) must fail before the fix. New cases for `fixedStateBytes` and `attentionReachLayers` (DeepSeek-V3 → 61; V3.2 → 2048 × 61 / 163840 ≈ 0.76).
- **`test/ui/route.test.ts`** — `#info/arch/<id>` round-trips; unknown id falls back to the list.
- **`test/check-skill-sync.test.ts`** — the `## Figures` extractor: full coverage passes, a missing variant is reported by name.
- **Svelte components** — no unit tests (presentational, matches precedent); verified in-browser on the twelve exemplars and all thirteen schematics.

## 8. Follow-ons (not v1)

- Compare tab: a block-stack strip per candidate row from the same `GeometryModel`.
- Catalog list: the block-stack bar as a 12px inline strip beside each model name.
- CLI: `llm-calc figure <model-id>` emitting the geometry SVG for docs and PRs.
- Dark colour set if the app grows theme tokens.

## Architecture rationale

The two figure types have different sources of truth, so they get different authoring modes. Schematics describe a mechanism defined by a paper; they are hand-drawn once, cited, and keyed by discriminant so the type system says when one is missing. Geometry describes a catalog entry; it is computed from `ModelArch` with an exhaustive switch and cross-checked against the engine, because a hand-transcribed number on a figure is the same drift the repo's data discipline exists to prevent. The prototype was that hand transcription, and it is the reason this spec draws the line where it does. Pure module plus dumb component is the Info tab's existing pattern and is what makes the Compare strip and the CLI export cheap later. A diagram library would add a runtime for thirteen fixed drawings and eleven computed ones.
