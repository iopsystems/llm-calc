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
