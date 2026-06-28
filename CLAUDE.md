# CLAUDE.md

This file provides guidance to Claude Code (claude.com/claude-code) when working with code in this repository.

## Project Overview

`llm-calc` is a roofline-style performance calculator for dense and MoE decoder-only LLM inference. Given a (accelerator, model, quantization, workload) tuple it computes the theoretical memory + throughput limits (prefill TTFT, decode TPOT, KV-cache footprint, arithmetic intensity vs. the hardware roofline) and renders them interactively. Two front-ends share one TypeScript engine: a Svelte 5 web app (Vite-built, deployed to Cloudflare Workers) and a Node CLI.

Extracted from `iopsystems/llm-perf` (the Rust benchmarking tool) in 2026-06; the two are now separate repos.

## Common Development Commands

```bash
npm run dev          # Vite dev server (interactive web app)
npm run build        # production build → dist/
npm run check        # svelte-check (TypeScript + Svelte type checking)
npm test             # Vitest (engine + UI logic, node env, no DOM)
npm run test:watch   # Vitest watch mode
npm run cli -- ...   # run the CLI (bin/llm-calc.mjs)
npm run deploy       # deploy to Cloudflare Workers (wrangler)
npm run deploy:dry   # build + wrangler dry-run
npm run check:skill-sync  # verify model/attention enums stay in sync with the skill docs
```

Always run `npm run check` and `npm test` before considering a change done.

## Architecture

- `src/engine/` — pure, framework-free calc engine. `calc.ts` orchestrates; `memory.ts` (KV cache, weights, activations, per-rank divisors), `prefill.ts` / `decode.ts` (roofline per phase), `roofline.ts` (compute/memory/comms max), `parallelism.ts` (TP/PP/EP/DP divisors + comms bytes), `queueModel.ts` (under-load N-sweep: `computeNMax`, `loadCurve`), `opPoints.ts`, `dtypes.ts`, `derivation.ts`. No Svelte, no DOM — importable from CLI and tests.
- `src/data/` — the hardware/model database. `models.ts` (ModelArch entries), `accelerators.ts` / `interconnects.ts` / `systems.ts` (hardware), `sources.ts` (citation registry), `workload-presets.ts` (benchmark presets). This is sourced data — see the discipline section below.
- `src/ui/` — Svelte 5 components + stores. `stores.ts` holds all input state; URL hash encodes shareable state (`share.ts`). Tabs: Calculator, Simulator (single-request + under-load), Info (catalog).
- `src/cli.ts` / `bin/llm-calc.mjs` — CLI front-end over the same engine.
- `scripts/` — horizontal-scaling survey enumerations (`comms-survey.ts`, etc.).
- `test/` — Vitest, mirrors `src/` paths. Engine integration tests in `test/engine/calc.test.ts`; UI logic + store tests in `test/ui/`.

## Development Practice

Follow Test-Driven Development: write the failing test first, confirm red, implement minimally to green, refactor. Never write implementation before a failing test exists. Engine changes especially — a mis-entered KV-byte or attention dim looks fine until it silently breaks a real workload's roofline.

Comments capture *why*, not *what* — intent, non-obvious constraints, decisions a future reader couldn't recover from the code. Drop comments that restate the next line.

## Data discipline (the database is sourced, not invented)

The `src/data/` entries are the product's credibility. Read [`docs/data-philosophy.md`](docs/data-philosophy.md) before touching them.

- **Adding a model** → use the `adding-a-model` skill. Source architecture from HuggingFace `config.json` (the file the model actually loads from), `paramCount` from the official card, novel attention from the paper. Write the test first.
- **Adding hardware** (accelerator / interconnect / system) → use the `adding-hardware-sku` skill.
- **Writing any citable perf number** (TFLOPS per dtype, sustained bandwidth, achievable-vs-theoretical) → use the `verifying-achievable-perf-numbers` skill. Never infer a date/URL/value from adjacent knowledge — verify the source. Research-agent summaries and "I remember when that shipped" are systematically wrong about hardware variants.

Never fabricate. If a value can't be cleanly sourced, drop the entry rather than ship a plausible-but-wrong number.

## Design docs

Specs and implementation plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/` (dated `YYYY-MM-DD-<topic>`). New non-trivial features brainstorm → spec → plan → implement.

## Deploy

Cloudflare Workers via `wrangler.toml`. The web app is a static build (`dist/`) served by a Worker. `npm run deploy:dry` to validate before a real `npm run deploy`.
