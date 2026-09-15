# Model Math Audit Corrections Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task by task with regression verification.

**Goal:** Correct the nine findings in the local model-math audit and deliver a tested PR.
**Architecture:** Separate attention operation counts from cached storage. Compute runtime on physical ranks with explicit topology and routing assumptions. Keep capacity at peak sequence length while averaging runtime over generated positions.
**Tech stack:** TypeScript, Vitest, Svelte 5.
**Spec:** Local review `.git/review-feedback/threads/origin-main/001-codex-be79928.md`, baseline be79928; user authorized fixes and PR.

## Constraints

Primary configs for model geometry; no new dependencies. Test independent operation counts and weight conservation. Preserve existing phase/result interfaces where possible. Communicate analytical assumptions in README/PR. Do not mix peak capacity with average traffic. Do not merge the PR.

## Task 1: Model data and cache geometry

Owner: memory worker. Files: models.ts, memory.ts, memory/queue regression tests as needed. Correct Qwen27 dense classification, Delta head counts (27B48,35B32), 35B FFN512. Delta recurrent storage FP32; V4 single shared KV vector. Add exported `cacheBytesAtSequence(model, kvDtype, seqlen, tp=1, pp=1): {bytes:number, recurrentBytes:number}` and `averageDecodeCacheBytes(model,kvDtype,prompt,output,tp=1,pp=1)` returning the same shape. Peak cache uses full prompt+output; average positions are prompt through prompt+max(output,1)-1. TP shards main GQA/state heads where possible; shared MLA/MSA-index/V4 cache replicated. PP divides layer storage uniformly. Average caps at each position, using closed forms, not O(context) loops.

- [x] Write independent red tests: Qwen dense active27B; Delta FP32 invariant; V4 half-main cache; MSA index retained; average full/window cache including cap-crossing.
- [x] Implement minimal storage helpers and integrate peak memory without changing runtime functions.
- [x] Run owned tests; report helper interface and any expected legacy-test changes.

## Task 2: Attention operation counting

Owner: attention worker. Files: new attention.ts, new attention.test.ts. Export `prefillAttentionFlops(model,prompt)` and `averageDecodeAttentionFlops(model,prompt,output)`. Include both QK and AV. Full causal prefill sums visible positions1..prompt. Decode averages prompt..prompt+max(output,1)-1. For full/GQA cost per pair=4*heads*headDim. Absorbed MLA decode=2*heads*(2*latent+rope); expanded MLA prefill=2*heads*(nope+rope+value). Cap sliding/topK per query; sum hybrid layers. Preserve indexer-compute omission as documented assumption. Recurrent compute remains in callers.

- [x] Red independent matrix/pair enumeration tests: full, sliding below/above/crossing cap, hybrid, MLA128heads, DSA/MSA, compressed attention.
- [x] Implement pure helpers and run tests. Do not edit prefill/decode/memory owned by other workers.

## Task 3: Physical parallelism and runtime integration

Owner: controller. Files: parallelism.ts, prefill.ts, decode.ts, calc.ts, queueModel.ts, associated tests and UI/documentation.

- [x] Red tests for default MoE weight conservation and TP8 dense runtime scaling; use TP/PP defaults only so default degrees never overcommit.
- [x] Require TP*PP*EP*DP <= physical count for independent mesh axes. Infer shared/routed pool from P,A,E,k with bounds; only routed weights divide by EP.
- [x] Add expected batched weight reads: shared+routed*(1-(1-k/E)^batch), documented independent uniform routing. Test batch1, saturation, dense identity.
- [x] Integrate attention and average cache helpers. TP divides per-rank work, DP partitions requests; PP stages execute sequentially for latency. Keep aggregate metric volume fields and document rank-local rate calculations.
- [x] Keep original concurrency meaning (decode batch); prefill models one prompt, so prefill compute/communications both use one prompt. Capacity remains conservative for concurrent prompts.
- [x] Update derivation descriptions, mathematical assumptions, and tests whose obsolete literal values encoded incorrect formulas. Preserve explicit MTP ceiling labeling rather than introducing a speculative runtime subsystem.

## Task 4: Integration, review, and delivery

- [x] Run all tests, type check, skill-sync, build, diff check. Resolve failures according to independent math, not by loosening assertions.
- [x] Independent full-diff review and fixes. Document residual approximations.
- [ ] Commit and push feature branch; create PR explaining concrete numerical corrections and validation.
- [ ] Record local review resolution with commits and PR link.

## Verification outcome

526 tests passed across 39 files. Svelte check: zero errors/warnings. Skill sync and production build passed (build retains a >500 kB chunk advisory). Independent review identified uneven-DP aggregate traffic, now regression-tested and fixed, and EP ideal-balance labeling, now explicit. No unresolved material review findings within these assumptions.
