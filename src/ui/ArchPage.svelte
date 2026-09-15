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
