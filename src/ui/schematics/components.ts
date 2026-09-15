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
