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

// Partial until Task 12 adds the remaining drawings and tightens this to Record.
export const SCHEMATIC_COMPONENTS: Partial<Record<SchematicId, Component>> = {
  'full': FullSchematic,
  'sliding': SlidingSchematic,
  'hybrid': HybridSchematic,
  'partial': PartialSchematic,
  'mla': MlaSchematic,
  'mla-dsa': MlaDsaSchematic,
  'msa-hybrid': MsaSchematic,
  'csa-hca-hybrid': CsaHcaSchematic,
}
