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
