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
