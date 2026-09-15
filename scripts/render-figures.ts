// Renders every figure to a standalone file so the drawings can be eyeballed
// without a browser. Nothing in the app imports this; it exists because label
// overflow and collisions are invisible to the unit tests.
//
//   npm run render:figures [outDir]      (default .figures-out/)
import { mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { createServer } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import type { Component } from 'svelte'

const SVG_NS = 'http://www.w3.org/2000/svg'

// css: 'injected' so scoped styles reach render()'s `head`; dev: false because
// the dev-mode element tracker needs a client runtime we do not have here.
async function devServer() {
  return createServer({
    configFile: false,
    server: { middlewareMode: true },
    appType: 'custom',
    optimizeDeps: { noDiscovery: true, include: [] },
    plugins: [svelte({ compilerOptions: { css: 'injected', dev: false } })],
  })
}

// A component whose root is <svg> becomes a standalone .svg once it carries the
// namespace and the scoped styles; anything else stays an HTML fragment.
function toFile(head: string, body: string): { ext: string; text: string } {
  // Svelte wraps SSR output in hydration markers; they are not valid SVG.
  const trimmed = body.trim().replace(/^<!--\[-->/, '').replace(/<!--\]-->$/, '').trim()
  if (!trimmed.startsWith('<svg')) return { ext: 'html', text: `${head}\n${trimmed}\n` }
  const styles = head.trim()
  const withNs = trimmed.replace(/^<svg\b/, `<svg xmlns="${SVG_NS}"`)
  const withStyles = styles ? withNs.replace(/^(<svg[^>]*>)/, `$1${styles}`) : withNs
  return { ext: 'svg', text: `${withStyles}\n` }
}

type Render = (c: Component, o: { props: Record<string, unknown> }) => { head: string; body: string }

function write(render: Render, dir: string, name: string, component: Component, props: Record<string, unknown>): string {
  const { head, body } = render(component, { props })
  const { ext, text } = toFile(head, body)
  const file = join(dir, `${name}.${ext}`)
  writeFileSync(file, text)
  return file
}

async function main() {
  const outDir = resolve(process.argv[2] ?? '.figures-out')
  rmSync(outDir, { recursive: true, force: true })
  mkdirSync(outDir, { recursive: true })

  const server = await devServer()
  try {
    // Through the server, so it is the same svelte instance the components compiled against.
    const { render } = (await server.ssrLoadModule('svelte/server')) as { render: Render }
    const { SCHEMATIC_COMPONENTS } = await server.ssrLoadModule('/src/ui/schematics/components.ts')
    const { schematicFor } = await server.ssrLoadModule('/src/ui/schematics/meta.ts')
    const SchematicFrame = (await server.ssrLoadModule('/src/ui/schematics/SchematicFrame.svelte')).default
    const ArchGeometry = (await server.ssrLoadModule('/src/ui/ArchGeometry.svelte')).default
    const { geometryModel } = await server.ssrLoadModule('/src/ui/archGeometry.ts')
    const { MODELS } = await server.ssrLoadModule('/src/data/index.ts')

    let n = 0
    for (const [id, component] of Object.entries(SCHEMATIC_COMPONENTS as Record<string, Component>)) {
      write(render, outDir, `schematic-${id}`, SchematicFrame, { meta: schematicFor(id), component })
      n++
    }
    for (const m of MODELS as { id: string }[]) {
      write(render, outDir, `geometry-${m.id}`, ArchGeometry, { geometry: geometryModel(m) })
      n++
    }
    console.log(`${n} figures → ${outDir}`)
  } finally {
    await server.close()
  }
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
