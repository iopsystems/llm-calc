import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'
import { readCompareUrlIntoStores, startCompareUrlSync } from './ui/compareShare'
import { initRouteSync } from './ui/route'
import { initNativeDtypeSync, seedCompareOnFirstEntry } from './ui/stores'

readUrlIntoStores()
// Seed compare from the calc selection on first entry to the tab, unless the
// URL already carries a compare payload (a shared compare link must win).
const hasCompareUrl = typeof window !== 'undefined'
  && window.location.hash.replace(/^#/, '').startsWith('compare?')
readCompareUrlIntoStores()
initNativeDtypeSync()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
// Seed before the URL writer is wired so the first compare URL carries the seed.
if (!hasCompareUrl) seedCompareOnFirstEntry()
startCompareUrlSync()
initRouteSync()
export default app
