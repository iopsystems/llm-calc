import { mount } from 'svelte'
import App from './ui/App.svelte'
import { readUrlIntoStores, startUrlSync } from './ui/share'
import { readCompareUrlIntoStores, startCompareUrlSync } from './ui/compareShare'
import { initRouteSync } from './ui/route'
import { initNativeDtypeSync, seedCompareFromCalc } from './ui/stores'

readUrlIntoStores()
// Seed compare from the calc selection unless the URL already carries a compare
// payload (a shared compare link must win over the seed).
const hasCompareUrl = typeof window !== 'undefined'
  && window.location.hash.replace(/^#/, '').startsWith('compare?')
readCompareUrlIntoStores()
if (!hasCompareUrl) seedCompareFromCalc()
initNativeDtypeSync()
const app = mount(App, { target: document.getElementById('app')! })
startUrlSync()
startCompareUrlSync()
initRouteSync()
export default app
