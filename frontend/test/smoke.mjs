import assert from 'node:assert/strict'
import { unlink } from 'node:fs/promises'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { build } from 'esbuild'

const output = new URL('./.app-smoke.mjs', import.meta.url)
try {
  await build({ entryPoints: ['src/Dashboard.jsx'], outfile: output.pathname, bundle: true, platform: 'node', format: 'esm', packages: 'external' })
  const { default: Dashboard } = await import(output.href)
  const html = renderToStaticMarkup(createElement(Dashboard))
  for (const label of ['Headcount', 'Store footfall', 'Checkout pulse', 'Spatial Analysis and Reports', 'Video Analysis', 'Task Queues', 'System Logs']) {
    assert.ok(html.includes(label), `Missing dashboard content: ${label}`)
  }
  assert.ok(!html.includes('STORE FLOOR PLAN') && !html.includes('Inventory task queue'), 'Overview includes another page')
  assert.ok(html.includes('COGNO') && html.includes('VALLEY') && !/TRACK\s*K|TRACKK|>TK</i.test(html), 'Dashboard still shows the old project label')
  assert.ok(html.includes('Shopper count unavailable') && html.includes('aria-controls="primary-navigation"'), 'Empty state or mobile navigation is inaccessible')
  assert.ok(!html.includes('NO ANALYSIS YET') && !html.includes('class="site-pill"'), 'Overview includes redundant empty-state chrome')
  for (const [hash, expected] of [['#reports', 'Spatial analysis &amp; reports'], ['#cameras', 'Video analysis'], ['#tasks', 'Inventory task queue'], ['#logs', 'System activity']]) {
    globalThis.window = { location: { hash } }
    const page = renderToStaticMarkup(createElement(Dashboard))
    assert.ok(page.includes(expected), `Missing ${hash} page`)
    if (hash === '#reports') {
      assert.ok(page.includes('No measured activity for this session') && page.includes('STORE FLOOR PLAN') && page.includes('HEATMAP') && page.includes('EXPORT REPORT'), 'Spatial report page is incomplete')
      assert.equal((page.match(/STORE FLOOR PLAN/g) ?? []).length, 1, 'Spatial report duplicates the floor plan')
      assert.ok(!page.includes('Final test video') && !page.includes('3D scene orbit') && !page.includes('Window shopping'), 'Spatial report includes video processing')
      const electronics = renderToStaticMarkup(createElement(Dashboard, { initialSite: 'electronics' }))
      assert.ok(electronics.includes('<model-viewer') && electronics.includes('/api/electronics/scene.glb?v=5') && electronics.includes('camera-orbit="32deg 68deg 15m"') && electronics.includes('camera-target="-4m 2.6m -1m"') && electronics.includes('3D scene orbit') && electronics.includes('Ground floor') && electronics.includes('Upper') && electronics.includes('Refrigeration') && electronics.includes('STORE FLOOR PLAN'), 'Electronics floor plan is missing the shop model')
      assert.ok(electronics.includes('camera-controls') && electronics.includes('auto-rotate') && electronics.includes('Two-floor electronics shop'), 'Electronics orbit controls are missing')
      assert.ok(electronics.includes('class="shopper-dot"') && electronics.includes('HEATMAP') && electronics.includes('TRACKS') && electronics.includes('Entry crossings') && electronics.includes('Checkout service') && electronics.includes('Zone engagement') && electronics.includes('Shelf findings') && electronics.includes('Flagship phone display'), 'Electronics analysis is missing')
      assert.ok(!electronics.includes('DEMO DATA') && !electronics.includes('presentation mock') && !electronics.includes('NOT MEASURED'), 'Electronics page still labels the figures as a mock')
      assert.ok(!electronics.includes('Window shopping') && !electronics.includes('STAFF ALERTED') && !electronics.includes('Mark greeted'), 'Electronics greeting watch should stay removed')
      assert.ok(!electronics.includes('>Dairy<') && !electronics.includes('Snacks') && !electronics.includes('Beverages') && !electronics.includes('Canned foods'), 'Electronics floor plan still shows the grocery aisles')
    }
    if (hash === '#cameras') {
      assert.ok(page.includes('Final test video') && page.includes('3D scene orbit') && page.includes('UNAVAILABLE'), 'Video workspace is incomplete')
      assert.ok(page.includes('Review video') && page.includes('VIDEO EVENTS'), 'Camera review workflow is incomplete')
      assert.ok(!page.includes('STORE FLOOR PLAN') && !page.includes('EXPORT REPORT') && !page.includes('ZONE DETAIL'), 'Video analysis includes spatial reports')
      assert.ok(!page.includes('STORE ACTIVITY REPORT') && !page.includes('RECENT ANALYSIS'), 'Camera review must not present a replay as measured analysis')
      assert.ok(!page.includes('Define camera regions') && !page.includes('Calibrate this angle'), 'Manager view includes camera setup controls')
      assert.ok(!page.includes('00:25') && !page.includes('Dairy facing removed'), 'Scripted cues must not look like detected events')
      assert.ok(!page.includes('class="shopper-dot"'), 'Unprocessed camera page shows shopper marks')
    }
  }
  const observedAt = new Date().toISOString()
  const measuredFixture = {
    mode: 'vision', measurement_source: 'model_inference', updated_at: observedAt, current_scenario: 'final_test',
    measurements: { visible_shoppers: true, footfall: true, queues: true, positions: true, shelf_gaps: true },
    live_occupancy: 1,
    footfall: { in: 3, out: 1, hourly: [{ time: observedAt, in: 3, out: 1 }] },
    checkout: { counter_01: { queue_count: 4, wait_seconds: 12 }, counter_02: { queue_count: 0, status: 'active', response_delay_seconds: 2 } },
    occupancy: { count: 1, density_pct: 4, zones: { lane1: { visits: 1, headcount: 0, dwell_hours: 0.01 }, shelf1: { visits: 1, headcount: 1, dwell_hours: 0.02 } }, heatmap: { size: 2, seconds: [0, 2, 3, 0] } },
    zones: [{ id: 'lane1', title: 'Counter 01', category: 'Checkout', left: 10, top: 70, width: 20, height: 15 }, { id: 'shelf1', title: 'Dairy - 01', category: 'Dairy', left: 15, top: 15, width: 15, height: 40 }],
    task_queue: [{ id: 'fixture-ticket', item: 'Fixture carton', category: 'Dairy', location: 'Fixture shelf', missing: 1, status: 'Open', shelf: 'shelf1', evidence: { camera_id: 'cam_entrance', video_seconds: 25 } }],
    system_logs: [], shoppers: [{ id: 'cam_entrance_1', x: 50, y: 50, zone: 'shelf1', trail: [] }], camera_feeds: {},
  }
  globalThis.window = { location: { hash: '#reports' } }
  const populatedReport = renderToStaticMarkup(createElement(Dashboard, { initialTelemetry: measuredFixture }))
  for (const label of ['Entry crossings', 'Checkout service', 'Zone engagement', 'Fixture carton', 'VIEW FRAME · 00:25', 'TRACKS', 'HEATMAP', 'shopper-dot', 'LAST SIGNAL']) {
    assert.ok(populatedReport.includes(label), `Measured report missing: ${label}`)
  }
  const completedReport = renderToStaticMarkup(createElement(Dashboard, { initialTelemetry: {
    ...measuredFixture,
    queue_alerts: [{ video_seconds: 43, count: 4 }],
    checkout: { ...measuredFixture.checkout, counter_01: { queue_count: 0, wait_seconds: 0 } },
  } }))
  assert.ok(completedReport.includes('Queue length 4: 1 alert at 00:43') && !completedReport.includes('ANALYSIS REPLAY'), 'Completed report must show the queue alert count and clip time without replay controls')
  globalThis.window.location.hash = '#overview'
  const populatedOverview = renderToStaticMarkup(createElement(Dashboard, { initialTelemetry: measuredFixture }))
  assert.ok(populatedOverview.includes('entries') && populatedOverview.includes('Exits'), 'Footfall key is missing')
  const referenceFixture = { ...measuredFixture, measurement_source: 'isaac_ground_truth', task_queue: [], checkout: { counter_01: { queue_count: 4 }, counter_02: { queue_count: 0, status: 'standby' } } }
  const referenceOverview = renderToStaticMarkup(createElement(Dashboard, { initialTelemetry: referenceFixture }))
  assert.ok(referenceOverview.includes('ISAAC REFERENCE') && referenceOverview.includes('Reference headcount') && !referenceOverview.includes('LIMITED VIDEO ANALYSIS'), 'Isaac replay must be labeled as simulator reference')
  globalThis.window.location.hash = '#reports'
  const referenceReport = renderToStaticMarkup(createElement(Dashboard, { initialTelemetry: referenceFixture }))
  assert.ok(referenceReport.includes('ISAAC REFERENCE') && referenceReport.includes('No shelf gaps detected.'), 'Reference report does not show baseline findings')
  delete globalThis.window
  console.log('Dashboard smoke check passed')
} finally {
  await unlink(output).catch(() => {})
}
