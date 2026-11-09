// Tironian's page and status-bar chip for Hermes Desktop.
// Loaded uncompiled by the Desktop plugin SDK, so it uses jsx() calls instead of JSX syntax.
import * as sdk from '@hermes/plugin-sdk'
import { Button, EmptyState, ROUTES_AREA, SIDEBAR_NAV_AREA, STATUSBAR_AREAS, host } from '@hermes/plugin-sdk'
// Optional: Desktop builds without transcript directives don't export it, and a named import
// of a missing export would stop the whole plugin from loading.
const TRANSCRIPT_DIRECTIVE_AREA = sdk.TRANSCRIPT_DIRECTIVE_AREA
import { useCallback, useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const PATH = '/tironian'

const pct = x => `${Math.round((x || 0) * 100)}%`
const usd = x => (x >= 0.01 || !x ? `$${(x || 0).toFixed(2)}` : `$${x.toFixed(4)}`)

const STATUS = {
  candidate: ['learning', 'var(--ui-text-tertiary)'],
  trained: ['trained, not tested', 'var(--ui-text-secondary)'],
  evaluated: ['failed its test', 'var(--ui-text-secondary)'],
  shadow: ['shadow', 'var(--ui-accent)'],
  live: ['answering', 'var(--ui-accent)'],
  demoted: ['stopped', 'var(--ui-text-tertiary)']
}

function useSummary(ctx) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const load = useCallback(() => {
    ctx.rest('/summary').then(d => { setData(d); setError(null) }, e => setError(String(e?.message || e)))
  }, [ctx])
  useEffect(() => {
    load()
    const stopTimer = ctx.setInterval(load, 5000)
    const stopEvents = ctx.onEvent('*', ev => { if (String(ev?.type || '').startsWith('plugin.tironian.')) load() })
    return () => { stopTimer(); stopEvents() }
  }, [ctx, load])
  return [data, error, load]
}

function Tile({ label, value, note }) {
  return jsxs('div', {
    className: 'flex min-w-[9rem] flex-1 flex-col gap-0.5 rounded-md border border-(--ui-stroke-secondary) px-3 py-2',
    children: [
      jsx('div', { className: 'text-[0.6875rem] uppercase tracking-wide text-(--ui-text-tertiary)', children: label }),
      jsx('div', { className: 'text-xl font-medium tabular-nums', children: value }),
      note ? jsx('div', { className: 'text-[0.6875rem] text-(--ui-text-tertiary)', children: note }) : null
    ]
  })
}

function Section({ title, children, right }) {
  return jsxs('section', {
    className: 'flex flex-col gap-2',
    children: [
      jsxs('div', {
        className: 'flex items-baseline justify-between',
        children: [jsx('h2', { className: 'text-sm font-medium', children: title }), right || null]
      }),
      children
    ]
  })
}

function Progress({ value }) {
  return jsx('div', {
    className: 'h-1.5 w-full overflow-hidden rounded bg-(--ui-stroke-secondary)',
    children: jsx('div', { className: 'h-full bg-(--ui-accent) transition-all', style: { width: pct(value) } })
  })
}

function JobCard({ ctx, job, onChange }) {
  const cancel = () => ctx.rest('/train/cancel', { method: 'POST', body: { job: job.id } }).then(onChange)
  return jsxs('div', {
    className: 'flex flex-col gap-2 rounded-md border border-(--ui-stroke-secondary) px-3 py-2',
    children: [
      jsxs('div', {
        className: 'flex items-center justify-between text-sm',
        children: [
          jsx('span', { children: `Training ${job.family_id} on ${job.backend === 'local' ? 'your GPU' : 'Modal'}` }),
          jsx(Button, { size: 'sm', variant: 'ghost', onClick: cancel, children: 'Cancel' })
        ]
      }),
      jsx(Progress, { value: job.progress }),
      jsx('div', {
        className: 'text-xs text-(--ui-text-tertiary)',
        children: `${job.status} · ${job.message || ''}${job.cost_est_usd ? ` · estimate ${usd(job.cost_est_usd)}` : ''}`
      })
    ]
  })
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`
const KIND = { first: 'first steps', follow: 'after a tool result' }

function TestResult({ f }) {
  if (f.eval_n == null) return jsx('span', { className: 'text-(--ui-text-quaternary)', children: 'not tested' })
  const ok = !!f.gate_passed
  const steps = f.steps || {}
  const serve = f.serve_steps || []
  // Passed for some kinds of step only: show each kind on its own, as the gate judged them.
  if (ok && serve.length && Object.keys(steps).length > serve.length) {
    return jsxs('span', {
      title: f.gate_reason || '',
      style: { display: 'inline-flex', gap: 16, flexWrap: 'wrap' },
      children: Object.entries(steps).map(([k, s]) => jsx('span', {
        key: k,
        style: { color: serve.includes(k) ? 'var(--ui-accent)' : 'var(--ui-text-tertiary)' },
        children: serve.includes(k)
          ? `${KIND[k] || k}: ${s.passed}/${s.n}, lower bound ${pct(s.wilson_lo)}, passes`
          : `${KIND[k] || k}: ${s.passed}/${s.n}, left to your model`
      }))
    })
  }
  return jsx('span', {
    title: f.gate_reason || '',
    className: ok ? 'text-(--ui-accent)' : 'text-(--ui-text-secondary)',
    children: `${f.eval_passed}/${f.eval_n} passed, ${pct(f.wilson_lo)} to ${pct(f.wilson_hi)}`
  })
}

function FamilyRow({ ctx, f, minTurns, busy, onChange }) {
  const [quote, setQuote] = useState(null)
  const [note, setNote] = useState(null)
  const ready = f.n_good >= minTurns
  const [label, color] = f.status === 'candidate' && ready
    ? ['ready to train', 'var(--ui-text-secondary)']
    : STATUS[f.status] || [f.status, 'var(--ui-text-tertiary)']
  const mode = m => ctx.rest(`/families/${f.id}/mode`, { method: 'POST', body: { mode: m } })
    .then(r => { setNote(r.message); onChange() })
  const prepare = () => ctx.rest('/train/prepare', { method: 'POST', body: { family: f.id } })
    .then(setQuote, e => setNote(String(e?.message || e)))
  const start = () => ctx.rest('/train/start', { method: 'POST', body: { family: f.id, dataset_id: quote.dataset_id } })
    .then(() => { setQuote(null); onChange() }, e => setNote(String(e?.message || e)))

  let action = null
  if (quote) {
    action = quote.enough
      ? jsxs('div', {
          className: 'flex items-center gap-2',
          children: [
            jsx('span', {
              className: 'text-xs text-(--ui-text-tertiary)',
              children: `${quote.n_train} to learn from, ${quote.n_heldout} held out, about ${Math.max(1, Math.round(quote.estimate_s / 60))} min, ${quote.backend === 'local' ? 'free' : usd(quote.estimate_usd)}`
            }),
            jsx(Button, { size: 'sm', onClick: start, children: 'Start training' }),
            jsx(Button, { size: 'sm', variant: 'ghost', onClick: () => setQuote(null), children: 'Not now' })
          ]
        })
      : jsx('span', { className: 'text-xs text-(--ui-text-tertiary)', children: `Only ${quote.n_train} usable examples so far.` })
  } else if (f.status === 'live') {
    action = jsx(Button, { size: 'sm', variant: 'ghost', onClick: () => mode('demote'), children: 'Stop' })
  } else if (f.status === 'shadow') {
    action = jsx(Button, { size: 'sm', onClick: () => mode('promote'), children: 'Let it answer' })
  } else if (f.gate_passed) {
    action = jsx(Button, { size: 'sm', onClick: () => mode('shadow'), children: 'Try in shadow' })
  } else if (ready && !busy) {
    action = jsx(Button, { size: 'sm', variant: 'outline', onClick: prepare, children: f.student_id ? 'Train again' : 'Train' })
  } else if (!ready) {
    action = jsx('span', { className: 'text-xs text-(--ui-text-quaternary)', children: `needs ${minTurns - f.n_good} more good` })
  }

  // Inline styles on purpose: Desktop ships only the Tailwind classes the app itself uses,
  // so arbitrary layout classes from a plugin silently do nothing.
  const progress = Math.min(1, f.n_good / Math.max(1, minTurns))
  return jsxs('div', {
    style: { display: 'flex', flexDirection: 'column', gap: 6, padding: '12px 0', borderBottom: '1px solid var(--ui-stroke-secondary)' },
    children: [
      jsxs('div', {
        style: { display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 72px 72px 120px minmax(0,auto)', alignItems: 'center', gap: 12, fontSize: 13.5 },
        children: [
          jsxs('div', {
            style: { minWidth: 0 },
            children: [
              jsx('div', { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontWeight: 500 }, children: f.name }),
              jsx('div', { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 11, color: 'var(--ui-text-quaternary)' }, children: `${f.id} · ${(f.tools || []).join(', ') || 'no tools'}` })
            ]
          }),
          jsx('div', { style: { color: 'var(--ui-text-secondary)', fontVariantNumeric: 'tabular-nums' }, children: `${f.n_turns} req` }),
          jsx('div', { style: { color: 'var(--ui-text-secondary)', fontVariantNumeric: 'tabular-nums' }, children: `${f.n_good} good` }),
          jsx('div', { style: { color, fontSize: 12 }, children: label }),
          jsx('div', { style: { display: 'flex', justifyContent: 'flex-end' }, children: action })
        ]
      }),
      f.status === 'candidate'
        ? jsx('div', {
            title: `${f.n_good} of ${minTurns} good examples`,
            style: { height: 4, borderRadius: 3, background: 'var(--ui-stroke-secondary)', overflow: 'hidden' },
            children: jsx('div', { style: { width: pct(progress), height: '100%', background: 'var(--ui-accent)', transition: 'width .4s' } })
          })
        : null,
      jsxs('div', {
        style: { display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 12, color: 'var(--ui-text-tertiary)' },
        children: [
          jsx(TestResult, { f }),
          f.shadow_n ? jsx('span', { children: `shadow: agreed ${f.shadow_agree} of ${f.shadow_n}` }) : null,
          f.new_places && f.new_places.n
            ? jsx('span', {
                title: `Trained on ${(f.new_places.trained_on || []).join(', ')}. Tested on ${(f.new_places.trees || []).join(', ')}, which it never saw.`,
                children: `new folders: ${f.new_places.passed} of ${f.new_places.n}`
              })
            : null,
          f.spot_n ? jsx('span', { title: 'Your usual model answers a share of live requests and the student is judged against it, to catch drift', children: `spot checks: agreed ${f.spot_agree} of ${f.spot_n}` }) : null,
          f.status === 'shadow' && (f.status_reason || '').startsWith('drift')
            ? jsx('span', { style: { color: 'var(--ui-text-secondary)' }, children: `back in shadow: ${f.status_reason.replace(/^drift: /, '')}` })
            : null,
          f.served ? jsx('span', { children: f.routed
            ? `answered ${f.served} of ${f.routed} locally (${pct(f.served / f.routed)}), ${(f.tokens_avoided || 0).toLocaleString()} cloud tokens avoided`
            : `answered ${plural(f.served, 'request')}` }) : null,
          note ? jsx('span', { style: { color: 'var(--ui-text-secondary)' }, children: note }) : null
        ]
      })
    ]
  })
}

function TironianPage({ ctx }) {
  const [data, error, reload] = useSummary(ctx)
  const [fams, setFams] = useState({ families: [], min_family_turns: 25 })
  const [demo, setDemo] = useState(null)   // null, 'running', or the demo's lines
  const runDemo = () => {
    setDemo('running')
    ctx.rest('/demo', { method: 'POST' }).then(r => setDemo(r.lines || []), e => setDemo([`The demo failed: ${e?.message || e}`]))
  }
  const loadFams = useCallback(() => { ctx.rest('/families').then(setFams, () => {}) }, [ctx])
  useEffect(() => { loadFams(); return ctx.setInterval(loadFams, 8000) }, [ctx, loadFams])
  const refreshAll = () => { reload(); loadFams() }

  if (error && !data) {
    return jsx(EmptyState, {
      title: 'Tironian is not reachable',
      description: 'Enable the plugin in Settings, Plugins, then restart the gateway.'
    })
  }
  if (!data) return jsx('div', { className: 'p-6 text-sm text-(--ui-text-tertiary)', children: 'Loading…' })

  const { stats, report, settings, job } = data
  const savings = data.savings || { tokens_avoided: 0, families: {} }
  const empty = !report.turns
  return jsx('div', {
    className: 'h-full overflow-y-auto',
    children: jsxs('div', {
      className: 'mx-auto flex max-w-5xl flex-col gap-6 p-6',
      children: [
        jsxs('header', {
          className: 'flex items-end justify-between gap-4',
          children: [
            jsxs('div', {
              children: [
                jsx('h1', { className: 'text-lg font-medium', children: 'Tironian' }),
                jsx('p', {
                  className: 'text-sm text-(--ui-text-tertiary)',
                  children: empty
                    ? 'Nothing to learn from yet. Use Hermes as usual and come back.'
                    : `${pct(report.routine_share)} of your ${report.turns} requests are tasks you repeat. Those are what a model of your own can take over.`
                })
              ]
            }),
            jsxs('div', {
              style: { display: 'flex', gap: 8 },
              children: [
                jsx(Button, { size: 'sm', onClick: runDemo, disabled: demo === 'running',
                  children: demo === 'running' ? 'Running the demo…' : 'Run the demo' }),
                jsx(Button, {
                  size: 'sm', variant: 'ghost',
                  onClick: () => ctx.rest('/refresh', { method: 'POST' }).then(refreshAll),
                  children: 'Re-read history'
                })
              ]
            })
          ]
        }),
        Array.isArray(demo)
          ? jsxs('section', {
              style: { border: '1px solid var(--ui-stroke-secondary)', borderRadius: 8, padding: '10px 12px' },
              children: [
                jsxs('div', {
                  style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
                  children: [
                    jsx('div', { style: { fontSize: 13, fontWeight: 500 }, children: 'Demo: real Tironian code, scripted history and models, in its own folder' }),
                    jsx(Button, { size: 'sm', variant: 'ghost', onClick: () => setDemo(null), children: 'Close' })
                  ]
                }),
                jsx('pre', { style: { margin: 0, fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-wrap', color: 'var(--ui-text-secondary)', maxHeight: 420, overflowY: 'auto' }, children: demo.join('\n') })
              ]
            })
          : null,
        jsxs('div', {
          className: 'flex flex-wrap gap-3',
          children: [
            jsx(Tile, { label: 'Requests learned', value: stats.turns, note: `${stats.good} graded good` }),
            jsx(Tile, { label: 'Answered locally', value: stats.local_share > 0 && stats.local_share < 0.01 ? '<1%' : pct(stats.local_share), note: plural(stats.served, 'request') }),
            jsx(Tile, { label: 'Saved', value: usd(stats.saved_usd), note: report.spend_30d_usd ? `of ${usd(report.spend_30d_usd)} in 30 days` : 'model spend' }),
            jsx(Tile, { label: 'Cloud tokens avoided', value: (savings.tokens_avoided || 0).toLocaleString(), note: 'prompt tokens that never left this machine' }),
            jsx(Tile, { label: 'Silent fallbacks', value: stats.fallbacks, note: 'your model answered instead' }),
            jsx(Tile, { label: 'Models', value: `${stats.families_live} live`, note: `${stats.families_shadow} in shadow` })
          ]
        }),
        !settings.routing_enabled && (stats.families_live || stats.families_shadow)
          ? jsx('div', {
              className: 'rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-sm text-(--ui-text-secondary)',
              children: 'Students may not answer yet, so your own model answers everything (shadow mode still compares in the background). Turn on "Let promoted students answer" in Settings, Plugins, Tironian.'
            })
          : null,
        job ? jsx(JobCard, { ctx, job, onChange: refreshAll }) : null,
        jsx(Section, {
          title: 'Task families',
          right: jsx('span', { className: 'text-xs text-(--ui-text-tertiary)', children: `a model needs ${fams.min_family_turns} good examples, then passes a test at ${pct(settings.promotion_threshold)} or better` }),
          children: (() => {
            // One-off requests aren't a task yet; list them as a count instead of rows.
            const shown = fams.families.filter(f => f.n_turns >= 3 || f.status !== 'candidate')
            const hidden = fams.families.length - shown.length
            return shown.length
              ? jsxs('div', {
                  className: 'flex flex-col',
                  children: [
                    ...shown.map(f => jsx(FamilyRow, { key: f.id, ctx, f, minTurns: fams.min_family_turns, busy: !!job, onChange: refreshAll })),
                    hidden ? jsx('div', { key: 'hidden', style: { paddingTop: 10, fontSize: 12, color: 'var(--ui-text-quaternary)' }, children: `${plural(hidden, 'one-off group')} not shown` }) : null
                  ]
                })
              : jsx('div', { className: 'text-sm text-(--ui-text-tertiary)', children: 'No recurring tasks found yet.' })
          })()
        }),
        jsx('footer', {
          className: 'text-xs text-(--ui-text-quaternary)',
          children: `${stats.redactions || 0} secrets and personal details were masked before anything was stored. Tironian sends nothing anywhere unless you start a training run (then only to your own Modal account, capped at ${usd(settings.max_usd_per_run)} a run) or switch on the judge.`
        })
      ]
    })
  })
}

// The card under a reply whose step your own model answered. Attributes come from model-visible
// text, so each one is checked and the card renders nothing if any looks wrong.
function ProofCard({ attrs }) {
  const a = attrs || {}
  const frac = /^\d{1,4}\/\d{1,4}$/
  if (!frac.test(a.proven || '') || !/^\d{1,3}$/.test(a.bound || '') || (a.shadow && !frac.test(a.shadow))) return null
  const name = String(a.family || '').slice(0, 60)
  const tokens = /^\d{1,7}$/.test(a.tokens || '') ? Number(a.tokens) : 0
  const bits = [
    `proven ${a.proven} on ${a.steps === 'first steps' ? 'first steps' : 'steps'} it never saw (lower bound ${a.bound}%)`,
    a.shadow ? `agreed ${a.shadow} in shadow mode` : null,
    tokens ? `about ${tokens.toLocaleString()} cloud tokens kept local` : null
  ].filter(Boolean)
  return jsxs('div', {
    style: { display: 'flex', gap: 10, alignItems: 'flex-start', padding: '8px 12px', marginTop: 4, borderRadius: 8,
      border: '1px solid var(--ui-stroke-secondary)', background: 'var(--ui-bg-secondary, transparent)', fontSize: 12.5, lineHeight: 1.45 },
    children: [
      jsx('span', { 'aria-hidden': true, style: { fontSize: 14, lineHeight: '18px' }, children: '⚡' }),
      jsxs('div', {
        children: [
          jsxs('div', { style: { fontWeight: 500 }, children: [`Answered in part by your own model`, name ? ` · ${name}` : ''] }),
          jsx('div', { style: { color: 'var(--ui-text-tertiary)' }, children: bits.join(' · ') }),
          jsx('button', { type: 'button', onClick: () => host.navigate(PATH),
            style: { marginTop: 4, padding: 0, border: 0, background: 'none', color: 'var(--ui-accent)', cursor: 'pointer', fontSize: 12 },
            children: 'See the evidence' })
        ]
      })
    ]
  })
}

function Chip({ ctx }) {
  const [data] = useSummary(ctx)
  if (!data) return null
  const { stats, job } = data
  const text = job
    ? `Tironian training ${pct(job.progress)}`
    : stats.served
      ? `Tironian ${stats.local_share > 0 && stats.local_share < 0.01 ? '<1%' : pct(stats.local_share)} local`
      : `Tironian ${stats.turns} learned`
  return jsx('button', {
    type: 'button',
    title: 'Open Tironian',
    className: 'px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) hover:text-(--ui-text-secondary)',
    onClick: () => host.navigate(PATH),
    children: text
  })
}

export default {
  id: 'tironian',
  name: 'Tironian',
  register(ctx) {
    const items = [
      { id: 'page', area: ROUTES_AREA, data: { path: PATH }, render: () => jsx(TironianPage, { ctx }) },
      { id: 'nav', area: SIDEBAR_NAV_AREA, data: { path: PATH, label: 'Tironian', codicon: 'mortar-board' } },
      { id: 'chip', area: STATUSBAR_AREAS.right, order: 140, render: () => jsx(Chip, { ctx }) }
    ]
    // Older Desktop builds have no transcript directives; the page and chip still work there.
    if (TRANSCRIPT_DIRECTIVE_AREA) {
      items.push({ id: 'proof', area: TRANSCRIPT_DIRECTIVE_AREA,
        data: { name: 'tironian-proof', render: ({ attrs }) => jsx(ProofCard, { attrs }) } })
    }
    ctx.registerMany(items)
  }
}
