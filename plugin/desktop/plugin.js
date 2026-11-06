// Tironian's page and status-bar chip for Hermes Desktop.
// Loaded uncompiled by the Desktop plugin SDK, so it uses jsx() calls instead of JSX syntax.
import { Button, EmptyState, ROUTES_AREA, SIDEBAR_NAV_AREA, STATUSBAR_AREAS, host } from '@hermes/plugin-sdk'
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

function TestResult({ f }) {
  if (f.eval_n == null) return jsx('span', { className: 'text-(--ui-text-quaternary)', children: 'not tested' })
  const ok = !!f.gate_passed
  return jsx('span', {
    title: f.gate_reason || '',
    className: ok ? 'text-(--ui-accent)' : 'text-(--ui-text-secondary)',
    children: `${f.eval_passed}/${f.eval_n} passed, ${pct(f.wilson_lo)} to ${pct(f.wilson_hi)}`
  })
}

function FamilyRow({ ctx, f, minTurns, busy, onChange }) {
  const [quote, setQuote] = useState(null)
  const [note, setNote] = useState(null)
  const [label, color] = STATUS[f.status] || [f.status, 'var(--ui-text-tertiary)']
  const ready = f.n_good >= minTurns
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

  return jsxs('div', {
    className: 'flex flex-col gap-1 border-b border-(--ui-stroke-secondary) py-2 last:border-b-0',
    children: [
      jsxs('div', {
        className: 'grid grid-cols-[minmax(0,1fr)_5rem_5rem_8rem_minmax(0,14rem)] items-center gap-3 text-sm',
        children: [
          jsxs('div', {
            className: 'min-w-0',
            children: [
              jsx('div', { className: 'truncate', children: f.name }),
              jsx('div', { className: 'truncate text-[0.6875rem] text-(--ui-text-quaternary)', children: `${f.id} · ${(f.tools || []).join(', ') || 'no tools'}` })
            ]
          }),
          jsx('div', { className: 'tabular-nums text-(--ui-text-secondary)', children: `${f.n_turns} req` }),
          jsx('div', { className: 'tabular-nums text-(--ui-text-secondary)', children: `${f.n_good} good` }),
          jsx('div', { className: 'text-xs', style: { color }, children: label }),
          jsx('div', { className: 'flex justify-end', children: action })
        ]
      }),
      jsxs('div', {
        className: 'flex gap-4 text-xs text-(--ui-text-tertiary)',
        children: [
          jsx(TestResult, { f }),
          f.shadow_n ? jsx('span', { children: `shadow: agreed ${f.shadow_agree} of ${f.shadow_n}` }) : null,
          f.served ? jsx('span', { children: `answered ${f.served} requests` }) : null,
          note ? jsx('span', { className: 'text-(--ui-text-secondary)', children: note }) : null
        ]
      })
    ]
  })
}

function TironianPage({ ctx }) {
  const [data, error, reload] = useSummary(ctx)
  const [fams, setFams] = useState({ families: [], min_family_turns: 25 })
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
            jsx(Button, {
              size: 'sm', variant: 'ghost',
              onClick: () => ctx.rest('/refresh', { method: 'POST' }).then(refreshAll),
              children: 'Re-read history'
            })
          ]
        }),
        jsxs('div', {
          className: 'flex flex-wrap gap-3',
          children: [
            jsx(Tile, { label: 'Requests learned', value: stats.turns, note: `${stats.good} graded good` }),
            jsx(Tile, { label: 'Answered locally', value: pct(stats.local_share), note: `${stats.served} requests` }),
            jsx(Tile, { label: 'Saved', value: usd(stats.saved_usd), note: report.spend_30d_usd ? `of ${usd(report.spend_30d_usd)} in 30 days` : 'model spend' }),
            jsx(Tile, { label: 'Silent fallbacks', value: stats.fallbacks, note: 'your model answered instead' }),
            jsx(Tile, { label: 'Models', value: `${stats.families_live} live`, note: `${stats.families_shadow} in shadow` })
          ]
        }),
        !settings.routing_enabled && (stats.families_live || stats.families_shadow)
          ? jsx('div', {
              className: 'rounded-md border border-(--ui-stroke-secondary) px-3 py-2 text-sm text-(--ui-text-secondary)',
              children: 'Routing is off, so your own model answers everything. Turn on "Let promoted students answer" in Settings, Plugins, Tironian.'
            })
          : null,
        job ? jsx(JobCard, { ctx, job, onChange: refreshAll }) : null,
        jsx(Section, {
          title: 'Task families',
          right: jsx('span', { className: 'text-xs text-(--ui-text-tertiary)', children: `a model needs ${fams.min_family_turns} good examples, then passes a test at ${pct(settings.promotion_threshold)} or better` }),
          children: fams.families.length
            ? jsx('div', {
                className: 'flex flex-col',
                children: fams.families.map(f => jsx(FamilyRow, { key: f.id, ctx, f, minTurns: fams.min_family_turns, busy: !!job, onChange: refreshAll }))
              })
            : jsx('div', { className: 'text-sm text-(--ui-text-tertiary)', children: 'No recurring tasks found yet.' })
        }),
        jsx('footer', {
          className: 'text-xs text-(--ui-text-quaternary)',
          children: `${stats.redactions || 0} secrets and personal details were masked before anything was stored. Nothing leaves this computer until you start a training run, and then only to your own Modal account (capped at ${usd(settings.max_usd_per_run)} a run).`
        })
      ]
    })
  })
}

function Chip({ ctx }) {
  const [data] = useSummary(ctx)
  if (!data) return null
  const { stats, job } = data
  const text = job
    ? `Tironian training ${pct(job.progress)}`
    : stats.served
      ? `Tironian ${pct(stats.local_share)} local`
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
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: PATH }, render: () => jsx(TironianPage, { ctx }) },
      { id: 'nav', area: SIDEBAR_NAV_AREA, data: { path: PATH, label: 'Tironian', codicon: 'mortar-board' } },
      { id: 'chip', area: STATUSBAR_AREAS.right, order: 140, render: () => jsx(Chip, { ctx }) }
    ])
  }
}
