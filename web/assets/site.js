// Small, dependency-free behaviour for the Tironian site.
(() => {
  const $ = (s, el = document) => el.querySelector(s)
  const $$ = (s, el = document) => [...el.querySelectorAll(s)]
  const INSTALL = 'hermes plugins install angelraph/tironian'

  // Toast
  const toast = document.createElement('div')
  toast.className = 'toast'
  toast.setAttribute('role', 'status')
  document.body.appendChild(toast)
  let toastTimer
  const say = msg => {
    toast.textContent = msg
    toast.classList.add('show')
    clearTimeout(toastTimer)
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1600)
  }

  const copy = async (text, btn) => {
    try {
      await navigator.clipboard.writeText(text)
      if (btn) {
        const old = btn.textContent
        btn.textContent = 'Copied'
        btn.classList.add('done')
        setTimeout(() => { btn.textContent = old; btn.classList.remove('done') }, 1500)
      } else {
        say('Copied: ' + text)
      }
    } catch {
      say('Select the text and copy it with Ctrl+C')
    }
  }

  // Copy buttons: data-copy="text", or the text of the nearest <code>/<pre>
  $$('[data-copy]').forEach(btn => btn.addEventListener('click', () => {
    const v = btn.dataset.copy
    const target = v ? null : btn.closest('.cmd, .tab-panel, .term, pre')?.querySelector('code, pre')
    copy(v || (target ? target.innerText.replace(/^\$ /gm, '') : ''), btn)
  }))

  // Mobile menu
  const head = $('.site-head')
  $('.menu-btn')?.addEventListener('click', e => {
    const open = head.classList.toggle('menu-open')
    e.currentTarget.setAttribute('aria-expanded', String(open))
  })
  $$('.nav a').forEach(a => a.addEventListener('click', () => head.classList.remove('menu-open')))

  // Tabs (install tabs and the loop): [role=tablist] > button[aria-controls]
  $$('[role="tablist"]').forEach(list => {
    const tabs = $$('[role="tab"]', list)
    const show = tab => {
      tabs.forEach(t => {
        const on = t === tab
        t.setAttribute('aria-selected', String(on))
        t.tabIndex = on ? 0 : -1
        const panel = document.getElementById(t.getAttribute('aria-controls'))
        if (panel) panel.hidden = !on
      })
    }
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => { show(t); list.dataset.auto = 'off' })
      t.addEventListener('keydown', e => {
        if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return
        const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length]
        next.focus(); show(next); list.dataset.auto = 'off'
      })
    })
    // The loop advances on its own until someone clicks it (like Phoenix's tabs).
    if (list.dataset.auto === 'on') {
      setInterval(() => {
        if (list.dataset.auto !== 'on' || document.hidden) return
        const rect = list.getBoundingClientRect()
        if (rect.bottom < 0 || rect.top > innerHeight) return
        const i = tabs.findIndex(t => t.getAttribute('aria-selected') === 'true')
        show(tabs[(i + 1) % tabs.length])
      }, 5200)
    }
  })

  // Keyboard shortcuts, shown as chips on buttons
  const shortcuts = {
    i: () => copy(INSTALL),
    d: () => { location.href = '/docs' },
    f: () => { location.href = '/faq' },
    g: () => { location.href = 'https://github.com/angelraph/tironian' },
    h: () => { location.href = '/' }
  }
  document.addEventListener('keydown', e => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return
    const t = e.target
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return
    const fn = shortcuts[e.key.toLowerCase()]
    if (fn) { e.preventDefault(); fn() }
  })

  // Docs: mobile section picker and active link while scrolling
  const picker = $('.toc-mobile select')
  picker?.addEventListener('change', () => { location.hash = picker.value })
  const sideLinks = $$('.doc-side a[href^="#"]')
  if (sideLinks.length && 'IntersectionObserver' in window) {
    const map = new Map(sideLinks.map(a => [a.getAttribute('href').slice(1), a]))
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (!en.isIntersecting) return
        sideLinks.forEach(a => a.classList.remove('on'))
        map.get(en.target.id)?.classList.add('on')
      })
    }, { rootMargin: '-20% 0px -70% 0px' })
    map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el) })
  }

  // FAQ: open the question named in the URL hash
  if (location.hash) {
    const d = document.getElementById(location.hash.slice(1))
    if (d && d.tagName === 'DETAILS') d.open = true
  }

  // Cost calculator
  const calc = $('#calc')
  if (calc) {
    const rate = { L4: 0.000222, A10G: 0.000306 }
    const perEx = { '1.7b': 0.55, '4b': 1.3 }
    const overhead = { '1.7b': 420, '4b': 600 }
    let gpu = 'L4', model = '1.7b'
    const ex = $('#calc-ex'), runs = $('#calc-runs')
    const render = () => {
      const n = +ex.value, r = +runs.value
      const secs = overhead[model] + n * 2 * perEx[model]
      const one = secs * rate[gpu]
      $('#calc-ex-v').textContent = n
      $('#calc-runs-v').textContent = r
      $('#calc-one').textContent = '$' + one.toFixed(2)
      $('#calc-total').innerHTML = '$' + (one * r).toFixed(2) + '<small>a month</small>'
      $('#calc-min').textContent = Math.round(secs / 60) + ' min'
      const month = one * r
      $('#calc-credit').textContent = month <= 1
        ? 'Covered by Modal’s free credit, even without a card.'
        : month <= 30
          ? 'Covered by Modal’s $30 monthly free credit once a card is on file.'
          : 'Above Modal’s monthly free credit.'
    }
    $$('[data-gpu]', calc).forEach(b => b.addEventListener('click', () => {
      gpu = b.dataset.gpu; $$('[data-gpu]', calc).forEach(x => x.setAttribute('aria-pressed', String(x === b))); render()
    }))
    $$('[data-model]', calc).forEach(b => b.addEventListener('click', () => {
      model = b.dataset.model; $$('[data-model]', calc).forEach(x => x.setAttribute('aria-pressed', String(x === b))); render()
    }))
    ex.addEventListener('input', render); runs.addEventListener('input', render)
    render()
  }

  // Reveal on scroll
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target) }
    }), { rootMargin: '0px 0px -8% 0px' })
    $$('.reveal').forEach(el => io.observe(el))
  } else {
    $$('.reveal').forEach(el => el.classList.add('in'))
  }
})()
