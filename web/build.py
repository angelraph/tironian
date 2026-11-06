"""Wrap each page in web/src/ with the shared head, header and footer.

Each source file starts with one comment line of settings:
<!-- title: Page title | desc: One sentence for search and link previews | nav: docs -->

Run: python web/build.py   (writes web/<page>.html, which Vercel serves as /<page>)
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
SITE = "https://tironian.vercel.app"
REPO = "https://github.com/angelraph/tironian"

NAV = [
    ("how-it-works", "How it works", "/how-it-works"),
    ("docs", "Docs", "/docs"),
    ("pricing", "Cost", "/pricing"),
    ("privacy", "Privacy", "/privacy"),
    ("faq", "FAQ", "/faq"),
    ("changelog", "Changelog", "/changelog"),
]

GITHUB_ICON = ('<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 '
               '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-'
               '1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
               '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 '
               '2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
               '.21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')


def head(title: str, desc: str, path: str) -> str:
    full = "Tironian" if title == "Tironian" else f"{title} · Tironian"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{full}</title>
<meta name="description" content="{desc}">
<meta name="theme-color" content="#000000">
<link rel="canonical" href="{SITE}{path}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Tironian">
<meta property="og:title" content="{full}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{SITE}{path}">
<meta property="og:image" content="{SITE}/assets/og.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{SITE}/assets/og.png">
<link rel="icon" type="image/png" sizes="32x32" href="/assets/mark-32.png">
<link rel="apple-touch-icon" href="/assets/mark-180.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/assets/site.css">
</head>
<body>
<a class="sr-only" href="#main">Skip to content</a>"""


def header(active: str) -> str:
    links = "\n        ".join(
        f'<a href="{href}"{" aria-current=\"page\"" if key == active else ""}>{label}</a>' for key, label, href in NAV)
    return f"""
<header class="site-head">
  <div class="wrap">
    <a class="brand" href="/" aria-label="Tironian home"><img src="/assets/mark-180.png" alt="" width="30" height="30">Tironian</a>
    <div class="nav-wrap">
      <nav class="nav" aria-label="Main">
        {links}
      </nav>
      <div class="head-actions">
        <a class="btn btn-ghost btn-sm" href="{REPO}">{GITHUB_ICON}GitHub <kbd>G</kbd></a>
        <a class="btn btn-primary btn-sm" href="/docs#install">Install <kbd>I</kbd></a>
      </div>
    </div>
    <button class="menu-btn" type="button" aria-label="Menu" aria-expanded="false"><span></span></button>
  </div>
</header>
<main id="main">"""


FOOTER = f"""
</main>
<footer class="site-foot">
  <div class="wrap">
    <div class="foot-grid">
      <div class="foot-brand">
        <img src="/assets/wordmark.png" alt="Tironian. Train, route, ship." width="160">
        <p>A Hermes Agent plugin that turns the work you already do into a small model of your own.</p>
      </div>
      <div class="foot-col">
        <h4>Product</h4>
        <a href="/how-it-works">How it works</a>
        <a href="/pricing">Cost</a>
        <a href="/privacy">Privacy and safety</a>
        <a href="/changelog">Changelog</a>
      </div>
      <div class="foot-col">
        <h4>Docs</h4>
        <a href="/docs#install">Install</a>
        <a href="/docs#first-model">Your first model</a>
        <a href="/docs#cli">CLI reference</a>
        <a href="/docs#settings">Settings</a>
        <a href="/docs#troubleshooting">Troubleshooting</a>
      </div>
      <div class="foot-col">
        <h4>Project</h4>
        <a href="{REPO}">Source code</a>
        <a href="{REPO}/issues">Report a problem</a>
        <a href="{REPO}/blob/master/README.md">README</a>
        <a href="{REPO}/blob/master/LICENSE">MIT license</a>
      </div>
      <div class="foot-col">
        <h4>Hermes</h4>
        <a href="https://hermes-agent.nousresearch.com">Hermes Agent</a>
        <a href="https://hermes-agent.nousresearch.com/docs/plugins">Plugin marketplace</a>
        <a href="https://hermes-agent.nousresearch.com/docs/developer-guide/plugins">Plugin guide</a>
        <a href="/faq">FAQ</a>
      </div>
    </div>
    <div class="foot-base">
      <span>Named after Tiro, Cicero's secretary, who invented a shorthand to carry his master's work forward.</span>
      <span>MIT licensed · No telemetry · Keyboard: <kbd>I</kbd> copy install <kbd>D</kbd> docs <kbd>F</kbd> FAQ <kbd>G</kbd> GitHub</span>
    </div>
  </div>
</footer>
<script src="/assets/site.js" defer></script>
</body>
</html>
"""

META = re.compile(r"^\s*<!--(.*?)-->\s*", re.S)


def build() -> list[str]:
    out = []
    for src in sorted(SRC.glob("*.html")):
        text = src.read_text(encoding="utf-8")
        m = META.match(text)
        meta = {}
        if m:
            for part in m.group(1).split("|"):
                k, _, v = part.partition(":")
                meta[k.strip()] = v.strip()
            text = text[m.end():]
        name = src.stem
        path = "/" if name == "index" else f"/{name}"
        page = head(meta.get("title", "Tironian"), meta.get("desc", ""), path) + header(meta.get("nav", "")) + text + FOOTER
        if "—" in page or "–" in page:
            raise SystemExit(f"{src.name}: contains a long dash; rewrite the sentence")
        (HERE / f"{name}.html").write_text(page, encoding="utf-8")
        out.append(name)
    return out


if __name__ == "__main__":
    print("built:", ", ".join(build()))
