"""Screenshots of the real Tironian page in the Hermes web app, for the demo video.

usage: python demo/capture_desktop.py
Needs webapp.enabled: true in Hermes's config. Asks `hermes webapp` for a one-use sign-in link.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "out"


def link() -> str:
    out = subprocess.run(["hermes", "webapp", "--skip-build", "--no-open"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", shell=True).stdout
    return re.search(r"http://127\.0\.0\.1:\d+/webapp/launch#ticket=\S+", out).group(0)


def main() -> None:
    url = link()
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1600, "height": 760}, device_scale_factor=2, color_scheme="dark")
        # The Desktop half is switched on per browser; a fresh one starts with it undecided.
        page.add_init_script("""localStorage.setItem('hermes.desktop.pluginDecisions.v2', '{"tironian":true}');
            localStorage.setItem('hermes-desktop-onboarded-v1', '1'); localStorage.setItem('hermes.desktop.intro-splash.v1', 'true');""")
        page.goto(url)
        page.wait_for_timeout(8000)
        page.goto(url.split("/webapp/")[0] + "/#/tironian")
        page.wait_for_selector("text=first steps", timeout=60000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "desktop1.png"))
        # Scroll the page's own scroller so the families list and the privacy line show.
        page.evaluate("""() => { const s = [...document.querySelectorAll('div')].find(d => d.scrollHeight > d.clientHeight + 40
            && getComputedStyle(d).overflowY === 'auto' && d.innerText.includes('Task families')); if (s) s.scrollTop = 320; }""")
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "desktop2.png"))
        b.close()
    print("wrote desktop1.png and desktop2.png")


if __name__ == "__main__":
    main()
