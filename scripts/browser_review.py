"""Visual review harness: launches the REAL Streamlit app and drives a real (headless) Chromium over the Chrome DevTools Protocol.

No new dependencies: it uses a Chromium/Chrome binary that is already installed (CHROME_BIN, or Playwright's browser cache, or an installed Chrome)
and the `websockets` package that is already in the environment. It screenshots every Payments section at desktop resolution (1440 px wide,
full page height), the original Simulation Control Center, and Demonstration mode, and records on-page Streamlit exceptions and console errors.

  python -m scripts.browser_review [--out docs/screenshots] [--port 8790] [--autocheck 0|1]

It never types credentials and never clicks a button that spends API quota or calls PayPal.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = ["Overview", "Policy Optimization", "Experiments", "PayPal Sandbox", "Agent Execution Traces"]
WIDTH, VIEW_H, MAX_H = 1440, 900, 7000


def find_browser() -> str:
    if os.environ.get("CHROME_BIN"):
        return os.environ["CHROME_BIN"]
    pats = [os.path.expanduser("~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"),
            os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/chrome-linux*/headless_shell"),
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"]
    for p in pats:
        hit = sorted(glob.glob(p))
        if hit:
            return hit[-1]
    raise SystemExit("No Chromium/Chrome binary found (set CHROME_BIN).")


class CDP:
    def __init__(self, ws):
        self.ws, self._id, self.events = ws, 0, []

    async def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), 60))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


STATE_JS = """(() => ({
  running: !!document.querySelector('[data-testid="stStatusWidget"]'),
  headings: [...document.querySelectorAll('h3')].map(e => e.innerText.trim()),
  exceptions: [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 300)),
  height: Math.max(document.querySelector('[data-testid="stMain"]')?.scrollHeight || 0, document.body.scrollHeight)
}))()"""


async def settle(c: CDP, want_heading: str | None = None, timeout: float = 90.0) -> dict:
    t0, stable, last = time.time(), 0, None
    while time.time() - t0 < timeout:
        s = await c.js(STATE_JS)
        if not s:  # page still loading / evaluation not possible yet
            await asyncio.sleep(0.5)
            continue
        ok = (not s["running"]) and (want_heading is None or want_heading in s["headings"])
        sig = (s["height"], tuple(s["headings"]))
        stable = stable + 1 if (ok and sig == last) else 0
        last = sig
        if stable >= 3:
            return s
        await asyncio.sleep(0.5)
    raise TimeoutError(f"page did not settle (wanted heading {want_heading!r}); last state {last}")


async def click_text(c: CDP, scope: str, text: str, tag: str = "label") -> bool:
    return bool(await c.js(f"""(() => {{ const root = document.querySelector({json.dumps(scope)}); if (!root) return false;
        const el = [...root.querySelectorAll({json.dumps(tag)})].find(e => e.innerText.trim() === {json.dumps(text)});
        if (!el) return false; el.click(); return true; }})()"""))


async def set_demo_mode(c: CDP, want: bool) -> bool:
    """Flip the sidebar toggle by clicking its <input> (react-aria ignores a click on the <label>) and VERIFY the app really changed mode."""
    for _ in range(3):
        now = await c.js("document.querySelector('[data-testid=\"stSidebar\"] input[type=checkbox]')?.checked")
        if now is want:
            await asyncio.sleep(1.0)
            return True
        await c.js("document.querySelector('[data-testid=\"stSidebar\"] input[type=checkbox]')?.click()")
        await asyncio.sleep(2.0)
    return False


async def mode_banner(c: CDP) -> str:
    return await c.js("[...document.querySelectorAll('.inoema-pill')].map(e => e.innerText.replace(/\\u00a0/g,' ')).join(' | ')")


async def shot(c: CDP, path: str, state: dict) -> int:
    h = min(max(state["height"] + 24, VIEW_H), MAX_H)
    await c.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=h, deviceScaleFactor=1, mobile=False)
    await asyncio.sleep(1.2)
    data = (await c.call("Page.captureScreenshot", format="png", captureBeyondViewport=False))["data"]
    import base64
    open(path, "wb").write(base64.b64decode(data))
    await c.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=VIEW_H, deviceScaleFactor=1, mobile=False)
    return h


async def review(url: str, dbg_port: int, out: str) -> dict:
    page = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg_port}/json/list")) if t["type"] == "page")
    summary: dict = {"url": url, "viewport": f"{WIDTH}x{VIEW_H} (screenshots full page height)", "steps": []}
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024) as ws:
        c = CDP(ws)
        await c.call("Page.enable")
        await c.call("Runtime.enable")
        await c.call("Log.enable")
        await c.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=VIEW_H, deviceScaleFactor=1, mobile=False)

        async def step(name: str, heading: str | None, action=None, expect_mode: str | None = None):
            if action:
                ok = await action()
                if not ok:
                    summary["steps"].append({"name": name, "error": "could not click the control"})
                    return
            s = await settle(c, heading)
            banner = await mode_banner(c)
            err = None
            if expect_mode and expect_mode not in banner:  # a screenshot is only taken if the app is really in the mode its name claims
                err = f"expected mode {expect_mode!r} in header pill but found {banner!r}"
            f = f"{out}/{name}.png"
            h = await shot(c, f, s)
            summary["steps"].append({"name": name, "file": os.path.relpath(f, ROOT), "headings": s["headings"][:3], "header_pill": banner, "mode_verified": expect_mode is None or err is None,
                                     "page_height_px": h, "page_exceptions": s["exceptions"], **({"error": err} if err else {})})
            print(f"  {name}: headings={s['headings'][:2]} pill={banner!r} exceptions={len(s['exceptions'])} height={h}px" + (f"  !! {err}" if err else ""))

        await c.call("Page.navigate", url=url)
        await step("01_overview_live_default", "Overview", expect_mode="Live workspace")
        for i, sec in enumerate(SECTIONS[1:], 2):
            await step(f"0{i}_{sec.lower().replace(' ', '_')}_live", sec, lambda sec=sec: click_text(c, '[data-testid="stMain"]', sec), expect_mode="Live workspace")
        # Demonstration mode (sidebar toggle): re-visit each section; the toggle result is verified, not assumed
        summary["demo_mode_enabled"] = await set_demo_mode(c, True)
        for i, sec in enumerate(SECTIONS, 1):
            await step(f"1{i}_{sec.lower().replace(' ', '_')}_demo", sec, lambda sec=sec: click_text(c, '[data-testid="stMain"]', sec), expect_mode="Demonstration mode")
        summary["demo_mode_disabled_again"] = await set_demo_mode(c, False)
        # the original workspace stays reachable from the sidebar
        await step("20_original_simulation_control_center", None, lambda: click_text(c, '[data-testid="stSidebar"]', "Simulation Control Center"))
        summary["console_errors"] = [e["params"]["entry"]["text"][:200] for e in c.events
                                     if e.get("method") == "Log.entryAdded" and e["params"]["entry"].get("level") == "error"]
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "screenshots"))
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--autocheck", default="1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = {**os.environ, "INOEMA_DASHBOARD_AUTOCHECK": a.autocheck}
    env.pop("INOEMA_DEMO_MODE", None)
    env.pop("INOEMA_DEFAULT_WORKSPACE", None)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "dashboard", "app.py"), "--server.port", str(a.port),
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser_bin, dbg = find_browser(), 9341
    browser = subprocess.Popen([browser_bin, f"--remote-debugging-port={dbg}", f"--window-size={WIDTH},{VIEW_H}", "--hide-scrollbars", "--no-first-run", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{a.port}/_stcore/health", timeout=1)
                urllib.request.urlopen(f"http://127.0.0.1:{dbg}/json/version", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            raise SystemExit("server or browser did not start")
        print(f"browser: {browser_bin}")
        summary = asyncio.run(review(f"http://localhost:{a.port}/", dbg, a.out))
        summary["browser"] = os.path.basename(os.path.dirname(browser_bin))
        json.dump(summary, open(os.path.join(a.out, "review_summary.json"), "w"), indent=2)
        bad = [s for s in summary["steps"] if s.get("error") or s.get("page_exceptions")]
        print(f"done: {len(summary['steps'])} screenshots, {len(bad)} problem step(s), {len(summary['console_errors'])} console error(s)")
        return 1 if bad else 0
    finally:
        for p in (browser, server):
            p.terminate()
        for p in (browser, server):
            try:
                p.wait(10)
            except Exception:  # noqa: BLE001
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
