"""Capture the screenshots used in docs/article.md.

    pip install playwright && python -m playwright install chromium
    python scripts/capture_screenshots.py [--base https://vishalmysore.github.io/layaAsRagJudge/] [--no-model] [--chrome PATH]

Drives the real pages in headless Chromium. The first shots use the recorded results (no download); the live-model
shots load Laya (422 MB) and the embedder once into a persistent profile under .cache/.
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
args = sys.argv[1:]
BASE = args[args.index("--base") + 1] if "--base" in args else "https://vishalmysore.github.io/layaAsRagJudge/"
WITH_MODEL = "--no-model" not in args
# An installed Chrome/Chromium (skips `playwright install`); default: the Playwright-managed build
CHROME = args[args.index("--chrome") + 1] if "--chrome" in args else None


def shot(page, name, selector=None, clip_rows=None):
    path = OUT / f"{name}.png"
    if clip_rows:
        boxes = [page.locator(s).first.bounding_box() for s in clip_rows]
        x0 = min(b["x"] for b in boxes); y0 = min(b["y"] for b in boxes)
        x1 = max(b["x"] + b["width"] for b in boxes); y1 = max(b["y"] + b["height"] for b in boxes)
        page.screenshot(path=str(path), clip={"x": x0 - 8, "y": y0 - 8, "width": x1 - x0 + 16, "height": y1 - y0 + 16})
    elif selector:
        page.locator(selector).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path))
    print("saved", path.relative_to(ROOT))


def open_page(page, name, fresh=False):
    page.goto(BASE + name)
    page.wait_for_function("() => window.__lrj && window.__lrj.state && window.__lrj.state.corpus", timeout=60_000)
    if fresh:
        page.evaluate("() => { Object.keys(localStorage).filter(k => k.startsWith('lrj.')).forEach(k => localStorage.removeItem(k)); }")
        page.goto(BASE + name)
        page.wait_for_function("() => window.__lrj && window.__lrj.state && window.__lrj.state.corpus", timeout=60_000)
    page.wait_for_timeout(1200)


def pick_claim(page, claim_id, dataset):
    page.select_option("#dataset", dataset)
    page.dispatch_event("#dataset", "change")
    page.click(f'#examples button[data-id="{claim_id}"]')
    page.wait_for_timeout(600)


def open_details(page):
    page.evaluate("() => document.querySelectorAll('#result details').forEach(d => d.open = true)")
    page.wait_for_timeout(300)


def select_run(page, label_part, recorded=True):
    page.evaluate("""([part, rec]) => { const s = document.getElementById('runSel');
        const o = [...s.options].find(o => o.text.includes(part) && o.value.startsWith('rec|') === rec);
        s.value = o.value; s.dispatchEvent(new Event('change')); }""", [label_part, recorded])
    page.wait_for_timeout(900)


def load_models(page):
    page.wait_for_function("() => !document.getElementById('loadBtn').disabled", timeout=60_000)
    page.click("#loadBtn")
    page.wait_for_function("() => window.__lrj.models.laya", timeout=900_000)
    page.wait_for_timeout(500)


def verify(page):
    page.evaluate("() => window.__lrj.verify()")
    page.wait_for_function("() => !document.getElementById('verifyBtn').disabled && !document.getElementById('result').hidden", timeout=120_000)
    page.wait_for_timeout(600)


with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        str(ROOT / ".cache" / "chromium-profile"), headless=True, executable_path=CHROME,
        viewport={"width": 1440, "height": 960}, device_scale_factor=1.25, color_scheme="light",
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.on("pageerror", lambda e: print("PAGE ERROR:", e))

    # 1. Verify page, recorded result: lighthouse claim, right verdict but held for review
    open_page(page, "index.html", fresh=True)
    pick_claim(page, "enc-lighthouse-2", "encyclopedic")
    shot(page, "01-verify-overview")
    shot(page, "02-verify-result", "#resultCard")

    # 2. Evaluate page, recorded runs
    open_page(page, "rag-eval.html", fresh=True)
    select_run(page, "Two options · 2 sent/chunk · k=3 · document")
    shot(page, "03-eval-summary", "#summaryCard")
    shot(page, "04-eval-coverage-breakdowns", "main.main > section.two")
    # click the dot of a hallucination the judge let through (API keys) -> its details in the table
    page.dispatch_event('#strip .dot[data-id="pol-api-4"]', "click")  # dots can overlap, so no pointer hit-test
    page.wait_for_timeout(1200)
    shot(page, "05-eval-dot-detail", clip_rows=["#claims tr.focus", "#claims tr.focus + tr.detailrow"])
    select_run(page, "Three options · 2 sent/chunk · k=3 · document")
    shot(page, "06-eval-three-options", "#summaryCard")
    select_run(page, "Two options · 2 sent/chunk · k=3 · corpus")
    shot(page, "07-eval-whole-corpus", "#summaryCard")

    # 3. Live models on the Verify page
    if WITH_MODEL:
        open_page(page, "index.html")
        load_models(page)
        shot(page, "08-model-card", "#modelCard")
        # the API-keys contradiction, live, with every passage and the exact token sequence Laya reads
        pick_claim(page, "pol-api-4", "policy")
        verify(page)
        open_details(page)
        shot(page, "09-live-contradiction", "#resultCard")
        # a claim the corpus never mentions
        page.fill("#claim", "The study shows eating more carbs helps students sleep.")
        page.dispatch_event("#claim", "input")
        page.select_option("#doc", "news-sleep")
        verify(page)
        page.evaluate("() => document.querySelectorAll('#result details').forEach(d => d.open = false)")
        shot(page, "10-live-unsupported-block", "#resultCard")
        # your own evidence text
        page.click('label:has(input[name=source][value=pasted])')  # the radio itself is visually hidden
        page.fill("#pasted", "The Kestrel 5 e-bike has a 540 Wh battery. The manufacturer says it can travel up to 90 kilometres "
                  "on one charge in eco mode. Charging from empty takes about five hours with the standard charger. The frame is "
                  "made of aluminium and the bike weighs 24 kilograms. It comes with a two-year warranty on the motor and battery.")
        page.fill("#claim", "The Kestrel 5 can ride 90 km per charge in eco mode.")
        page.dispatch_event("#claim", "input")
        verify(page)
        shot(page, "11-live-pasted-auto", ".layout")

    # 4. Dark mode at phone width
    dark = ctx.new_page()
    dark.emulate_media(color_scheme="dark")
    dark.set_viewport_size({"width": 390, "height": 844})
    open_page(dark, "rag-eval.html")
    select_run(dark, "Two options · 2 sent/chunk · k=3 · document")
    dark.evaluate("() => document.getElementById('summaryCard').scrollIntoView()")
    dark.wait_for_timeout(500)
    shot(dark, "12-dark-mobile")
    dark.close()

    ctx.close()
print("done")
