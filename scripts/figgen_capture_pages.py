"""Capture product-page screenshots from the running OneStopMarket Docker.

Captures (a) raw and (b) intervention-rewritten versions of the same product
page, plus three product pages for the Sony Cybershot triad used in Paradigm A
illustrations.
"""
from __future__ import annotations
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = "http://localhost:7770"
OUT  = Path(__file__).resolve().parents[1].parent / "paper" / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# Sony triad from controlled_triads_decoy.csv (triad_id 0, geometry_id 0 — range)
TRIAD = [
    ("target",     "sony-cybershot-dscw70-7-2mp-digital-camera-with-3x-optical-zoom.html",
                   {"price": "$58.19", "rating_pct": 82}),
    ("competitor", "sony-cybershot-dsc-s650-7-2-mp-3x-optical-zoom-digital-camera-silver.html",
                   {"price": "$41.71", "rating_pct": 77}),
    ("decoy",      "sony-cybershot-dscs730-7-2mp-digital-camera-with-3x-optical-zoom.html",
                   {"price": "$58.19", "rating_pct": 80}),
]

# JS injected into the page to perform a small live demo of the price/rating
# rewrites the proxy normally does. This mirrors the abxlab.choices.shop
# interventions but runs client-side so we can also screenshot the *raw* page
# first, then the *rewritten* one.
INJECT_JS = r"""
(spec) => {
  // ---- price ----
  document.querySelectorAll('span.price').forEach(el => {
    el.textContent = spec.price;
  });
  document.querySelectorAll('[data-price-amount]').forEach(el => {
    el.setAttribute('data-price-amount',
      String(parseFloat(spec.price.replace(/[^0-9.]/g, ''))));
  });
  // ---- rating star bar ----
  document.querySelectorAll('div.rating-result > span').forEach(el => {
    el.style.width = spec.rating_pct + '%';
  });
  document.querySelectorAll('div.rating-result').forEach(el => {
    el.setAttribute('title', spec.rating_pct + '%');
  });
  // numeric rating badge: insert one if missing
  document.querySelectorAll('div.product-reviews-summary').forEach(el => {
    if (!el.querySelector('.injected-rating-badge')) {
      const b = document.createElement('span');
      b.className = 'injected-rating-badge';
      b.textContent = ' Rating: ' + spec.rating_pct + '%';
      b.style.marginLeft = '8px';
      b.style.color = '#1f4e79';
      b.style.fontWeight = '600';
      el.appendChild(b);
    }
  });
  // ---- review count harmonised to 50 ----
  document.querySelectorAll('a.action.view').forEach(el => {
    el.textContent = '50 Reviews';
  });
  // ---- sanitize title (strip pack tokens) ----
  document.querySelectorAll('h1.page-title span').forEach(el => {
    el.textContent = el.textContent
      .replace(/\(?Pack of\s*\d+\)?/gi, '')
      .replace(/\(?Set of\s*\d+\)?/gi, '')
      .replace(/\s+/g, ' ').trim();
  });
  // ---- ablate: remove descriptive copy, social, sku ----
  ['div.product.attribute.description',
   'div.product-social-links',
   'div.product-info-stock-sku'
  ].forEach(sel => {
    document.querySelectorAll(sel).forEach(el => el.remove());
  });
}
"""

def main():
    storage = Path(__file__).resolve().parents[1] / ".auth" / "shopping_state.json"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1180, "height": 820},
            storage_state=str(storage) if storage.exists() else None,
        )
        page = context.new_page()

        # --- 1. Single product before/after for fig:intervention_before_after ---
        # Use the target Sony page.
        spec = TRIAD[0][2]
        page.goto(f"{BASE}/{TRIAD[0][1]}", wait_until="networkidle")
        before_path = OUT / "intervention_before.png"
        page.screenshot(path=str(before_path), full_page=False)
        page.evaluate(INJECT_JS, spec)
        page.wait_for_timeout(200)
        after_path = OUT / "intervention_after.png"
        page.screenshot(path=str(after_path), full_page=False)
        print("wrote", before_path, after_path)

        # --- 2. Three Sony product pages for fig:paradigm_a_schematic ---
        for role, slug, sp in TRIAD:
            page.goto(f"{BASE}/{slug}", wait_until="networkidle")
            page.evaluate(INJECT_JS, sp)
            page.wait_for_timeout(200)
            outp = OUT / f"paradigm_a_{role}.png"
            page.screenshot(path=str(outp), full_page=False)
            print("wrote", outp)

        # --- 3. Capture the URL-leakage example (Frank-and-Sal cheese) ---
        # Find a candidate cheese product with size in URL.
        cheese_url = "frank-and-sal-genuine-grana-padano-aged-24-months-italian-import-3-pound.html"
        page.goto(f"{BASE}/{cheese_url}", wait_until="networkidle")
        outp = OUT / "url_anon_before.png"
        page.screenshot(path=str(outp), full_page=False)
        print("wrote", outp)

        browser.close()

if __name__ == "__main__":
    main()
