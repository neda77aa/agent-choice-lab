"""
Render screenshots for human-experiment stimuli.

For each chosen product, generates 4 PNG screenshots:
  - 2option_v2.png  (baseline, popcorn pricing)
  - 3option_v2.png  (treat-medium, popcorn pricing)
  - 2option_v3.png  (baseline, absolute-dominance pricing)
  - 3option_v3.png  (treat-medium, absolute-dominance pricing)

Output: human_experiments/<product_slug>/{2,3}option_{v2,v3}.png

Requires:
  - Running OneStopMarket Magento docker at localhost:7770
  - playwright (sync_api)
  - tasks/size_decoy_geometries_v2_gs10.csv  (popcorn)
  - tasks/size_decoy_geometries_v3.csv       (absolute dominance)
"""
from __future__ import annotations
import argparse
import re
import tempfile
from pathlib import Path

import pandas as pd
import requests
from playwright.sync_api import sync_playwright

from abxlab.choices.shop import options as opts_mod
from abxlab.choices.shop import product as prod_mod


BASE_URL = "http://localhost:7770/"
ABLATE_ELEMS = ["breadcrumbs", "price-box price-final_price"]

# Top 20 cleanest products from the cleanliness scorer (script-side).
DEFAULT_TITLES = [
    "Animal Crackers Original",
    "Bourbon Cookies",
    "Caramel Apple Candy Corn",
    "Cheese Wafer",
    "Condensed Milk Rusks",
    "Cream Crackers",
    "Danish Style Butter Cookies",
    "Dark Chocolate Nonpareils",
    "Fresh Snacking Mozzarella Cheese",
    "Garlic Parmesan Bread Chips",
    "Greek Yogurt Coated Blueberries",
    "Lemon Creme Milk Chocolate",
    "Opera Creme Milk Chocolate",
    "Parmesan Cheese",
    "Sweet Rolls",
    "Traditional Christmas Wafers",
    "Ultimate Holiday Cookie Collection",
    "Chocolate Covered Roasted Espresso Beans",
    "Enamelon Fluoride Toothpaste",
    "French Cookies LU Barquettes Strawberry",
]


def slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


def apply_chain(html_bytes: bytes, row: pd.Series, show_medium: bool) -> str:
    """Apply v2/v3 intervention chain. `show_medium` toggles 2-option vs 3-option."""
    sid = str(int(row["small_id"]))
    mid = str(int(row["medium_id"]))
    lid = str(int(row["large_id"]))
    if show_medium:
        keep_ids = [sid, mid, lid]
        labels = {sid: row["small_name"], mid: row["medium_name"], lid: row["large_name"]}
        prices = {sid: float(row["P_small"]), mid: float(row["P_medium"]), lid: float(row["P_large"])}
    else:
        keep_ids = [sid, lid]
        labels = {sid: row["small_name"], lid: row["large_name"]}
        prices = {sid: float(row["P_small"]), lid: float(row["P_large"])}

    h = html_bytes
    h, _ = opts_mod.set_title(h, value=row["template_title"])
    h, _ = opts_mod.inject_rating(h.encode() if isinstance(h, str) else h, value=80, review_count=50)
    h, _ = opts_mod.keep_options(h.encode() if isinstance(h, str) else h, value_ids=keep_ids)
    h, _ = opts_mod.set_option_label(h.encode() if isinstance(h, str) else h, mapping=labels)
    h, _ = opts_mod.set_option_price(h.encode() if isinstance(h, str) else h, mapping=prices)
    h, _ = prod_mod.ablate(h.encode() if isinstance(h, str) else h, elems=ABLATE_ELEMS)
    return h


def inject_base_url(html: str, base_url: str) -> str:
    """Inject <base href> into <head> so file:// rendering can resolve relative URLs."""
    base_tag = f'<base href="{base_url}">'
    if "<head>" in html:
        return html.replace("<head>", f"<head>{base_tag}", 1)
    if "<head ".lower() in html.lower():
        return re.sub(r"(<head[^>]*>)", lambda m: m.group(1) + base_tag, html, count=1)
    return html.replace("<html>", f"<html><head>{base_tag}</head>", 1)


def render_one(page, html: str, out_path: Path):
    """Save html to a temp file, navigate, screenshot the product card area only.

    Cosmetic patches applied via injected CSS/JS just before screenshotting
    (do NOT alter the HTML the agents see):
      - hide the description tabs and details block below the cart
      - replace the Magento CSS-sprite star bar with self-contained Unicode stars
        (★ filled / ☆ empty) so the rating renders reliably on file:// URLs
    """
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html",
                                      dir="/tmp", encoding="utf-8") as f:
        f.write(html)
        tmp_path = f.name
    try:
        page.goto(f"file://{tmp_path}", wait_until="load", timeout=15000)
        page.wait_for_timeout(800)  # let JS / CSS settle

        # Hide the tabs section and description below the cart, AND
        # suppress Magento's CSS-sprite rating visualization (we replace it
        # with self-contained Unicode stars below).
        page.add_style_tag(content="""
            .product.data.items, .block.upsell, .block.related,
            .product-info-detailed, .columns .sidebar-additional,
            footer, .page-footer, .page-bottom, .compare.wrapper,
            .breadcrumbs, .panel.header, .nav-sections, .page-header
                { display: none !important; }
            /* Kill Magento's original rating bar so only our Unicode
               replacement shows. */
            .rating-result, .rating-result *::before,
            .rating-result *::after, .rating-result::before,
            .rating-result::after { display: none !important; }
            body { padding-top: 16px !important; }
        """)

        # Build a self-contained Unicode star bar and put it where the
        # original rating-result lived.
        page.evaluate("""
        () => {
          const rr = document.querySelector('div.rating-result');
          if (!rr) return;
          const title = rr.getAttribute('title') || '';
          const m = title.match(/(\\d+(?:\\.\\d+)?)\\s*%/);
          const pct = m ? parseFloat(m[1]) : 80;
          const filled = Math.round(pct / 20);
          const stars = '★'.repeat(filled) + '☆'.repeat(5 - filled);
          const wrap = document.createElement('span');
          wrap.style.cssText =
            'display:inline-block; font-size:18px; color:#fb4f1f; '
            + 'letter-spacing:2px; vertical-align:middle;';
          wrap.textContent = stars;
          rr.parentNode.insertBefore(wrap, rr);
        }
        """)

        page.wait_for_timeout(200)

        # Tight clip: just enough to fit (title + image + info card).
        try:
            edges = page.evaluate("""
            () => {
              const cands = ['div.product-info-main', 'div.product.media',
                             'div.gallery-placeholder', 'div.fotorama__stage',
                             'div.box-tocart', 'h1.page-title'];
              let bottom = 0, right = 0;
              for (const sel of cands) {
                const el = document.querySelector(sel);
                if (!el) continue;
                const r = el.getBoundingClientRect();
                bottom = Math.max(bottom, r.bottom);
                right  = Math.max(right,  r.right);
              }
              return {bottom, right};
            }
            """)
            page.screenshot(
                path=str(out_path),
                clip={
                    "x": 0,
                    "y": 0,
                    "width":  min(page.viewport_size["width"],  int(edges["right"])  + 8),
                    "height": min(page.viewport_size["height"], int(edges["bottom"]) + 8),
                },
            )
        except Exception:
            page.screenshot(path=str(out_path), full_page=False)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def fetch_html(url: str) -> bytes:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="human_experiments")
    ap.add_argument("--titles", nargs="+", default=None,
                    help="Substring matches against template_title (default: top-20).")
    ap.add_argument("--all", action="store_true",
                    help="Render every product present in tasks/size_decoy_geometries_v3.csv.")
    args = ap.parse_args()

    if args.all:
        v3_all = pd.read_csv("tasks/size_decoy_geometries_v3.csv")
        titles = sorted(v3_all.template_title.unique().tolist())
    else:
        titles = args.titles if args.titles else DEFAULT_TITLES
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    v2 = pd.read_csv("tasks/size_decoy_geometries_v2_gs10.csv")
    v3 = pd.read_csv("tasks/size_decoy_geometries_v3.csv")

    # Build the work list: (geometry_label, arm_label, geometry_df) for the 4 outputs
    plan = [
        ("v2", False, v2[v2.arm == "treat-medium"], "2option_v2.png"),
        ("v2", True,  v2[v2.arm == "treat-medium"], "3option_v2.png"),
        ("v3", False, v3[v3.arm == "treat-medium"], "2option_v3.png"),
        ("v3", True,  v3[v3.arm == "treat-medium"], "3option_v3.png"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Narrow viewport so the Magento layout compresses and there's no
        # large empty area to the right of the cart info.
        ctx = browser.new_context(viewport={"width": 900, "height": 1400},
                                  device_scale_factor=2)
        page = ctx.new_page()

        for needle in titles:
            # Find the row in v3 (use v3 as canonical, both share the same product set)
            mask = v3.template_title.str.contains(needle, case=False, regex=False)
            matches = v3[mask].template_title.unique()
            if not len(matches):
                print(f"[skip] no match: {needle!r}")
                continue
            full_title = matches[0]
            print(f"\n=== {full_title} ===")

            url_template = v3[v3.template_title == full_title].iloc[0]["url"]
            url = url_template.replace("${env.abxlab_url}", BASE_URL)
            try:
                raw_html = fetch_html(url)
            except Exception as e:
                print(f"  fetch failed: {e}")
                continue

            sl = slug(full_title)
            prod_dir = out_dir / sl
            prod_dir.mkdir(exist_ok=True)

            for geom_label, show_medium, df, fname in plan:
                row = df[df.template_title == full_title].iloc[0]
                rendered = apply_chain(raw_html, row, show_medium=show_medium)
                rendered_str = rendered if isinstance(rendered, str) else rendered.decode()
                rendered_str = inject_base_url(rendered_str, BASE_URL)
                out_path = prod_dir / fname
                try:
                    render_one(page, rendered_str, out_path)
                    print(f"  wrote {out_path.relative_to(out_dir.parent)}")
                except Exception as e:
                    print(f"  FAILED {fname}: {e}")

        browser.close()
    print(f"\nDone. See {out_dir}/")


if __name__ == "__main__":
    main()
