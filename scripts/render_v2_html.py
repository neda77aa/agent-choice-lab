"""
Render the post-intervention HTML for selected v2 size-decoy products.

Fetches the live Magento page, applies the v2 intervention chain
(set_title, inject_rating, keep_options, set_option_label, set_option_price,
ablate) using the same functions the experiment harness uses, and writes
before.html and after.html into analysis_output/size_decoy_v2_html/.

Run after:
    python scripts/select_size_decoy_geometries.py --preset v2 \
        --output tasks/size_decoy_geometries_v2.csv

Usage:
    python scripts/render_v2_html.py                       # default 3 demo products
    python scripts/render_v2_html.py --titles ENAMELON RAISINS TWISTS
    python scripts/render_v2_html.py --all-faked           # all 16 faked products
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from abxlab.choices.shop import options as opts_mod
from abxlab.choices.shop import product as prod_mod


GEOM_CSV = "tasks/size_decoy_geometries_v2.csv"
OUT_DIR = Path("analysis_output/size_decoy_v2_html")
ABLATE_ELEMS = ["breadcrumbs", "price-box price-final_price"]

# Default 3 demo products (substring match against template_title)
DEFAULT_TITLES = ["Enamelon", "Milk Chocolate Covered Raisins", "Triple Flavor Twists"]


def slug(template_title: str) -> str:
    """File-safe slug from a product title (collapses repeated separators)."""
    import re as _re
    s = template_title.lower()
    s = _re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


def apply_v2_chain(html: str, row: pd.Series) -> tuple[str, list[dict]]:
    """Apply the v2 intervention chain in order. Returns (final_html, log)."""
    sid = str(int(row["small_id"]))
    mid = str(int(row["medium_id"]))
    lid = str(int(row["large_id"]))
    keep_ids = [sid, mid, lid]
    labels = {sid: row["small_name"], mid: row["medium_name"], lid: row["large_name"]}
    prices = {sid: float(row["P_small"]),
              mid: float(row["P_medium"]),
              lid: float(row["P_large"])}

    log = []
    html, info = opts_mod.set_title(html.encode(), value=row["template_title"])
    log.append(info)
    html, info = opts_mod.inject_rating(html.encode(), value=80, review_count=50)
    log.append(info)
    html, info = opts_mod.keep_options(html.encode(), value_ids=keep_ids)
    log.append(info)
    html, info = opts_mod.set_option_label(html.encode(), mapping=labels)
    log.append(info)
    html, info = opts_mod.set_option_price(html.encode(), mapping=prices)
    log.append(info)
    html, info = prod_mod.ablate(html.encode(), elems=ABLATE_ELEMS)
    log.append(info)
    return html, log


def render_one(row: pd.Series, out_dir: Path) -> None:
    title = row["template_title"]
    sl = slug(title)
    dst = out_dir / sl
    dst.mkdir(parents=True, exist_ok=True)

    url = str(row["url"]).replace("${env.abxlab_url}", "http://localhost:7770/")
    print(f"  [{title[:50]}] fetching {url}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    before = resp.text

    after, log = apply_v2_chain(before, row)

    (dst / "before.html").write_text(before)
    (dst / "after.html").write_text(after)
    (dst / "info.txt").write_text(
        f"product:           {title}\n"
        f"url:               {url}\n"
        f"category:          {row['category']}\n"
        f"q_S/q_M/q_L_show:  {row['small_qty']}/{row['medium_qty']}/{row['large_qty']}\n"
        f"q_L_real:          {row.get('large_qty_real', row['large_qty'])}\n"
        f"P_S/P_M/P_L:       ${row['P_small']:.2f} / ${row['P_medium']:.2f} / ${row['P_large']:.2f}\n"
        f"value_ids (S/M/L): {int(row['small_id'])}/{int(row['medium_id'])}/{int(row['large_id'])}\n"
        f"\nIntervention log:\n" + "\n".join(f"  {x}" for x in log)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometries", default=GEOM_CSV)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--titles", nargs="+", default=None,
                    help="Substring match against template_title. Default: 3 demo products.")
    ap.add_argument("--all-faked", action="store_true",
                    help="Render all 16 q_L-faked products.")
    ap.add_argument("--arm", default="treat-medium",
                    choices=["baseline", "treat-strong", "treat-medium", "treat-weak"])
    args = ap.parse_args()

    geo = pd.read_csv(args.geometries)
    geo = geo[geo.arm == args.arm].reset_index(drop=True)

    if args.all_faked:
        chosen = geo[geo.large_qty != geo.large_qty_real]
    elif args.titles:
        mask = pd.Series(False, index=geo.index)
        for needle in args.titles:
            mask |= geo.template_title.str.contains(needle, case=False, regex=False)
        chosen = geo[mask]
    else:
        mask = pd.Series(False, index=geo.index)
        for needle in DEFAULT_TITLES:
            mask |= geo.template_title.str.contains(needle, case=False, regex=False)
        chosen = geo[mask]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(chosen)} product(s) at arm={args.arm} -> {out_dir}/")
    for _, row in chosen.iterrows():
        render_one(row, out_dir)
    print(f"\nDone. Open the after.html files in a browser to inspect.")


if __name__ == "__main__":
    main()
