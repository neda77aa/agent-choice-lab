"""
Compute popcorn-style decoy price geometries for the within-product size-decoy
experiment.

Input:
    tasks/product_size_ladders-decoy.csv
        One row per candidate product, with the chosen S/M/L sizes,
        Magento option IDs, natural price, and template title.

Output:
    tasks/size_decoy_geometries.csv
        One row per (product × arm). For each product, emits four arms:
            - baseline: only S and L visible (no decoy)
            - treat-strong: S + M + L, M-L gap = 5% of P_L
            - treat-medium: S + M + L, M-L gap = 10% of P_L
            - treat-weak:   S + M + L, M-L gap = 15% of P_L

Pricing formula (per arm, anchored to the product's natural catalog price):
    P_L = natural_price                    # L = catalog price for the largest size
    P_M = P_L × (1 − gap_pct)              # M priced just below L
    P_S = q_S × (P_M / q_M)                # S inherits M's per-unit rate

Dominance check (popcorn): per-unit price at M must be strictly worse than at L,
i.e. q_L > q_M / (1 − gap_pct). Products that fail this check at any treatment
arm are dropped from the experiment.

This script does NOT touch any existing function in abxlab/ — it only consumes
the products CSV and emits a new geometries CSV that the experiment generator
will read.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# v1 (original) defaults — preserved so the script reproduces the previous
# geometry exactly when invoked with no flags.
GAP_LEVELS = [
    ("treat-strong", 0.05),
    ("treat-medium", 0.10),
    ("treat-weak",   0.15),
]

# v2 preset — bundles the three changes documented in the revised geometry
# section: a per-unit S penalty, a cap on the displayed q_L/q_M ratio, and
# a tightened dose range for the absolute medium-large gap.
PRESET_V2 = {
    "g_s": 0.15,
    "qL_cap": 1.5,
    "gap_levels": [("treat-strong", 0.08),
                   ("treat-medium", 0.10),
                   ("treat-weak",   0.15)],
}


def parse_price(x) -> float:
    """Catalog natural_price strings come as '$15.63'."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x).replace("$", "").replace(",", "").strip())


def fake_qL_int(q_m: float, target_ratio: float) -> int:
    """Smallest integer >= q_m * target_ratio that strictly exceeds q_m."""
    return max(int(q_m) + 1, round(q_m * target_ratio))


def relabel_qty(original_name: str, new_qty: float) -> str:
    """Rewrite the first numeric token in `original_name` to `new_qty`.

    e.g. relabel_qty("12 Pack", 4) -> "4 Pack"
         relabel_qty("Pack of 12", 4) -> "Pack of 4"
         relabel_qty("1.5 Pound", 1) -> "1 Pound"
    """
    new_qty_str = (str(int(new_qty)) if float(new_qty).is_integer()
                   else f"{new_qty:.2f}".rstrip("0").rstrip("."))
    return re.sub(r"\d+(?:\.\d+)?", new_qty_str, str(original_name), count=1)


def popcorn_prices(natural_price, q_s, q_m, q_l_real,
                   gap_pct: float, g_s: float = 0.0,
                   qL_cap: float | None = None,
                   s_anchor_gap: float | None = None):
    """Return (P_S, P_M, P_L, q_l_show) for a popcorn-style decoy.

    Args:
      natural_price: catalogue price for the product at q_l_real.
      q_s, q_m, q_l_real: native quantity ladder.
      gap_pct:  absolute (P_L - P_M) / P_L  (governs M's visible price).
                Negative values place M above L (absolute-dominance variant).
      g_s:      per-unit gap from M to S; P_S anchor = per_unit_M_anchor × (1 + g_s).
                Default 0.0 reproduces the v1 identity per_unit_S == per_unit_M.
      qL_cap:   if set and q_l_real / q_m > qL_cap, the displayed q_L is
                reduced to fake_qL_int(q_m, qL_cap) and P_L is rescaled at
                the natural per-unit rate. The underlying option_id is
                untouched; only the visible label and price change.
      s_anchor_gap: optional separate gap used to anchor S's per-unit price
                (defaults to gap_pct). When the main gap is negative
                (absolute-dominance M), passing a positive s_anchor_gap
                keeps S at popcorn-style per-unit pricing instead of
                inheriting M's high price.

    Raises ValueError if the M-vs-L dominance condition
    q_L > q_M / (1 - gap_pct) fails on the *displayed* q_L.
    """
    P_L_natural = parse_price(natural_price)

    if qL_cap is not None and q_l_real / q_m > qL_cap:
        q_l_show = fake_qL_int(q_m, qL_cap)
        per_unit_L = P_L_natural / q_l_real
        P_L = round(q_l_show * per_unit_L, 2)
    else:
        q_l_show = q_l_real
        P_L = round(P_L_natural, 2)

    threshold = q_m / (1 - gap_pct)
    if q_l_show <= threshold:
        raise ValueError(
            f"Ladder cannot support {gap_pct:.0%} popcorn gap: "
            f"need q_L > {threshold:.3f}; got q_L = {q_l_show}."
        )

    P_M = round(P_L * (1 - gap_pct), 2)

    # S is anchored to its own (possibly different) gap. Defaults to gap_pct.
    # Round P_M_for_S to 2 decimals to match the historical v1/v2 rounding
    # path (which rounded P_M before deriving per_unit_S).
    s_gap = s_anchor_gap if s_anchor_gap is not None else gap_pct
    P_M_for_S = round(P_L * (1 - s_gap), 2)
    per_unit_S = (P_M_for_S / q_m) * (1 + g_s)
    P_S = round(q_s * per_unit_S, 2)
    return P_S, P_M, P_L, q_l_show


def baseline_prices(natural_price, q_s, q_m, q_l_real,
                    g_s: float = 0.0, qL_cap: float | None = None,
                    baseline_gap: float = 0.10,
                    s_anchor_gap: float | None = None):
    """Baseline arm: only S and L visible. Uses the SAME geometry as the
    treat-medium arm so the only difference vs treatment is the presence of M.

    `baseline_gap` defaults to 0.10 (the historical convention) but should be
    set to the gap of the treat-medium arm whenever that differs (e.g. v3
    absolute-dominance geometry uses -0.05). `s_anchor_gap` follows the same
    rule as in the treat-medium arm so P_S is identical across baseline and
    treatment.
    """
    P_S, _, P_L, q_l_show = popcorn_prices(
        natural_price, q_s, q_m, q_l_real,
        gap_pct=baseline_gap, g_s=g_s, qL_cap=qL_cap,
        s_anchor_gap=s_anchor_gap,
    )
    return P_S, None, P_L, q_l_show


def parse_gap_levels(specs: list[str]) -> list[tuple[str, float]]:
    """Parse --gap_levels arguments like 'treat-strong:0.08'."""
    out = []
    for spec in specs:
        if ":" not in spec:
            raise argparse.ArgumentTypeError(
                f"--gap_levels item must be 'name:gap_pct', got {spec!r}"
            )
        name, val = spec.split(":", 1)
        out.append((name.strip(), float(val)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="tasks/product_size_ladders-decoy.csv",
        help="Input products CSV.",
    )
    parser.add_argument(
        "--output",
        default="tasks/size_decoy_geometries.csv",
        help="Output geometries CSV.",
    )
    parser.add_argument(
        "--g_s", type=float, default=None,
        help=("Per-unit penalty for the small option: per_unit_S = "
              "per_unit_M × (1 + g_s). Default 0.0 (S and M share per-unit "
              "price, the v1 behavior). The v2 preset uses 0.15."),
    )
    parser.add_argument(
        "--qL_cap", type=float, default=None,
        help=("If set, cap the displayed q_L/q_M ratio at this value. When "
              "the native ratio exceeds the cap, the largest option's label "
              "and price are rewritten to a smaller integer at the natural "
              "per-unit rate (Magento option_id is unchanged). Default: no "
              "cap (v1). The v2 preset uses 1.5."),
    )
    parser.add_argument(
        "--gap_levels", nargs="+", default=None,
        help=("Override the dose-response gap levels. Format: "
              "'name:gap_pct', e.g. 'treat-strong:0.08 treat-medium:0.10 "
              "treat-weak:0.15'. Default reproduces v1: "
              "treat-strong=0.05, treat-medium=0.10, treat-weak=0.15."),
    )
    parser.add_argument(
        "--s_anchor_gap", type=float, default=None,
        help=("Optional separate gap used to anchor S's per-unit price. "
              "Default: same as the treat-medium gap (current behavior). "
              "Useful with negative main gaps (v3 absolute-dominance): pass "
              "a positive value (e.g. 0.05) to keep S at popcorn-style "
              "pricing rather than inheriting M's high price."),
    )
    parser.add_argument(
        "--preset", choices=["v1", "v2"], default=None,
        help=("Convenience preset. 'v1' (default behavior) reproduces the "
              "original geometry. 'v2' bundles --g_s 0.15 --qL_cap 1.5 "
              "--gap_levels treat-strong:0.08 treat-medium:0.10 "
              "treat-weak:0.15."),
    )
    args = parser.parse_args()

    # Resolve presets, then let explicit flags override.
    g_s     = 0.0
    qL_cap  = None
    gap_lvl = list(GAP_LEVELS)
    if args.preset == "v2":
        g_s     = PRESET_V2["g_s"]
        qL_cap  = PRESET_V2["qL_cap"]
        gap_lvl = list(PRESET_V2["gap_levels"])
    if args.g_s        is not None: g_s     = args.g_s
    if args.qL_cap     is not None: qL_cap  = args.qL_cap
    if args.gap_levels is not None: gap_lvl = parse_gap_levels(args.gap_levels)

    # Anchor baseline prices to the treat-medium arm (so baseline and
    # treat-medium share identical S and L prices). Falls back to 0.10
    # if no treat-medium arm is configured.
    baseline_gap = next((g for n, g in gap_lvl if n == "treat-medium"), 0.10)

    df = pd.read_csv(args.input)
    rows = []
    skipped = []

    for _, p in df.iterrows():
        q_s = float(p["small_qty"])
        q_m = float(p["medium_qty"])
        q_l = float(p["large_qty"])
        nat = parse_price(p["natural_price"])

        # Compute the displayed q_L (may equal q_l, or be capped).
        if qL_cap is not None and q_l / q_m > qL_cap:
            q_l_show = float(fake_qL_int(q_m, qL_cap))
        else:
            q_l_show = q_l

        # Validate dominance at the worst-case (largest) gap level we use.
        max_gap = max(g for _, g in gap_lvl)
        if q_l_show <= q_m / (1 - max_gap):
            skipped.append((p["template_title"], q_s, q_m, q_l_show,
                            f"M-L dominance: need q_L > {q_m/(1-max_gap):.2f}"))
            continue

        # Validate S<M absolute-price monotonicity under the per-unit S penalty.
        # P_S < P_M iff q_S × per_unit_M × (1 + g_s) < q_M × per_unit_M
        # iff q_M / q_S > (1 + g_s).
        if g_s > 0 and q_m / q_s <= (1 + g_s):
            skipped.append((p["template_title"], q_s, q_m, q_l_show,
                            f"S-M monotonicity: need q_M/q_S > {1+g_s:.2f} "
                            f"(got {q_m/q_s:.3f})"))
            continue

        large_name_show = (relabel_qty(p["large_name"], q_l_show)
                           if q_l_show != q_l else p["large_name"])

        # Baseline arm
        P_S, _, P_L, _ = baseline_prices(nat, q_s, q_m, q_l, g_s=g_s,
                                         qL_cap=qL_cap, baseline_gap=baseline_gap,
                                         s_anchor_gap=args.s_anchor_gap)
        rows.append({
            "category": p["category"],
            "template_title": p["template_title"],
            "url": p["url"],
            "arm": "baseline",
            "gap_pct": None,
            "n_options_visible": 2,
            "small_id": int(p["small_id"]),
            "medium_id": int(p["medium_id"]),    # kept for reference; hidden in baseline
            "large_id": int(p["large_id"]),
            "small_name": p["small_name"],
            "medium_name": p["medium_name"],
            "large_name": large_name_show,
            "small_qty": q_s,
            "medium_qty": q_m,
            "large_qty": q_l_show,
            "large_qty_real": q_l,
            "unit": p["unit"],
            "P_small": P_S,
            "P_medium": None,
            "P_large": P_L,
            "natural_price": nat,
            "group_id": int(p["group_id"]),
            "product_id": int(p["product_id"]),
            "multi_option_group": bool(p["multi_option_group"]),
        })

        # Treatment arms
        for arm_label, gap in gap_lvl:
            P_S, P_M, P_L, _ = popcorn_prices(
                nat, q_s, q_m, q_l, gap_pct=gap, g_s=g_s, qL_cap=qL_cap,
                s_anchor_gap=args.s_anchor_gap,
            )
            rows.append({
                "category": p["category"],
                "template_title": p["template_title"],
                "url": p["url"],
                "arm": arm_label,
                "gap_pct": gap,
                "n_options_visible": 3,
                "small_id": int(p["small_id"]),
                "medium_id": int(p["medium_id"]),
                "large_id": int(p["large_id"]),
                "small_name": p["small_name"],
                "medium_name": p["medium_name"],
                "large_name": large_name_show,
                "small_qty": q_s,
                "medium_qty": q_m,
                "large_qty": q_l_show,
                "large_qty_real": q_l,
                "unit": p["unit"],
                "P_small": P_S,
                "P_medium": P_M,
                "P_large": P_L,
                "natural_price": nat,
                "group_id": int(p["group_id"]),
                "product_id": int(p["product_id"]),
                "multi_option_group": bool(p["multi_option_group"]),
            })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    n_products = out["product_id"].nunique()
    n_arms = len(out)
    print(f"Wrote {n_arms} arms across {n_products} products → {args.output}")
    print(f"  geometry params: g_s={g_s}, qL_cap={qL_cap}, "
          f"gap_levels={[(n, g) for n, g in gap_lvl]}")
    print(f"  baseline:     {(out['arm']=='baseline').sum()}")
    for label, _ in gap_lvl:
        print(f"  {label:14s} {(out['arm']==label).sum()}")
    if qL_cap is not None:
        n_faked = (out["large_qty"] != out["large_qty_real"]).sum() // (1 + len(gap_lvl))
        print(f"  q_L faked on {n_faked} products (cap = {qL_cap})")
    if skipped:
        print(f"\nSkipped {len(skipped)} products:")
        for t, s, m, l, reason in skipped:
            print(f"  - {t[:55]}: q={s}/{m}/{l}  ({reason})")


if __name__ == "__main__":
    main()
