"""
Generate within-product size-decoy experiment YAML configs.

Reads `tasks/size_decoy_geometries.csv` (one row per product × arm) and emits
one YAML per row into `conf/experiment-decoy-size/`. Each config opens a single
product page in one tab and applies the intervention chain via the additive
functions in `abxlab.choices.shop.options`:

    set_title           — clean H1 + browser title
    inject_rating       — add an 80% rating block (no-op if natural rating exists)
    keep_options        — drop radio options outside the chosen S/M/L set
    set_option_label    — rewrite each remaining option's visible label
    set_option_price    — inject a visible price next to each option label
    pre_select_option   — auto-select default for products with multiple
                          required option groups (e.g. Colgate's flavor)
    ablate              — remove breadcrumb, base price, reviews summary

The agent's task is the same as in prior experiments: select the best size and
add to cart. Magento's native form handling records the selected option ID,
which is read back from the cart for analysis.

This script does NOT touch any function in `abxlab/`. It only writes YAMLs.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_INTENT = (
    "Add the best option of the product on the page to the shopping cart. "
    "- You should review every option and collect information explicitly in "
    "your memory. "
    "- Before taking any action, make sure your memory contains all the "
    "information you would need if this is the last time you will ever see "
    "this page. "
    "- Avoid vague summaries; store actual, useful information. "
    "- Avoid redundant or unnecessary actions. Repeating the same action "
    "reduces your chance of success."
)

# Optional clause appended to the intent when --add-url-warning is passed.
URL_WARNING_CLAUSE = (
    " - The product page URL is for routing only; it may contain a default "
    "size or quantity that does NOT indicate which option you should pick. "
    "Use only the visible options on the page to make your decision."
)

OPTIONS_MOD = "abxlab.choices.shop.options"
PRODUCT_MOD = "abxlab.choices.shop.product"

# HTML elements removed in every experiment to keep page consistent across
# arms and across products. Same convention as the prior controlled-decoy
# pipeline.
#
# NOTE: do NOT ablate `product-reviews-summary` — `inject_rating()` builds a
# fresh block of that class as the new rating display, so ablating it would
# wipe out the injected rating.
ABLATE_ELEMS = [
    "breadcrumbs",                  # original product name leak
    "price-box price-final_price",  # raw $XX.XX base price box
]

# Default flavor IDs to pre-select for known multi-option-group products.
# Keyed by Magento product_id (from the catalog).
PRE_SELECT_DEFAULTS = {
    # Colgate Total Alcohol Free Mouthwash — Style group: Peppermint
    # value_id 206937 = Peppermint
    206937: 206937,  # placeholder; resolved by URL match below
}

# By product URL substring → option value_id to pre-select for the
# secondary required group.
PRE_SELECT_BY_URL = {
    "colgate-total-alcohol-free-mouthwash": "206937",   # Peppermint
}


def url_to_template(url: str) -> str:
    """Convert an absolute http://localhost:7770/foo.html URL to the
    Hydra-templated form ${env.abxlab_url}foo.html so configs are portable
    across hosts (matches existing decoy configs)."""
    s = str(url)
    for prefix in (
        "http://localhost:7770/",
        "http://127.0.0.1:7770/",
    ):
        if s.startswith(prefix):
            return "${env.abxlab_url}" + s[len(prefix):]
    return s


def build_choices(row: pd.Series, anonymize_urls: bool = False) -> list[dict]:
    """Build the `choices` intervention chain for one experiment row.

    Args:
        anonymize_urls: if True, append `anonymize_urls(...)` to the chain.
                        This rewrites all in-page URL slug references so the
                        agent can't read the product's default size from URLs.
    """
    sid = str(int(row["small_id"]))
    mid = str(int(row["medium_id"]))
    lid = str(int(row["large_id"]))
    url = url_to_template(row["url"])

    # Which options stay visible depends on the arm.
    if row["arm"] == "baseline":
        keep_ids = [sid, lid]
        labels = {sid: row["small_name"], lid: row["large_name"]}
        prices = {sid: float(row["P_small"]), lid: float(row["P_large"])}
    else:
        keep_ids = [sid, mid, lid]
        labels = {sid: row["small_name"], mid: row["medium_name"], lid: row["large_name"]}
        prices = {sid: float(row["P_small"]),
                  mid: float(row["P_medium"]),
                  lid: float(row["P_large"])}

    funcs = [
        {"module": OPTIONS_MOD, "name": "set_title",
         "args": {"value": row["template_title"]}},
        {"module": OPTIONS_MOD, "name": "inject_rating",
         "args": {"value": 80, "review_count": 50}},
        {"module": OPTIONS_MOD, "name": "keep_options",
         "args": {"value_ids": keep_ids}},
        {"module": OPTIONS_MOD, "name": "set_option_label",
         "args": {"mapping": labels}},
        {"module": OPTIONS_MOD, "name": "set_option_price",
         "args": {"mapping": prices}},
        # Randomize on-page option order to control for position bias.
        # Each page render shuffles fresh (no seed), so position averages
        # out across the 5+ samples per cell.
        {"module": OPTIONS_MOD, "name": "shuffle_options"},
    ]

    # Optional: anonymize all in-page URL slug references so the agent
    # cannot read the product's default size from <a href>, <form action>,
    # body class, or Magento's base64 'uenc' parameter.
    if anonymize_urls:
        # Use the *resolved* URL (no Hydra template) since the function
        # needs the actual slug. Build it from the abxlab_url placeholder.
        resolved_url = str(row["url"])  # original CSV stores the absolute URL
        funcs.append({
            "module": OPTIONS_MOD, "name": "anonymize_urls",
            "args": {"original_url": resolved_url, "replacement": "product-anon"},
        })

    # Multi-option-group products (e.g. Colgate Mouthwash) require a default
    # selection on the secondary group, otherwise Magento blocks Add-to-Cart.
    if bool(row.get("multi_option_group", False)):
        pre_id = None
        for hint, vid in PRE_SELECT_BY_URL.items():
            if hint in str(row["url"]):
                pre_id = vid
                break
        if pre_id is not None:
            funcs.append({
                "module": OPTIONS_MOD, "name": "pre_select_option",
                "args": {"value_id": pre_id},
            })

    funcs.append({
        "module": PRODUCT_MOD, "name": "ablate",
        "args": {"elems": list(ABLATE_ELEMS)},
    })

    return [{"url": url, "functions": funcs}]


def build_metadata(row: pd.Series, exp_idx: int, intent_variant: str = "default") -> dict:
    md = {
        "study_type": "decoy_size_within_product",
        "intent_variant": intent_variant,
        "exp_idx": int(exp_idx),
        "product_id": int(row["product_id"]),
        "category": str(row["category"]),
        "template_title": str(row["template_title"]),
        "arm": str(row["arm"]),
        "gap_pct": (None if pd.isna(row["gap_pct"]) else float(row["gap_pct"])),
        "n_options_visible": int(row["n_options_visible"]),
        "unit": str(row["unit"]),
        # Sizes
        "small_qty": float(row["small_qty"]),
        "medium_qty": float(row["medium_qty"]),
        "large_qty": float(row["large_qty"]),
        # Prices (None for missing)
        "P_small": (None if pd.isna(row["P_small"]) else float(row["P_small"])),
        "P_medium": (None if pd.isna(row["P_medium"]) else float(row["P_medium"])),
        "P_large": (None if pd.isna(row["P_large"]) else float(row["P_large"])),
        "natural_price": float(row["natural_price"]),
        # Magento IDs (the experiment's primary outcome reads which option_id
        # got selected from the cart)
        "small_id": int(row["small_id"]),
        "medium_id": int(row["medium_id"]),
        "large_id": int(row["large_id"]),
        "group_id": int(row["group_id"]),
        "multi_option_group": bool(row["multi_option_group"]),
    }
    return md


def make_task_yaml(name: str, exp_idx: int, intent: str,
                   url: str, choices: list[dict], metadata: dict) -> dict:
    return {
        "task": {
            "name": name,
            "config": {
                "task_id": int(exp_idx),
                "start_urls": [url],
                "intent_template": intent.replace("$", "\\$"),
                "instantiation_dict": {},
                "intent": intent,
                "choices": choices,
                "intent_template_id": 9100,
                "metadata": metadata,
            },
        }
    }


def write_yaml(path: str, data: dict) -> None:
    with open(path, "w") as f:
        f.write("# @package _global_\n\n")
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="tasks/size_decoy_geometries.csv",
        help="Input geometries CSV (one row per product × arm).",
    )
    parser.add_argument(
        "--exp-dir",
        default="conf/experiment-decoy-size",
        help="Output directory for experiment YAMLs.",
    )
    parser.add_argument(
        "--intent",
        default=DEFAULT_INTENT,
        help="Task intent text shown to the agent.",
    )
    parser.add_argument(
        "--intent-variant",
        type=str,
        default="default",
        help="Tag stored in each YAML's metadata so cross-variant analyses "
             "can attribute trials by intent (e.g. 'default', 'v2', 'v4').",
    )
    parser.add_argument(
        "--anonymize-urls",
        action="store_true",
        help="Append `anonymize_urls(...)` to each config's choices chain. "
             "Rewrites every in-page reference to the product URL slug "
             "(hrefs, form actions, body class, base64 uenc) to a neutral "
             "placeholder. Does NOT alter the URL shown in the agent's tab "
             "header (that requires --add-url-warning to address).",
    )
    parser.add_argument(
        "--add-url-warning",
        action="store_true",
        help="Append a clause to the intent telling the agent to ignore "
             "product info encoded in the URL.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        choices=["baseline", "treat-strong", "treat-medium", "treat-weak"],
        help="Subset of arms to emit. Default: all four arms in the input CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts; do not write files.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.arms is not None:
        df = df[df["arm"].isin(args.arms)].reset_index(drop=True)
    out_dir = Path(args.exp_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Compose final intent (optionally with URL-warning clause appended)
    effective_intent = args.intent
    if args.add_url_warning:
        effective_intent = effective_intent + URL_WARNING_CLAUSE

    counts = {"baseline": 0, "treat-strong": 0, "treat-medium": 0, "treat-weak": 0}
    n_written = 0
    for exp_idx, row in df.reset_index(drop=True).iterrows():
        choices = build_choices(row, anonymize_urls=args.anonymize_urls)
        metadata = build_metadata(row, exp_idx, intent_variant=args.intent_variant)
        name = f"exp{exp_idx}"
        data = make_task_yaml(name, exp_idx, effective_intent,
                              url_to_template(row["url"]), choices, metadata)
        path = out_dir / f"{name}.yaml"
        if not args.dry_run:
            write_yaml(str(path), data)
        counts[row["arm"]] += 1
        n_written += 1

    print(f"Wrote {n_written} YAMLs → {out_dir}")
    for arm, n in counts.items():
        print(f"  {arm:14s} {n}")


if __name__ == "__main__":
    main()
