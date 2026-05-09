"""
Preprocess aggregated ABxLab results for the within-product size-decoy
experiments.

Sibling to scripts/preprocess_decoy_results.py, but adapted for the new design
where each experiment is a SINGLE product page with multiple Magento size
options. Role (small / medium / large) is resolved from the clicked radio
button's `value` attribute, not from a URL match.

Pipeline:
    Input:  aggregated CSV from scripts/collect_results.py
    Output: per-trial CSV with role-aware columns ready for plotting / stats

Per row added:
    arm                  - baseline / treat-strong / treat-medium / treat-weak
    gap_pct              - 0.05 / 0.10 / 0.15 / NaN (baseline)
    chosen_value_id      - Magento option ID of the clicked radio
    chosen_role          - small / medium / large / unknown
    chosen_price         - price of the chosen option (from arm metadata)
    chose_target         - True iff chosen_role == 'large'
    chose_decoy          - True iff chosen_role == 'medium'
    chose_competitor     - True iff chosen_role == 'small'
    completed            - reward > 0 (Add-to-Cart succeeded)

Filters:
    - keeps only study_type == 'decoy_size_within_product'
    - keeps only completed trials (reward > 0)
"""

import argparse
import ast
import logging
import re

import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Pattern for an Add-to-Cart click — used to skip ATC and find the radio click
ATC_ID_HINT = re.compile(r"addtocart", re.I)


def safe_eval(value):
    """ast.literal_eval that returns None on failure / NaN."""
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def step_columns(df: pd.DataFrame, max_steps: int = 15) -> list[int]:
    """Return list of step indices that have action columns in df."""
    out = []
    for i in range(max_steps):
        if f"step_{i}.action.name" in df.columns:
            out.append(i)
    return out


def find_chosen_radio(row: pd.Series, step_idxs: list[int]) -> tuple[str, str, bool]:
    """Walk through this row's steps, return the LAST click on a Magento radio
    option. Returns (clicked_value_id, clicked_label, atc_clicked_yn).

    The aggregated CSV from collect_results.py flattens `elem_info.attrs` into
    one column per attribute (e.g. `step_1.elem_info.attrs.value`,
    `step_1.elem_info.attrs.class`), so we read these directly.
    """
    chosen_value_id = None
    chosen_label = None
    atc_clicked = False

    for i in step_idxs:
        action_name = row.get(f"step_{i}.action.name")
        if not isinstance(action_name, str) or action_name != "click":
            continue

        elem_id  = row.get(f"step_{i}.elem_info.attrs.id")
        elem_cls = row.get(f"step_{i}.elem_info.attrs.class")
        elem_val = row.get(f"step_{i}.elem_info.attrs.value")

        # Detect Add-to-Cart click
        if isinstance(elem_id, str) and ATC_ID_HINT.search(elem_id):
            atc_clicked = True
            continue

        # Detect a Magento custom-option radio click — class may be a list
        # (post json_normalize) or a space-separated string
        cls_tokens = []
        if isinstance(elem_cls, list):
            cls_tokens = elem_cls
        elif isinstance(elem_cls, str):
            # Could be a Python repr of a list, or a literal class string
            parsed = safe_eval(elem_cls)
            if isinstance(parsed, list):
                cls_tokens = parsed
            else:
                cls_tokens = elem_cls.split()
        if "product-custom-option" not in cls_tokens:
            continue

        if elem_val is not None and not (isinstance(elem_val, float) and pd.isna(elem_val)):
            # Normalize to int-string so we can match against small_id/etc.
            try:
                chosen_value_id = str(int(float(elem_val)))
            except (ValueError, TypeError):
                chosen_value_id = str(elem_val)
        chosen_label = row.get(f"step_{i}.elem_info.attrs.aria-label") or chosen_label

    return chosen_value_id, chosen_label, atc_clicked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_files", type=str, nargs="+", required=True,
                    help="Aggregated CSVs from scripts/collect_results.py")
    ap.add_argument("--output_file", type=str, required=True,
                    help="Where to write the preprocessed per-trial CSV")
    ap.add_argument("--max_steps", type=int, default=15,
                    help="Max step index to consider (matches benchmark cap).")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.input_files], ignore_index=True)
    logger.info("Loaded %d raw rows from %d input file(s)",
                len(df), len(args.input_files))

    # Keep only size-decoy rows
    if "cfg.task.config.metadata.study_type" in df.columns:
        before = len(df)
        df = df[df["cfg.task.config.metadata.study_type"] == "decoy_size_within_product"].copy()
        logger.info("Kept %d / %d rows after study_type filter", len(df), before)
    if df.empty:
        logger.warning("No size-decoy rows to process. Exiting.")
        return

    # Surface metadata columns into top-level names
    md_prefix = "cfg.task.config.metadata."
    df["arm"]            = df.get(md_prefix + "arm")
    df["gap_pct"]        = pd.to_numeric(df.get(md_prefix + "gap_pct"), errors="coerce")
    df["category"]       = df.get(md_prefix + "category")
    df["template_title"] = df.get(md_prefix + "template_title")
    df["product_id"]     = pd.to_numeric(df.get(md_prefix + "product_id"), errors="coerce").astype("Int64")
    df["unit"]           = df.get(md_prefix + "unit")
    df["small_qty"]      = pd.to_numeric(df.get(md_prefix + "small_qty"), errors="coerce")
    df["medium_qty"]     = pd.to_numeric(df.get(md_prefix + "medium_qty"), errors="coerce")
    df["large_qty"]      = pd.to_numeric(df.get(md_prefix + "large_qty"), errors="coerce")
    df["P_small"]        = pd.to_numeric(df.get(md_prefix + "P_small"), errors="coerce")
    df["P_medium"]       = pd.to_numeric(df.get(md_prefix + "P_medium"), errors="coerce")
    df["P_large"]        = pd.to_numeric(df.get(md_prefix + "P_large"), errors="coerce")
    df["natural_price"]  = pd.to_numeric(df.get(md_prefix + "natural_price"), errors="coerce")
    # Cast IDs to int-string to match the chosen_value_id normalization above
    def _to_int_str(s):
        try:
            return str(int(float(s)))
        except (ValueError, TypeError):
            return None
    df["small_id"]       = df.get(md_prefix + "small_id").map(_to_int_str)
    df["medium_id"]      = df.get(md_prefix + "medium_id").map(_to_int_str)
    df["large_id"]       = df.get(md_prefix + "large_id").map(_to_int_str)
    # Also surface the intent_variant tag for cross-variant analyses
    if md_prefix + "intent_variant" in df.columns:
        df["intent_variant"] = df.get(md_prefix + "intent_variant").fillna("default")

    # Model name (study fields populated by collect_results.py)
    df["agent_name"] = df.get("study.agent_name")
    df["model_name"] = df.get("study.chat_model_args.model_name")

    # Outcome from summary_info.json
    df["reward"] = pd.to_numeric(df.get("summary.cum_reward"), errors="coerce")
    df["n_steps"] = pd.to_numeric(df.get("summary.n_steps"), errors="coerce").astype("Int64")
    df["completed"] = df["reward"].fillna(0) > 0

    # Resolve chosen radio per row
    step_idxs = step_columns(df, max_steps=args.max_steps)
    logger.info("Scanning %d step columns per row", len(step_idxs))

    def _resolve(row):
        vid, label, atc = find_chosen_radio(row, step_idxs)
        return pd.Series({"chosen_value_id": vid,
                          "chosen_label": label,
                          "atc_clicked": atc})

    resolved = df.progress_apply(_resolve, axis=1)
    df = pd.concat([df, resolved], axis=1)

    # Map chosen value_id to role
    def _role(row):
        vid = row["chosen_value_id"]
        if not isinstance(vid, str) or vid in ("", "None"):
            return "unknown"
        if vid == row["small_id"]:  return "small"
        if vid == row["medium_id"]: return "medium"
        if vid == row["large_id"]:  return "large"
        return "unknown"

    df["chosen_role"] = df.apply(_role, axis=1)
    df["chose_target"]     = df["chosen_role"] == "large"
    df["chose_decoy"]      = df["chosen_role"] == "medium"
    df["chose_competitor"] = df["chosen_role"] == "small"

    # Chosen price for convenience
    def _chosen_price(row):
        return {
            "small":  row["P_small"],
            "medium": row["P_medium"],
            "large":  row["P_large"],
        }.get(row["chosen_role"])
    df["chosen_price"] = df.apply(_chosen_price, axis=1)

    # Final per-trial table — keep just the analysis-ready columns
    keep = [
        "experiment_id", "agent_name", "model_name",
        "intent_variant",
        "product_id", "category", "template_title",
        "arm", "gap_pct", "unit",
        "small_qty", "medium_qty", "large_qty",
        "P_small", "P_medium", "P_large", "natural_price",
        "small_id", "medium_id", "large_id",
        "reward", "n_steps", "completed", "atc_clicked",
        "chosen_value_id", "chosen_label", "chosen_role", "chosen_price",
        "chose_target", "chose_decoy", "chose_competitor",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out.to_csv(args.output_file, index=False)
    logger.info("Wrote per-trial table → %s (%d rows)", args.output_file, len(out))

    # Quick console summary so the user sees something immediately
    completed = out[out["completed"]]
    if completed.empty:
        logger.warning("No completed trials in output.")
        return

    print("\n=== Per-arm shares (completed trials) ===")
    summary = (
        completed
        .groupby("arm", dropna=False)
        .agg(
            n=("chosen_role", "size"),
            target_share=("chose_target", "mean"),
            decoy_share=("chose_decoy", "mean"),
            competitor_share=("chose_competitor", "mean"),
        )
        .round(3)
        .reset_index()
    )
    print(summary.to_string(index=False))

    if "baseline" in summary["arm"].values:
        baseline = float(summary.loc[summary["arm"] == "baseline", "target_share"].iloc[0])
        print("\n=== Decoy lift (Δtarget vs baseline) ===")
        for _, r in summary[summary["arm"] != "baseline"].iterrows():
            print(f"  {r['arm']:14s}  Δtarget = {r['target_share'] - baseline:+.3f}  (n={r['n']})")


if __name__ == "__main__":
    main()
