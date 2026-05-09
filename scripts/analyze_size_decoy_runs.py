"""Per-model analysis for the within-product size-decoy experiments.

For one or more model run-roots, walks each sample's experiments, resolves the
agent's chosen size option from action history, and prints:
  - per-arm shares (target / decoy / competitor)
  - Δtarget vs baseline with paired t-test
  - per-sample replicability table
  - per-trial diagnostics (completion rate, n_steps, mean cost)

Run roots are expected to follow:
    results/decoy_size_v5a_<model_label>/sample_<1..N>/run-*/exp*/

Examples:
    # Single model
    python3 scripts/analyze_size_decoy_runs.py \\
        --run-root results/decoy_size_v5a_gpt52 \\
        --model-label gpt-5-2

    # Multiple models side-by-side
    python3 scripts/analyze_size_decoy_runs.py \\
        --run-root results/decoy_size_v5a_gpt52 \\
        --model-label gpt-5-2 \\
        --run-root results/decoy_size_v5a_claude46 \\
        --model-label claude-opus-4-6 \\
        --run-root results/decoy_size_v5a_gemini25 \\
        --model-label gemini-2.5-pro

    # Save trial-level CSV for downstream stats / plots
    python3 scripts/analyze_size_decoy_runs.py \\
        --run-root results/decoy_size_v5a_gpt52 \\
        --model-label gpt-5-2 \\
        --out-csv analysis_output/v5a_gpt52_trials.csv
"""
import argparse
import gzip
import pickle
import re
import sys
from pathlib import Path

# Allow loading agentlab/browsergym pickles
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from scipy import stats


CLICK_RE = re.compile(r"click\(['\"](\w+)['\"]\)")


def load_step(p: Path):
    with gzip.open(p) as f:
        return pickle.load(f)


def find_chosen_value_id(run_subdir: Path) -> str | None:
    """Walk the agent's steps, return the LAST clicked Magento radio's value
    attribute, computed by mapping the clicked bid to the option's value via
    the step's pruned HTML."""
    from browsergym.utils.obs import flatten_dom_to_str, prune_html
    chosen = None
    radio_map: dict[str, str] = {}
    for sp in sorted(run_subdir.glob("step_*.pkl.gz")):
        try:
            s = load_step(sp)
        except Exception:
            continue
        try:
            dom_txt = flatten_dom_to_str(
                s.obs["dom_object"],
                extra_properties=s.obs["extra_element_properties"],
                with_visible=True,
                filter_visible_only=True,
            )
            soup = BeautifulSoup(prune_html(dom_txt), "lxml")
            for inp in soup.select("input.product-custom-option[type=radio]"):
                bid = inp.get("bid", "")
                vid = inp.get("value", "")
                if bid and vid:
                    radio_map[bid] = vid
        except Exception:
            pass
        if not s.action:
            continue
        m = CLICK_RE.match(str(s.action))
        if m and m.group(1) in radio_map:
            chosen = radio_map[m.group(1)]
    return chosen


def collect_one_model(run_root: Path, model_label: str) -> pd.DataFrame:
    """Walk all sample_*/run-*/exp*/ folders under run_root and produce one
    row per completed trial."""
    rows = []
    for sample_dir in sorted(run_root.glob("sample_*")):
        try:
            sample_idx = int(sample_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for cfg_path in sorted(sample_dir.glob("run-*/exp*/config.yaml")):
            exp_dir = cfg_path.parent
            cfg = yaml.safe_load(cfg_path.read_text())
            md = cfg["task"]["config"]["metadata"]
            sf = exp_dir / "summary_df_trial_1_of_1.csv"
            if not sf.exists():
                continue
            summary = pd.read_csv(sf).iloc[0]
            reward = float(summary["avg_reward"])
            n_steps = int(summary["avg_steps"])
            cost = float(summary["cum_cost"])
            run_subdir = next(
                (d for d in exp_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()),
                None,
            )
            chosen_vid = find_chosen_value_id(run_subdir) if run_subdir else None
            role_by_id = {
                str(int(md["small_id"])):  "small",
                str(int(md["medium_id"])): "medium",
                str(int(md["large_id"])):  "large",
            }
            chosen_role = role_by_id.get(chosen_vid, "unknown")
            rows.append({
                "model": model_label,
                "sample": sample_idx,
                "exp_name": exp_dir.name,
                "arm": md["arm"],
                "product_id": int(md["product_id"]),
                "category": md["category"],
                "template_title": md["template_title"],
                "unit": md["unit"],
                "qty_S": float(md["small_qty"]),
                "qty_M": float(md["medium_qty"]),
                "qty_L": float(md["large_qty"]),
                "P_S": md.get("P_small"),
                "P_M": md.get("P_medium"),
                "P_L": md.get("P_large"),
                "reward": reward,
                "n_steps": n_steps,
                "cost": cost,
                "chosen_value_id": chosen_vid,
                "chosen_role": chosen_role,
                "completed": reward > 0,
                "chose_target": chosen_role == "large",
                "chose_decoy": chosen_role == "medium",
                "chose_competitor": chosen_role == "small",
            })
    return pd.DataFrame(rows)


def per_arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm shares of small/medium/large among completed trials."""
    completed = df[df["completed"]]
    return (
        completed.groupby("arm", dropna=False)
        .agg(
            n=("chosen_role", "size"),
            target=("chose_target", "mean"),
            decoy=("chose_decoy", "mean"),
            competitor=("chose_competitor", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("arm")
    )


def paired_test(df: pd.DataFrame) -> tuple[float, float, float, int, float, float]:
    """Within-product paired t-test of Δtarget = treat-medium − baseline.
    Returns (mean_delta, t, p, n, ci_lo, ci_hi)."""
    completed = df[df["completed"]]
    pivot = (
        completed.pivot_table(
            index=["product_id", "sample"],
            columns="arm",
            values="chose_target",
        ).dropna()
    )
    if "baseline" not in pivot.columns or "treat-medium" not in pivot.columns:
        return float("nan"), float("nan"), float("nan"), 0, float("nan"), float("nan")
    delta = pivot["treat-medium"] - pivot["baseline"]
    n = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std())
    sem = sd / np.sqrt(n) if n > 0 else float("nan")
    t, p = stats.ttest_1samp(delta, 0)
    return mean, float(t), float(p), n, mean - 1.96 * sem, mean + 1.96 * sem


def per_sample_table(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["completed"]]
    return (
        completed.groupby(["sample", "arm"], dropna=False)
        .agg(
            n=("chosen_role", "size"),
            target=("chose_target", "mean"),
            decoy=("chose_decoy", "mean"),
        )
        .round(3)
        .reset_index()
    )


def diagnostics(df: pd.DataFrame) -> dict:
    return {
        "total_trials": len(df),
        "completed": int(df["completed"].sum()),
        "completion_rate": round(float(df["completed"].mean()), 4),
        "median_steps": int(df["n_steps"].median()),
        "max_steps": int(df["n_steps"].max()),
        "n_steps_high_5plus": int((df["n_steps"] >= 5).sum()),
        "n_unknown_role": int((df["chosen_role"] == "unknown").sum()),
        "mean_cost_per_trial": round(float(df["cost"].mean()), 4),
        "total_cost": round(float(df["cost"].sum()), 2),
    }


def print_model_block(df: pd.DataFrame, model_label: str):
    print(f"\n{'='*78}")
    print(f"MODEL: {model_label}")
    print(f"{'='*78}")
    diag = diagnostics(df)
    print("\n  Diagnostics:")
    for k, v in diag.items():
        print(f"    {k:20s} {v}")

    summary = per_arm_summary(df)
    print("\n  Per-arm shares (completed trials):")
    print("  " + summary.to_string(index=False).replace("\n", "\n  "))

    mean, t, p, n, lo, hi = paired_test(df)
    if not np.isnan(mean):
        print(f"\n  Paired Δtarget (treat-medium − baseline), within-product:")
        print(f"    N pairs (product × sample): {n}")
        print(f"    Mean Δ = {mean:+.4f}    95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    Paired t-test:  t = {t:.3f}, p = {p:.4f}")

    per_sample = per_sample_table(df)
    if not per_sample.empty:
        print("\n  Per-sample replicability:")
        print("  " + per_sample.to_string(index=False).replace("\n", "\n  "))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", action="append", required=True,
                    help="One or more run-root directories (e.g. results/decoy_size_v5a_gpt52)")
    ap.add_argument("--model-label", action="append", required=True,
                    help="Label per --run-root (must come in same order)")
    ap.add_argument("--out-csv", default=None,
                    help="If set, writes the combined per-trial table here")
    args = ap.parse_args()

    if len(args.run_root) != len(args.model_label):
        raise SystemExit("--run-root and --model-label must be paired (same count)")

    all_dfs = []
    for root, label in zip(args.run_root, args.model_label):
        root_p = Path(root)
        if not root_p.exists():
            print(f"WARN: {root} not found, skipping")
            continue
        print(f"Collecting {label} from {root} ...", flush=True)
        df = collect_one_model(root_p, label)
        print_model_block(df, label)
        all_dfs.append(df)

    if args.out_csv and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out_csv, index=False)
        print(f"\n→ wrote combined trial table to {args.out_csv} ({len(combined)} rows)")

    # Cross-model summary if more than one
    if len(all_dfs) > 1:
        print(f"\n\n{'='*78}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*78}")
        rows = []
        for df in all_dfs:
            label = df["model"].iloc[0]
            mean, t, p, n, lo, hi = paired_test(df)
            sm = per_arm_summary(df).set_index("arm")
            rows.append({
                "model": label,
                "baseline_target":      sm.loc["baseline",      "target"]       if "baseline" in sm.index else None,
                "baseline_competitor":  sm.loc["baseline",      "competitor"]  if "baseline" in sm.index else None,
                "treat_target":         sm.loc["treat-medium", "target"]       if "treat-medium" in sm.index else None,
                "treat_decoy":          sm.loc["treat-medium", "decoy"]        if "treat-medium" in sm.index else None,
                "delta_target":         round(mean, 3),
                "ci_lo":                round(lo, 3),
                "ci_hi":                round(hi, 3),
                "p_value":              round(p, 4),
            })
        comp = pd.DataFrame(rows)
        print()
        print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
