"""
Analyze the URL-mask × warning factorial across models and samples.

Walks results/url_mask_compare/<cond>/<model>/sample_<N>/run-*/exp*/ for the
four conditions (v2_nowarn, v2_warn_new, v3_nowarn, v3_warn_new) and prints:
  1. A per-sample table (one row per condition × model)
  2. An across-samples summary (mean Δtarget pooled over all sample×product pairs)

Usage:
    python scripts/analyze_url_mask_compare.py
    python scripts/analyze_url_mask_compare.py --samples 1,2,3
    python scripts/analyze_url_mask_compare.py --out-csv analysis_output/url_mask_compare.csv
"""
import argparse
import gzip
import pickle
import re
import sys
from pathlib import Path

# Ensure project root (parent of scripts/) is on sys.path so pickled StepInfo
# instances that reference agentlab.* modules can be reconstructed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path("results/url_mask_compare")

CONDS = [
    ("v2",        "nowarn",   "v2_nowarn"),
    ("v2",        "warn_new", "v2_warn_new"),
    ("v2_strong", "nowarn",   "v2_strong_nowarn"),
    ("v2_strong", "warn_new", "v2_strong_warn_new"),
    ("v2_weak",   "nowarn",   "v2_weak_nowarn"),
    ("v2_weak",   "warn_new", "v2_weak_warn_new"),
    ("v3",        "nowarn",   "v3_nowarn"),
    ("v3",        "warn_new", "v3_warn_new"),
]

MODELS = [
    "claude-opus-4-6",
    "gpt-5-2",
    "gemini-2-5-pro",   # on-disk dir is hyphenated (run_url_mask_pair.sh strips only "vertex-")
    "deepseek-v3-2",
    "qwen-3-5",
]


def find_chosen_value_id(run_subdir: Path) -> str | None:
    """Walk steps to recover the radio value the agent clicked."""
    step_files = sorted(run_subdir.glob("step_*.pkl.gz"))
    if not step_files:
        return None
    bid_to_value: dict[str, str] = {}
    for sf in step_files:
        try:
            d = pickle.load(gzip.open(sf))
        except (EOFError, OSError):
            continue  # mid-write or corrupt
        obs = getattr(d, "obs", None) or {}
        html = obs.get("pruned_html", "") if isinstance(obs, dict) else ""
        for m in re.finditer(r'bid="([^"]+)"[^>]*type="radio"[^>]*value="(\d+)"', html):
            bid_to_value[m.group(1)] = m.group(2)
    last = None
    for sf in step_files:
        try:
            d = pickle.load(gzip.open(sf))
        except (EOFError, OSError):
            continue
        action = getattr(d, "action", None)
        if not action:
            continue
        m = re.match(r"click\(['\"]([^'\"]+)['\"]\)", action.strip())
        if m and m.group(1) in bid_to_value:
            last = bid_to_value[m.group(1)]
    return last


def collect_one(cond_label: str, model: str, sample_dir: Path) -> pd.DataFrame:
    """Build one trial-level row per exp under sample_dir."""
    rows = []
    sample_idx = int(sample_dir.name.split("_")[1])
    # Take the latest run-* if multiple exist
    run_dirs = sorted(sample_dir.glob("run-*"))
    if not run_dirs:
        return pd.DataFrame()
    run_dir = run_dirs[-1]
    for cfg_path in sorted(run_dir.glob("exp*/config.yaml")):
        exp_dir = cfg_path.parent
        cfg = yaml.safe_load(cfg_path.read_text())
        md = cfg["task"]["config"]["metadata"]
        sf = exp_dir / "summary_df_trial_1_of_1.csv"
        if not sf.exists():
            continue
        summary = pd.read_csv(sf).iloc[0]
        reward = float(summary["avg_reward"])
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
            "cond":          cond_label,
            "model":         model,
            "sample":        sample_idx,
            "exp_name":      exp_dir.name,
            "arm":           md["arm"],
            "product_id":    int(md["product_id"]),
            "reward":        reward,
            "completed":     reward > 0,
            "chosen_role":   chosen_role,
            "chose_target":  chosen_role == "large",
            "chose_decoy":   chosen_role == "medium",
            "chose_competitor": chosen_role == "small",
        })
    return pd.DataFrame(rows)


def collect_all(samples: list[int] | None) -> pd.DataFrame:
    """Walk the full results/url_mask_compare/ tree and concatenate."""
    frames = []
    for geom, warn, dirname in CONDS:
        cond_root = ROOT / dirname
        if not cond_root.exists():
            continue
        for model in MODELS:
            mdir = cond_root / model
            if not mdir.exists():
                continue
            for sdir in sorted(mdir.glob("sample_*")):
                try:
                    sn = int(sdir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if samples is not None and sn not in samples:
                    continue
                cond_label = f"{geom}_{warn}"
                frames.append(collect_one(cond_label, model, sdir))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def per_cell_stats(df: pd.DataFrame) -> dict:
    """Per (sample, model, cond): paired Δtarget over products.

    The treatment arm name varies across conditions: v2/v3 use "treat-medium",
    v2_strong uses "treat-strong", v2_weak uses "treat-weak". Pick whichever
    non-baseline arm is present in this slice.
    """
    df_c = df[df["completed"]]
    pivot = (
        df_c.pivot_table(
            index="product_id",
            columns="arm",
            values="chose_target",
        ).dropna()
    )
    treat_arms = [c for c in pivot.columns if c.startswith("treat-")]
    if "baseline" not in pivot.columns or not treat_arms:
        return dict(n=0, baseline_L=np.nan, treat_L=np.nan, treat_M=np.nan,
                    delta=np.nan, t=np.nan, p=np.nan, n_pos=0, n_neg=0, n_zero=0)
    treat_arm = treat_arms[0]  # one treatment arm per cell by design
    delta = pivot[treat_arm] - pivot["baseline"]
    n = len(delta)
    sd = float(delta.std())
    if sd == 0 or np.isnan(sd):
        t, p = float("nan"), float("nan")
    else:
        t, p = stats.ttest_1samp(delta, 0)
    bl_L = float(df_c[df_c["arm"] == "baseline"]["chose_target"].mean())
    tr_L = float(df_c[df_c["arm"] == treat_arm]["chose_target"].mean())
    tr_M = float(df_c[df_c["arm"] == treat_arm]["chose_decoy"].mean())
    return dict(
        n=n, baseline_L=bl_L, treat_L=tr_L, treat_M=tr_M,
        delta=float(delta.mean()), t=float(t), p=float(p),
        n_pos=int((delta > 0).sum()),
        n_neg=int((delta < 0).sum()),
        n_zero=int((delta == 0).sum()),
    )


def print_per_sample_tables(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data found under results/url_mask_compare/")
        return
    samples = sorted(df["sample"].unique())
    cond_order = [f"{g}_{w}" for g, w, _ in CONDS]
    for s in samples:
        print(f"\n{'='*92}")
        print(f"SAMPLE {s}")
        print('='*92)
        header = f'{"Cond":<14} {"Model":<18} {"n":>3}  {"BslL%":>6}  {"TrtL%":>6}  {"TrtM%":>6}  {"Δ pp":>7}  {"p":>7}  {"+/-/=":>10}'
        print(header)
        print('-' * len(header))
        for cond in cond_order:
            for model in MODELS:
                sub = df[(df["sample"] == s) & (df["model"] == model) & (df["cond"] == cond)]
                if sub.empty:
                    continue
                st = per_cell_stats(sub)
                p_str = f"{st['p']:.4f}" if not np.isnan(st['p']) else "  nan "
                print(
                    f"{cond:<14} {model:<18} {st['n']:>3}  "
                    f"{st['baseline_L']*100:>6.1f}  {st['treat_L']*100:>6.1f}  "
                    f"{st['treat_M']*100:>6.1f}  {st['delta']*100:>+7.1f}  {p_str:>7}  "
                    f"{st['n_pos']:>3}/{st['n_neg']}/{st['n_zero']}"
                )


def print_pooled_table(df: pd.DataFrame) -> None:
    """Pool across samples: per (model, cond) pair-level deltas across all sample×product pairs."""
    if df.empty:
        return
    cond_order = [f"{g}_{w}" for g, w, _ in CONDS]
    print(f"\n{'='*92}")
    print("POOLED ACROSS SAMPLES (per-product paired deltas, all samples)")
    print('='*92)
    header = f'{"Cond":<14} {"Model":<18} {"#smp":>5}  {"N pairs":>8}  {"Δ pp":>7}  {"95% CI":>20}  {"p":>7}'
    print(header)
    print('-' * len(header))
    for cond in cond_order:
        for model in MODELS:
            sub = df[(df["model"] == model) & (df["cond"] == cond) & df["completed"]]
            if sub.empty:
                continue
            n_samples = sub["sample"].nunique()
            pivot = sub.pivot_table(
                index=["sample", "product_id"], columns="arm", values="chose_target"
            ).dropna()
            treat_arms = [c for c in pivot.columns if c.startswith("treat-")]
            if "baseline" not in pivot.columns or not treat_arms:
                continue
            treat_arm = treat_arms[0]
            delta = pivot[treat_arm] - pivot["baseline"]
            n = len(delta)
            mean = float(delta.mean())
            sd = float(delta.std())
            sem = sd / np.sqrt(n) if n > 0 else float("nan")
            if sd == 0 or np.isnan(sd):
                t, p = float("nan"), float("nan")
            else:
                t, p = stats.ttest_1samp(delta, 0)
            ci_lo = (mean - 1.96 * sem) * 100
            ci_hi = (mean + 1.96 * sem) * 100
            p_str = f"{p:.4f}" if not np.isnan(p) else "  nan "
            print(
                f"{cond:<14} {model:<18} {n_samples:>5}  {n:>8}  "
                f"{mean*100:>+7.1f}  [{ci_lo:>+5.1f}, {ci_hi:>+5.1f}]   {p_str:>7}"
            )


def print_all_models_pooled(df: pd.DataFrame) -> None:
    """Pool across BOTH samples AND models per condition.

    Two views per condition:
      (a) trial-level pool: every (model, sample, product) Δ is one observation,
          equal weight per observation
      (b) model-equal-weight: average per-model means, equal weight per model.
    """
    if df.empty:
        return
    cond_order = [f"{g}_{w}" for g, w, _ in CONDS]
    print(f"\n{'='*92}")
    print("POOLED ACROSS ALL MODELS AND SAMPLES (per condition)")
    print('='*92)
    header = (
        f'{"Cond":<14} {"#models":>7} {"#smp":>5} {"N pairs":>8}  '
        f'{"Δ pooled":>9} {"95% CI":>20} {"p":>8}   {"Δ model-avg":>12}'
    )
    print(header)
    print('-' * len(header))
    for cond in cond_order:
        sub = df[(df["cond"] == cond) & df["completed"]]
        if sub.empty:
            continue
        # (a) trial-level pool over (model, sample, product)
        pivot = sub.pivot_table(
            index=["model", "sample", "product_id"],
            columns="arm", values="chose_target",
        ).dropna()
        treat_arms = [c for c in pivot.columns if c.startswith("treat-")]
        if "baseline" not in pivot.columns or not treat_arms:
            continue
        treat_arm = treat_arms[0]
        delta = pivot[treat_arm] - pivot["baseline"]
        n = len(delta)
        n_models = sub["model"].nunique()
        n_samples = sub["sample"].nunique()
        mean = float(delta.mean())
        sd = float(delta.std())
        sem = sd / np.sqrt(n) if n > 0 else float("nan")
        if sd == 0 or np.isnan(sd):
            p = float("nan")
        else:
            _, p = stats.ttest_1samp(delta, 0)
        ci_lo = (mean - 1.96 * sem) * 100
        ci_hi = (mean + 1.96 * sem) * 100
        p_str = f"{p:.4f}" if not np.isnan(p) else "  nan "
        # (b) per-model means → average
        per_model = (
            pivot.groupby(level="model")
            .apply(lambda g: (g[treat_arm] - g["baseline"]).mean())
        )
        model_avg = float(per_model.mean()) if len(per_model) > 0 else float("nan")
        print(
            f'{cond:<14} {n_models:>7} {n_samples:>5} {n:>8}  '
            f'{mean*100:>+8.1f}  [{ci_lo:>+5.1f}, {ci_hi:>+5.1f}]   {p_str:>7}   '
            f'{model_avg*100:>+11.1f}'
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples", default=None,
        help="Comma-separated list of sample indices to include (default: all).",
    )
    ap.add_argument(
        "--out-csv", default=None,
        help="If set, write the trial-level dataframe to this CSV.",
    )
    args = ap.parse_args()

    samples = None
    if args.samples is not None:
        samples = [int(s) for s in args.samples.split(",")]

    df = collect_all(samples)
    print_per_sample_tables(df)
    print_pooled_table(df)
    print_all_models_pooled(df)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote trial-level data → {args.out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
