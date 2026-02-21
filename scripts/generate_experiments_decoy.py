"""
Generate decoy experiment configs from triad CSV.

For each triad, this script writes two configs:
- baseline: Target vs Competitor (2 URLs)
- treatment: Target vs Competitor vs Decoy (3 URLs)

It randomizes URL order per arm and stores role->position mapping in metadata
so analysis can recover which chosen URL corresponds to T/C/D.
"""

import os
import argparse
import random
import yaml
import pandas as pd

SEED = 42
EXP_DIR = "conf/experiment-decoy"
DEFAULT_INTENT = (
    "Add the best product from the open tabs to the shopping cart. "
    "- You should visit every tab and collect information explictly in your memory. "
    "- Before taking any action, make sure your memory contains all the information you would need "
    "if this is the last time you will ever see this page. "
    "- Avoid vague summaries; store actual, useful information. "
    "- Avoid redundant or unnecessary actions. Repeating the same action reduces your chance of success."
)


def role_positions(start_urls: list[str], role_to_url: dict[str, str]) -> dict[str, int]:
    return {f"{role}_pos": start_urls.index(url) for role, url in role_to_url.items() if url in start_urls}


def maybe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def build_metadata(row: pd.Series, triad_id: int, arm: str, start_urls: list[str], role_to_url: dict[str, str]) -> dict:
    md = {
        "study_type": "decoy_pricerating",
        "triad_id": int(triad_id),
        "arm": arm,
        "category": row.get("category", None),
        "target_url": role_to_url.get("target"),
        "competitor_url": role_to_url.get("competitor"),
        "decoy_url": role_to_url.get("decoy"),
    }

    md.update(role_positions(start_urls, role_to_url))

    # Keep diagnostics if present in input
    for c in [
        "tc_price_premium_pct",
        "tc_rating_adv",
        "dt_price_premium_pct",
        "td_rating_adv",
        "target_price",
        "target_rating",
        "competitor_price",
        "competitor_rating",
        "decoy_price",
        "decoy_rating",
    ]:
        if c in row:
            md[c] = maybe_float(row[c])

    return md


def make_task_data(
    task_name: str,
    task_id: int,
    start_urls: list[str],
    intent: str,
    intent_template_id: int,
    metadata: dict,
) -> dict:
    return {
        "task": {
            "name": task_name,
            "config": {
                "task_id": int(task_id),
                "start_urls": start_urls,
                "intent_template": intent.replace("$", "\\$"),
                "instantiation_dict": {},
                "intent": intent,
                "choices": [],
                "intent_template_id": int(intent_template_id),
                "metadata": metadata,
            },
        }
    }


def write_yaml(path: str, data: dict):
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


def generate_experiments(
    input_file: str,
    exp_dir: str,
    intent: str,
    seed: int,
    dry_run: bool,
):
    random.seed(seed)
    df = pd.read_csv(input_file)

    required_cols = {
        "target_url",
        "competitor_url",
        "decoy_url",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not os.path.exists(exp_dir) and not dry_run:
        os.makedirs(exp_dir)

    n_baseline = 0
    n_treatment = 0

    exp_idx = 0
    for triad_id, row in df.reset_index(drop=True).iterrows():
        role_to_url = {
            "target": row["target_url"],
            "competitor": row["competitor_url"],
            "decoy": row["decoy_url"],
        }

        # baseline arm: T,C randomized
        baseline_urls = [role_to_url["target"], role_to_url["competitor"]]
        random.shuffle(baseline_urls)
        baseline_md = build_metadata(row, triad_id=triad_id, arm="baseline", start_urls=baseline_urls, role_to_url=role_to_url)

        baseline_name = f"exp{exp_idx}"
        baseline_data = make_task_data(
            task_name=baseline_name,
            task_id=exp_idx,
            start_urls=baseline_urls,
            intent=intent,
            intent_template_id=9001,
            metadata=baseline_md,
        )

        if not dry_run:
            write_yaml(os.path.join(exp_dir, f"{baseline_name}.yaml"), baseline_data)

        exp_idx += 1
        n_baseline += 1

        # treatment arm: T,C,D randomized
        treatment_urls = [role_to_url["target"], role_to_url["competitor"], role_to_url["decoy"]]
        random.shuffle(treatment_urls)
        treatment_md = build_metadata(row, triad_id=triad_id, arm="treatment", start_urls=treatment_urls, role_to_url=role_to_url)

        treatment_name = f"exp{exp_idx}"
        treatment_data = make_task_data(
            task_name=treatment_name,
            task_id=exp_idx,
            start_urls=treatment_urls,
            intent=intent,
            intent_template_id=9002,
            metadata=treatment_md,
        )

        if not dry_run:
            write_yaml(os.path.join(exp_dir, f"{treatment_name}.yaml"), treatment_data)

        exp_idx += 1
        n_treatment += 1

    print("=" * 50)
    print(f"Input triads: {len(df)}")
    print(f"Baseline configs: {n_baseline}")
    print(f"Treatment configs: {n_treatment}")
    print(f"Total configs: {n_baseline + n_treatment}")
    print(f"Output dir: {exp_dir}")
    print(f"Dry run: {dry_run}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate decoy experiment YAMLs from triad CSV.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to triad CSV file.")
    parser.add_argument("--exp-dir", type=str, default=EXP_DIR, help="Directory to store generated experiment YAML files.")
    parser.add_argument("--intent", type=str, default=DEFAULT_INTENT, help="Task intent text.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for URL order randomization.")
    parser.add_argument("--dry-run", action="store_true", help="Compute counts only; do not write files.")

    args = parser.parse_args()

    generate_experiments(
        input_file=args.input_file,
        exp_dir=args.exp_dir,
        intent=args.intent,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
