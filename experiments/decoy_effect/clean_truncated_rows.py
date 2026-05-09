"""Filter truncated reasoning rows out of a literature_decoy results CSV.

Truncated rows are deliberative/knowledge trials whose raw_text never reached
"Therefore, I choose X" — typically because the original run used max_tokens=400.
The recorded choice_key in those rows is the regex fallback (last [A-C] mention),
not the model's stated final answer.

Usage:
    python3 experiments/decoy_effect/clean_truncated_rows.py \\
        --in  results/literature_decoy/full_claude_run/<file>.csv \\
        --out results/literature_decoy/claude_topup/cleaned.csv

Then resume the runner against the cleaned file to top up the missing samples:

    python3 experiments/decoy_effect/run_literature_decoy.py \\
        --agent vertex-claude-opus-4-6 --max-tokens 2500 --samples 10 \\
        --resume-from results/literature_decoy/claude_topup/cleaned.csv
    # repeat with --for-me for the for-me framing
"""
import argparse
from pathlib import Path

import pandas as pd

# Modes that require an explicit "Therefore, I choose X" to be considered complete.
# Fast mode emits a bare letter and is always considered complete here.
REASONING_MODES = {
    "deliberative", "knowledge",
    "deliberative_for_me", "knowledge_for_me",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="Path to the source results CSV (untouched).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to the cleaned CSV (created/overwritten).")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    n0 = len(df)

    txt = df["raw_text"].fillna("").str.lower()
    has_therefore = txt.str.contains("therefore")
    is_reasoning = df["prompting_mode"].isin(REASONING_MODES)

    # Drop reasoning rows without an explicit "therefore"
    truncated = is_reasoning & ~has_therefore
    cleaned = df[~truncated].copy()

    # Per-mode summary
    summary = (
        df.groupby("prompting_mode").size().rename("original").to_frame()
        .join(cleaned.groupby("prompting_mode").size().rename("kept"))
        .fillna(0).astype(int)
    )
    summary["dropped"] = summary["original"] - summary["kept"]
    print(summary.to_string())
    print(f"\nKept {len(cleaned):,} / {n0:,} rows ({100*len(cleaned)/n0:.1f}%).")
    print(f"Dropped {int(truncated.sum()):,} truncated reasoning trials.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    print(f"\nWrote -> {args.out}")
    print("\nNext: re-run with --resume-from pointing at this file.")


if __name__ == "__main__":
    main()
