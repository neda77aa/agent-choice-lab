"""Find stimuli where knowledge-mode prompting flips Claude (debias) vs GPT (reinforce).

Logic:
  1. Compute per-stimulus Δ = P(Target|3) - P(Target|2) for each (model, mode, framing).
  2. Keep stimuli where deliberative Δ > 0 for BOTH claude-opus-4-6 and gpt-5-2.
  3. Score divergence:  switch = (Δ_knowledge_gpt - Δ_deliberative_gpt)
                              - (Δ_knowledge_claude - Δ_deliberative_claude)
     A large positive switch means GPT amplified while Claude attenuated/flipped.
  4. For the top candidates, dump representative knowledge-mode reasoning traces:
       - Claude trials in 3-opt where it chose Rival (avoided decoy)
       - GPT    trials in 3-opt where it chose Target (reinforced)
"""
import argparse
import textwrap
from pathlib import Path
import pandas as pd

CLAUDE_CSV = Path("/Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab/results/literature_decoy/full_claude_run/literature_decoy_vertex-claude-opus-4-6_20260303_120816.csv")
GPT_CSV = Path("/Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab/results/literature_decoy/full_gpt52/literature_decoy_gpt-5-2_20260303_122202.csv")


def short_model(name: str) -> str:
    n = str(name).lower()
    if "claude" in n:
        return "claude"
    if "gpt" in n:
        return "gpt"
    return n


def load() -> pd.DataFrame:
    parts = []
    for p in (CLAUDE_CSV, GPT_CSV):
        df = pd.read_csv(p)
        df["short"] = df["agent"].apply(short_model)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["is_target"] = (df["choice_key"] == "Target").astype(float)
    # split mode label into base mode + framing
    df["framing"] = df["prompting_mode"].apply(
        lambda m: "for_me" if str(m).endswith("_for_me") else "regular"
    )
    df["base_mode"] = df["prompting_mode"].apply(
        lambda m: str(m).replace("_for_me", "")
    )
    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["short", "framing", "base_mode", "stimulus_id", "condition"])
        ["is_target"].mean().reset_index()
    )
    pivot = g.pivot_table(
        index=["short", "framing", "base_mode", "stimulus_id"],
        columns="condition",
        values="is_target",
    ).reset_index()
    pivot["delta"] = pivot.get("3_opt") - pivot.get("2_opt")
    return pivot


def find_divergent(deltas: pd.DataFrame, framing: str = "regular") -> pd.DataFrame:
    sub = deltas[deltas["framing"] == framing].copy()
    wide = sub.pivot_table(
        index="stimulus_id", columns=["short", "base_mode"], values="delta"
    )
    # need all four cells present
    needed = [
        ("claude", "deliberative"), ("claude", "knowledge"),
        ("gpt", "deliberative"),    ("gpt", "knowledge"),
    ]
    wide = wide.dropna(subset=needed)
    # Flatten MultiIndex columns to simple "model_mode" strings.
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]

    pos_delib = (wide["claude_deliberative"] > 0) & (wide["gpt_deliberative"] > 0)
    wide = wide[pos_delib].copy()

    wide["claude_shift"] = wide["claude_knowledge"] - wide["claude_deliberative"]
    wide["gpt_shift"]    = wide["gpt_knowledge"]    - wide["gpt_deliberative"]
    wide["switch_score"] = wide["gpt_shift"] - wide["claude_shift"]

    # claude must have moved DOWN (debias) while gpt moved UP (reinforce)
    wide = wide[(wide["claude_shift"] < 0) & (wide["gpt_shift"] > 0)]
    wide = wide.sort_values("switch_score", ascending=False)
    return wide


def dump_examples(df: pd.DataFrame, stimulus_id: str, framing: str,
                  n_each: int = 3, max_chars: int = 1400) -> str:
    out = [f"\n=== Stimulus: {stimulus_id}  (framing={framing}) ==="]

    # Claude knowledge trials in 3-opt that chose Rival (looks like debiasing)
    claude_rival = df[
        (df["short"] == "claude")
        & (df["framing"] == framing)
        & (df["base_mode"] == "knowledge")
        & (df["stimulus_id"] == stimulus_id)
        & (df["condition"] == "3_opt")
        & (df["choice_key"] == "Rival")
    ].head(n_each)

    # GPT knowledge trials in 3-opt that chose Target (looks like reinforcing)
    gpt_target = df[
        (df["short"] == "gpt")
        & (df["framing"] == framing)
        & (df["base_mode"] == "knowledge")
        & (df["stimulus_id"] == stimulus_id)
        & (df["condition"] == "3_opt")
        & (df["choice_key"] == "Target")
    ].head(n_each)

    out.append(f"\n-- Claude (knowledge, 3-opt, chose Rival) --  n={len(claude_rival)}")
    for _, r in claude_rival.iterrows():
        txt = str(r["raw_text"]).replace("\\n", " ")[:max_chars]
        out.append(f"\n[T={r['temperature']} order={r['option_order']}]\n{textwrap.fill(txt, 100)}")

    out.append(f"\n-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n={len(gpt_target)}")
    for _, r in gpt_target.iterrows():
        txt = str(r["raw_text"]).replace("\\n", " ")[:max_chars]
        out.append(f"\n[T={r['temperature']} order={r['option_order']}]\n{textwrap.fill(txt, 100)}")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framing", default="regular", choices=["regular", "for_me"])
    ap.add_argument("--top", type=int, default=5, help="Top stimuli to dump examples for")
    ap.add_argument("--n-each", type=int, default=2, help="Examples per side per stimulus")
    ap.add_argument("--out", type=Path,
                    default=Path("/Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab/experiments/decoy_effect/knowledge_divergence_report.md"))
    args = ap.parse_args()

    df = load()
    deltas = compute_deltas(df)
    diverg = find_divergent(deltas, framing=args.framing)

    print(f"\nFraming = {args.framing}")
    print(f"Stimuli where Claude debiases AND GPT reinforces under knowledge mode: {len(diverg)}")
    cols_show = [
        "claude_deliberative", "claude_knowledge", "claude_shift",
        "gpt_deliberative",    "gpt_knowledge",    "gpt_shift",
        "switch_score",
    ]
    print(diverg[cols_show].round(3).to_string())

    # Write report
    lines = [f"# Knowledge-mode divergence (framing={args.framing})\n",
             "Δ = P(Target|3-opt) − P(Target|2-opt) per stimulus, mode, model.",
             "Filtered to stimuli where deliberative Δ > 0 for both models, ",
             "Claude shift (knowledge − deliberative) < 0, and GPT shift > 0.\n",
             "## Ranking table\n",
             diverg[cols_show].round(3).to_markdown()]
    for sid in diverg.head(args.top).index:
        lines.append(dump_examples(df, sid, args.framing, n_each=args.n_each))
    args.out.write_text("\n".join(lines))
    print(f"\nWrote report -> {args.out}")


if __name__ == "__main__":
    main()
