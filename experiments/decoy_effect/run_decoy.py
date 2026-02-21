from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from dotenv import load_dotenv
try:
    from scipy.stats import chi2_contingency  # type: ignore
except Exception:  # pragma: no cover
    chi2_contingency = None


SYSTEM_PROMPT = (
    "You are a careful consumer agent. Choose the option you would buy. "
    "Reply with ONLY the option ID and nothing else."
)


ATTR_ORDER = [
    "price",
    "rating",
    "reviews",
    "shipping_days",
    "warranty_years",
]


@dataclass
class Option:
    option_id: str
    name: str
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass
class Task:
    task_id: str
    context: str
    target: Option
    competitor: Option
    decoy: Option


def load_tasks(path: Path) -> List[Task]:
    tasks: List[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(
                Task(
                    task_id=raw["task_id"],
                    context=raw["context"],
                    target=_opt_from(raw["target"]),
                    competitor=_opt_from(raw["competitor"]),
                    decoy=_opt_from(raw["decoy"]),
                )
        )
    return tasks


def _opt_from(raw: Dict) -> Option:
    raw = dict(raw)
    option_id = raw.pop("id")
    name = raw.pop("name")
    return Option(option_id=option_id, name=name, attrs=raw)


def _format_attr(key: str, value: object) -> str:
    if key == "price":
        return f"$ {float(value):.2f}"
    if key == "rating":
        return f"{float(value):.1f} stars"
    if key == "reviews":
        return f"{int(value):,} reviews"
    if key == "shipping_days":
        return f"{int(value)} days shipping"
    if key == "warranty_years":
        return f"{int(value)} year warranty"
    return f"{key}: {value}"


def _option_line(opt: Option) -> str:
    parts: List[str] = []
    for key in ATTR_ORDER:
        if key in opt.attrs:
            parts.append(_format_attr(key, opt.attrs[key]))
    for key in sorted(opt.attrs.keys()):
        if key in ATTR_ORDER:
            continue
        parts.append(_format_attr(key, opt.attrs[key]))
    attrs = " | ".join(parts)
    return f"{opt.option_id}. {opt.name} | {attrs}" if attrs else f"{opt.option_id}. {opt.name}"


def build_prompt(task: Task, options: List[Option]) -> str:
    lines = [
        task.context,
        "",
        "Options:",
    ]
    for opt in options:
        lines.append(_option_line(opt))
    lines.append("")
    lines.append("Pick the option you would buy. Reply with ONLY the option ID.")
    return "\n".join(lines)


def parse_choice(text: str, valid_ids: List[str]) -> str:
    text = text.strip()
    if text in valid_ids:
        return text
    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in valid_ids:
        return match.group(1)
    return ""


def run_task(
    client: OpenAI,
    model: str,
    temperature: float,
    task: Task,
    condition: str,
    options: List[Option],
) -> Dict:
    prompt = build_prompt(task, options)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=5,
    )
    raw = response.choices[0].message.content or ""
    choice = parse_choice(raw, [o.option_id for o in options])
    return {
        "task_id": task.task_id,
        "condition": condition,
        "option_order": ",".join([o.option_id for o in options]),
        "choice_id": choice,
        "raw_text": raw.strip(),
    }


def summarize(rows: List[Dict], target_id: str = "A") -> Dict:
    summary: Dict[str, Dict[str, float]] = {}
    for row in rows:
        cond = row["condition"]
        summary.setdefault(cond, {"n": 0, "target_share": 0.0})
        summary[cond]["n"] += 1
        if row["choice_id"] == target_id:
            summary[cond]["target_share"] += 1.0
    for cond in summary:
        n = summary[cond]["n"]
        summary[cond]["target_share"] = summary[cond]["target_share"] / n if n else 0.0
    return summary


def _count_choices(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        cond = row["condition"]
        choice = row["choice_id"] or "NA"
        counts.setdefault(cond, {})
        counts[cond][choice] = counts[cond].get(choice, 0) + 1
    return counts


def _chi2(rows: List[Dict], target_id: str = "A") -> Tuple[float, float] | None:
    if chi2_contingency is None:
        return None
    baseline = [r for r in rows if r["condition"] == "baseline"]
    decoy = [r for r in rows if r["condition"] == "decoy"]
    if not baseline or not decoy:
        return None
    base_target = sum(1 for r in baseline if r["choice_id"] == target_id)
    base_other = len(baseline) - base_target
    decoy_target = sum(1 for r in decoy if r["choice_id"] == target_id)
    decoy_other = len(decoy) - decoy_target
    chi2, p, _, _ = chi2_contingency([[base_target, base_other], [decoy_target, decoy_other]])
    return float(chi2), float(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a decoy effect experiment with an LLM.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).parent / "tasks.jsonl",
        help="Path to tasks JSONL.",
    )
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "decoy_effect",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    rng = random.Random(args.seed)
    client = OpenAI(api_key=api_key)

    tasks = load_tasks(args.tasks)
    rows: List[Dict] = []

    for task in tasks:
        for condition in ("baseline", "decoy"):
            base_options = [task.target, task.competitor]
            if condition == "decoy":
                base_options = [task.target, task.competitor, task.decoy]
            for _ in range(args.repeats):
                options = base_options[:]
                rng.shuffle(options)
                row = run_task(client, args.model, args.temperature, task, condition, options)
                row.update(
                    {
                        "model": args.model,
                        "temperature": args.temperature,
                        "seed": args.seed,
                    }
                )
                rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"decoy_results_{ts}.csv"
    json_path = args.out_dir / f"decoy_summary_{ts}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "condition",
                "option_order",
                "choice_id",
                "raw_text",
                "model",
                "temperature",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    counts = _count_choices(rows)
    stats = _chi2(rows)
    summary_out = {
        "summary": summary,
        "choice_counts": counts,
        "chi2_test": {"chi2": stats[0], "p_value": stats[1]} if stats else None,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
