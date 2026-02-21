"""
This script selects product triads (target, competitor, decoy) for attraction-effect
(Price-Rating decoy) experiments.

Target (T): higher rated and more expensive than Competitor (C).
Decoy  (D): asymmetrically dominated by Target on price/rating and closer to Target.
"""

import random
import argparse
import multiprocessing
import functools
import pandas as pd
import dspy
from tqdm import tqdm
from typing import Any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="tasks/products.csv", help="Path to the input products CSV file.")
    parser.add_argument("--output_file", default="tasks/product_triads-decoy-pricerating.csv", help="Path to output CSV file for triads.")

    # T vs C tradeoff constraints
    parser.add_argument("--min_rating_adv", type=float, default=0.3, help="Minimum rating(T) - rating(C).")
    parser.add_argument("--max_rating_adv", type=float, default=0.8, help="Maximum rating(T) - rating(C).")
    parser.add_argument("--min_price_premium", type=float, default=0.10, help="Minimum (price(T)-price(C))/price(C).")
    parser.add_argument("--max_price_premium", type=float, default=0.30, help="Maximum (price(T)-price(C))/price(C).")

    # D vs T dominated constraints
    parser.add_argument("--min_decoy_rating_gap", type=float, default=0.2, help="Minimum rating(T)-rating(D).")
    parser.add_argument("--max_decoy_rating_gap", type=float, default=1.0, help="Maximum rating(T)-rating(D).")
    parser.add_argument("--min_decoy_price_premium", type=float, default=0.00, help="Minimum (price(D)-price(T))/price(T).")
    parser.add_argument("--max_decoy_price_premium", type=float, default=0.20, help="Maximum (price(D)-price(T))/price(T).")
    parser.add_argument("--strict_decoy_price_gap", type=float, default=0.05, help="Strict decoy price premium threshold (OR condition).")
    parser.add_argument("--strict_decoy_rating_gap", type=float, default=0.4, help="Strict decoy rating gap threshold (OR condition).")

    # Asymmetry / confound controls
    parser.add_argument("--require_asymmetry", action="store_true", help="Require decoy to be closer to target than competitor (normalized distance).")
    parser.add_argument("--max_review_ratio_tc", type=float, default=2.0, help="Maximum review count ratio between T and C (set <=0 to disable).")

    parser.add_argument("--strategy", choices=["sequential", "random"], default="sequential", help="Triad generation strategy.")
    parser.add_argument("--max_triads", type=int, default=100, help="Maximum number of triads to generate.")

    parser.add_argument("--use_llm_filter", action="store_true", help="Enable LLM-based title filtering.")
    parser.add_argument("--llm_model", default="gpt-4.1-nano", help="LLM model to use for filtering.")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel processes to use.")
    args = parser.parse_args()

    # Load and clean data
    df: pd.DataFrame = pd.read_csv(args.input_file)
    df.dropna(subset=["rating", "price"], inplace=True)
    df["rating"] = df["rating"].map(parse_rating)
    df = df[df["rating"] > 0]

    if args.use_llm_filter:
        print("Filtering products with LLM...")
        chunks = [df_part for _, df_part in df.groupby(df.index // max(1, (len(df) // max(1, args.num_workers))))]
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        functools.partial(filter_products_chunk, llm_model=args.llm_model),
                        chunks,
                        chunksize=1,
                    ),
                    total=len(chunks),
                    desc="Filtering products",
                    position=0,
                )
            )
        df = pd.concat(results)
        print(f"Filtered down to {len(df)} products.")

    df["price"] = df["price"].map(parse_price)
    df["reviews"] = df["reviews"].map(parse_reviews)
    df = df[df["price"] > 0]
    df = df[~df["has_options"]]

    grouped = df.groupby("category")
    all_triads: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    print(f"Processing {len(grouped)} categories...")

    for _, data in tqdm(grouped):
        if len(data) < 3:
            continue

        if args.strategy == "sequential":
            triads = generate_sequential_triads(data, args)
        else:
            triads = generate_random_triads(data, args)

        all_triads.extend(triads)

    all_triads = deduplicate_triads(all_triads)
    print(f"Generated {len(all_triads)} triads before subsampling.")

    triad_records = []
    for t, c, d in all_triads:
        triad_records.append(
            {
                "category": t["category"],
                "target_name": t["product_name"],
                "target_url": t["product_url"],
                "target_price": t["price"],
                "target_rating": t["rating"],
                "target_reviews": t["reviews"],
                "competitor_name": c["product_name"],
                "competitor_url": c["product_url"],
                "competitor_price": c["price"],
                "competitor_rating": c["rating"],
                "competitor_reviews": c["reviews"],
                "decoy_name": d["product_name"],
                "decoy_url": d["product_url"],
                "decoy_price": d["price"],
                "decoy_rating": d["rating"],
                "decoy_reviews": d["reviews"],
                "tc_price_premium_pct": safe_pct(t["price"] - c["price"], c["price"]),
                "tc_rating_adv": t["rating"] - c["rating"],
                "dt_price_premium_pct": safe_pct(d["price"] - t["price"], t["price"]),
                "td_rating_adv": t["rating"] - d["rating"],
            }
        )

    output_df = pd.DataFrame(triad_records)

    if len(output_df) > args.max_triads:
        output_df = output_df.sample(n=args.max_triads, random_state=42).reset_index(drop=True)
        print(f"Subsampled to {args.max_triads} triads.")

    output_df.to_csv(args.output_file, index=False)
    print(f"Saved {len(output_df)} triads to {args.output_file}")


def parse_rating(x: Any) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0


def parse_price(x: Any) -> float:
    if pd.isna(x):
        return 0.0
    s = str(x).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0


def parse_reviews(x: Any) -> int:
    if pd.isna(x):
        return 0
    s = str(x).replace(",", "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        return int(digits) if digits else 0
    except Exception:
        return 0


def safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def category_scales(data: pd.DataFrame) -> dict[str, float]:
    price_min, price_max = float(data["price"].min()), float(data["price"].max())
    rating_min, rating_max = float(data["rating"].min()), float(data["rating"].max())
    return {
        "price_span": max(price_max - price_min, 1e-9),
        "rating_span": max(rating_max - rating_min, 1e-9),
    }


def normalized_distance(a: dict[str, Any], b: dict[str, Any], scales: dict[str, float]) -> float:
    dp = (a["price"] - b["price"]) / scales["price_span"]
    dr = (a["rating"] - b["rating"]) / scales["rating_span"]
    return (dp * dp + dr * dr) ** 0.5


def valid_tradeoff_tc(target: dict[str, Any], competitor: dict[str, Any], args: argparse.Namespace) -> bool:
    if not (target["price"] > competitor["price"]):
        return False

    rating_adv = target["rating"] - competitor["rating"]
    if rating_adv < args.min_rating_adv or rating_adv > args.max_rating_adv:
        return False

    price_premium = safe_pct(target["price"] - competitor["price"], competitor["price"])
    if price_premium < args.min_price_premium or price_premium > args.max_price_premium:
        return False

    if args.max_review_ratio_tc > 0:
        t_rev = max(1, int(target.get("reviews", 0)))
        c_rev = max(1, int(competitor.get("reviews", 0)))
        ratio = max(t_rev / c_rev, c_rev / t_rev)
        if ratio > args.max_review_ratio_tc:
            return False

    return True


def valid_decoy_d_for_tc(
    target: dict[str, Any],
    competitor: dict[str, Any],
    decoy: dict[str, Any],
    scales: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    if decoy["product_url"] in {target["product_url"], competitor["product_url"]}:
        return False

    # Dominated by target on both dimensions
    decoy_price_premium = safe_pct(decoy["price"] - target["price"], target["price"])
    decoy_rating_gap = target["rating"] - decoy["rating"]

    if decoy_price_premium < args.min_decoy_price_premium or decoy_price_premium > args.max_decoy_price_premium:
        return False
    if decoy_rating_gap < args.min_decoy_rating_gap or decoy_rating_gap > args.max_decoy_rating_gap:
        return False

    # Require at least one stronger margin
    strong_price_gap = safe_pct(decoy["price"] - target["price"], target["price"]) >= args.strict_decoy_price_gap
    strong_rating_gap = (target["rating"] - decoy["rating"]) >= args.strict_decoy_rating_gap
    if not (strong_price_gap or strong_rating_gap):
        return False

    # Optional asymmetry requirement: decoy closer to target than competitor
    if args.require_asymmetry:
        if normalized_distance(decoy, target, scales) >= normalized_distance(decoy, competitor, scales):
            return False

    # Avoid trivial global-dominance where competitor strongly dominates decoy too
    if competitor["price"] <= decoy["price"] and competitor["rating"] >= decoy["rating"]:
        both_strict = (competitor["price"] < decoy["price"]) and (competitor["rating"] > decoy["rating"])
        if both_strict:
            return False

    return True


def generate_sequential_triads(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    triads = []
    products = data.sort_values(by=["price", "rating"], ascending=[True, False]).to_dict("records")
    scales = category_scales(data)

    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            c = products[i]
            t = products[j]
            if not valid_tradeoff_tc(t, c, args):
                continue

            # pick best decoy by proximity to target
            candidates = [
                d
                for d in products
                if valid_decoy_d_for_tc(t, c, d, scales, args)
            ]
            if not candidates:
                continue

            candidates.sort(key=lambda d: normalized_distance(d, t, scales))
            triads.append((t, c, candidates[0]))

    return triads


def generate_random_triads(
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    triads = []
    products = data.to_dict("records")
    scales = category_scales(data)

    max_iters = min(2000, max(200, len(products) * 25))
    for _ in range(max_iters):
        t, c, d = random.sample(products, 3)
        if valid_tradeoff_tc(t, c, args) and valid_decoy_d_for_tc(t, c, d, scales, args):
            triads.append((t, c, d))

    return triads


def deduplicate_triads(
    triads: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    seen = set()
    out = []
    for t, c, d in triads:
        key = (t["product_url"], c["product_url"], d["product_url"])
        if key in seen:
            continue
        seen.add(key)
        out.append((t, c, d))
    return out


def filter_products_chunk(df_chunk: pd.DataFrame, llm_model: str) -> pd.DataFrame:
    lm = dspy.LM(model=llm_model)
    dspy.settings.configure(lm=lm)
    title_checker = TitleChecker()

    try:
        process_num = int(multiprocessing.current_process().name.split("-")[-1])
    except Exception:
        process_num = 1

    tqdm.pandas(desc=f"Filtering titles in chunk-{process_num}", position=process_num, leave=False)
    contains_suggestion, is_multipack, mentions_quantity = zip(
        *df_chunk["product_name"].progress_apply(lambda title: filter_title_with_llm(title, title_checker))
    )
    df_chunk["contains_suggestion"] = contains_suggestion
    df_chunk["is_multipack"] = is_multipack
    df_chunk["mentions_quantity"] = mentions_quantity
    return df_chunk[~(df_chunk["contains_suggestion"] | df_chunk["is_multipack"] | df_chunk["mentions_quantity"])]


class CheckTitle(dspy.Signature):
    """Detect if a product title contains suggestion language, multipack signals, or quantity mentions."""

    title: str = dspy.InputField(desc="The product title to inspect.")
    contains_suggestion: bool = dspy.OutputField(desc="A boolean indicating if suggestion language is present.")
    is_multipack: bool = dspy.OutputField(desc="A boolean indicating if the product is a multi-pack or bundle.")
    mentions_quantity: bool = dspy.OutputField(desc="A boolean indicating if the title mentions a quantity.")


class TitleChecker(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chain_of_thought = dspy.ChainOfThought(CheckTitle)

    def forward(self, title: str) -> dspy.Prediction:
        return self.chain_of_thought(title=title)


def filter_title_with_llm(title: str, title_checker: TitleChecker) -> tuple[bool, bool, bool]:
    try:
        prediction = title_checker(title)
        return prediction.contains_suggestion, prediction.is_multipack, prediction.mentions_quantity
    except Exception as error:
        print(f"Error processing title '{title}': {error}")
        return False, False, False


if __name__ == "__main__":
    main()
