"""
Preprocess aggregated ABxLab results for decoy experiments.

Input: raw CSV from scripts/collect_results.py
Output: analysis-ready CSV with role-aware fields for baseline (TC) and treatment (TCD).
"""

import argparse
import ast
import logging
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from page_utils import get_price_for_product, get_rating_for_product


tqdm.pandas()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def safe_eval_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def fetch_single_product_data(url: str) -> tuple[str, float, float]:
    try:
        rating = float(get_rating_for_product(url).replace("%", ""))
        price = float(get_price_for_product(url).replace("$", "").replace(",", ""))
        return url, rating, price
    except Exception as error:
        logger.warning("Failed to fetch data for %s: %s", url, error)
        return url, 0.0, 0.0


def fetch_product_data_parallel(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    logger.info("Fetching product ratings and prices with %d workers", max_workers)

    all_urls = set()
    for urls in df["cfg.task.config.start_urls"]:
        all_urls.update(urls)
    all_urls = list(all_urls)

    url_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_product_data, url): url for url in all_urls}

        with tqdm(total=len(futures), desc="Fetching product data") as pbar:
            for future in as_completed(futures):
                url, rating, price = future.result()
                url_data[url] = {"rating": rating, "price": price}
                pbar.update(1)

    df["ratings"] = df["cfg.task.config.start_urls"].map(lambda urls: [url_data[url]["rating"] for url in urls])
    df["prices"] = df["cfg.task.config.start_urls"].map(lambda urls: [url_data[url]["price"] for url in urls])
    return df


def lookup_attr(row: pd.Series, role_url: str, field_name: str):
    if not isinstance(role_url, str):
        return None
    urls = row["cfg.task.config.start_urls"]
    vals = row[field_name]
    if not isinstance(urls, list) or not isinstance(vals, list):
        return None
    try:
        idx = urls.index(role_url)
        return vals[idx]
    except Exception:
        return None


def chosen_role(row: pd.Series) -> str:
    cu = row["chosen_url"]
    if cu == row.get("target_url"):
        return "target"
    if cu == row.get("competitor_url"):
        return "competitor"
    if cu == row.get("decoy_url"):
        return "decoy"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs="+", required=True, help="Input CSV files from collect_results.py")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV path")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers for product attribute fetch")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.input_files], ignore_index=True)

    # Parse URL arrays and keep valid rows
    df["cfg.task.config.start_urls"] = df["cfg.task.config.start_urls"].map(safe_eval_list)
    df = df[df["cfg.task.config.start_urls"].map(lambda x: isinstance(x, list) and len(x) >= 2)].copy()
    df = df[df["final_step.url"].notnull()].copy()

    # Keep decoy study rows when marker exists
    if "cfg.task.config.metadata.study_type" in df.columns:
        d2 = df[df["cfg.task.config.metadata.study_type"] == "decoy_pricerating"].copy()
        if len(d2) > 0:
            df = d2

    logger.info("Rows after decoy filter: %d", len(df))

    df = fetch_product_data_parallel(df, max_workers=args.num_workers)

    # Metadata columns produced by generate_experiments_decoy.py
    df["arm"] = df.get("cfg.task.config.metadata.arm")
    df["triad_id"] = pd.to_numeric(df.get("cfg.task.config.metadata.triad_id"), errors="coerce").astype("Int64")
    df["category"] = df.get("cfg.task.config.metadata.category")
    df["target_url"] = df.get("cfg.task.config.metadata.target_url")
    df["competitor_url"] = df.get("cfg.task.config.metadata.competitor_url")
    df["decoy_url"] = df.get("cfg.task.config.metadata.decoy_url")
    df["target_pos"] = pd.to_numeric(df.get("cfg.task.config.metadata.target_pos"), errors="coerce").astype("Int64")
    df["competitor_pos"] = pd.to_numeric(df.get("cfg.task.config.metadata.competitor_pos"), errors="coerce").astype("Int64")
    df["decoy_pos"] = pd.to_numeric(df.get("cfg.task.config.metadata.decoy_pos"), errors="coerce").astype("Int64")
    df["model_family"] = df.get("study.chat_model_args.model_name")

    # Resolve choice
    df["set_size"] = df["cfg.task.config.start_urls"].map(len)
    df["chosen_url"] = df["final_step.url"]
    df["chose_idx"] = df.apply(
        lambda row: row["cfg.task.config.start_urls"].index(row["chosen_url"])
        if row["chosen_url"] in row["cfg.task.config.start_urls"] else None,
        axis=1,
    ).astype("Int64")

    df = df[df["chose_idx"].notnull()].copy()

    # Keep add-to-cart completions when element ID exists
    if "final_step.elem_info.attrs.id" in df.columns:
        df = df[df["final_step.elem_info.attrs.id"].notnull()].copy()
        df = df[df["final_step.elem_info.attrs.id"].map(lambda x: "addtocart" in str(x).lower())].copy()

    df["chosen_role"] = df.apply(chosen_role, axis=1)
    df["chose_target"] = df["chosen_role"] == "target"
    df["chose_competitor"] = df["chosen_role"] == "competitor"
    df["chose_decoy"] = df["chosen_role"] == "decoy"

    # Role attribute lookup from shown URLs
    df["target_price"] = df.apply(lambda row: lookup_attr(row, row["target_url"], "prices"), axis=1)
    df["target_rating"] = df.apply(lambda row: lookup_attr(row, row["target_url"], "ratings"), axis=1)
    df["competitor_price"] = df.apply(lambda row: lookup_attr(row, row["competitor_url"], "prices"), axis=1)
    df["competitor_rating"] = df.apply(lambda row: lookup_attr(row, row["competitor_url"], "ratings"), axis=1)
    df["decoy_price"] = df.apply(lambda row: lookup_attr(row, row["decoy_url"], "prices"), axis=1)
    df["decoy_rating"] = df.apply(lambda row: lookup_attr(row, row["decoy_url"], "ratings"), axis=1)

    # Derived metrics
    df["decoy_present"] = df["set_size"] >= 3
    df["tc_price_premium_pct"] = (df["target_price"] - df["competitor_price"]) / df["competitor_price"]
    df["tc_rating_adv"] = df["target_rating"] - df["competitor_rating"]
    df["dt_price_premium_pct"] = (df["decoy_price"] - df["target_price"]) / df["target_price"]
    df["td_rating_adv"] = df["target_rating"] - df["decoy_rating"]

    # Conditional target choice among {T,C} only
    df["choice_in_tc"] = df["chosen_role"].isin(["target", "competitor"])
    df["chose_target_cond_tc"] = df.apply(
        lambda row: (row["chosen_role"] == "target") if row["choice_in_tc"] else None,
        axis=1,
    )

    df.to_csv(args.output_file, index=False)
    logger.info("Saved decoy-preprocessed data to %s", args.output_file)


if __name__ == "__main__":
    main()
