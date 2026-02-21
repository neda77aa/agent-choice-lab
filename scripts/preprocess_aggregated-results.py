# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nikhil.u.singh@dartmouth.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This script processes aggregated results to make them ready for the analysis in R.
"""

import logging
import argparse
import ast
import pandas as pd
from urllib.parse import urlparse
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from page_utils import get_price_for_product, get_rating_for_product


# Configure tqdm for pandas
tqdm.pandas()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", type=str, nargs='+', required=True, help="Input CSV file paths")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path for processed data")
    parser.add_argument("--num_workers", type=int, default=8, help="(Maximum) number of parallel workers for fetching data")
    parser.add_argument("--product_list", type=str, help="Optional list of product URLs to include metadata.")
    parser.add_argument(
        "--decoy_mode",
        action="store_true",
        help="Enable preprocessing for decoy experiments (supports 2- and 3-option sets)."
    )
    args = parser.parse_args()

    df_list = [pd.read_csv(file) for file in args.input_files]
    df = pd.concat(df_list, ignore_index=True)

    if args.decoy_mode:
        preprocess_decoy(df=df, output_file=args.output_file, num_workers=args.num_workers)
        return

    # Filter to pairs
    df["cfg.task.config.start_urls"] = df["cfg.task.config.start_urls"].map(eval)
    df = df[df["cfg.task.config.start_urls"].map(len) == 2].copy()
    logger.info("Found %d pairs with exactly 2 start URLs", len(df))

    if args.product_list:
        df_products = pd.read_csv(args.product_list)
        product_map = df_products.set_index("product_url")["category"].to_dict()
        product_map = {urlparse(url).path: category for url, category in product_map.items()}
        df["category"] = df["cfg.task.config.start_urls"].map(lambda urls: product_map[urlparse(urls[0]).path])

    df["choices"] = df["cfg.task.config.choices"].map(eval)

    # Identify nudged choice
    nudge_types_to_ignore = ["Matching Review Count", "Matching Price"]

    df["nudged_choice_url"] = df["choices"].map(
        lambda x: x[0]["url"] if ((len(x) > 0) and (x[0]["nudge"] not in nudge_types_to_ignore)) else None
    )
    df["nudge_type"] = df["choices"].map(
        lambda x: x[0]["nudge"] if ((len(x) > 0) and (x[0]["nudge"] not in nudge_types_to_ignore)) else None
    )

    df["nudge_text"] = df["choices"].map(
        lambda x: x[0]["functions"][0]["args"]["value"] if ((len(x) > 0) and (x[0]["nudge"] not in nudge_types_to_ignore)) else None
    )

    df["chose_nudged_product"] = df.apply(
        lambda row: row["final_step.url"] == row["nudged_choice_url"],
        axis=1
    )

    # Fetch product data
    df = fetch_product_data_parallel(df, max_workers=args.num_workers)

    # Prepare data for regression
    # df_reg = df[df["nudged_choice_url"].notnull()].copy()
    df_reg = df.copy()
    df_reg["nudge_trial"] = df_reg["nudged_choice_url"].notnull()

    # Get product indices
    df_reg["nudged_idx"] = df_reg.apply(
        lambda row: row["cfg.task.config.start_urls"].index(row["nudged_choice_url"]) if row["nudge_trial"] else None,
        axis=1
    ).astype("Int64")
    df_reg["chose_idx"] = df_reg.apply(
        lambda row: row["cfg.task.config.start_urls"].index(row["final_step.url"]) if row["final_step.url"] in row["cfg.task.config.start_urls"] else None,
        axis=1
    ).astype("Int64")
    df_reg["other_idx"] = 1 - df_reg["nudged_idx"]

    # Remove any invalid choices
    df_reg = df_reg[df_reg["chose_idx"].notnull()]
    df_reg = df_reg[df_reg["final_step.elem_info.attrs.id"].notnull()]
    df_reg = df_reg[df_reg["final_step.elem_info.attrs.id"].map(lambda x: "addtocart" in x)]

    # Extract prices and ratings based on nudged/other product
    df_reg["price_nudged"] = df_reg.apply(lambda row: row["prices"][row["nudged_idx"]] if row["nudge_trial"] else None, axis=1)
    df_reg["price_other"] = df_reg.apply(lambda row: row["prices"][row["other_idx"]] if row["nudge_trial"] else None, axis=1)
    df_reg["rating_nudged"] = df_reg.apply(lambda row: row["ratings"][row["nudged_idx"]] if row["nudge_trial"] else None, axis=1)
    df_reg["rating_other"] = df_reg.apply(lambda row: row["ratings"][row["other_idx"]] if row["nudge_trial"] else None, axis=1)

    df_reg["avg_price"] = df_reg["prices"].apply(lambda x: sum(x) / len(x))
    df_reg["price_diff_lr"] = df_reg["prices"].apply(lambda x: abs(x[1] - x[0]))
    df_reg["price_diff_lr_pct"] = df_reg["price_diff_lr"] / df_reg["avg_price"]

    df_reg["chose_cheaper"] = df_reg.apply(
        lambda row: row["prices"][row["chose_idx"]] < row["prices"][1 - row["chose_idx"]],
        axis=1
    )
    df_reg["cheaper_idx"] = df_reg.apply(
        lambda row: min(enumerate(row["prices"]), key=lambda x: x[1])[0],
        axis=1
    )
    df_reg["better_rated_idx"] = df_reg.apply(
        lambda row: max(enumerate(row["ratings"]), key=lambda x: x[1])[0],
        axis=1
    )
    df_reg["chose_better_rated"] = df_reg.apply(
        lambda row: row["ratings"][row["chose_idx"]] > row["ratings"][1 - row["chose_idx"]],
        axis=1
    )

    # Calculate differences
    df_reg["price_diff"] = df_reg["price_nudged"] - df_reg["price_other"]
    df_reg["rating_diff"] = df_reg["rating_nudged"] - df_reg["rating_other"]

    # Get nudge type
    df_reg["nudge_type"] = df_reg["choices"].map(lambda x: x[0]["nudge"] if len(x) > 0 else None)

    # Create model_family column
    df_reg["model_family"] = df_reg["study.chat_model_args.model_name"]

    # Save the processed data for further inspection
    df_reg.to_csv(args.output_file, index=False)
    logger.info("Saved preprocessed data to %s", args.output_file)


def preprocess_decoy(df: pd.DataFrame, output_file: str, num_workers: int = 8):
    df["cfg.task.config.start_urls"] = df["cfg.task.config.start_urls"].map(safe_eval_list)
    df = df[df["cfg.task.config.start_urls"].map(lambda x: isinstance(x, list) and len(x) >= 2)].copy()
    df = df[df["final_step.url"].notnull()].copy()

    if "cfg.task.config.metadata.study_type" in df.columns:
        d2 = df[df["cfg.task.config.metadata.study_type"] == "decoy_pricerating"].copy()
        if len(d2) > 0:
            df = d2

    logger.info("Found %d candidate decoy experiment rows", len(df))

    df = fetch_product_data_parallel(df, max_workers=num_workers)

    df["set_size"] = df["cfg.task.config.start_urls"].map(len)
    df["arm"] = df.get("cfg.task.config.metadata.arm", pd.Series([None] * len(df)))
    df["triad_id"] = pd.to_numeric(df.get("cfg.task.config.metadata.triad_id"), errors="coerce").astype("Int64")
    df["category"] = df.get("cfg.task.config.metadata.category")
    df["target_url"] = df.get("cfg.task.config.metadata.target_url")
    df["competitor_url"] = df.get("cfg.task.config.metadata.competitor_url")
    df["decoy_url"] = df.get("cfg.task.config.metadata.decoy_url")
    df["target_pos"] = pd.to_numeric(df.get("cfg.task.config.metadata.target_pos"), errors="coerce").astype("Int64")
    df["competitor_pos"] = pd.to_numeric(df.get("cfg.task.config.metadata.competitor_pos"), errors="coerce").astype("Int64")
    df["decoy_pos"] = pd.to_numeric(df.get("cfg.task.config.metadata.decoy_pos"), errors="coerce").astype("Int64")
    df["model_family"] = df.get("study.chat_model_args.model_name")

    df["chose_idx"] = df.apply(
        lambda row: row["cfg.task.config.start_urls"].index(row["final_step.url"])
        if row["final_step.url"] in row["cfg.task.config.start_urls"] else None,
        axis=1,
    ).astype("Int64")

    df = df[df["chose_idx"].notnull()].copy()

    if "final_step.elem_info.attrs.id" in df.columns:
        df = df[df["final_step.elem_info.attrs.id"].notnull()].copy()
        df = df[df["final_step.elem_info.attrs.id"].map(lambda x: "addtocart" in str(x).lower())].copy()

    df["chosen_url"] = df["final_step.url"]
    df["chose_target"] = df["chosen_url"] == df["target_url"]
    df["chose_competitor"] = df["chosen_url"] == df["competitor_url"]
    df["chose_decoy"] = df["chosen_url"] == df["decoy_url"]
    df["chosen_role"] = df.apply(
        lambda row: (
            "target" if row["chose_target"] else
            "competitor" if row["chose_competitor"] else
            "decoy" if row["chose_decoy"] else
            "unknown"
        ),
        axis=1,
    )

    df["target_price"] = df.apply(lambda row: lookup_attr(row, row["target_url"], "prices"), axis=1)
    df["target_rating"] = df.apply(lambda row: lookup_attr(row, row["target_url"], "ratings"), axis=1)
    df["competitor_price"] = df.apply(lambda row: lookup_attr(row, row["competitor_url"], "prices"), axis=1)
    df["competitor_rating"] = df.apply(lambda row: lookup_attr(row, row["competitor_url"], "ratings"), axis=1)
    df["decoy_price"] = df.apply(lambda row: lookup_attr(row, row["decoy_url"], "prices"), axis=1)
    df["decoy_rating"] = df.apply(lambda row: lookup_attr(row, row["decoy_url"], "ratings"), axis=1)

    df["decoy_present"] = df["set_size"] >= 3
    df["tc_price_premium_pct"] = (df["target_price"] - df["competitor_price"]) / df["competitor_price"]
    df["tc_rating_adv"] = df["target_rating"] - df["competitor_rating"]
    df["dt_price_premium_pct"] = (df["decoy_price"] - df["target_price"]) / df["target_price"]
    df["td_rating_adv"] = df["target_rating"] - df["decoy_rating"]

    df["choice_in_tc"] = df["chosen_role"].isin(["target", "competitor"])
    df["chose_target_cond_tc"] = df.apply(
        lambda row: (row["chosen_role"] == "target") if row["choice_in_tc"] else None,
        axis=1,
    )

    df.to_csv(output_file, index=False)
    logger.info("Saved decoy-preprocessed data to %s", output_file)


def fetch_single_product_data(url: str) -> tuple[str, float, float]:
    try:
        rating = float(get_rating_for_product(url).replace("%", ""))
        price = float(get_price_for_product(url).replace("$", "").replace(",", ""))
        return url, rating, price
    except Exception as error:
        logger.warning("Failed to fetch data for %s: %s" % (url, error))
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

    df["ratings"] = df["cfg.task.config.start_urls"].map(
        lambda urls: [url_data[url]["rating"] for url in urls]
    )
    df["prices"] = df["cfg.task.config.start_urls"].map(
        lambda urls: [url_data[url]["price"] for url in urls]
    )

    return df


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


if __name__ == "__main__":
    main()
