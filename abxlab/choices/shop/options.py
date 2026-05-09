# Copyright (c) 2026
# Additive interventions for size-decoy experiments. Authored separately from
# product.py so existing experiment configs and runs are not affected.

"""
Interventions for within-product size-decoy experiments.

These functions modify Magento product pages that have a `product-options-wrapper`
block (multi-option products) to support a within-product asymmetric-dominance
(decoy) manipulation:

  - Rewrite the product title (so the agent sees a clean, size-free name).
  - Keep only specified options (so the baseline arm shows only Small + Large).
  - Rewrite each option's visible label.
  - Inject a visible price label next to each option.
  - Pre-select an option for products with multiple required option groups
    (so the agent only has to pick a Size, not also a Flavor/Style).

All functions follow the existing convention: take HTML bytes/str, return a
(modified_html, metadata) tuple.
"""

import random
from typing import Mapping, Optional
from bs4 import BeautifulSoup


def set_title(original_html: bytes, value: str) -> tuple[str, dict]:
    """Rewrite the product H1, the inner page-title span, and the <title> tag.

    Replaces every visible textual surface that shows the product name, so the
    agent sees a single consistent title across the page and the tab title.
    """
    soup = BeautifulSoup(original_html, "lxml")

    # Page <title> (browser tab name)
    if soup.title:
        soup.title.string = value

    # H1 inside <div class="page-title-wrapper product"><h1 class="page-title">
    h1 = soup.select_one("div.page-title-wrapper.product h1.page-title")
    if h1:
        # Rewrite all the inner spans to the new value (Magento can have
        # multiple decorative spans inside the H1).
        for span in h1.find_all(["span"]):
            span.string = value
        if not h1.find_all(["span"]):
            h1.string = value

    # Schema.org itemprop="name" anywhere on the page
    for el in soup.find_all(attrs={"itemprop": "name"}):
        # Only override text nodes, not nested structures
        if not el.find(True):  # leaf element
            el.string = value

    # Breadcrumb's last item (the product name)
    breadcrumb_product = soup.select_one("div.breadcrumbs li.item.product strong")
    if breadcrumb_product:
        breadcrumb_product.string = value

    return str(soup), {"intervention": "set_title", "value": value}


def keep_options(
    original_html: bytes,
    value_ids: list[str | int],
) -> tuple[str, dict]:
    """Drop radio-button options from `product-options-wrapper` whose value
    attribute is not in `value_ids`.

    Used for the baseline arm: keep only Small and Large, drop Medium (the decoy).

    Args:
        value_ids: list of Magento option value IDs (as strings) to keep.
                   IDs not in this list are removed from the page entirely
                   (radio input + its label wrapper).
    """
    keep = {str(v) for v in value_ids}
    soup = BeautifulSoup(original_html, "lxml")

    wrapper = soup.select_one("div.product-options-wrapper")
    if not wrapper:
        return str(soup), {"intervention": "keep_options", "removed": 0,
                           "kept": list(keep)}

    removed = 0
    for inp in wrapper.select("input.product-custom-option[type=radio]"):
        if inp.get("value", "") in keep:
            continue
        # Remove the parent <div class="field choice ..."> wrapping this radio
        choice_div = inp.find_parent("div", class_="field")
        target = choice_div if choice_div is not None else inp
        target.decompose()
        removed += 1

    return str(soup), {"intervention": "keep_options",
                       "removed": removed, "kept": list(keep)}


def set_option_label(
    original_html: bytes,
    mapping: Mapping[str | int, str],
) -> tuple[str, dict]:
    """Rewrite the visible <label><span> text for specified option value IDs.

    `mapping` is {value_id: new_label_text}.

    Magento's option markup looks like:
      <input id="options_29728_2" value="184404" .../>
      <label for="options_29728_2"><span>1 Pound</span></label>

    We locate each input by value, find its label by `for=` attribute, and
    rewrite the inner span text.
    """
    str_map = {str(k): v for k, v in mapping.items()}
    soup = BeautifulSoup(original_html, "lxml")

    wrapper = soup.select_one("div.product-options-wrapper")
    if not wrapper:
        return str(soup), {"intervention": "set_option_label", "rewritten": 0}

    rewritten = 0
    for inp in wrapper.select("input.product-custom-option[type=radio]"):
        vid = inp.get("value", "")
        if vid not in str_map:
            continue
        input_id = inp.get("id", "")
        if not input_id:
            continue
        label_span = wrapper.select_one(f'label[for="{input_id}"] > span')
        if label_span is not None:
            label_span.string = str_map[vid]
            rewritten += 1

    return str(soup), {"intervention": "set_option_label",
                       "rewritten": rewritten,
                       "mapping": str_map}


def set_option_price(
    original_html: bytes,
    mapping: Mapping[str | int, float],
) -> tuple[str, dict]:
    """Inject a visible price label next to each option's existing label.

    `mapping` is {value_id: price_in_dollars}.

    The injected element is a <span class="opt-price"> placed AFTER the
    existing label span, formatted as " — $X.XX". The agent reads it as
    plain text in the pruned HTML.
    """
    str_map = {str(k): float(v) for k, v in mapping.items()}
    soup = BeautifulSoup(original_html, "lxml")

    wrapper = soup.select_one("div.product-options-wrapper")
    if not wrapper:
        return str(soup), {"intervention": "set_option_price", "added": 0}

    added = 0
    for inp in wrapper.select("input.product-custom-option[type=radio]"):
        vid = inp.get("value", "")
        if vid not in str_map:
            continue
        input_id = inp.get("id", "")
        if not input_id:
            continue
        label = wrapper.select_one(f'label[for="{input_id}"]')
        if label is None:
            continue
        # Don't double-inject if rerun
        if label.find("span", class_="opt-price"):
            continue
        price_span = soup.new_tag(
            "span",
            attrs={
                "class": "opt-price",
                "style": (
                    "margin-left: 10px; "
                    "font-weight: bold; "
                    "color: rgb(30, 109, 182);"
                ),
            },
        )
        price_span.string = f" \u2014 ${str_map[vid]:.2f}"
        label.append(price_span)
        added += 1

    return str(soup), {"intervention": "set_option_price",
                       "added": added,
                       "mapping": str_map}


def inject_rating(
    original_html: bytes,
    value: int,
    review_count: int = 50,
) -> tuple[str, dict]:
    """Inject a complete rating block on products that don't have one.

    For products in the catalog with no natural rating, the
    `product-reviews-summary` block contains only a "Be the first to review
    this product" link — there is no `rating-result` div for `set_rating()`
    to operate on. This function builds the entire structure (star bar +
    numeric rating + review count link) and inserts it into the
    `product-info-price` block where the rating normally sits.

    If a `rating-result` already exists, this function is a no-op (use
    `set_rating()` from `abxlab.choices.shop.product` instead).
    """
    soup = BeautifulSoup(original_html, "lxml")

    # If a real rating-result already exists, defer to set_rating().
    if soup.select_one("div.rating-result"):
        return str(soup), {"intervention": "inject_rating", "skipped": True,
                           "reason": "rating already present"}

    anchor = soup.select_one("div.product-info-price")
    if anchor is None:
        return str(soup), {"intervention": "inject_rating", "skipped": True,
                           "reason": "no product-info-price anchor"}

    # Remove any existing "Be the first to review" placeholder.
    placeholder = soup.select_one("div.product-reviews-summary")
    if placeholder is not None:
        placeholder.decompose()

    # Build the rating block from scratch.
    block = soup.new_tag("div", attrs={
        "class": "product-reviews-summary",
        "itemprop": "aggregateRating",
        "itemscope": "",
        "itemtype": "http://schema.org/AggregateRating",
    })

    rating_summary = soup.new_tag("div", attrs={"class": "rating-summary"})
    rating_result = soup.new_tag("div", attrs={
        "class": "rating-result",
        "title": f"{value}%",
    })
    star_span = soup.new_tag("span", attrs={"style": f"width: {value}%;"})
    rating_result.append(star_span)
    rating_summary.append(rating_result)
    block.append(rating_summary)

    # Numeric rating display (parallel to product.rating()'s injection)
    rating_details = soup.new_tag("span", attrs={
        "class": "product-rating-details",
        "style": (
            "display: inline-block; "
            "margin-top: 4px; "
            "margin-right: 10px; "
            "color: rgb(251, 79, 31);"
        ),
    })
    rating_details.string = f"Rating: {value}%"
    block.append(rating_details)

    # Review count link
    actions = soup.new_tag("div", attrs={"class": "reviews-actions"})
    review_link = soup.new_tag("a", attrs={"class": "action view"})
    review_count_span = soup.new_tag("span", attrs={"itemprop": "reviewCount"})
    review_count_span.string = str(review_count)
    review_link.append(review_count_span)
    review_link.append(" Reviews")
    actions.append(review_link)
    block.append(actions)

    # Place the rating at the start of product-info-price (above the price)
    anchor.insert(0, block)

    return str(soup), {
        "intervention": "inject_rating",
        "value": value,
        "review_count": review_count,
        "skipped": False,
    }


def anonymize_urls(
    original_html: bytes,
    original_url: str,
    replacement: str = "product-anon",
) -> tuple[str, dict]:
    """Replace all in-page references to the original product URL slug.

    The product URL slug (e.g. 'frank-and-sal-genuine-grana-padano-...-3-pound')
    can leak the original product's default size to the agent through:
      - <a href> attributes (review links, "add your review", etc.)
      - <form action> attributes (cart submission URL)
      - Magento's base64 'uenc' parameter inside the cart form action
      - <link rel="canonical">, <meta property="og:url">, etc.

    This intervention rewrites every occurrence of the slug to a neutral
    placeholder. The browser's actual URL (shown in the agent's tab header
    by BrowserGym) is NOT changed by this function — that requires either a
    Magento URL rewrite or an obs-preprocessor patch. Combine this with an
    explicit "ignore the URL" instruction in the intent for full coverage.

    Args:
        original_url: the full original URL of the product page (the
                      generator already knows this).
        replacement:  the placeholder slug used in rewrites
                      (default 'product-anon').
    """
    import base64
    import re as _re
    from urllib.parse import urlparse

    parsed = urlparse(original_url)
    # Slug = path without leading slash and without ".html"
    slug = parsed.path.lstrip("/")
    if slug.endswith(".html"):
        slug = slug[:-len(".html")]
    if not slug:
        return str(BeautifulSoup(original_html, "lxml")), {
            "intervention": "anonymize_urls", "n_replaced": 0,
            "reason": "empty slug",
        }

    soup = BeautifulSoup(original_html, "lxml")
    n_replaced = 0

    def _rewrite_string(val: str) -> tuple[str, int]:
        """Rewrite slug + uenc inside one string. Returns (new_str, n_changes)."""
        n = 0
        new_val = val
        if slug in new_val:
            new_val = new_val.replace(slug, replacement)
            n += 1
        # uenc: base64 alphabet contains '/', so anchor on the end with /product/
        m = _re.search(r"/uenc/([A-Za-z0-9_+/=]+?)/product/", new_val)
        if m:
            try:
                # Pad to multiple of 4 for safe decoding
                b64 = m.group(1)
                pad = "=" * (-len(b64) % 4)
                decoded = base64.b64decode(b64 + pad).decode("utf-8", errors="ignore")
                if slug in decoded:
                    rewritten = decoded.replace(slug, replacement)
                    new_uenc = base64.b64encode(rewritten.encode()).decode().rstrip("=")
                    new_val = new_val.replace(m.group(1), new_uenc)
                    n += 1
            except Exception:
                pass
        return new_val, n

    # Walk every element and rewrite any attribute containing the slug.
    # Handle both string and list (e.g. class) attribute values.
    for el in soup.find_all(True):
        for attr, val in list(el.attrs.items()):
            if isinstance(val, str):
                new_val, n = _rewrite_string(val)
                if new_val != val:
                    el[attr] = new_val
                    n_replaced += n
            elif isinstance(val, list):
                new_list = []
                changed = False
                for item in val:
                    if isinstance(item, str):
                        new_item, n = _rewrite_string(item)
                        if new_item != item:
                            changed = True
                            n_replaced += n
                        new_list.append(new_item)
                    else:
                        new_list.append(item)
                if changed:
                    el[attr] = new_list

    # Also rewrite plain text occurrences inside the body (rare but possible
    # in description/breadcrumb if not ablated).
    for text_node in list(soup.find_all(string=True)):
        if slug in text_node:
            text_node.replace_with(text_node.replace(slug, replacement))
            n_replaced += 1

    return str(soup), {
        "intervention": "anonymize_urls",
        "n_replaced": n_replaced,
        "slug": slug,
        "replacement": replacement,
    }


def shuffle_options(
    original_html: bytes,
    seed: Optional[int] = None,
) -> tuple[str, dict]:
    """Randomize the on-page order of the radio button options.

    Used to control for position bias — agents may show first-pick or
    last-pick bias regardless of an option's attributes. By shuffling per
    page render, position effects average out across products and samples.

    Place this LATE in the choices chain (after `keep_options`,
    `set_option_label`, `set_option_price`) so labels/prices are already
    bound to their radios before reordering.

    Args:
        seed: optional integer seed for a reproducible shuffle. If None
              (default), uses a fresh random shuffle on every call.

    Returns metadata including the original and new orders so post-hoc
    analyses can recover the rendered position of each option.
    """
    soup = BeautifulSoup(original_html, "lxml")
    options_list = soup.select_one("div.options-list.nested")
    if options_list is None:
        return str(soup), {"intervention": "shuffle_options", "shuffled": 0}

    # Each option is a <div class="field choice ..."> child of options-list.
    # Use direct children only so we don't capture nested fields by mistake.
    choice_divs = [
        d for d in options_list.find_all("div", recursive=False)
        if d.get("class") and "choice" in d.get("class")
    ]
    if len(choice_divs) <= 1:
        return str(soup), {"intervention": "shuffle_options", "shuffled": 0}

    def _value_for(d):
        inp = d.find("input", class_="product-custom-option")
        return inp.get("value", "") if inp is not None else ""

    original_value_ids = [_value_for(d) for d in choice_divs]

    rng = random.Random(seed) if seed is not None else random.Random()
    new_order = list(choice_divs)
    rng.shuffle(new_order)

    new_value_ids = [_value_for(d) for d in new_order]

    # Detach + reinsert in the shuffled order.
    for d in choice_divs:
        d.extract()
    for d in new_order:
        options_list.append(d)

    return str(soup), {
        "intervention": "shuffle_options",
        "shuffled": len(new_order),
        "original_order": original_value_ids,
        "new_order": new_value_ids,
        "seed": seed,
    }


def pre_select_option(
    original_html: bytes,
    value_id: str | int,
) -> tuple[str, dict]:
    """Mark a given option's radio button as `checked="checked"`.

    Used for products with multiple required option groups (e.g. Colgate
    Mouthwash with both Size and Style/flavor required) — the secondary
    group is auto-selected so the agent only has to pick from the primary
    (Size) group.

    Magento's frontend JS will respect the `checked` attribute on page load.
    """
    vid = str(value_id)
    soup = BeautifulSoup(original_html, "lxml")

    wrapper = soup.select_one("div.product-options-wrapper")
    if not wrapper:
        return str(soup), {"intervention": "pre_select_option",
                           "selected": False, "value_id": vid}

    inp = wrapper.select_one(
        f'input.product-custom-option[type=radio][value="{vid}"]'
    )
    if inp is None:
        return str(soup), {"intervention": "pre_select_option",
                           "selected": False, "value_id": vid}

    inp["checked"] = "checked"
    return str(soup), {"intervention": "pre_select_option",
                       "selected": True, "value_id": vid}
