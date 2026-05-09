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
This module contains interventions to modify product pages from config files.
"""

from bs4 import BeautifulSoup


def subtitle(
    original_html: bytes,
    value: str,
    elem_id: str = "page-title-wrapper product"
) -> tuple[str, dict]:
    """Inserts a subtitle below the product title."""

    soup = BeautifulSoup(original_html, "lxml")

    element = soup.find("div", class_=elem_id)

    span_tag = soup.new_tag("h2", attrs={"class":"product-title-details",
                                         "visible":""})
    span_tag["style"] = (
        "display: inline-block; "
        "padding: 4px 8px; "
        "border: 1px solid rgb(30, 109, 182); "
        "border-radius: 12px; "
        "color: rgb(30, 109, 182); "
        "font-size: 2em;"
    )
    span_tag.string = value

    element.insert_after(span_tag)

    modified_html = str(soup)
    return modified_html, {}


def stock(
    original_html: bytes,
    value: str,
    elem_id: str = "product-info-stock-sku"
) -> tuple[str, dict]:
    """Replaces stock information for the product."""

    soup = BeautifulSoup(original_html, "lxml")

    element = soup.find("div", class_=elem_id)

    span_tag = soup.new_tag("span", attrs={"class":"product-stock-details"})
    span_tag["style"] = (
        "display: inline-block; "
        "padding: 4px 8px; "
        "margin-top: 10px; "
        "border: 1px solid rgb(30, 109, 182); "
        "border-radius: 2px; "
        "color: rgb(30, 109, 182); "
        "font-size: 0.9em;"
    )
    span_tag.string = value

    element.insert_after(span_tag)

    modified_html = str(soup)
    return modified_html, {}


def out_of_stock(
    original_html: bytes,
) -> tuple[str, dict]:
    """Marks a product as out of stock (phantom decoy).

    - Changes the stock status text to "Out of Stock"
    - Disables and greys out the "Add to Cart" button
    - Removes the quantity input field
    This makes the product visible for comparison but unpurchasable,
    implementing a phantom decoy design (Pratkanis & Farquhar, 1992).
    """

    soup = BeautifulSoup(original_html, "lxml")

    # Change stock text from "In stock" to "Out of Stock"
    stock_div = soup.find("div", class_="stock")
    if stock_div:
        stock_div["class"] = ["stock", "unavailable"]
        stock_div.string = "Out of Stock"

    # Disable the Add to Cart button
    add_btn = soup.find("button", class_="tocart")
    if add_btn:
        add_btn["disabled"] = "disabled"
        add_btn["style"] = (
            "opacity: 0.4; "
            "cursor: not-allowed; "
            "pointer-events: none;"
        )
        add_btn.string = "Out of Stock"

    # Remove the quantity input box
    qty_box = soup.find("div", class_="box-tocart")
    if qty_box:
        qty_input = qty_box.find("input", attrs={"type": "number"})
        if qty_input:
            qty_input.decompose()

    modified_html = str(soup)
    return modified_html, {}


def price(
    original_html: bytes,
    value: float
) -> str:
    """Replaces the product price."""

    soup = BeautifulSoup(original_html, "lxml")

    # Change price in span
    price = soup.find("span", class_="price")
    if price:
        price.string = "$" + f"{value:.2f}"

    # Change data-price-amount in price-wrapper
    price_wrapper = soup.find("span", class_="price-wrapper")
    if price_wrapper:
        price_wrapper["data-price-amount"] = f"{value:.2f}"

    modified_html = str(soup)
    return modified_html, {}


def review_count(
    original_html: bytes,
    value: int
) -> str:
    """Replaces the review count for the product."""

    soup = BeautifulSoup(original_html, "lxml")

    # Change review count on the right next to rating
    review_count_ratings = soup.find("span", itemprop="reviewCount")
    if review_count_ratings:
        review_count_ratings.string = str(value)

    # Change review count in the tab at the bottom
    review_count_tab = soup.find("span", class_="counter")
    if review_count_tab:
        review_count_tab.string = str(value)

    modified_html = str(soup)
    return modified_html, {}


def set_rating(
    original_html: bytes,
    value: int
) -> tuple[str, dict]:
    """Replaces the product rating with a specific percentage value.

    Also updates the numeric text span created by rating() if it already ran
    (task.process_html calls rating() before choices interventions execute).
    """

    soup = BeautifulSoup(original_html, "lxml")

    rating_result = soup.find("div", class_="rating-result")
    if rating_result:
        # Update title attribute (used by the rating() display function)
        rating_result["title"] = f"{value}%"

        # Update the star bar width
        top_span = rating_result.find("span", recursive=False)
        if top_span:
            top_span["style"] = f"width:{value}% !important"

        # Update the itemprop ratingValue text
        rating_value = rating_result.find("span", itemprop="ratingValue")
        if rating_value:
            rating_value.string = str(value)

        # Magento injects a <script> that hardcodes the original star width,
        # e.g. element.style.width = '60%'; — this overrides our inline style.
        # Find and update that script to use the new value.
        rating_id = rating_result.get("id", "")
        if rating_id:
            for script in soup.find_all("script"):
                if script.string and rating_id in script.string and "style.width" in script.string:
                    import re
                    script.string = re.sub(
                        r"style\.width\s*=\s*'[^']*'",
                        f"style.width = '{value}%'",
                        script.string,
                    )

    # Update the numeric rating text span if rating() already inserted it
    rating_text = soup.find("span", class_="product-rating-details")
    if rating_text:
        rating_text.string = f"Rating: {value}%"

    modified_html = str(soup)
    return modified_html, {}


def sanitize_title(
    original_html: bytes,
) -> tuple[str, dict]:
    """Removes pack-size and quantity mentions from the product title.

    This prevents the agent from using quantity info (e.g. "Pack of 8" vs
    "Pack of 3") to rationalize price differences, which would undermine
    the decoy manipulation.
    """
    import re
    soup = BeautifulSoup(original_html, "lxml")

    # Patterns to strip from the product title
    pack_patterns = [
        r',?\s*\d+-?\s*(?:pack|pk)\b',           # "3-Pack", "3 Pack", ", 3-pk"
        r',?\s*pack\s+of\s+\d+',                  # "Pack of 3"
        r',?\s*set\s+of\s+\d+',                   # "Set of 2"
        r',?\s*\d+\s*(?:packs|count)\b',           # "3 Packs", "12 Count"
        r'\(\s*\d+-?\s*(?:pack|pk|packs)\s*\)',    # "(3-Pack)", "(3 Pack)"
        r'\(\s*pack\s+of\s+\d+\s*\)',              # "(Pack of 3)"
        r',?\s*(?:twin|double|triple)\s*(?:pack)?\b',  # "Twin Pack"
    ]
    combined = '|'.join(pack_patterns)

    # Update the main product title (h1.page-title span)
    title_span = soup.find("h1", class_="page-title")
    if title_span:
        span = title_span.find("span")
        if span and span.string:
            span.string = re.sub(combined, '', span.string, flags=re.IGNORECASE).strip()

    # Also update the <title> tag
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title_tag.string = re.sub(combined, '', title_tag.string, flags=re.IGNORECASE).strip()

    modified_html = str(soup)
    return modified_html, {}


################################################################################
# Functions below modify the HTML without returning additional metadata
# These are useful when are called all the time (e.g. ABxLabShopTask in task.py)
################################################################################

def rating(
    original_html: bytes,
    elem_id: str = "rating-summary"
) -> str:
    """Inserts the rating explicitly in percentage to avoid confusion with the stars by default."""

    soup = BeautifulSoup(original_html, "lxml")

    rating = soup.find("div", class_="rating-result")

    if rating:
        element = soup.find("div", class_=elem_id)

        span_tag = soup.new_tag("span", attrs={"class":"product-rating-details"})
        span_tag["style"] = (
            "display: inline-block; "
            "margin-top: 4px; "
            "margin-right: 10px; "
            "color: rgb(251, 79, 31); "
        )
        span_tag.string = "Rating: " + rating["title"]

        element.insert_after(span_tag)

    modified_html = str(soup)
    return modified_html


def ablate(
    original_html: bytes,
    elems: list[str] = ["product-reviews-summary", "price-box price-final_price"]
) -> tuple[str, dict]:
    """Removes specified elements from the product page.

    Searches for any tag type (div, span, etc.) matching the given class names.
    """

    soup = BeautifulSoup(original_html, "lxml")

    for elem in elems:
        element = soup.find(class_=elem)
        if element:
            element.decompose()

    modified_html = str(soup)
    return modified_html, {}
