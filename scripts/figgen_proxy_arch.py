"""Generate fig:proxy_architecture: Agent <-> Proxy <-> Magento store diagram.

Layout: three side-by-side cards, with the request/response arrows drawn ABOVE
the cards (so they never overlap card body text). Below the cards a single
narrative line summarises the design intent.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1].parent / "paper" / "figures" / "generated" / "proxy_architecture.png"

# ----- canvas -----
fig, ax = plt.subplots(figsize=(11.0, 5.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

# ----- palette -----
C_AGENT = "#e8702a"
C_PROXY = "#1f4e79"
C_STORE = "#5b8c5a"
LIGHT   = {"agent": "#fde9d8", "proxy": "#dde6f0", "store": "#e2eedf"}

def card(x, y, w, h, fill, edge, title, subtitle, body):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.4,rounding_size=1.2",
                         linewidth=1.6, edgecolor=edge,
                         facecolor=fill, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 3.2, title,
            ha="center", va="top", fontsize=12.5, fontweight="bold", color=edge)
    ax.text(x + w/2, y + h - 7.2, subtitle,
            ha="center", va="top", fontsize=8.6, color="#555555", style="italic")
    for i, line in enumerate(body):
        ax.text(x + 1.6, y + h - 12.2 - i*3.0, line,
                ha="left", va="top", fontsize=8.4, color="#222222")

# Cards anchored at bottom y=4, height 36 -> tops at y=40.
card(2, 4, 28, 36, LIGHT["agent"], C_AGENT,
     "1.  Agent",
     "LLM driving a browser",
     [
        "• GPT-5.2  /  Claude Opus 4.6",
        "• Gemini 2.5 Pro  /  GPT-4.1 Mini",
        "",
        "Per step the agent emits:",
        "  <think>  reasoning",
        "  <memory> notepad of facts",
        "  <action> click / scroll / tab",
     ])

card(36, 4, 28, 36, LIGHT["proxy"], C_PROXY,
     "2.  Intervention Proxy",
     "edits every page in flight",
     [
        "Per HTTP fetch, applies an",
        "ordered chain of edits:",
        "  • price()      • set_rating()",
        "  • sanitize_title()",
        "  • keep_options()",
        "  • set_option_label/price()",
        "  • shuffle_options()",
        "  • anonymize_urls()",
        "  • ablate(), out_of_stock()",
     ])

card(70, 4, 28, 36, LIGHT["store"], C_STORE,
     "3.  OneStopMarket",
     "local Magento Docker",
     [
        "• 96k product pages",
        "• 27 top-level categories",
        "• Real prices, ratings,",
        "  reviews, images, SKUs",
        "• Working cart + checkout",
        "",
        "Served at  localhost:7770",
     ])

# ----- arrows above the cards -----
# Top row at y=49: rightward "request" flow
def arr(x1, y1, x2, y2, color, mut=18, lw=1.8, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=mut,
                        color=color, linewidth=lw, linestyle=ls, zorder=3)
    ax.add_patch(a)
    return a

Y_TOP = 51
Y_BOT = 45
arr(16, Y_TOP, 50, Y_TOP, "#7a7a7a")
arr(50, Y_TOP, 84, Y_TOP, "#7a7a7a")
ax.text(33, Y_TOP + 0.9, "click / scroll / goto",
        ha="center", va="bottom", fontsize=8.6, color="#7a7a7a", fontweight="semibold")
ax.text(67, Y_TOP + 0.9, "fetch (unmodified)",
        ha="center", va="bottom", fontsize=8.6, color="#7a7a7a", fontweight="semibold")

arr(84, Y_BOT, 50, Y_BOT, C_STORE)
arr(50, Y_BOT, 16, Y_BOT, C_PROXY)
ax.text(67, Y_BOT - 1.0, "raw page",
        ha="center", va="top", fontsize=8.6, color=C_STORE, fontweight="semibold")
ax.text(33, Y_BOT - 1.0, "rewritten page",
        ha="center", va="top", fontsize=8.6, color=C_PROXY, fontweight="semibold")

# ----- title and footer -----
ax.text(50, 58.5, "How the experiment is wired up",
        ha="center", va="top", fontsize=13.5, fontweight="bold")
ax.text(50, 1.2,
        "The agent always sees a normal-looking product page; every controlled "
        "attribute is rewritten in flight by the proxy.",
        ha="center", va="bottom", fontsize=9, color="#444444", style="italic")

plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
