"""Compose fig:intervention_before_after as a side-by-side panel with arrows
and callouts marking the elements changed by the intervention chain."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path

GEN = Path(__file__).resolve().parents[1].parent / "paper" / "figures" / "generated"
OUT = GEN / "intervention_before_after.png"

before = Image.open(GEN / "intervention_before.png")
after  = Image.open(GEN / "intervention_after.png")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.10)

for ax, img, title, subtitle in [
    (axes[0], before, "BEFORE",
     "raw page from Magento"),
    (axes[1], after,  "AFTER",
     "after the intervention chain"),
]:
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold",
                 color={"BEFORE": "#7a7a7a", "AFTER": "#1f4e79"}[title], pad=22)
    ax.text(0.5, 1.025, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, style="italic", color="#555555")

# Highlight the changed regions on the AFTER image with coloured rectangles
# (image is 1180x820 pixels at viewport).
ax = axes[1]
W, H = after.size

def highlight(x, y, w, h, color, label):
    rect = Rectangle((x, y), w, h, linewidth=2.0, edgecolor=color,
                     facecolor="none", linestyle="-", alpha=0.95)
    ax.add_patch(rect)
    ax.annotate(label, xy=(x + w + 4, y + h/2),
                xytext=(x + w + 14, y + h/2),
                fontsize=8.6, color=color, fontweight="semibold",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0))

# Coordinate guesses tuned for the 1180x820 viewport-based screenshot
highlight(660, 410, 165, 30, "#c0392b", "set_rating(82)\nreview_count(50)")
highlight(670, 460, 70, 20, "#1f4e79", "price(58.19)")
highlight(660, 380, 215, 90, "#7a4c1c", "ablate(stock+SKU)")

ax.set_xlim(0, W)
ax.set_ylim(H, 0)

fig.suptitle("A product page before and after the intervention chain",
             fontsize=14, fontweight="bold", y=0.995)

plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
