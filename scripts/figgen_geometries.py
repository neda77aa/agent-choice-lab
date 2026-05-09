"""Generate fig:geometries_visual: three price-vs-rating scatter plots showing
real T/R/D positions from the controlled triad data, one per decoy geometry."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / "paper" / "figures" / "generated" / "geometries_visual.png"

triads = pd.read_csv(REPO / "data" / "controlled_triads_decoy.csv")

# Map geometry_id to geometry name. From paper:
# Range: ids 0,1; Frequency: 2,3; Inferior: 4,5
GEO_NAME = {
    "range": "Range decoy",
    "frequency": "Frequency decoy",
    "inferior": "Inferior decoy",
}
GEO_DESC = {
    "range": "Same price as target,\nlower rating",
    "frequency": "Same rating as target,\nhigher price",
    "inferior": "Worse on both,\nclose on one",
}

# Pick one representative triad per geometry (the digital-cameras triad is illustrative).
representative = (
    triads.sort_values(["decoy_type", "triad_id", "geometry_id"])
          .groupby("decoy_type", as_index=False)
          .first()
)

fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.16, wspace=0.18)

ROLE_STYLE = {
    "T": dict(marker="o", color="#1f4e79", s=200, zorder=4, label="Target (T)"),
    "R": dict(marker="s", color="#7c7c7c", s=180, zorder=3, label="Rival (R)"),
    "D": dict(marker="^", color="#c0392b", s=210, zorder=4, label="Decoy (D)"),
}

for ax, geo in zip(axes, ["range", "frequency", "inferior"]):
    row = representative.loc[representative["decoy_type"] == geo].iloc[0]
    pts = {
        "T": (row["set_price_target"],     row["set_rating_target"]),
        "R": (row["set_price_competitor"], row["set_rating_competitor"]),
        "D": (row["set_price_decoy"],      row["set_rating_decoy"]),
    }

    # Light dominance rectangle: region dominated by target on price-rating space.
    tx, ty = pts["T"]
    ax.axhspan(0, ty, xmin=0, xmax=1, color="#1f4e79", alpha=0.0)
    ax.add_patch(plt.Rectangle((tx, 0), 200, ty,
                               facecolor="#1f4e79", alpha=0.06, zorder=1))

    # plot points
    for role, (x, y) in pts.items():
        ax.scatter([x], [y], **ROLE_STYLE[role])
        offset = {"T": (8, 6), "R": (8, -2), "D": (-16, -14)}[role]
        ax.annotate(role, (x, y), xytext=offset, textcoords="offset points",
                    fontsize=12, fontweight="bold",
                    color=ROLE_STYLE[role]["color"])

    # dotted dominance arrow from D to T (only when distinguishable)
    if abs(pts["D"][0] - pts["T"][0]) + abs(pts["D"][1] - pts["T"][1]) > 0.01:
        ax.annotate("", xy=pts["T"], xytext=pts["D"],
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                    lw=1.0, ls=":", alpha=0.65))

    ax.set_title(GEO_NAME[geo], fontsize=12, pad=22, fontweight="semibold")
    ax.text(0.5, 1.015, GEO_DESC[geo], transform=ax.transAxes,
            fontsize=8.5, color="#666666", ha="center", va="bottom",
            style="italic")
    ax.set_xlabel("Price ($)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(38, 66)
    ax.set_ylim(72, 90)

axes[0].set_ylabel("Rating (%)", fontsize=9)

# unified legend at top
handles = [plt.Line2D([0], [0], marker=ROLE_STYLE[r]["marker"],
                       color=ROLE_STYLE[r]["color"], linestyle="",
                       markersize=10, label=ROLE_STYLE[r]["label"])
            for r in ("T", "R", "D")]
fig.legend(handles=handles, loc="upper center",
           ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.005))

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
