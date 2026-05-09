"""
Generate paper-ready figures for the agent-choice-lab paper.

Produces clean, consistent matplotlib figures into
    /Users/neda/Desktop/UBC/PHD/LLM_research/paper/figures/generated/
that replace the placeholder \fbox{[... to be created]} blocks in main.tex.
"""

from __future__ import annotations

import os
import re
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch

REPO = Path("/Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab")
PAPER = Path("/Users/neda/Desktop/UBC/PHD/LLM_research/paper")
OUT = PAPER / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent paper-style typography and color palette
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PALETTE = {
    "catalogue": "#9CA3AF",   # grey
    "proxy":     "#2563EB",   # blue
    "agent":     "#F97316",   # orange
    "analytics": "#10B981",   # green
    "target":    "#1F2937",   # ink
    "rival":     "#9CA3AF",
    "decoy":     "#DC2626",
    "small":     "#94A3B8",
    "medium":    "#FBBF24",
    "large":     "#0EA5E9",
    "accent":    "#7C3AED",
    "panel":     "#F8FAFC",
    "panel_dark": "#E5E7EB",
}

MODEL_COLORS = {
    "gpt-5-2":         "#0EA5E9",
    "claude-opus-4-6": "#F97316",
    "gemini-2.5-pro":  "#10B981",
    "claude-opus-4-6_nourl": "#7C3AED",
}
MODEL_LABEL = {
    "gpt-5-2": "GPT-5.2",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-opus-4-6_nourl": "Claude Opus 4.6 (no-URL)",
}


# --------------------------------------------------------------------------
#  Drawing helpers
# --------------------------------------------------------------------------

def rounded_box(ax, x, y, w, h, *, fc="white", ec="#374151", lw=1.4,
                title=None, title_fs=10, title_color="#111827",
                body=None, body_fs=9, body_color="#1F2937",
                pad=0.05, bold_title=True, body_align="left",
                radius=0.04):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    if title is not None:
        ax.text(x + w/2, y + h - pad - 0.01, title,
                ha="center", va="top", fontsize=title_fs,
                color=title_color, weight="bold" if bold_title else "normal",
                zorder=3)
    if body is not None:
        if body_align == "left":
            ax.text(x + pad, y + h - pad - (0.18 if title else 0), body,
                    ha="left", va="top", fontsize=body_fs,
                    color=body_color, zorder=3, family="monospace" if "→" in body or "<" in body else "DejaVu Sans")
        else:
            ax.text(x + w/2, y + h/2, body,
                    ha="center", va="center", fontsize=body_fs,
                    color=body_color, zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color="#374151", lw=1.5, ls="-",
          mut=14, label=None, label_offset=(0, 0.02), label_fs=9,
          curve=0.0):
    style = f"Simple,head_length=8,head_width=6,tail_width=1"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, linestyle=ls, color=color,
        mutation_scale=1.0, lw=lw,
        connectionstyle=f"arc3,rad={curve}",
        zorder=2)
    ax.add_patch(arr)
    if label is not None:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha="center", va="bottom", fontsize=label_fs,
                color="#374151")


def setup_blank_ax(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("auto")


# --------------------------------------------------------------------------
#  1. Paradigm B effect figure (data-driven from v5c trial extraction)
# --------------------------------------------------------------------------

def extract_v5c_trials() -> pd.DataFrame:
    from browsergym.experiments.loop import get_exp_result
    ROOTS = {
        "gpt-5-2":        REPO / "results/decoy_size_v5c_gpt52/sample_1",
        "claude-opus-4-6": REPO / "results/decoy_size_v5c_claude46/sample_1",
        "gemini-2.5-pro":  REPO / "results/decoy_size_v5c_gemini25/sample_1",
        "claude-opus-4-6_nourl": REPO / "results/decoy_size_v5c_nourl_claude46/sample_1",
    }
    rows = []
    for model, root in ROOTS.items():
        if not root.exists():
            continue
        for exp_dir in sorted(root.glob("*/exp*")):
            cfg = exp_dir / "config.yaml"
            if not cfg.exists():
                continue
            with open(cfg) as f:
                meta = yaml.safe_load(f)["task"]["config"]["metadata"]
            sub = next(
                (d for d in sorted(os.listdir(exp_dir))
                 if re.match(r"\d{4}-\d{2}-\d{2}_", d) and (exp_dir / d).is_dir()),
                None)
            if sub is None:
                continue
            try:
                er = get_exp_result(str(exp_dir / sub))
                steps = er.steps_info
            except Exception:
                continue
            completed = bool(steps and (steps[-1].reward or 0) > 0)
            chosen_value = None
            for st in steps:
                action = getattr(st, "action", None)
                if not action:
                    continue
                m = re.match(r"click\(['\"]?(\w+)['\"]?\)", action)
                if not m:
                    continue
                bid = m.group(1)
                obs = getattr(st, "obs", None) or {}
                html = obs.get("pruned_html", "") or ""
                if "type=\"radio\"" not in html:
                    continue
                pat1 = re.compile(
                    r'<input[^>]*\bbid=["\']' + re.escape(bid) +
                    r'["\'][^>]*?value=["\'](\d+)["\']', re.I)
                pat2 = re.compile(
                    r'<input[^>]*?value=["\'](\d+)["\'][^>]*\bbid=["\']' +
                    re.escape(bid) + r'["\']', re.I)
                m2 = pat1.search(html) or pat2.search(html)
                if m2:
                    chosen_value = m2.group(1)
            sid, mid, lid = (
                meta.get("small_id"), meta.get("medium_id"), meta.get("large_id"))
            role = None
            if chosen_value is not None:
                for rid, name in [(sid, "small"), (mid, "medium"), (lid, "large")]:
                    if rid is not None and str(int(rid)) == str(chosen_value):
                        role = name
                        break
            rows.append(dict(
                model=model,
                product_id=meta.get("product_id"),
                arm=meta.get("arm"),
                gap_pct=meta.get("gap_pct"),
                completed=completed,
                chosen_role=role,
                template_title=meta.get("template_title"),
            ))
    return pd.DataFrame(rows)


def fig_paradigm_b_effect():
    cache = OUT / "_v5c_trials_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = extract_v5c_trials()
        df.to_csv(cache, index=False)

    df = df[df["completed"]].copy()
    df["chose_L"] = (df["chosen_role"] == "large").astype(float)
    df["chose_M"] = (df["chosen_role"] == "medium").astype(float)
    df["chose_S"] = (df["chosen_role"] == "small").astype(float)

    rng = np.random.default_rng(42)

    def paired_diff_ci(b: pd.Series, t: pd.Series, n_boot=4000):
        b, t = b.align(t, join="inner")
        d = (t - b).dropna().to_numpy()
        if len(d) == 0:
            return 0.0, 0.0, 0.0, 0
        mean = d.mean()
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return mean, lo, hi, len(d)

    def per_model_paired(model_key):
        sub = df[df["model"] == model_key]
        bl = sub[sub["arm"] == "baseline"].set_index("product_id")["chose_L"]
        tr = sub[sub["arm"] == "treat-medium"].set_index("product_id")["chose_L"]
        return paired_diff_ci(bl, tr)

    primary_models = ["gpt-5-2", "claude-opus-4-6", "gemini-2.5-pro"]
    eff = {m: per_model_paired(m) for m in primary_models}
    pooled_b = (df[df["arm"] == "baseline"]
                .groupby("product_id")["chose_L"].mean())
    pooled_t = (df[(df["arm"] == "treat-medium")
                   & (df["model"].isin(primary_models))]
                .groupby("product_id")["chose_L"].mean())
    eff["__pooled__"] = paired_diff_ci(pooled_b, pooled_t)

    # URL anonymisation ablation: claude with vs without anonymisation
    url_models = ["claude-opus-4-6", "claude-opus-4-6_nourl"]
    url_eff = {m: per_model_paired(m) for m in url_models}

    # ------------------------------------------------------------------
    #  Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.5, 1.0],
                          wspace=0.35)

    # Panel A: per-model L-share by arm (paired across products)
    axA = fig.add_subplot(gs[0, 0])
    arms = ["baseline", "treat-medium"]
    arm_label = {"baseline": "Baseline\n(S, L)",
                 "treat-medium": "Treatment\n(S, M, L) g=0.10"}
    width = 0.20
    x = np.arange(len(arms))
    for i, m in enumerate(primary_models):
        sub = df[df["model"] == m]
        means = [sub[sub["arm"] == a]["chose_L"].mean() for a in arms]
        ses = [sub[sub["arm"] == a]["chose_L"].sem() for a in arms]
        axA.bar(x + (i - 1) * width, means, width=width,
                yerr=ses, capsize=3,
                color=MODEL_COLORS[m], edgecolor="white",
                label=MODEL_LABEL[m])
    axA.set_xticks(x)
    axA.set_xticklabels([arm_label[a] for a in arms])
    axA.set_ylim(0, 1.05)
    axA.set_ylabel(r"Target ($L$) share")
    axA.set_title("(a) Target share by arm, per model",
                  loc="left", weight="bold", fontsize=11)
    axA.legend(loc="upper left", frameon=False, fontsize=8.5)
    axA.grid(axis="y", alpha=0.25)

    # Panel B: within-product paired Δ with 95% bootstrap CI
    axB = fig.add_subplot(gs[0, 1])
    keys = primary_models + ["__pooled__"]
    labels = [MODEL_LABEL[m] for m in primary_models] + ["Pooled\n(3 models)"]
    colors = [MODEL_COLORS[m] for m in primary_models] + ["#1F2937"]
    means = [eff[k][0] for k in keys]
    los = [eff[k][1] for k in keys]
    his = [eff[k][2] for k in keys]
    ks = [eff[k][3] for k in keys]
    y = np.arange(len(keys))[::-1]
    axB.axvline(0, color="#374151", ls="--", lw=1, zorder=1)
    for yi, m, lo, hi, c, k in zip(y, means, los, his, colors, ks):
        axB.errorbar([m], [yi],
                     xerr=[[m - lo], [hi - m]],
                     fmt="o", color=c, ecolor=c, lw=2.0, ms=8,
                     capsize=4, zorder=3)
        axB.text(hi + 0.015, yi, f"  Δ={m:+.2f} (k={k})",
                 va="center", ha="left", fontsize=8.5, color=c)
    axB.set_yticks(y)
    axB.set_yticklabels(labels)
    axB.set_xlabel(r"Within-product paired difference $\Delta_{\rm paired}^B$")
    axB.set_xlim(-0.10, 0.62)
    axB.set_title(r"(b) Treatment $-$ baseline at $g=0.10$",
                  loc="left", weight="bold", fontsize=11)
    axB.grid(axis="x", alpha=0.25)

    # Panel C: URL anonymisation ablation for Claude
    axC = fig.add_subplot(gs[0, 2])
    bar_x = np.arange(2)
    keys2 = ["claude-opus-4-6_nourl", "claude-opus-4-6"]
    labels2 = ["Without\nURL anon.", "With\nURL anon."]
    means2 = [url_eff[m][0] for m in keys2]
    los2 = [url_eff[m][1] for m in keys2]
    his2 = [url_eff[m][2] for m in keys2]
    cs2 = ["#9CA3AF", MODEL_COLORS["claude-opus-4-6"]]
    yerr_low = [m - l for m, l in zip(means2, los2)]
    yerr_high = [h - m for m, h in zip(means2, his2)]
    axC.bar(bar_x, means2, yerr=[yerr_low, yerr_high],
            capsize=4, color=cs2, edgecolor="white", width=0.55)
    axC.axhline(0, color="#374151", ls="--", lw=1, zorder=1)
    for xi, m, h in zip(bar_x, means2, his2):
        axC.text(xi, max(h, m) + 0.025, f"{m:+.2f}",
                 ha="center", va="bottom", fontsize=10, weight="bold",
                 color="#0F172A")
    axC.set_xticks(bar_x)
    axC.set_xticklabels(labels2)
    axC.set_ylabel(r"$\Delta_{\rm paired}^B$ (Claude Opus 4.6)")
    axC.set_ylim(-0.10, 0.52)
    axC.set_title("(c) URL-leakage ablation",
                  loc="left", weight="bold", fontsize=11)
    axC.grid(axis="y", alpha=0.25)

    fig.suptitle(
        r"Paradigm B treatment effect: medium decoy ($g=0.10$) under v5c personal-shopping intent",
        fontsize=11.5, fontweight="bold", y=1.02)
    fig.savefig(OUT / "paradigm_b_effect.png", dpi=220)
    fig.savefig(OUT / "paradigm_b_effect.pdf")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------
#  2. Paradigm A schematic (two browser-tab rows)
# --------------------------------------------------------------------------

def fig_paradigm_a_schematic():
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    setup_blank_ax(ax, xlim=(0, 11.5), ylim=(0, 5.4))

    def draw_tab(ax, x, y, w, h, label, price, rating, *, role,
                 highlight=False):
        # Tab "header" strip
        rounded_box(ax, x, y + h, w, 0.32,
                    fc=PALETTE["panel_dark"], ec="#9CA3AF", lw=0.8,
                    title=label, title_fs=8.5, bold_title=False,
                    pad=0.04)
        # Card body
        ec = "#7C3AED" if highlight else "#374151"
        lw = 2.0 if highlight else 1.2
        rounded_box(ax, x, y, w, h, fc="white", ec=ec, lw=lw,
                    title=None)
        # Product image placeholder
        img_box = Rectangle((x + 0.18, y + h - 1.05), w - 0.36, 0.85,
                            facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8)
        ax.add_patch(img_box)
        ax.text(x + w/2, y + h - 0.62, "[product image]",
                ha="center", va="center", fontsize=7.5, color="#94A3B8",
                style="italic")
        # Title
        ax.text(x + w/2, y + h - 1.30, "Cheese Aged 24mo",
                ha="center", va="center", fontsize=8.5, weight="bold",
                color="#111827")
        # Rating bar
        full_w = w - 0.6
        filled = full_w * (rating / 100)
        ax.add_patch(Rectangle((x + 0.30, y + h - 1.65), full_w, 0.10,
                               facecolor="#E5E7EB", edgecolor="none"))
        ax.add_patch(Rectangle((x + 0.30, y + h - 1.65), filled, 0.10,
                               facecolor="#FBBF24", edgecolor="none"))
        ax.text(x + 0.30 + full_w + 0.02, y + h - 1.60,
                f"{rating}%", fontsize=7.5, va="center",
                color="#92400E")
        # Price
        ax.text(x + w/2, y + h - 2.05, f"${price:.2f}",
                ha="center", va="center", fontsize=11, weight="bold",
                color="#0F172A")
        # 50 reviews
        ax.text(x + w/2, y + h - 2.32, "50 reviews",
                ha="center", va="center", fontsize=7, color="#6B7280")
        # Add to cart button
        rounded_box(ax, x + 0.30, y + 0.10, w - 0.6, 0.36,
                    fc="#0F172A", ec="#0F172A", lw=0.5,
                    title="Add to Cart", title_fs=8.5,
                    title_color="white", radius=0.05, pad=0.05)
        # Role badge
        rcolor = {"T": PALETTE["target"], "R": PALETTE["rival"], "D": PALETTE["decoy"]}[role]
        ax.add_patch(FancyBboxPatch((x + w - 0.50, y + h - 0.10), 0.36, 0.30,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=rcolor, ec="none", zorder=4))
        ax.text(x + w - 0.32, y + h + 0.05, role, ha="center", va="center",
                fontsize=10, weight="bold", color="white", zorder=5)

    # ---- top row : baseline arm (2 tabs)
    ax.text(0.05, 5.05, "Baseline arm", fontsize=11.5, weight="bold",
            color="#111827")
    ax.text(0.05, 4.80, r"$P_{\mathrm{choose}}(2\text{-opt})$",
            fontsize=10, color="#6B7280", style="italic")
    draw_tab(ax, 1.5, 2.6, 2.4, 2.2, "Tab 0 - Target",
             price=58.19, rating=82, role="T")
    draw_tab(ax, 4.3, 2.6, 2.4, 2.2, "Tab 1 - Rival",
             price=41.71, rating=77, role="R")
    arrow(ax, 7.0, 3.6, 8.6, 3.6, color="#374151", lw=2.0,
          label=r"$P_{\rm choose}^{2}$",
          label_offset=(0, 0.10), label_fs=10)

    # ---- bottom row: treatment arm (3 tabs)
    ax.text(0.05, 1.85, "Treatment arm", fontsize=11.5, weight="bold",
            color="#111827")
    ax.text(0.05, 1.60, "+ asymmetric-dominance decoy",
            fontsize=9.5, color="#7C3AED")
    draw_tab(ax, 1.5, -0.65, 2.0, 2.2, "Tab 0 - Target",
             price=58.19, rating=82, role="T")
    draw_tab(ax, 3.7, -0.65, 2.0, 2.2, "Tab 1 - Rival",
             price=41.71, rating=77, role="R")
    draw_tab(ax, 5.9, -0.65, 2.0, 2.2, "Tab 2 - Decoy",
             price=58.19, rating=80, role="D", highlight=True)
    arrow(ax, 8.05, 0.45, 8.6, 0.45, color="#374151", lw=2.0,
          label=r"$P_{\rm choose}^{3}$",
          label_offset=(0, 0.10), label_fs=10)

    # ---- right side : effect equation
    rounded_box(ax, 8.7, 1.4, 2.7, 1.5,
                fc=PALETTE["panel"], ec="#7C3AED", lw=1.6,
                radius=0.08)
    ax.text(8.7 + 1.35, 2.55, r"Decoy effect", ha="center", va="center",
            fontsize=10, weight="bold", color="#7C3AED")
    ax.text(8.7 + 1.35, 2.10,
            r"$\Delta_{\rm lit} = P_{\rm choose}^{3} - P_{\rm choose}^{2}$",
            ha="center", va="center", fontsize=11, color="#111827")
    ax.text(8.7 + 1.35, 1.65,
            "Range / Frequency / Inferior\ngeometries (Table 1)",
            ha="center", va="center", fontsize=8.5, color="#374151",
            style="italic")

    # bottom note
    ax.text(0.05, -0.95,
            "Tab order is independently randomised per trial.  "
            "Decoy geometry callout: range / frequency / inferior. ",
            fontsize=8.5, color="#6B7280", style="italic")

    fig.savefig(OUT / "paradigm_a_schematic.png", dpi=220)
    fig.savefig(OUT / "paradigm_a_schematic.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  3. Paradigm B schematic (within-product radio panels)
# --------------------------------------------------------------------------

def _fig_paradigm_b_schematic_synthetic():
    """Older purely synthetic version, kept for reference."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    setup_blank_ax(ax, xlim=(0, 11.5), ylim=(0, 5.4))

    def draw_product_card(ax, x, y, w, h, *, title, options,
                          callout=None):
        rounded_box(ax, x, y, w, h, fc="white", ec="#374151", lw=1.4,
                    radius=0.06)
        # tab header strip
        rounded_box(ax, x, y + h, w, 0.32,
                    fc=PALETTE["panel_dark"], ec="#9CA3AF", lw=0.8,
                    title=title, title_fs=9, bold_title=False,
                    pad=0.04)
        # image placeholder
        img = Rectangle((x + 0.25, y + h - 1.20), w - 0.5, 0.95,
                        facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8)
        ax.add_patch(img)
        ax.text(x + w/2, y + h - 0.74, "[granola product image]",
                ha="center", va="center", fontsize=7.5,
                color="#94A3B8", style="italic")
        # product title
        ax.text(x + w/2, y + h - 1.45, "Golden Crunchy Granola",
                ha="center", va="center", fontsize=9, weight="bold",
                color="#111827")
        # rating
        ax.text(x + w/2, y + h - 1.65, "Rating: 80%   |   50 reviews",
                ha="center", va="center", fontsize=7.5, color="#6B7280")
        # Size selector heading
        ax.text(x + 0.30, y + h - 1.95, "Choose size:",
                ha="left", va="center", fontsize=8.5, weight="bold",
                color="#111827")
        # radios
        for j, (label, sub, role, highlight) in enumerate(options):
            yy = y + h - 2.30 - j * 0.55
            color = "white"
            ec = "#374151"
            if highlight:
                color = "#FEF3C7"
                ec = "#7C3AED"
            ax.add_patch(plt.Circle((x + 0.40, yy + 0.10), 0.07,
                                    facecolor=color, edgecolor=ec, lw=1.4,
                                    zorder=2))
            ax.text(x + 0.55, yy + 0.17, label,
                    ha="left", va="center", fontsize=9, color="#0F172A")
            ax.text(x + 0.55, yy - 0.04, sub,
                    ha="left", va="center", fontsize=7.5, color="#6B7280",
                    style="italic")
            # role tag
            if role:
                rcolor = {"S": PALETTE["small"], "M": PALETTE["medium"],
                          "L": PALETTE["large"]}[role]
                rounded_box(ax, x + w - 0.45, yy - 0.02, 0.30, 0.30,
                            fc=rcolor, ec="none",
                            title=role, title_fs=8.5, bold_title=True,
                            title_color="white", radius=0.06, pad=0.04)
        # Add to Cart
        rounded_box(ax, x + 0.30, y + 0.10, w - 0.6, 0.36,
                    fc="#0F172A", ec="#0F172A", lw=0.5,
                    title="Add to Cart", title_fs=8.5, title_color="white",
                    radius=0.05, pad=0.05)
        if callout:
            ax.text(x + w/2, y - 0.20, callout,
                    ha="center", va="top", fontsize=9, color="#7C3AED",
                    style="italic", weight="bold")

    # baseline (left)
    draw_product_card(ax, 0.5, 0.6, 3.3, 4.0,
                      title="Baseline arm",
                      options=[("0.5 lb - $5.31", "= $10.62 / lb",
                                "S", False),
                               ("1.5 lb - $11.25", "= $7.50 / lb",
                                "L", False)])

    # treatment (centre)
    draw_product_card(ax, 4.2, 0.6, 3.3, 4.0,
                      title=r"Treatment arm ($g=0.10$)",
                      options=[("0.5 lb - $5.31", "= $10.62 / lb",
                                "S", False),
                               ("0.75 lb - $7.50", "= $10.00 / lb (decoy)",
                                "M", True),
                               ("1.5 lb - $11.25", "= $7.50 / lb",
                                "L", False)])

    # Inset: popcorn-pricing equation + dose-response strip
    rounded_box(ax, 7.9, 2.85, 3.3, 1.75,
                fc=PALETTE["panel"], ec="#7C3AED", lw=1.4, radius=0.06)
    ax.text(7.9 + 1.65, 4.40, "Popcorn pricing", ha="center", va="center",
            fontsize=10, weight="bold", color="#7C3AED")
    ax.text(7.9 + 1.65, 4.05, r"$P_M = (1-g)\, P_L,\quad P_S = q_S\, P_M / q_M$",
            ha="center", va="center", fontsize=10, color="#111827")
    ax.text(7.9 + 1.65, 3.60, r"$g \in \{0.05,\ 0.10,\ 0.15\}$",
            ha="center", va="center", fontsize=10, color="#111827")
    ax.text(7.9 + 1.65, 3.20,
            "smaller g  =>  larger predicted decoy effect",
            ha="center", va="center", fontsize=8.5, color="#374151",
            style="italic")

    # Dose-response thumbnail
    rounded_box(ax, 7.9, 0.6, 3.3, 2.05,
                fc="white", ec="#374151", lw=1.0, radius=0.06)
    ax.text(7.9 + 1.65, 2.45, "Four-arm dose-response design",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#0F172A")
    ax.text(7.9 + 0.18, 2.10, r"target $L$ share",
            ha="left", va="bottom", fontsize=7.5, color="#6B7280")
    # mini bars (4 arms, evenly spaced)
    arm_centres = np.linspace(8.30, 10.85, 4)
    bh = [0.18, 0.65, 0.45, 0.28]
    blab = ["baseline", "g=0.05", "g=0.10", "g=0.15"]
    bsub = ["(S, L)", "strong", "medium", "weak"]
    cols = ["#9CA3AF", PALETTE["large"], PALETTE["large"], PALETTE["large"]]
    for xi, h, lb, sb, c in zip(arm_centres, bh, blab, bsub, cols):
        ax.add_patch(Rectangle((xi - 0.22, 1.20), 0.44, h,
                               facecolor=c, edgecolor="white", lw=0.5,
                               zorder=4))
        ax.text(xi, 1.18, lb, ha="center", va="top", fontsize=7.2,
                color="#0F172A", weight="bold", zorder=5)
        ax.text(xi, 1.00, sb, ha="center", va="top", fontsize=6.8,
                color="#6B7280", style="italic", zorder=5)
    # baseline reference line at top of baseline bar
    ax.plot([8.30 - 0.22, 10.85 + 0.22], [1.38, 1.38],
            color="#9CA3AF", ls=":", lw=0.8, zorder=3)

    fig.savefig(OUT / "paradigm_b_schematic_synthetic.png", dpi=220)
    plt.close(fig)


def fig_paradigm_b_schematic():
    """Real-screenshot version: uses the rendered Cream Crackers product page
    captured by scripts/render_decoy_examples (analysis_output/decoy_size_examples)
    as the treatment arm; synthesises the matched baseline by masking out the
    middle radio.  Adds the popcorn-pricing inset and dose-response thumbnail
    on the right."""
    src_dir = REPO / "analysis_output/decoy_size_examples"
    after = src_dir / "03_Cream_Crackers_after.png"
    if not after.exists():
        # fall back to synthetic
        _fig_paradigm_b_schematic_synthetic()
        return

    img = plt.imread(after)
    H, W = img.shape[:2]

    # The "after" render shows three radios with the popcorn geometry.
    # We synthesise the baseline by hiding the middle (medium) row with a
    # white rectangle overlay on a copy of the same image.
    # Cream Crackers image is 1280 x 1100; middle radio (Pack of 4) sits
    # around y in [0.420, 0.450] and x in [0.55, 0.80].
    import numpy as _np
    img_baseline = img.copy()
    y0 = int(H * 0.420); y1 = int(H * 0.452)
    x0 = int(W * 0.55);  x1 = int(W * 0.83)
    if img_baseline.shape[2] == 4:
        img_baseline[y0:y1, x0:x1, :3] = 1.0
        img_baseline[y0:y1, x0:x1, 3] = 1.0
    else:
        img_baseline[y0:y1, x0:x1, :] = 1.0

    fig = plt.figure(figsize=(14.0, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.95],
                          wspace=0.10)

    # Baseline arm
    axL = fig.add_subplot(gs[0, 0])
    axL.imshow(img_baseline)
    axL.set_xticks([]); axL.set_yticks([])
    for s in axL.spines.values():
        s.set_color("#9CA3AF"); s.set_linewidth(2)
    axL.set_title("Baseline arm   (only S, L visible)",
                  loc="left", fontsize=11, weight="bold",
                  color="#0F172A", pad=8)

    # Treatment arm
    axT = fig.add_subplot(gs[0, 1])
    axT.imshow(img)
    axT.set_xticks([]); axT.set_yticks([])
    for s in axT.spines.values():
        s.set_color("#7C3AED"); s.set_linewidth(2)
    axT.set_title(r"Treatment arm   (S, $M$ decoy, L)   $g=0.10$",
                  loc="left", fontsize=11, weight="bold",
                  color="#7C3AED", pad=8)

    # Right column: popcorn pricing + dose-response thumbnail
    axR = fig.add_subplot(gs[0, 2])
    setup_blank_ax(axR, xlim=(0, 1), ylim=(0, 1))
    rounded_box(axR, 0.02, 0.55, 0.96, 0.42,
                fc=PALETTE["panel"], ec="#7C3AED", lw=1.4, radius=0.05)
    axR.text(0.5, 0.92, "Popcorn pricing", ha="center", va="center",
             fontsize=11.5, weight="bold", color="#7C3AED")
    axR.text(0.5, 0.82,
             r"$P_M = (1-g)\, P_L,\quad P_S = q_S\, P_M / q_M$",
             ha="center", va="center", fontsize=10.5, color="#111827")
    axR.text(0.5, 0.72, r"$g \in \{0.05,\ 0.10,\ 0.15\}$",
             ha="center", va="center", fontsize=10.5, color="#111827")
    axR.text(0.5, 0.62, "smaller g  ⇒  stronger predicted decoy effect",
             ha="center", va="center", fontsize=8.5, color="#374151",
             style="italic")
    axR.text(0.5, 0.55,
             r"This page (Cream Crackers): " + "\n" +
             r"$q_S{=}2$, $q_M{=}4$, $q_L{=}6$;   " +
             r"$P_S{=}\$4.95$,  $P_M{=}\$9.89$,  $P_L{=}\$10.99$",
             ha="center", va="top", fontsize=8, color="#374151")

    rounded_box(axR, 0.02, 0.04, 0.96, 0.45,
                fc="white", ec="#374151", lw=1.0, radius=0.05)
    axR.text(0.5, 0.43, "Four-arm dose-response design",
             ha="center", va="center", fontsize=10.5, weight="bold",
             color="#0F172A")
    axR.text(0.06, 0.34, r"target $L$ share", ha="left", va="bottom",
             fontsize=8.5, color="#6B7280")
    arm_centres = _np.linspace(0.18, 0.92, 4)
    bh = [0.06, 0.22, 0.16, 0.10]
    blab = ["baseline", "g=0.05", "g=0.10", "g=0.15"]
    bsub = ["(S, L)", "strong", "medium", "weak"]
    cols = ["#9CA3AF", PALETTE["large"], PALETTE["large"], PALETTE["large"]]
    for xi, h, lb, sb, c in zip(arm_centres, bh, blab, bsub, cols):
        axR.add_patch(Rectangle((xi - 0.07, 0.16), 0.14, h,
                                facecolor=c, edgecolor="white", lw=0.5,
                                zorder=4, transform=axR.transAxes))
        axR.text(xi, 0.13, lb, ha="center", va="top", fontsize=8,
                 color="#0F172A", weight="bold", transform=axR.transAxes)
        axR.text(xi, 0.09, sb, ha="center", va="top", fontsize=7,
                 color="#6B7280", style="italic", transform=axR.transAxes)
    axR.plot([arm_centres[0] - 0.07, arm_centres[-1] + 0.07],
             [0.22, 0.22], color="#9CA3AF", ls=":", lw=0.8, zorder=3,
             transform=axR.transAxes)

    fig.suptitle("Paradigm B — within-product size decoys "
                 "(real Magento renders + popcorn-pricing inset)",
                 fontsize=12, weight="bold", y=1.02)
    fig.savefig(OUT / "paradigm_b_schematic.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUT / "paradigm_b_schematic.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
#  4. Framework overview (4-stage pipeline flowchart)
# --------------------------------------------------------------------------

def fig_framework_overview():
    fig, ax = plt.subplots(figsize=(13.5, 5.3))
    setup_blank_ax(ax, xlim=(0, 13.5), ylim=(0, 5.3))

    # Stage palette
    cs = {1: "#E5E7EB", 2: "#DBEAFE", 3: "#FED7AA", 4: "#D1FAE5"}
    eg = {1: "#9CA3AF", 2: "#2563EB", 3: "#F97316", 4: "#10B981"}

    # (i) Catalogue ingestion
    rounded_box(ax, 0.2, 1.7, 2.4, 2.6, fc=cs[1], ec=eg[1], lw=1.6,
                title="(i) Catalogue\ningestion", title_fs=10.5,
                title_color=eg[1], pad=0.10, radius=0.08)
    ax.text(1.4, 3.45, "OneStopMarket Magento\nDocker container",
            ha="center", va="center", fontsize=8, color="#374151",
            style="italic")
    ax.text(1.4, 2.85, "products.csv\n~96k rows", ha="center", va="center",
            fontsize=8.5, color="#111827", family="monospace")
    ax.text(1.4, 2.30, "size_ladders\n39 products", ha="center", va="center",
            fontsize=8.5, color="#111827", family="monospace")

    # (ii) Stimulus construction
    rounded_box(ax, 3.0, 1.7, 3.6, 2.6, fc=cs[2], ec=eg[2], lw=1.6,
                title="(ii) Stimulus\nconstruction", title_fs=10.5,
                title_color=eg[2], pad=0.10, radius=0.08)
    rounded_box(ax, 3.2, 2.45, 1.55, 1.30, fc="white", ec=eg[2], lw=1.0,
                title="Paradigm A", title_fs=8.5, title_color=eg[2],
                pad=0.05, radius=0.05)
    ax.text(3.2 + 0.78, 3.20, "Jaccard >= 0.15\nbetween-product",
            ha="center", va="center", fontsize=7.2, color="#374151")
    ax.text(3.2 + 0.78, 2.75, "6 attraction-\neffect templates",
            ha="center", va="center", fontsize=7.2, color="#111827",
            weight="bold")
    rounded_box(ax, 4.85, 2.45, 1.55, 1.30, fc="white", ec=eg[2], lw=1.0,
                title="Paradigm B", title_fs=8.5, title_color=eg[2],
                pad=0.05, radius=0.05)
    ax.text(4.85 + 0.78, 3.20, "Within-product\nsize ladders",
            ha="center", va="center", fontsize=7.2, color="#374151")
    ax.text(4.85 + 0.78, 2.75, "popcorn pricing\ng in {.05,.10,.15}",
            ha="center", va="center", fontsize=7.2, color="#111827",
            weight="bold")
    ax.text(4.80, 2.05,
            "~500-1,500 YAML configs\n(triad x arm x ablation)",
            ha="center", va="center", fontsize=7.6, color="#374151",
            style="italic")

    # (iii) Live execution
    rounded_box(ax, 7.0, 1.7, 3.6, 2.6, fc=cs[3], ec=eg[3], lw=1.6,
                title="(iii) Live execution", title_fs=10.5,
                title_color=eg[3], pad=0.10, radius=0.08)
    rounded_box(ax, 7.2, 3.45, 3.2, 0.55, fc="white", ec=eg[3], lw=1.0,
                title="ABxLab proxy", title_fs=9.0, title_color=eg[3],
                pad=0.04, radius=0.04)
    ax.text(7.2 + 1.6, 3.05,
            "price -> set_rating -> review_count\n-> sanitize_title -> ablate "
            "(P-A)",
            ha="center", va="center", fontsize=7, color="#374151",
            family="monospace")
    ax.text(7.2 + 1.6, 2.55,
            "set_title -> inject_rating -> keep_options\n"
            "-> set_option_label -> set_option_price\n"
            "-> shuffle_options -> anonymize_urls (P-B)",
            ha="center", va="center", fontsize=7, color="#374151",
            family="monospace")
    ax.text(7.2 + 1.6, 1.93,
            "Agent: <think>/<memory>/<action>",
            ha="center", va="center", fontsize=7.5, weight="bold",
            color=eg[3])

    # (iv) Outcome
    rounded_box(ax, 11.0, 1.7, 2.3, 2.6, fc=cs[4], ec=eg[4], lw=1.6,
                title="(iv) Outcome\nand analysis", title_fs=10.5,
                title_color=eg[4], pad=0.10, radius=0.08)
    ax.text(12.15, 3.40,
            "Recover chosen URL /\nvalue_id from final\naction; per-trial table",
            ha="center", va="center", fontsize=7.7, color="#374151")
    ax.text(12.15, 2.55,
            r"Within-triad LPM:" + "\n" + r"$\Delta_{\rm lit}$ per agent x geometry",
            ha="center", va="center", fontsize=8, color="#111827",
            weight="bold")

    # arrows between stages
    for x1, x2, mid in [(2.6, 3.0, "stimuli"),
                         (6.6, 7.0, "configs"),
                         (10.6, 11.0, "trials")]:
        arrow(ax, x1, 3.0, x2, 3.0, color="#374151", lw=1.8)
        ax.text((x1 + x2) / 2, 3.20, mid,
                ha="center", va="bottom", fontsize=7.5, color="#6B7280",
                style="italic")

    # title bar at top
    rounded_box(ax, 0.0, 4.55, 13.5, 0.55, fc="#0F172A", ec="#0F172A",
                lw=0.5, radius=0.04)
    ax.text(13.5 / 2, 4.83,
            "Study 2 pipeline: catalogue -> stimuli -> live execution -> within-triad analysis",
            ha="center", va="center", fontsize=11, weight="bold",
            color="white")

    # legend strip at bottom
    legend_handles = [
        Patch(facecolor=cs[1], edgecolor=eg[1], label="Catalogue"),
        Patch(facecolor=cs[2], edgecolor=eg[2], label="Proxy / stimuli"),
        Patch(facecolor=cs[3], edgecolor=eg[3], label="Agent execution"),
        Patch(facecolor=cs[4], edgecolor=eg[4], label="Analytics"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.05),
              ncol=4, frameon=False, fontsize=9)

    fig.savefig(OUT / "framework_overview.png", dpi=220)
    fig.savefig(OUT / "framework_overview.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  5. Proxy architecture (3-box wiring diagram)
# --------------------------------------------------------------------------

def fig_proxy_architecture():
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    setup_blank_ax(ax, xlim=(0, 12.0), ylim=(0, 4.6))

    # Agent box
    rounded_box(ax, 0.4, 1.0, 3.0, 2.6, fc="#FFEDD5", ec="#F97316", lw=1.8,
                title="(1) Agent", title_fs=11, title_color="#9A3412",
                radius=0.08, pad=0.10)
    ax.text(1.9, 3.20, "LLM + BrowserGym",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#7C2D12")
    ax.text(1.9, 2.75, "<think> + <memory>\n+ <action>",
            ha="center", va="center", fontsize=8.5, family="monospace",
            color="#374151")
    ax.text(1.9, 2.10,
            "Actions:\nclick / fill / goto / scroll /\n"
            "select_option / tab_focus / ...",
            ha="center", va="center", fontsize=8, color="#374151")
    ax.text(1.9, 1.30, "10-step budget",
            ha="center", va="center", fontsize=9, color="#9A3412",
            weight="bold")

    # Proxy box
    rounded_box(ax, 4.5, 1.0, 3.0, 2.6, fc="#DBEAFE", ec="#2563EB", lw=1.8,
                title="(2) Proxy", title_fs=11, title_color="#1E3A8A",
                radius=0.08, pad=0.10)
    ax.text(6.0, 3.20, "Python service",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#1E3A8A")
    # intervention chip strip
    chips = [
        "price", "set_rating", "review_count", "sanitize_title",
        "keep_options", "set_option_label", "set_option_price",
        "anonymize_urls", "ablate",
    ]
    for j, c in enumerate(chips):
        col = j % 3
        row = j // 3
        cx = 4.7 + col * 0.95
        cy = 2.85 - row * 0.32
        rounded_box(ax, cx, cy, 0.85, 0.26,
                    fc="white", ec="#1E3A8A", lw=0.8,
                    title=c, title_fs=6.8, title_color="#1E3A8A",
                    bold_title=False, radius=0.06, pad=0.0)
    ax.text(6.0, 1.25, "Rewrites HTML on every fetch",
            ha="center", va="center", fontsize=8, color="#1E3A8A",
            style="italic")

    # Store box
    rounded_box(ax, 8.6, 1.0, 3.0, 2.6, fc="#E5E7EB", ec="#6B7280", lw=1.8,
                title="(3) OneStopMarket store", title_fs=11,
                title_color="#374151", radius=0.08, pad=0.10)
    ax.text(10.1, 3.10, "Magento (Docker)",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#374151")
    ax.text(10.1, 2.55,
            "Unmodified product\ncatalogue, cart, checkout",
            ha="center", va="center", fontsize=8.5, color="#374151")
    ax.text(10.1, 1.85, "~96,000 product pages\n27 categories",
            ha="center", va="center", fontsize=8.5, color="#6B7280")
    ax.text(10.1, 1.30, "shopping_final_0712.tar",
            ha="center", va="center", fontsize=7.5,
            family="monospace", color="#374151")

    # Arrows
    arrow(ax, 3.4, 2.75, 4.5, 2.75, color="#F97316", lw=2.0)
    ax.text(3.95, 2.95, "click / type",
            ha="center", va="bottom", fontsize=8, color="#9A3412",
            style="italic")
    arrow(ax, 7.5, 2.75, 8.6, 2.75, color="#2563EB", lw=2.0)
    ax.text(8.05, 2.95, "GET /page",
            ha="center", va="bottom", fontsize=8, color="#1E3A8A",
            style="italic")
    arrow(ax, 8.6, 1.85, 7.5, 1.85, color="#6B7280", lw=2.0)
    ax.text(8.05, 1.65, "raw HTML",
            ha="center", va="top", fontsize=8, color="#374151",
            style="italic")
    arrow(ax, 4.5, 1.85, 3.4, 1.85, color="#2563EB", lw=2.0)
    ax.text(3.95, 1.65, "modified HTML",
            ha="center", va="top", fontsize=8, color="#1E3A8A",
            style="italic")

    # Title strip
    rounded_box(ax, 0.0, 4.05, 12.0, 0.45, fc="#0F172A", ec="#0F172A",
                lw=0.5, radius=0.04)
    ax.text(6.0, 4.27,
            "How the experiment is wired up: Agent <-> Proxy <-> Store",
            ha="center", va="center", fontsize=11, weight="bold",
            color="white")

    fig.savefig(OUT / "proxy_architecture.png", dpi=220)
    fig.savefig(OUT / "proxy_architecture.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  6. URL anonymisation before / after diagram
# --------------------------------------------------------------------------

def fig_url_anon():
    """Two stacked panels showing browser-tab + DOM fragments before/after
    the anonymize_urls intervention."""
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    setup_blank_ax(ax, xlim=(0, 12.5), ylim=(0, 7.5))

    def render_panel(y0, *, title, ec, slug, fragments, note):
        x0, w, h = 0.4, 11.7, 3.30
        rounded_box(ax, x0, y0, w, h, fc="white", ec=ec, lw=1.8, radius=0.07,
                    title=title, title_fs=11.5, title_color=ec, pad=0.10)
        # browser tab + URL bar block
        rounded_box(ax, x0 + 0.25, y0 + h - 0.95, w - 0.50, 0.60,
                    fc="#F1F5F9", ec="#9CA3AF", lw=1.0, radius=0.04)
        ax.text(x0 + 0.45, y0 + h - 0.65, "● ● ●",
                ha="left", va="center", fontsize=9, color="#6B7280")
        ax.text(x0 + 1.15, y0 + h - 0.65,
                "Granola product page",
                ha="left", va="center", fontsize=9, color="#0F172A")
        # URL bar
        rounded_box(ax, x0 + 0.25, y0 + h - 1.55, w - 0.50, 0.45,
                    fc="#0F172A", ec="#0F172A", lw=0.5, radius=0.04)
        ax.text(x0 + 0.45, y0 + h - 1.32, "URL:",
                ha="left", va="center", fontsize=8, color="#94A3B8",
                family="monospace")
        ax.text(x0 + 0.95, y0 + h - 1.32, slug,
                ha="left", va="center", fontsize=8,
                family="monospace", color="white")
        # 3 DOM fragments laid out horizontally
        frag_w = (w - 0.60) / 3
        for j, (label, code) in enumerate(fragments):
            fx = x0 + 0.30 + j * frag_w
            rounded_box(ax, fx + 0.05, y0 + 0.50, frag_w - 0.10, 1.20,
                        fc="#F8FAFC", ec="#E5E7EB", lw=0.8, radius=0.03)
            ax.text(fx + 0.20, y0 + 1.55, label,
                    ha="left", va="top", fontsize=8,
                    color=ec, weight="bold", style="italic")
            ax.text(fx + 0.20, y0 + 1.30, code,
                    ha="left", va="top", fontsize=7.4,
                    family="monospace", color="#0F172A", wrap=True)
        # bottom note
        ax.text(x0 + w / 2, y0 + 0.22, note,
                ha="center", va="center", fontsize=8.5,
                color=ec, style="italic")

    # Real URL captured from results/decoy_size_v5c_gpt52/.../exp102 (rolled oats).
    # Agent's actual tab address contains the size token "10-pounds".
    raw_slug = "organic-rolled-oats-10-pounds-old-fashioned-100-whole-grain-non-gmo-kosher-bulk.html"
    raw_fragments = [
        ("wishlist <a href>",
         '<a class="action towishlist"\n'
         '  href="…/organic-rolled-\n'
         '   oats-10-pounds-…html">\n'
         '  Add to wishlist\n</a>'),
        ("cart <form action>",
         '<form action=\n'
         '  "/checkout/cart/add/\n'
         '   uenc/aHR0cD…cm9sbGVk\n'
         '   LW9hdHMtMTAtcG91…">'),
        ("uenc base64 (decoded)",
         '{"product":\n'
         '  ".../organic-rolled-\n'
         '   oats-10-pounds-…"}'),
    ]
    anon_slug = "product-anon.html"
    anon_fragments = [
        ("wishlist <a href>",
         '<a class="action towishlist"\n'
         '  href="…/product-anon\n'
         '   .html">\n'
         '  Add to wishlist\n</a>'),
        ("cart <form action>",
         '<form action=\n'
         '  "/checkout/cart/add/\n'
         '   uenc/aHR0cD…cHJvZHVj\n'
         '   dC1hbm9u…">\n'
         '[base64 re-encoded]'),
        ("uenc base64 (re-encoded)",
         '{"product":\n'
         '  ".../product-anon\n'
         '   .html"}\n'
         '[decoded → rewritten\n → re-encoded]'),
    ]

    render_panel(
        y0=3.85,
        title="(top) Raw Magento page — '10-pounds' token leaks in URL, link, and form",
        ec="#DC2626", slug=raw_slug, fragments=raw_fragments,
        note=("Real URL captured from a v5c run on 'Organic Rolled Oats'. "
              "Even after the H1 is sanitised, the agent still sees '10-pounds' three times."))
    render_panel(
        y0=0.35,
        title="(bottom) After  anonymize_urls — slug + base64 uenc rewritten to 'product-anon'",
        ec="#10B981", slug=anon_slug, fragments=anon_fragments,
        note=("Closes the URL channel; the agent can no longer infer the 'default' "
              "size from URL or form action."))

    # Down-arrow connecting the two panels
    arrow(ax, 6.25, 3.80, 6.25, 3.65, color="#7C3AED", lw=2.5)
    ax.text(6.45, 3.72, "  anonymize_urls",
            ha="left", va="center", fontsize=9.5, color="#7C3AED",
            weight="bold")

    fig.suptitle("URL leakage and the anonymize_urls intervention",
                 fontsize=12, weight="bold", y=0.99)
    fig.savefig(OUT / "url_anon.png", dpi=220)
    fig.savefig(OUT / "url_anon.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  7. Stimulus presentation 2x2 figure (Study 1)
# --------------------------------------------------------------------------

def fig_stimulus_examples():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.0))
    for ax in axes.ravel():
        setup_blank_ax(ax, xlim=(0, 1), ylim=(0, 1))

    def panel_header(ax, label, sub):
        rounded_box(ax, 0.02, 0.88, 0.96, 0.10, fc="#0F172A", ec="#0F172A",
                    lw=0.5, radius=0.02)
        ax.text(0.50, 0.93, label, ha="center", va="center",
                fontsize=11, weight="bold", color="white")
        ax.text(0.50, 0.83, sub, ha="center", va="center",
                fontsize=9, color="#374151", style="italic")

    def role_strip(ax, roles, x_positions):
        for r, x in zip(roles, x_positions):
            rcolor = {"T": PALETTE["target"], "R": PALETTE["rival"],
                      "D": PALETTE["decoy"]}[r]
            ax.add_patch(plt.Circle((x, 0.06), 0.025,
                                    facecolor=rcolor, edgecolor="white", lw=0.5))
            ax.text(x, 0.11, r, ha="center", va="bottom",
                    fontsize=10, weight="bold", color=rcolor)

    # ---- Top-left : Numeric, two-option
    ax = axes[0, 0]
    panel_header(ax, "Numeric, two-option",
                 "Hedgcock-Rao-Chen beer pair, plain monospaced text")
    rounded_box(ax, 0.06, 0.20, 0.88, 0.55, fc="#F8FAFC", ec="#374151",
                lw=1.0, radius=0.02)
    body = ("A: $4.40, quality 63%\n"
            "B: $6.50, quality 74%\n\n"
            "Which do you choose?")
    ax.text(0.50, 0.48, body, ha="center", va="center",
            fontsize=11, family="monospace", color="#0F172A")
    role_strip(ax, ["R", "T"], [0.32, 0.68])

    # ---- Top-right : Numeric, three-option
    ax = axes[0, 1]
    panel_header(ax, "Numeric, three-option",
                 "Asymmetric-dominance decoy added")
    rounded_box(ax, 0.06, 0.20, 0.88, 0.55, fc="#F8FAFC", ec="#374151",
                lw=1.0, radius=0.02)
    body = ("A: $4.40, quality 63%\n"
            "B: $6.50, quality 74%\n"
            "C: $6.50, quality 71%   <- decoy\n\n"
            "Which do you choose?")
    ax.text(0.50, 0.48, body, ha="center", va="center",
            fontsize=11, family="monospace", color="#0F172A")
    role_strip(ax, ["R", "T", "D"], [0.24, 0.50, 0.76])

    # ---- Bottom-left : Visual, two-option
    ax = axes[1, 0]
    panel_header(ax, "Visual, two-option",
                 "Frederick et al. (2014); product photographs")
    rounded_box(ax, 0.10, 0.22, 0.34, 0.50, fc="#F1F5F9", ec="#9CA3AF",
                lw=1.0, radius=0.04)
    ax.text(0.27, 0.47, "[image A]", ha="center", va="center",
            fontsize=10, color="#94A3B8", style="italic")
    rounded_box(ax, 0.56, 0.22, 0.34, 0.50, fc="#F1F5F9", ec="#9CA3AF",
                lw=1.0, radius=0.04)
    ax.text(0.73, 0.47, "[image B]", ha="center", va="center",
            fontsize=10, color="#94A3B8", style="italic")
    role_strip(ax, ["R", "T"], [0.27, 0.73])

    # ---- Bottom-right : Visual, phantom
    ax = axes[1, 1]
    panel_header(ax, "Visual, three-option (phantom)",
                 "Pettibone-Wedell phantom decoy: 'currently unavailable'")
    for i, (xpos, role) in enumerate(zip([0.06, 0.36, 0.66], ["R", "T", "D"])):
        rounded_box(ax, xpos, 0.22, 0.28, 0.50, fc="#F1F5F9", ec="#9CA3AF",
                    lw=1.0, radius=0.04)
        ax.text(xpos + 0.14, 0.47,
                f"[image {chr(ord('A') + i)}]",
                ha="center", va="center", fontsize=9, color="#94A3B8",
                style="italic")
        if role == "D":
            # translucent overlay banner
            ax.add_patch(Rectangle((xpos, 0.34), 0.28, 0.18,
                                   facecolor="#DC2626", alpha=0.78,
                                   edgecolor="none", zorder=4))
            ax.text(xpos + 0.14, 0.43, "Currently\nunavailable",
                    ha="center", va="center", fontsize=8.5,
                    weight="bold", color="white", zorder=5)
    role_strip(ax, ["R", "T", "D"], [0.20, 0.50, 0.80])

    fig.suptitle("Stimulus presentation styles in Study 1",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "stimulus_examples.png", dpi=220)
    fig.savefig(OUT / "stimulus_examples.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  8. Prompting modes example (3-column comparison)
# --------------------------------------------------------------------------

def fig_prompting_modes():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4))
    titles = ["Fast mode", "Deliberative mode", "Knowledge mode"]
    color = ["#0EA5E9", "#10B981", "#7C3AED"]
    sub = ["System-1 single token", "Chain-of-thought", "Two-turn metacognition"]

    PROMPT_BODY = (
        "Choose between:\n"
        r"  A: \$4.40, quality 63%" + "\n"
        r"  B: \$6.50, quality 74%" + "\n"
        r"  C: \$6.50, quality 71%" + "\n\n"
    )
    instructions = [
        "Respond with [A/B/C] only.\nNo explanation.",
        "Think step by step about the\ntrade-offs before answering.\n"
        "Finish with: 'Therefore, I\nchoose [X]'.",
        "Turn 1: Define the decoy effect\nand give one example.\n\n"
        "Turn 2: Use that knowledge to\nchoose between A, B, C.",
    ]
    responses = [
        "B",
        ("Option B beats A on\n"
         r"quality (74% vs 63%) for \$2.10" + "\n"
         "more. Option C is dominated\nby B (same price, lower\n"
         "quality). C makes B's\nquality advantage more salient.\n"
         "Therefore, I choose B."),
        ("Turn 1 (verbatim): 'The decoy\n"
         "effect (asymmetric dominance)\n"
         "occurs when adding a third\n"
         "option that is dominated by\n"
         "one of two existing options\n"
         "shifts preference toward the\n"
         "dominating option ...'\n\n"
         "Turn 2: 'C is clearly\n"
         "dominated by B; the higher\n"
         "quality of B becomes more\n"
         "compelling. I choose B.'"),
    ]

    for ax, title, c, s, instr, resp in zip(
            axes, titles, color, sub, instructions, responses):
        setup_blank_ax(ax, xlim=(0, 1), ylim=(0, 1))
        # header
        rounded_box(ax, 0.02, 0.93, 0.96, 0.06, fc=c, ec=c, lw=0.5,
                    radius=0.02)
        ax.text(0.5, 0.96, title, ha="center", va="center",
                fontsize=13, weight="bold", color="white")
        ax.text(0.5, 0.895, s, ha="center", va="center",
                fontsize=10, color="#374151", style="italic")
        # input prompt box
        rounded_box(ax, 0.02, 0.46, 0.96, 0.40,
                    fc="#F8FAFC", ec="#374151", lw=1.0,
                    title="Prompt (input to model)", title_fs=10,
                    title_color="#374151", radius=0.03, pad=0.04)
        ax.text(0.04, 0.79, PROMPT_BODY, ha="left", va="top",
                fontsize=10, family="monospace", color="#0F172A")
        ax.text(0.04, 0.60, instr, ha="left", va="top",
                fontsize=10, family="monospace", color="#0F172A")
        # response box
        rounded_box(ax, 0.02, 0.02, 0.96, 0.42,
                    fc="white", ec=c, lw=1.4,
                    title="Model response", title_fs=10,
                    title_color=c, radius=0.03, pad=0.04)
        ax.text(0.04, 0.37, resp, ha="left", va="top",
                fontsize=9.5, family="monospace", color="#0F172A")

    fig.tight_layout()
    fig.savefig(OUT / "prompting_modes.png", dpi=220)
    fig.savefig(OUT / "prompting_modes.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  9. Intervention before/after — uses real screenshots
# --------------------------------------------------------------------------

def fig_intervention_before_after():
    src_dir = REPO / "analysis_output/decoy_size_examples"
    # pick a clean example: Cream Crackers (#3)
    before = src_dir / "03_Cream_Crackers_before.png"
    after = src_dir / "03_Cream_Crackers_after.png"
    if not (before.exists() and after.exists()):
        # fallback to cookies
        before = src_dir / "01_Animal_Crackers_Original_before.png"
        after = src_dir / "01_Animal_Crackers_Original_after.png"

    img_b = plt.imread(before)
    img_a = plt.imread(after)

    fig = plt.figure(figsize=(13.0, 7.2))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    for col, (img, label, sub, color) in enumerate([
        (img_b, "Before", "Raw OneStopMarket product page (Magento)",
         "#9CA3AF"),
        (img_a, "After interventions",
         "price → set_rating → review_count → sanitize_title → ablate",
         "#7C3AED")]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(2.5)
        ax.set_title(label, fontsize=13, weight="bold",
                     color=color, loc="left", pad=22)
        ax.text(0.0, 1.025, sub, transform=ax.transAxes,
                fontsize=9, color="#374151", style="italic", va="bottom")

    fig.suptitle(
        "A product page before and after the ABxLab intervention chain  "
        "(real Magento + proxy renders)",
        fontsize=12, weight="bold", y=1.06)
    fig.savefig(OUT / "intervention_before_after.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUT / "intervention_before_after.pdf",
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 10. Decoy geometries: visual recap (3-panel price-vs-rating scatter)
# --------------------------------------------------------------------------

def fig_geometries_visual():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))

    # Common axis frame: price (x), rating (y); R cheap+low-rated, T pricier+higher-rated
    R = (0.35, 0.40)
    T = (0.65, 0.70)
    decoys = [
        ("Range",     (0.65, 0.55), "same price, lower rating",
         "$D$ dominated by $T$ on rating"),
        ("Frequency", (0.80, 0.70), "same rating, higher price",
         "$D$ dominated by $T$ on price"),
        ("Inferior",  (0.78, 0.55), "worse on both attributes",
         "$D$ dominated by $T$ on both"),
    ]

    for ax, (name, D, sub, dom) in zip(axes, decoys):
        ax.set_xlim(0.10, 0.95)
        ax.set_ylim(0.20, 0.95)
        ax.set_xlabel("Price  (lower → higher)")
        ax.set_ylabel("Rating  (lower → higher)")
        ax.grid(alpha=0.20)
        ax.set_title(name, weight="bold", fontsize=11.5, color="#0F172A",
                     loc="left", pad=18)
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=9,
                color="#6B7280", style="italic")

        # dominance rectangle (T strictly dominates everything strictly below+left)
        ax.add_patch(Rectangle((T[0], 0.20), 0.95 - T[0], T[1] - 0.20,
                               facecolor="#FBBF24", alpha=0.10,
                               edgecolor="none"))

        # Plot R, T, D
        for (x, y), name_, color, marker in [
            (R, "Rival ($R$)",   PALETTE["rival"], "o"),
            (T, "Target ($T$)",  PALETTE["target"], "o"),
            (D, "Decoy ($D$)",   PALETTE["decoy"], "s"),
        ]:
            ax.scatter([x], [y], s=240, color=color, edgecolor="white",
                       linewidth=1.8, marker=marker, zorder=5)
            label = name_.split(" ")[0][0]   # T / R / D
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=10.5, weight="bold", color="white", zorder=6)

        # arrow showing dominance direction
        ax.annotate("", xy=D, xytext=T,
                    arrowprops=dict(arrowstyle="->", color="#7C3AED",
                                    lw=1.6, shrinkA=8, shrinkB=8))
        # caption-like dominance note inside plot
        ax.text(0.50, 0.25, dom, ha="center", va="center",
                fontsize=8.5, color="#7C3AED", style="italic")

    fig.suptitle("The three decoy geometries used in Paradigm A "
                 "(price-vs-rating attribute space)",
                 fontsize=12, weight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / "geometries_visual.png", dpi=220)
    fig.savefig(OUT / "geometries_visual.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# 11. Intent variants comparison
# --------------------------------------------------------------------------

def fig_intent_variants():
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    setup_blank_ax(ax, xlim=(0, 13.5), ylim=(0, 8.0))

    variants = [
        ("v1  (default)",
         '"Add the best product from the\nopen tabs to the shopping cart."',
         "Largest", "L", "#9CA3AF",
         "Per-unit-value heuristic\ndominates."),
        ("v2  (advisor)",
         '"Add an option of this product\nto the shopping cart for me."',
         "Smallest", "S", "#94A3B8",
         "Pulls toward minimal\ncommitment."),
        ("v4  (recommend)",
         '"Add this product to the cart.\nChoose the option you would\nrecommend."',
         "Largest", "L", "#9CA3AF",
         "Reverts to per-unit-value\nframing."),
        ("v5a  (personal-shop., ‘right’)",
         '"Choose the right size of this\nproduct for me and add it to\nthe cart."',
         "Mixed", "S/L", "#FBBF24",
         "Partially recovers the\nattraction effect."),
        ("v5c  (personal-shop., ‘suitable’)",
         '"Choose a suitable size of this\nproduct for me and add it to the\nshopping cart."',
         "Largest when M present", "L*", "#0EA5E9",
         "Cleanest dose-response\nread."),
    ]

    row_h = 1.40
    top = 7.55

    # Title strip
    rounded_box(ax, 0.0, 7.55, 13.5, 0.40, fc="#0F172A", ec="#0F172A",
                lw=0.5, radius=0.04)
    ax.text(13.5/2, 7.75,
            "The five intent wordings compared in Paradigm B",
            ha="center", va="center", fontsize=12, weight="bold",
            color="white")
    # column headers
    ax.text(0.10 + 0.78, top - 0.20, "Variant",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#374151")
    ax.text(1.75 + 2.40, top - 0.20, "Intent prompt (verbatim)",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#374151")
    ax.text(6.70 + 1.78, top - 0.20, "Modal choice on 1.5 lb granola",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#374151")
    ax.text(10.40 + 1.50, top - 0.20, "Behavioural reading",
            ha="center", va="center", fontsize=9.5, weight="bold",
            color="#374151")

    for i, (vname, prompt, choice, choice_chip, chip_color, note) in enumerate(variants):
        y = top - 0.40 - (i + 1) * row_h
        # variant name
        rounded_box(ax, 0.10, y, 1.55, row_h - 0.12,
                    fc="#F1F5F9", ec="#9CA3AF", lw=1.0, radius=0.05,
                    title=vname.split("(")[0].strip(),
                    title_fs=10, title_color="#0F172A", pad=0.06)
        ax.text(0.10 + 0.78, y + 0.20,
                "(" + vname.split("(", 1)[1] if "(" in vname else "",
                ha="center", va="center", fontsize=8, color="#6B7280",
                style="italic")
        # prompt box (multi-line, narrower so it doesn't run into the chip)
        rounded_box(ax, 1.75, y, 4.80, row_h - 0.12,
                    fc="white", ec="#374151", lw=1.0, radius=0.04)
        ax.text(1.85, y + (row_h - 0.12) / 2, prompt,
                ha="left", va="center", fontsize=8.5, color="#0F172A",
                family="monospace")
        # choice chip
        rounded_box(ax, 6.70, y + 0.16, 3.55, row_h - 0.44,
                    fc=chip_color, ec=chip_color, lw=0.5, radius=0.05)
        ax.text(6.70 + 0.55, y + (row_h - 0.12) / 2, choice_chip,
                ha="center", va="center", fontsize=12, weight="bold",
                color="white")
        ax.text(6.70 + 2.20, y + (row_h - 0.12) / 2, choice,
                ha="center", va="center", fontsize=8.5, color="white",
                weight="bold")
        # note
        rounded_box(ax, 10.40, y, 3.00, row_h - 0.12,
                    fc=PALETTE["panel"], ec="#7C3AED", lw=0.8, radius=0.04)
        ax.text(10.50, y + (row_h - 0.12) / 2, note,
                ha="left", va="center", fontsize=8.5, color="#7C3AED",
                style="italic", wrap=True)

    ax.text(13.5/2, 0.10,
            "Identical product pages → different choices when only the prompt phrasing changes.",
            ha="center", va="center", fontsize=9.5, color="#374151",
            style="italic")

    fig.savefig(OUT / "intent_variants.png", dpi=220)
    fig.savefig(OUT / "intent_variants.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# 12. Agent reasoning trace example
# --------------------------------------------------------------------------

def fig_agent_trace():
    """Real 4-step annotated reasoning trace from a Paradigm A treatment trial.

    Source: results/decoy_pricepressure_baseline_gpt52/
            claude-4-6-run-2026-04-13_16-46-30/exp1
    Triad : Sony Cybershot DSCS730 (T) / DSC-S650 (R) / DSCW70 (D, range decoy).
    Real verbatim think text loaded from figures/generated/_real_trace.json.
    """
    import json, textwrap
    real_trace = OUT / "_real_trace.json"
    real_steps = json.loads(real_trace.read_text()).get("steps", []) \
                 if real_trace.exists() else []

    def short(text, w=44, max_lines=10):
        if not text:
            return ""
        # protect $ from mathtext, strip <memory>/<action> tags so we keep think only
        text = re.sub(r"<memory>.*?</memory>", "", text, flags=re.S)
        text = re.sub(r"<action>.*?</action>", "", text, flags=re.S).strip()
        text = text.replace("$", r"\$")
        out = []
        for line in text.split("\n"):
            if not line.strip():
                if out and out[-1] != "":
                    out.append("")
                continue
            out.extend(textwrap.wrap(line, width=w) or [line])
        return "\n".join(out[:max_lines])

    real_action = lambda i, default: real_steps[i]["action"] if i < len(real_steps) else default

    steps = [
        dict(
            t="Step 0 — read Tab 2 (Decoy: DSCW70, 85★, \\$61.19)",
            obs=("<title>Sony Cybershot DSCW70</title>\n"
                 "<span class='price'>\\$61.19</span>\n"
                 "<span class='rating'>85%</span>\n"
                 "<span>50 reviews</span>"),
            think=short(real_steps[0]["think"], 44, 10) if real_steps else "",
            action=real_action(0, "tab_focus(0)"),
            note="Encodes the third (decoy) option first",
        ),
        dict(
            t="Step 1 — read Tab 0 (Target: DSCS730, 82★, \\$61.19)",
            obs=("<title>Sony Cybershot DSCS730</title>\n"
                 "<span class='price'>\\$61.19</span>\n"
                 "<span class='rating'>82%</span>\n"
                 "<span>50 reviews</span>"),
            think=short(real_steps[1]["think"], 44, 10) if len(real_steps) > 1 else "",
            action=real_action(1, "tab_focus(1)"),
            note="Encodes the target option",
        ),
        dict(
            t="Step 2 — read Tab 1 (Rival: DSC-S650, 75★, \\$38.71)",
            obs=("<title>Sony Cybershot DSC-S650</title>\n"
                 "<span class='price'>\\$38.71</span>\n"
                 "<span class='rating'>75%</span>\n"
                 "<span>50 reviews</span>"),
            think=short(real_steps[2]["think"], 44, 10) if len(real_steps) > 2 else "",
            action=real_action(2, "tab_focus(2)"),
            note='Picks the highest-rated option (the decoy)',
        ),
        dict(
            t="Step 3 — Add to Cart (commit)",
            obs=("<button id='product-add-to-\n"
                 "cart-btn' bid='1443'>\n"
                 "  Add to Cart\n"
                 "</button>"),
            think=short(real_steps[3]["think"], 44, 10) if len(real_steps) > 3 else "",
            action=real_action(3, 'click("1443")'),
            note='Best-rating heuristic dominates the choice',
        ),
    ]

    fig, ax = plt.subplots(figsize=(14.5, 13.5))
    setup_blank_ax(ax, xlim=(0, 14.5), ylim=(0, 13.5))

    # title strip
    rounded_box(ax, 0.0, 12.95, 14.5, 0.45, fc="#0F172A", ec="#0F172A",
                lw=0.5, radius=0.04)
    ax.text(14.5/2, 13.18,
            "Real reasoning trace — Paradigm A treatment trial, range decoy "
            "(Claude Opus 4.6, Sony Cybershot triad)",
            ha="center", va="center", fontsize=12, weight="bold",
            color="white")

    row_h = 3.05
    top = 12.85
    for i, st in enumerate(steps):
        y = top - (i + 1) * row_h
        # step number badge
        ax.add_patch(plt.Circle((0.55, y + row_h - 0.50), 0.32,
                                facecolor="#0F172A", edgecolor="none"))
        ax.text(0.55, y + row_h - 0.50, str(i),
                ha="center", va="center", fontsize=12, weight="bold",
                color="white")
        ax.text(1.05, y + row_h - 0.40, st["t"],
                ha="left", va="center", fontsize=11, weight="bold",
                color="#0F172A")
        ax.text(1.05, y + row_h - 0.78, "» " + st["note"],
                ha="left", va="center", fontsize=9.5, color="#7C3AED",
                style="italic")
        # observation box
        rounded_box(ax, 1.05, y + 0.20, 3.95, row_h - 1.05,
                    fc="#F8FAFC", ec="#9CA3AF", lw=1.0,
                    title="Observation  (pruned HTML)", title_fs=9,
                    title_color="#374151", radius=0.05, pad=0.06)
        ax.text(1.20, y + row_h - 1.30, st["obs"],
                ha="left", va="top", fontsize=8.5, family="monospace",
                color="#0F172A")
        # think box (verbatim from the run)
        rounded_box(ax, 5.20, y + 0.20, 6.20, row_h - 1.05,
                    fc="white", ec="#0EA5E9", lw=1.2,
                    title="<think>  (verbatim model output)", title_fs=9,
                    title_color="#0369A1", radius=0.05, pad=0.06)
        ax.text(5.35, y + row_h - 1.30, st["think"],
                ha="left", va="top", fontsize=8.6, color="#0F172A",
                family="DejaVu Sans")
        # action box
        rounded_box(ax, 11.55, y + 0.20, 2.85, row_h - 1.05,
                    fc="white", ec="#F97316", lw=1.2,
                    title="<action>", title_fs=9,
                    title_color="#9A3412", radius=0.05, pad=0.06)
        ax.text(11.70, y + row_h / 2 - 0.30, st["action"],
                ha="left", va="center", fontsize=10.5, family="monospace",
                weight="bold", color="#9A3412")

    ax.text(14.5/2, 0.20,
            "Verbatim Claude Opus 4.6 reasoning. The agent visits all three tabs, then "
            "selects the option with the highest displayed rating (the decoy) "
            "— the choice rule the paper labels the 'best-rating heuristic'.",
            ha="center", va="center", fontsize=9.5, color="#374151",
            style="italic")

    fig.savefig(OUT / "agent_trace.png", dpi=200)
    fig.savefig(OUT / "agent_trace.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating figures into:", OUT)
    print(" - paradigm B effect (data-driven)")
    df = fig_paradigm_b_effect()
    print("   trials extracted:", len(df) if df is not None else "(cached)")
    print(" - paradigm A schematic")
    fig_paradigm_a_schematic()
    print(" - paradigm B schematic")
    fig_paradigm_b_schematic()
    print(" - framework overview")
    fig_framework_overview()
    print(" - proxy architecture")
    fig_proxy_architecture()
    print(" - URL anonymisation")
    fig_url_anon()
    print(" - stimulus examples")
    fig_stimulus_examples()
    print(" - prompting modes")
    fig_prompting_modes()
    print(" - intervention before/after")
    fig_intervention_before_after()
    print(" - geometries visual")
    fig_geometries_visual()
    print(" - intent variants")
    fig_intent_variants()
    print(" - agent trace")
    fig_agent_trace()
    print("Done.")
