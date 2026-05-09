# Agent Choice Lab — Decoy Effect in LLM Web Agents

> Code and data for the paper *Do AI Agents Inherit the Decoy Effect?
> Evidence from Literature-Based Replications and a Real-World Web
> Environment.*

This repository contains two complementary studies:

1. **Study 1 — Literature replication.** A factorial replication of 46
   decoy stimuli from six published consumer-choice experiments
   (Huber 1982, Simonson 1989, Prelec 1997, Pettibone 2000, Kim 2005,
   Hedgcock 2009, Frederick 2014) across five frontier LLMs
   (GPT-5.2, Claude Opus 4.6, Gemini 2.5 Pro, DeepSeek V3.2, Qwen 3.5),
   with crossed temperature, prompting mode, and framing variants
   (~96K trials). Code under [`experiments/decoy_effect/`](experiments/decoy_effect/).

2. **Study 2 — Within-product size-decoy on a real web shop.** Live
   LLM agents shop on a local Magento storefront via the
   [ABxLab](https://github.com/PapayaResearch/abxlab) man-in-the-middle
   proxy. The medium size on a real product page is added or hidden
   under two pricing schemes (absolute-dominance as the canonical
   primary design; popcorn-style ladder pricing at three gap settings
   as a robustness check), crossed with a URL-warning toggle. We close
   a previously-undocumented browser-tab URL-leakage channel via an
   observation-layer mask in the agent prompt.

The repository is a fork of the upstream
[ABxLab framework](https://github.com/PapayaResearch/abxlab); see the
[Acknowledgements](#acknowledgements) section.

---

## Prerequisites

- Python 3.11
- Node.js + npm (Playwright dependency)
- Docker (to host the WebArena Magento container)
- `gcloud` SDK if you want to run Vertex-served models
  (Claude Opus, Gemini 2.5 Pro, DeepSeek V3.2, Qwen 3.5)

## Installation

```bash
conda create -n agent-choice-lab python=3.11
conda activate agent-choice-lab

pip install -r requirements.txt
cd ./agentlab && pip install -e . && cd ..

playwright install
```

Optional (used by a few standalone scripts; conflicts with `hydra-ray-launcher`
so install separately):

```bash
pip install dspy==2.6.27
```

## Environment setup

Create `.env` in the repo root:

```bash
# Local WebArena (OneStopMarket Magento Docker)
BASE_WEB_AGENT_URL="http://localhost:7770/"
SHOPPING="${BASE_WEB_AGENT_URL}"
WA_SHOPPING="${SHOPPING}"
# (other WA_* / *_URL vars are required by BrowserGym; safe to point them all
#  at the same base URL if you only run the shopping environment)

# Where experiment results live (per-trial pickles, summaries, CSVs)
AGENTLAB_EXP_ROOT="results"

# LLM API keys (only the ones you use)
OPENAI_API_KEY="..."
ANTHROPIC_API_KEY="..."

# Vertex AI (for Claude on Vertex, Gemini, DeepSeek, Qwen)
VERTEX_PROJECT="<your-gcp-project>"
VERTEX_LOCATION="global"
VERTEX_PARTNER_LOCATION="asia-southeast1"  # serving region for Anthropic-on-Vertex
```

For Vertex models, authenticate **once per ~1-hour session**:

```bash
gcloud auth application-default login
```

The Magento container ships with the WebArena release; bring it up and
verify `curl http://localhost:7770/` returns `200`.

---

## Study 2: running the size-decoy experiments

The full evaluation crosses two pricing schemes × URL-warning toggle:

| Scheme | Geometry | Treatment arms | Predicted effect |
|---|---|---|---|
| **Absolute dominance** (primary) | $P_M = (1+\gamma) P_L$, $\gamma=0.05$ | one | strongest — $M$ is strictly worse than $L$ on every dimension |
| **Popcorn ladder** (robustness) | $P_M = (1-g) P_L$, $g \in \{0.08, 0.10, 0.15\}$ | three | smaller, monotone in $g$ — $M$ is a plausible compromise |

Each treatment cell has both a `nowarn` and a `warn_new` variant of the
v5c "personal-shopping" intent (see Methods §5.3 of the paper).

### 1. Generate experiment configs (one-time)

The geometry CSVs and YAML configs already live under [`tasks/`](tasks/) and
[`conf/`](conf/). To regenerate:

```bash
# Geometry CSVs (idempotent — produces the committed files)
python scripts/select_size_decoy_geometries.py --preset v2 --g_s 0.10 \
    --output tasks/size_decoy_geometries_v2_gs10.csv

python scripts/select_size_decoy_geometries.py \
    --g_s 0.10 --qL_cap 1.5 --gap_levels treat-medium:-0.05 --s_anchor_gap 0.10 \
    --output tasks/size_decoy_geometries_v3.csv

# YAMLs are produced by scripts/generate_experiments_size_decoy.py, one
# call per (geometry × warning × gap) condition. See the script's
# --help for the full argument list; the in-repo dirs were generated
# with --add-url-warning (or omitted) and the v5c intent text.
```

### 2. Run a single (model, sample, geometry) pair

The `run_url_mask_pair.sh` helper runs both warning conditions
(`nowarn` + `warn_new`) for one geometry, fitting comfortably inside
a one-hour Vertex auth window:

```bash
# Geometry options: v2 (popcorn medium), v2_strong, v2_weak, v3 (absdom)
# For Vertex models, refresh `gcloud auth application-default login` first.
bash scripts/run_url_mask_pair.sh vertex-claude-opus-4-6  1 v3
bash scripts/run_url_mask_pair.sh gpt-5-2                 1 v2_strong
```

Output goes to `results/url_mask_compare/{cond}/{model}/sample_{N}/run-*/`.

### 3. Analyze

The analyzer walks every `(condition, model, sample)` cell, computes
within-product paired Δtarget, and prints three tables:

```bash
python scripts/analyze_url_mask_compare.py                # all samples
python scripts/analyze_url_mask_compare.py --samples 1,2  # subset
python scripts/analyze_url_mask_compare.py --out-csv analysis_output/url_mask.csv
```

The pooled-across-models table at the bottom is the primary
cross-model summary; per-model rows show the dose-response on popcorn
and the strict-dominance effect on absdom separately.

---

## URL-mask methodology

Two layers of URL leakage are mitigated:

| Layer | Where it lives | What it leaks | Mitigation |
|---|---|---|---|
| In-page DOM | The HTML the proxy returns to the agent | Wishlist hrefs, cart form action, base64 `uenc` parameter, body CSS classes | `anonymize_urls` intervention (rewrites every in-page reference) |
| Browser tab header | `obs["open_pages_urls"]` rendered into the prompt's "Currently open tabs:" block | Original Magento slug, e.g. `...-3-pound.html` | `mask_tab_url` flag on `ObsFlags` (rewrites the tab block URL to `http://<host>/product-anon.html/<slugified-title>.html` before the prompt is built) |

The structural mask is configured via the
[`conf/agent/flags/obs/default_ax_tree_decoy.yaml`](conf/agent/flags/obs/default_ax_tree_decoy.yaml)
flags variant (used by `run_url_mask_pair.sh`).

---

## Project structure

```
agent-choice-lab/
├── abxlab/                          # ABxLab framework (proxy + interventions)
│   ├── browser.py                   # MITM browser env
│   ├── choices/shop/                # Per-page intervention primitives
│   │   ├── options.py               # set_title, set_option_price, anonymize_urls, …
│   │   └── product.py               # ablate, set_rating, …
│   └── …
├── agentlab/                        # Modified AgentLab (LLM chat APIs, prompt elems)
│   ├── llm/chat_api.py              # LiteLLM + Vertex backends
│   └── agents/dynamic_prompting.py  # Tabs prompt element (URL mask lives here)
├── conf/                            # Hydra configs
│   ├── agent/                       # Per-model agent definitions
│   ├── benchmark/                   # ABxLab benchmark
│   ├── task/                        # Task definitions
│   ├── config.yaml                  # Top-level config
│   └── experiment-decoy-size-newgeom-{gs10,absdom}-v5c[-{warn-new,nourlwarn}][-{strong,weak}]/
│                                    # 10 dirs: 4 v3 cells + 6 v2 cells (one per gap × warning)
├── tasks/                           # Geometry CSVs + Magento product catalogue
│   ├── size_decoy_geometries_v2_gs10.csv   # popcorn (g=0.08, 0.10, 0.15)
│   ├── size_decoy_geometries_v3.csv        # absolute-dominance (γ=0.05)
│   ├── product_size_ladders-decoy.csv
│   └── products{,_expanded,_resolved}.csv
├── scripts/
│   ├── run_url_mask_pair.sh                   # run one (model, sample, geom) pair
│   ├── analyze_url_mask_compare.py            # primary analyzer (3-table report)
│   ├── analyze_size_decoy_runs.py             # legacy analyzer
│   ├── generate_experiments_size_decoy.py     # YAML generator
│   ├── select_size_decoy_geometries.py        # geometry CSV generator
│   ├── preprocess_size_decoy_results.py       # paper-time preprocessing
│   ├── render_human_experiments.py            # human-study screenshot renderer
│   ├── render_v2_html.py                      # v2 HTML renderer
│   ├── figgen_*.py                            # paper figures (4 generators)
│   └── make_paper_figures.py
├── experiments/decoy_effect/         # Study 1 (literature replication)
├── results/url_mask_compare/         # Study 2 outputs (the active dataset)
└── analysis_output/                  # Per-trial CSVs for the paper
```

---

## Visualizing trial traces

The upstream AgentLab ships [AgentXray](https://github.com/ServiceNow/AgentLab),
a Gradio-based browser for the per-step pickles. Point it at a results dir
and launch:

```bash
export AGENTLAB_EXP_ROOT=./results/url_mask_compare/v3_warn_new/claude-opus-4-6/sample_1
agentlab-xray
```

---

## Acknowledgements

This repository is a fork of [ABxLab](https://github.com/PapayaResearch/abxlab)
([Cherep et al., 2026](https://arxiv.org/abs/2509.25609)) adapted for the
within-product size-decoy paradigm and extended with a structural
observation-layer URL mask, two pricing geometries, and a 5-model evaluation
panel. We are grateful to the ABxLab authors for the man-in-the-middle proxy
framework, and to the [AgentLab](https://github.com/ServiceNow/AgentLab) /
[BrowserGym](https://github.com/ServiceNow/BrowserGym) teams for the
underlying agent-evaluation infrastructure.

If you use this repository in your research, please cite both the
upstream ABxLab paper and our paper (citation forthcoming).
