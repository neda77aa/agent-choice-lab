#!/usr/bin/env bash
# Run one geometry pair (no-warn + new-warn) for a single model+sample.
# Two runs back-to-back, ~1 hour total — fits inside the 1-hour Vertex TTL.
#
# Usage:
#   bash scripts/run_url_mask_pair.sh <agent_config> <sample_n> <geom>
#
# Geom values:
#   v2          popcorn, treat-medium (g=0.10) — main popcorn cell
#   v2_strong   popcorn, treat-strong  (g=0.08) — tightest gap
#   v2_weak     popcorn, treat-weak    (g=0.15) — loosest gap
#   v3          absolute-dominance (gamma=0.05)
#
# Examples:
#   bash scripts/run_url_mask_pair.sh vertex-claude-opus-4-6 2 v2
#   bash scripts/run_url_mask_pair.sh vertex-claude-opus-4-6 1 v2_strong
#   bash scripts/run_url_mask_pair.sh vertex-gemini-2-5-pro 3 v3
#   bash scripts/run_url_mask_pair.sh gpt-5-2 1 v2_weak
#
# Refresh `gcloud auth application-default login` before each pair for Vertex models.

set -euo pipefail

AGENT="${1:?agent config required, e.g. vertex-claude-opus-4-6 / gpt-5-2 / vertex-gemini-2-5-pro / vertex-deepseek-v3-2 / vertex-qwen-3-5}"
SAMPLE="${2:?sample number required, e.g. 2}"
GEOM="${3:?geometry required: v2 / v2_strong / v2_weak / v3}"

# Strip "vertex-" prefix for the on-disk model label
MODEL_LABEL="${AGENT#vertex-}"
EXPS=$(echo exp{0..73} | tr ' ' ',')
ROOT=results/url_mask_compare

case "${GEOM}" in
  v2)
    NOWARN_GROUP="experiment-decoy-size-newgeom-gs10-v5c-nourlwarn"
    WARN_GROUP="experiment-decoy-size-newgeom-gs10-v5c-warn-new"
    NOWARN_DIR="v2_nowarn"
    WARN_DIR="v2_warn_new"
    ;;
  v2_strong)
    NOWARN_GROUP="experiment-decoy-size-newgeom-gs10-v5c-nourlwarn-strong"
    WARN_GROUP="experiment-decoy-size-newgeom-gs10-v5c-warn-new-strong"
    NOWARN_DIR="v2_strong_nowarn"
    WARN_DIR="v2_strong_warn_new"
    ;;
  v2_weak)
    NOWARN_GROUP="experiment-decoy-size-newgeom-gs10-v5c-nourlwarn-weak"
    WARN_GROUP="experiment-decoy-size-newgeom-gs10-v5c-warn-new-weak"
    NOWARN_DIR="v2_weak_nowarn"
    WARN_DIR="v2_weak_warn_new"
    ;;
  v3)
    NOWARN_GROUP="experiment-decoy-size-newgeom-absdom-v5c-nourlwarn"
    WARN_GROUP="experiment-decoy-size-newgeom-absdom-v5c-warn-new"
    NOWARN_DIR="v3_nowarn"
    WARN_DIR="v3_warn_new"
    ;;
  *)
    echo "ERROR: geom must be 'v2', 'v2_strong', 'v2_weak', or 'v3'" >&2
    exit 1
    ;;
esac

echo "=== ${AGENT} sample_${SAMPLE} ${GEOM}_nowarn ==="
AGENTLAB_EXP_ROOT=${ROOT}/${NOWARN_DIR}/${MODEL_LABEL}/sample_${SAMPLE} \
  .venv/bin/python run.py --multirun \
    "+${NOWARN_GROUP}=${EXPS}" \
    agent=${AGENT} \
    agent/flags/obs=default_ax_tree_decoy

echo "=== ${AGENT} sample_${SAMPLE} ${GEOM}_warn_new ==="
AGENTLAB_EXP_ROOT=${ROOT}/${WARN_DIR}/${MODEL_LABEL}/sample_${SAMPLE} \
  .venv/bin/python run.py --multirun \
    "+${WARN_GROUP}=${EXPS}" \
    agent=${AGENT} \
    agent/flags/obs=default_ax_tree_decoy

echo "=== ${AGENT} sample_${SAMPLE} ${GEOM} PAIR COMPLETE ==="
