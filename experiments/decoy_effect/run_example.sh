#!/bin/bash
# Example script to run one experiment with Gemini and Claude, then analyze.

# 1. Run the experiment for vertex-gemini-3-pro-preview
echo "Running Gemini 3 Pro Preview..."
python3 experiments/decoy_effect/run_literature_decoy.py \
  --agent vertex-gemini-3-pro-preview \
  --stimulus-id huber_1982_gambles_large \
  --samples 50

# 2. Run the experiment for vertex-claude-opus-4-6
echo "Running Claude Opus 4.6..."
python3 experiments/decoy_effect/run_literature_decoy.py \
  --agent vertex-claude-opus-4-6 \
  --stimulus-id huber_1982_gambles_large \
  --samples 50

# 3. Analyze the output CSVs in the results directory
echo "Analyzing and comparing against human results..."
python3 experiments/decoy_effect/analyze_literature_decoy.py \
  --csv-path results/literature_decoy \
  --out-dir results/literature_decoy/comparison_analysis

echo "Done! Check results/literature_decoy/comparison_analysis/report.md for the summary table."
