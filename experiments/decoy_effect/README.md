# Decoy Effect (LLM)

This is a decoy-effect replication using an LLM as the consumer agent. It compares the target option share between:

- `baseline`: target vs competitor
- `decoy`: target vs competitor vs a decoy dominated by the target

## Run

```bash
cd /Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab
source .venv/bin/activate
export OPENAI_API_KEY="..."
python experiments/decoy_effect/run_decoy.py --model gpt-4.1-mini --repeats 2
```

Outputs are written to:

- `/Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab/results/decoy_effect/decoy_results_*.csv`
- `/Users/neda/Desktop/UBC/PHD/LLM_research/agent-choice-lab/results/decoy_effect/decoy_summary_*.json`

## Notes

- If your account does not have access to `gpt-4.1-mini`, swap in a cheap model you do have access to.
- Increase `--repeats` to get a more stable estimate of the decoy effect.
- The output JSON also includes a simple chi-square test and raw choice counts.
