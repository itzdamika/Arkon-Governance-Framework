# Auditor Dataset Pipeline

This pipeline generates synthetic AuditorAgent rows in batches of 10, validates each batch, saves approved rows immediately, and continues until the target row count is reached.

## Files
- `auditor_dataset_pipeline.py`: main pipeline.
- `generator_prompt.txt`: generator prompt.
- `validator_prompt.txt`: validator prompt.
- `.env.example`: required Azure environment variables.

## Install
```bash
pip install openai
```

## Run
```bash
python auditor_dataset_pipeline.py --target-rows 1000 --batch-size 10 --model gpt-5-mini --output-dir auditor_dataset_run
```

## Output behavior
- Saves raw generator and validator responses for every batch.
- Appends approved rows to `auditor_dataset.jsonl` immediately after each batch.
- Rewrites `auditor_dataset.json` after every saved batch so progress is never lost.
- Supports resume from checkpoint by default.
