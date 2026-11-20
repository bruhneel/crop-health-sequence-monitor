# Crop Health Sequence Monitor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/9m5n3dabrm/badge.svg)](https://ask.thili.ai/b/9m5n3dabrm)

This project, 'Crop Health Sequence Monitor', provides farmers with a powerful tool for agriculture. By analyzing sequences of crop health data, it identifies common patterns and suggests next actions, aiding in predicting co-occurring pests or optimizing crop rotations.

## What you are building

**Goal.** You will build a tool that analyzes agricultural sequences to identify common patterns and predict likely next events, helping farmers make informed decisions.

**Done when.** The project is complete when you have successfully applied frequent itemset mining and language model continued pretraining to agricultural sequences, and the model's next-item predictions have been evaluated.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define agricultural sequences (e.g., crop rotations, pest observations) as 'baskets'.**  
      Hint: Define sequences as 'baskets' for market basket analysis.
- [ ] **2. Mine frequent co-occurring patterns using `thilimarketbasket`.**  
      Hint: Use `thilimarketbasket` to find frequent co-occurring patterns.
- [ ] **3. Train a small LM on these sequences for next-item prediction.**  
      Hint: Train a small LM on the generated sequences for next-item prediction.
- [ ] **4. Evaluate model's predictions on new, unseen agricultural sequences.**  
      Hint: Evaluate the model's predictions on a held-out set of agricultural sequences.
- [ ] **5. Implement a CLI to query likely next items based on current observations.**  
      Hint: Implement a command-line interface to query the trained model.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Enhance the CLI to allow users to input a partial sequence and get multiple top-k predictions.
- Implement a feature to visualize the most frequent patterns found.
- Add a confidence score or probability alongside the predicted next items.
