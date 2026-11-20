# Crop Health Sequence Monitor

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/9m5n3dabrm/badge.svg)](https://msrit.ask.thili.ai/b/9m5n3dabrm)

A local command-line demo that mines seasonal co-occurrences and ranks likely next items with a continued-pretrained GPT-2 model. **Synthetic sample data only.** All fields, observations, and action records are fictional. The program is an ML exercise, not agronomic advice.

## Install and run

Use Python 3.12 and Git from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# On Linux, install the CPU-only PyTorch wheel first:
# pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello
python -m app.cli generate-data
python -m app.cli mine
python -m app.cli train
python -m app.cli evaluate
python -m app.cli suggest --observed "wheat, aphids_observed, humid_season" --top 5
```

The first training run downloads the public [DistilGPT2 model and tokenizer](https://huggingface.co/distilbert/distilgpt2) into the Hugging Face cache without an API key. Once cached, use `HF_HUB_OFFLINE=1` for training or inference. The final model, tokenizer and bigram counts are Git-ignored under `artifacts/model/`. Keep `.env` local. The starter's optional hosted or Ollama chat helper is separate.

Run `HF_HUB_OFFLINE=1 pytest -q` to test with a tiny random model and local tokenizer. Use `python -m app.cli --help` for command and option details. Use `python -m app.cli interactive` to score several prefixes in one session. Type `/quit` to leave.

## Milestones

1. **Generate data.** Seed 42 produces 1,200 synthetic fields with four to six seasons, split by entire field into 840 training, 180 validation and 180 test fields. The generator samples persistent weather regimes, noisy crop rotations, crop- and weather-linked observations, and follow-up records. The vocabulary lists typed events in [app/core.py](app/core.py).
2. **Mine baskets.** A basket is an unordered set of events in one season. Mining pools baskets from training fields, with configurable `--min-support`, `--min-confidence`, and `--max-size`. The bare PyPI name had no release. The starter's [official thilimarketbasket GitHub dependency](https://github.com/thili-ai/thilimarketbasket) installed successfully, so the project uses and pins its bitset Apriori miner and rule generator.
3. **Train the LM and baseline.** A sequence preserves event order across seasons. Each event is encoded as one added special token, initialized from the model's embedding for its ordinary words. Continued pretraining starts from existing DistilGPT2 weights and updates them with full-vocabulary causal next-token loss. Training from scratch would start those weights randomly. We train on CPU for four epochs with AdamW and save the training-only smoothed bigram alongside the checkpoint. [Training log](results/training.json)
4. **Evaluate.** Compare both models on every proper prefix of the same test fields, including empty prefixes for first events. Evaluation checks the checkpoint's dataset hash and split seed. The bigram sees only the previous event, and DistilGPT2 uses the full prefix. Both distributions are normalized over the 19 event tokens. Perplexity excludes end-of-field targets. [Full results and mistakes](results/evaluation.md)
5. **Suggest.** Input is chronological, but rule matching treats the supplied items as a single basket. Output shows next-item probabilities, matching basket rules, and canned suggestions for reviewing the log. The latter neither come from the model nor recommend a treatment.

## Mined patterns

Support is the fraction of baskets containing the entire itemset. Confidence for A → B is support(A and B) divided by support(A). Lift divides confidence by the base support of B. A high lift indicates stronger co-occurrence than independence would imply in these baskets.

At 4% minimum support and 55% confidence the training data yielded 95 itemsets and 89 rules. The most supported multi-item patterns were mild season with no intervention (26.20%), drought stress with a dry season (21.36%), and drought stress with an irrigation record (18.35%). These are associations in the generator's output. [Pattern report](results/patterns.md)

## Held-out results

| Model | Top-1 | Top-3 | Top-5 | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| Continued DistilGPT2 | 60.67% | 86.87% | 95.49% | 3.084 |
| Smoothed bigram | 39.27% | 77.47% | 94.43% | 4.271 |

The LM improved top-1 accuracy by 21.40 percentage points over the bigram across 3,374 targets. Common errors included predicting mild conditions when the recorded event was aphids or drought stress, confusing a soybean rotation with chickpea, and predicting the next crop before a fallow entry. The generator supplies the structure the model learns. Real-world calibration, seasonal timing, and regional validity were not evaluated.

### Recorded query

```text
$ python -m app.cli suggest --observed "wheat, aphids_observed, humid_season" --top 5
Likely next items (synthetic model)
  scouting_visit           94.65%
  no_intervention           5.20%
  fungicide_applied         0.07%
  irrigation                0.05%
  humid_season              0.01%
```

The same run matched `aphids_observed, humid_season → scouting_visit` at 7.66% support, 57.66% confidence, and 2.61 lift. The highest-ranked item's canned follow-up was “Review or update the scouting log.” [Complete CLI session](results/example-session.txt)
