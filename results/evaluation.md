# Held-out next-item evaluation

180 fields and 3,374 proper prefixes (including empty prefixes).

| Model | Top-1 | Top-3 | Top-5 | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| Continued DistilGPT2 | 60.67% | 86.87% | 95.49% | 3.084 |
| Smoothed bigram | 39.27% | 77.47% | 94.43% | 4.271 |

Perplexity is over the fixed item vocabulary for both models. The evaluation counts only real next events and omits end-of-field tokens.

## Example predictions

| Prefix (last 8 displayed) | True event | LM top 3 | Bigram top 3 |
| --- | --- | --- | --- |
| cotton → drought_stress | dry_season | dry_season 95.0%; mild_season 3.5%; humid_season 1.4% | dry_season 85.3%; mild_season 7.7%; humid_season 6.6% |
| chickpea → drought_stress | dry_season | dry_season 95.2%; mild_season 3.8%; humid_season 0.9% | dry_season 85.3%; mild_season 7.7%; humid_season 6.6% |
| maize → aphids_observed → stem_borer | humid_season | humid_season 87.7%; drought_stress 7.3%; mild_season 2.3% | humid_season 69.2%; drought_stress 12.4%; mild_season 10.1% |
| cotton → drought_stress → dry_season → irrigation | fallow | maize 63.7%; fallow 16.7%; chickpea 14.7% | fallow 23.4%; soybean 23.2%; maize 19.9% |
| dry_season → irrigation → wheat → drought_stress → dry_season → irrigation → chickpea → drought_stress | mild_season | dry_season 98.2%; mild_season 1.3%; humid_season 0.4% | dry_season 85.3%; mild_season 7.7%; humid_season 6.6% |
| maize → aphids_observed | stem_borer | drought_stress 37.5%; stem_borer 20.5%; humid_season 16.0% | humid_season 34.9%; leaf_rust 21.2%; mild_season 16.2% |

## Most common LM errors

| True event | Predicted event | Count |
| --- | --- | ---: |
| aphids_observed | mild_season | 60 |
| soybean | chickpea | 59 |
| drought_stress | mild_season | 50 |
| scouting_visit | irrigation | 49 |
| fallow | maize | 46 |
| dry_season | drought_stress | 43 |
