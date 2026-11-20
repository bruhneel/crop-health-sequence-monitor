# Patterns mined from training fields

4,180 season baskets; minimum support 4.0%; minimum confidence 55.0%; itemsets up to size 4.

These are co-occurrences in a synthetic dataset.

| Frequent itemset | Count | Support |
| --- | ---: | ---: |
| mild_season, no_intervention | 1,095 | 26.20% |
| drought_stress, dry_season | 893 | 21.36% |
| drought_stress, irrigation | 767 | 18.35% |
| humid_season, no_intervention | 671 | 16.05% |
| drought_stress, dry_season, irrigation | 668 | 15.98% |
| dry_season, irrigation | 668 | 15.98% |
| aphids_observed, humid_season | 555 | 13.28% |
| maize, no_intervention | 548 | 13.11% |
| chickpea, no_intervention | 509 | 12.18% |
| humid_season, scouting_visit | 501 | 11.99% |
| no_intervention, soybean | 497 | 11.89% |
| aphids_observed, scouting_visit | 483 | 11.56% |

| Association | Support | Confidence | Lift |
| --- | ---: | ---: | ---: |
| fungicide_applied, humid_season → leaf_rust, wheat | 4.57% | 86.43% | 13.09 |
| leaf_rust, wheat → fungicide_applied, humid_season | 4.57% | 69.20% | 13.09 |
| fungicide_applied, wheat → humid_season, leaf_rust | 4.57% | 87.61% | 12.50 |
| humid_season, leaf_rust → fungicide_applied, wheat | 4.57% | 65.19% | 12.50 |
| fungicide_applied → leaf_rust, wheat | 5.22% | 68.99% | 10.45 |
| leaf_rust, wheat → fungicide_applied | 5.22% | 78.99% | 10.45 |
| fungicide_applied → humid_season, leaf_rust, wheat | 4.57% | 60.44% | 10.35 |
| humid_season, leaf_rust, wheat → fungicide_applied | 4.57% | 78.28% | 10.35 |
| fungicide_applied → leaf_rust | 7.56% | 100.00% | 10.22 |
| leaf_rust → fungicide_applied | 7.56% | 77.26% | 10.22 |
| fungicide_applied, humid_season → leaf_rust | 5.29% | 100.00% | 10.22 |
| fungicide_applied, wheat → leaf_rust | 5.22% | 100.00% | 10.22 |
