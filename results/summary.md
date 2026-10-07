# Results

Embeddings 2020 and 2025, reference GFC-2025-v1.13, seed 0.
Tolerated error 5% missed and 5% wrongly flagged, calibrated with 95% confidence and corrected for clustered plots.
Probe fitted on 179,982 pixels from the train areas (15.1% loss).

## Plots

| split | plots | deforested |
|---|---:|---:|
| calibration | 6,099 | 14.9% |
| test | 5,999 | 15.1% |
| transfer | 3,025 | 16.0% |

## Pixel level, per area

| area | method | AUC | average precision | IoU at 0.5 |
|---|---|---:|---:|---:|
| guaviare_miraflores | probe | 0.957 | 0.771 | 0.555 |
| guaviare_miraflores | cosine | 0.846 | 0.359 |  |
| guaviare_miraflores | dnbr | 0.756 | 0.261 |  |
| meta_tinigua | probe | 0.975 | 0.860 | 0.627 |
| meta_tinigua | cosine | 0.906 | 0.548 |  |
| meta_tinigua | dnbr | 0.774 | 0.339 |  |
| caqueta_solano | probe | 0.953 | 0.754 | 0.517 |
| caqueta_solano | cosine | 0.890 | 0.543 |  |
| caqueta_solano | dnbr | 0.779 | 0.347 |  |
| guaviare_retorno | probe | 0.964 | 0.536 | 0.390 |
| guaviare_retorno | cosine | 0.890 | 0.246 |  |
| guaviare_retorno | dnbr | 0.785 | 0.156 |  |
| meta_mapiripan | probe | 0.978 | 0.891 | 0.705 |
| meta_mapiripan | cosine | 0.935 | 0.641 |  |
| meta_mapiripan | dnbr | 0.711 | 0.411 |  |

## Plot ranking on the test plots

| method | AUC | average precision | plots without a score |
|---|---:|---:|---:|
| probe | 0.980 | 0.917 | 0.0% |
| cosine | 0.919 | 0.692 | 0.0% |
| dnbr | 0.806 | 0.484 | 0.3% |

## Triage, mean over 200 random calibration/test splits (5th to 95th percentile)

| method | tolerated error | cleared | flagged | review | missed | wrongly flagged | splits with missed above target |
|---|---:|---:|---:|---:|---:|---:|---:|
| probe | 10% | 78.6% | 21.4% | 0.0% (0.0% to 0.0%) | 4.9% (3.5% to 6.3%) | 8.5% (6.8% to 10.2%) | 0.0% |
| probe | 5% | 75.1% | 16.4% | 8.4% (6.5% to 11.1%) | 2.9% (1.5% to 4.6%) | 4.0% (2.9% to 5.1%) | 1.5% |
| probe | 2% | 52.5% | 12.1% | 35.5% (24.2% to 48.3%) | 0.6% (0.0% to 1.6%) | 1.5% (1.0% to 2.2%) | 1.5% |
| probe | 1% | 7.0% | 9.8% | 83.2% (37.9% to 91.5%) | 0.1% (0.0% to 0.5%) | 0.7% (0.4% to 1.1%) | 0.0% |
| cosine | 10% | 60.4% | 17.7% | 21.9% (18.4% to 25.2%) | 6.7% (3.9% to 9.6%) | 8.1% (6.3% to 9.9%) | 4.0% |
| cosine | 5% | 45.5% | 11.2% | 43.3% (36.7% to 50.0%) | 2.9% (1.1% to 4.7%) | 3.7% (2.6% to 5.0%) | 4.0% |
| cosine | 2% | 23.6% | 5.7% | 70.7% (59.6% to 76.9%) | 0.8% (0.2% to 1.8%) | 1.3% (0.7% to 1.9%) | 3.0% |
| cosine | 1% | 11.8% | 3.3% | 84.9% (77.1% to 96.9%) | 0.2% (0.0% to 0.7%) | 0.5% (0.2% to 1.0%) | 2.0% |
| dnbr | 10% | 28.9% | 13.9% | 57.2% (52.1% to 61.5%) | 7.0% (4.1% to 10.7%) | 8.2% (6.2% to 10.0%) | 9.5% |
| dnbr | 5% | 12.9% | 7.7% | 79.4% (73.9% to 83.1%) | 3.0% (1.2% to 5.5%) | 3.7% (2.6% to 4.8%) | 9.5% |
| dnbr | 2% | 3.6% | 2.9% | 93.5% (89.7% to 96.0%) | 0.8% (0.0% to 2.0%) | 1.3% (0.8% to 1.9%) | 4.5% |
| dnbr | 1% | 0.7% | 1.4% | 98.0% (95.6% to 99.2%) | 0.1% (0.0% to 0.8%) | 0.6% (0.3% to 1.0%) | 1.0% |

## Triage on the single split behind `plots.parquet` (seed 0)

| method | tolerated error | cleared | flagged | review | missed (cleared but deforested) | wrongly flagged |
|---|---:|---:|---:|---:|---:|---:|
| probe | 10% | 79.2% | 20.8% | 0.0% | 6.1% (55 of 905) | 7.9% |
| probe | 5% | 78.4% | 15.5% | 6.1% | 5.5% (50 of 905) | 3.1% |
| probe | 2% | 63.6% | 11.2% | 25.2% | 1.4% (13 of 905) | 1.0% |
| probe | 1% | 0.0% | 8.6% | 91.4% | 0.0% (0 of 905) | 0.4% |
| cosine | 10% | 62.1% | 17.1% | 20.8% | 7.3% (66 of 905) | 7.5% |
| cosine | 5% | 50.0% | 10.8% | 39.1% | 3.4% (31 of 905) | 3.5% |
| cosine | 2% | 22.4% | 5.0% | 72.7% | 0.8% (7 of 905) | 1.0% |
| cosine | 1% | 16.4% | 2.2% | 81.4% | 0.2% (2 of 905) | 0.3% |
| dnbr | 10% | 25.6% | 13.7% | 60.7% | 6.1% (55 of 905) | 7.5% |
| dnbr | 5% | 12.0% | 6.8% | 81.2% | 2.5% (23 of 905) | 2.9% |
| dnbr | 2% | 3.1% | 2.6% | 94.3% | 0.6% (5 of 905) | 1.1% |
| dnbr | 1% | 0.0% | 0.9% | 99.1% | 0.0% (0 of 905) | 0.2% |

## Probe calibrated on average: 200 splits at 5%

| rate | mean | 5th percentile | 95th percentile | splits above target |
|---|---:|---:|---:|---:|
| missed | 5.05% | 3.11% | 7.21% | 52.0% |
| wrongly flagged | 5.00% | 3.83% | 6.30% | 46.5% |
| review | 3.69% | 2.55% | 4.72% |  |

## Probe calibrated with 95% confidence, plots taken as independent: 200 splits at 5%

| rate | mean | 5th percentile | 95th percentile | splits above target |
|---|---:|---:|---:|---:|
| missed | 3.90% | 2.24% | 5.42% | 14.5% |
| wrongly flagged | 4.51% | 3.47% | 5.69% | 21.5% |
| review | 5.78% | 4.41% | 7.32% |  |

## Probe calibrated with 95% confidence, corrected for clustered plots: 200 splits at 5%

| rate | mean | 5th percentile | 95th percentile | splits above target |
|---|---:|---:|---:|---:|
| missed | 2.88% | 1.50% | 4.59% | 1.5% |
| wrongly flagged | 4.04% | 2.93% | 5.12% | 6.5% |
| review | 8.41% | 6.53% | 11.14% |  |

## Plot score: how to aggregate pixels

| aggregation | plot AUC | review share |
|---|---:|---:|
| mean of all pixels | 0.976 | 8.6% |
| mean of top 50 | 0.980 | 6.1% |
| single highest pixel | 0.967 | 12.6% |

## Area never seen by the model or the calibration

| thresholds | cleared | flagged | review | missed | wrongly flagged |
|---|---:|---:|---:|---:|---:|
| probe | 80.1% | 15.1% | 4.8% | 2.9% (14 of 485) | 1.5% |
| cosine | 68.1% | 12.9% | 18.9% | 1.4% (7 of 485) | 3.3% |
| dnbr | 27.4% | 2.5% | 70.1% | 14.6% (71 of 485) | 0.6% |
| probe_recalibrated_locally | 74.4% | 20.6% | 5.0% | 1.1% (3 of 268) | 4.4% |

## Label efficiency

| labelled pixels | seeds | plot AUC | review share (min to max) |
|---:|---:|---:|---|
| 300 | 3 | 0.957 | 21.8% (20.4% to 23.5%) |
| 3,000 | 3 | 0.970 | 14.6% (10.6% to 18.5%) |
| 30,000 | 3 | 0.979 | 7.7% (7.0% to 8.3%) |
| 179,982 | 1 | 0.980 | 6.1% (6.1% to 6.1%) |

## Error analysis

Wrongly flagged test plots: 159, of which 134 have some reference loss below the 0.5 ha line (median 27 pixels).

| main loss year | deforested test plots | cleared | review | flagged |
|---|---:|---:|---:|---:|
| 2021 | 287 | 4.9% | 6.6% | 88.5% |
| 2022 | 238 | 6.7% | 14.3% | 79.0% |
| 2023 | 62 | 12.9% | 14.5% | 72.6% |
| 2024 | 209 | 3.8% | 7.2% | 89.0% |
| 2025 | 109 | 3.7% | 8.3% | 88.1% |

## Precedents

Vote of the 5 most similar calibration plots, on 5,999 test plots: AUC 0.936, majority agrees with the reference on 92.6%. Inside the review zone (365 plots) the majority is right on 61.6%.

Runtime of the experiment on cached data: 398 s.
