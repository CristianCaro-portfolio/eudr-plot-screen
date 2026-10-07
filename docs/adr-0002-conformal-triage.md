# ADR 0002: Three way triage with conformal thresholds

## Context

A plot score between 0 and 1 is not a decision. Someone has to pick a cut, and in a due
diligence setting the two errors are not symmetric: clearing a plot that was deforested is the
expensive one. A threshold of 0.5 on a model probability says nothing about how often that
happens, and it changes meaning every time the model is retrained.

There is also no reason to force a yes or no on every plot. A reviewer exists; the question
is how few plots can be sent to them.

## Decision

Calibrate two thresholds on labelled plots that the model never saw, one per class
(class-conditional split conformal prediction):

- `t_clear` from the deforested calibration plots, so that a deforested plot is cleared with
  probability at most alpha.
- `t_flag` from the clean calibration plots, so that a clean plot is flagged with probability
  at most beta.

Scores below `t_clear` are cleared, above `t_flag` are flagged, the rest go to review. When
both rules apply the plot is flagged. Plots with no score go to review.

By default the thresholds are chosen so the targets hold for this particular calibration set
with 95% confidence (`confidence: 0.95`), not just on average, and the calibration size is
first shrunk by a design effect because neighbouring plots are correlated
(`cluster_correction: true`).

## Why

- The guarantee does not depend on the model being good or well calibrated. A weaker score
  gives a wider review zone, not more missed plots. That is what makes the comparison between
  methods fair: all of them carry the same guarantee and differ only in workload.
- Calibrating each class on its own makes the error rates independent of how common
  deforestation is. A supplier portfolio has far fewer deforested plots than the frontier
  areas used here, and the thresholds still mean the same thing.
- The plain conformal guarantee is an average. Over 200 random splits the mean missed rate
  was 5.05% for a 5% target, which also means 52% of the splits were above it. A compliance
  team calibrates once, so an average is not what they need.
- With a 95% confidence level and plots treated as independent, 14.5% of splits were still
  above target. The reason is spatial: plots in the same 1 km block rise and fall together,
  so 907 deforested calibration plots carry the information of about 337 independent ones
  (Kish design effect, intraclass correlation from a one-way analysis of variance).
- With that correction 1.5% of splits are above target for missed plots and 6.5% for wrongly
  flagged ones. Review share goes from 3.7% (plain) to 8.4%. That trade is the default.
- It is a few dozen lines of order statistics. There is nothing to tune and nothing to retrain.

## What was given up

- The guarantee assumes new plots look like the calibration plots within each class. It is
  measured, not assumed, on an area the calibration never saw (see the README), and it should
  be recalibrated with local labels before use in a new region.
- The design effect only sees correlation inside 1 km blocks. Longer range structure is not
  modelled, and the split bundled with the repo (seed 0) is one of the few that ends above
  target: 5.5% missed.
- Calibration plots were all 4 ha squares. Polygons of very different size or shape are scored
  the same way but sit outside what was calibrated.
