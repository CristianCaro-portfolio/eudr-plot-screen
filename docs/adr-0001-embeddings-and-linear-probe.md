# ADR 0001: Precomputed embeddings and a linear probe

## Context

The task is to tell, for a plot of land, whether forest was cleared between the end of 2020
and now. The usual ways to do that are (a) difference a spectral index between two cloud
free composites, (b) train a segmentation network on image time series, or (c) run a
foundation model over the imagery and classify its output.

All three start from raw Sentinel scenes. In the Amazon that means dozens of cloudy scenes per
tile and year, a compositing step that is easy to get wrong, and for (b) and (c) a GPU.

TESSERA publishes its output instead of only its weights: one 128 dimensional vector per 10 m
pixel per year, 2017 to 2025, global, built from the full year of Sentinel-1 radar and
Sentinel-2 optical data. The vectors sit in a public Zarr store and geotessera reads just the
pixels of a bounding box.

## Decision

Use the published TESSERA v1.1 embeddings for 2020 and 2025 as the only imagery input, and
put a standardised logistic regression on top of `[before, after, after - before]`.

## Why

- No compositing, no cloud masking, no GPU. An 11 km cell with two years of embeddings and its
  labels is a 20 to 30 second read and the whole experiment runs on two CPU cores.
- Radar is already inside the vector, so a cloudy year is not a missing year.
- A linear probe is the standard way to measure what a frozen representation knows. If a
  linear model works, the signal is in the embedding and not in a clever head.
- The fitted model is 1,153 numbers. It is stored as JSON, so it can be read, diffed and
  loaded without unpickling code from a repository.
- Measured in this repo, at a 5% tolerated error and the same guarantee for all three: the
  probe leaves 8.4% of plots for a person, the same embeddings without labels (cosine drift)
  leave 43.3%, and a dNBR baseline from Sentinel-2 composites leaves 79.4%.

## What was given up

- Annual resolution. This cannot say in which month a clearing happened and it is not an alert
  system. Loss late in the last year is only partly visible.
- Dependence on one provider and one model version. Embeddings from different TESSERA
  versions do not mix, so the version is recorded with every cube and the probe has to be
  refitted when it changes.
- A boosted tree model would probably rank a little better. It was not tried because the
  linear probe already leaves little to review and keeps the model auditable.
- The embeddings are CC BY-SA, which is fine for this use but has to be carried along with any
  redistributed sample.
