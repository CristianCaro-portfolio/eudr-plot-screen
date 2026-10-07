# ADR 0003: LanceDB for precedents

## Context

A reviewer who gets a plot in the review zone wants to know what similar cases turned out to
be. Each plot has a 128 dimensional change signature (mean embedding difference of its pixels,
weighted by the pixel probability), and the calibration plots have reference labels. That is
a nearest neighbour search over a few thousand vectors today and a few million if every
supplier plot of a commodity is stored.

## Decision

Store plots in LanceDB: one table with the vector, the decision, the score, the split and the
coordinates. Precedents are a cosine search filtered to labelled calibration plots.

## Why

- Embedded. It is a folder on disk and a Python import, so `plotscreen demo` builds the store
  in a temporary directory and CI needs no service container.
- The filter is applied before the vector search (`split = 'calibration'`), which matters
  here: a test plot must never be its own precedent.
- The table is Arrow on disk, so the same rows that back the search are readable from pandas,
  DuckDB or Polars for audits.
- It is the same store whether the table has six thousand rows (exact search, 8 ms per query
  measured here) or needs an ANN index later.

## Alternatives

- scikit-learn `NearestNeighbors`: fine for the experiment, but it is an in-memory index with
  no metadata filter and no persistence.
- pgvector: needs a running Postgres. Worth it when the plots already live in Postgres.
- FAISS: fast, but vectors only. Decisions and labels would live somewhere else and have to be
  kept in step by hand.

## What it is not

The precedent vote is context for a person, not a second classifier. On the test plots the
majority of the five nearest precedents agrees with the reference most of the time, but
inside the review zone, where it would matter, it is right much less often. The triage does
not use it.
