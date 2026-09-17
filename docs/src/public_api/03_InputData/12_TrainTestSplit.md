```@meta
Description = "Train/test split, public API of PortfolioOptimisers.jl: TrainTestSplit, TrainTestSplitResult, train_test_split."
```

# Train/test split

## Train/test splitting

A **holdout split** reserves the tail of the time-ordered observations as a test window and trains on the head. It comes in two forms: the free function [`train_test_split`](@ref), which cuts data into a train/test pair, and the estimator [`TrainTestSplit`](@ref) (alias `TTS`), which carries the protocol *inside* a [`Pipeline`](@ref) as its first step — so every fitted step downstream sees the training window alone, and `fit_predict(pipe, data)` evaluates on the held-out window in one line.

Sizes are row counts (`Integer`) or fractions of the observations (`AbstractFloat` in `(0, 1)`). Giving one side makes the other its complement; giving both **embargoes** the rows between the two windows. See `docs/adr/0031-holdout-split-as-a-pipeline-step.md`.

The keyword form returns a bare `(train, test)` tuple; the estimator form, `train_test_split(tts, data)`, returns the same [`TrainTestSplitResult`](@ref) a pipeline's split step produces, so one configured holdout can be reused inside and outside a pipeline.

## Types

```@docs
TrainTestSplit
TrainTestSplitResult
```

## Functions

```@docs
train_test_split
```
