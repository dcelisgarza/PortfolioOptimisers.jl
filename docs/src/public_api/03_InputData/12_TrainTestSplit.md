```@meta
Description = "Train/test split, public API of PortfolioOptimisers.jl: TrainTestSplit, TrainTestSplitResult, train_test_split."
```

# Train/test split

## Train/test splitting

A holdout split keeps the last observations as a test window and trains on the observations before them. It has two forms. [`train_test_split`](@ref) cuts data into a training part and a test part. [`TrainTestSplit`](@ref), alias `TTS`, is an estimator that does the same as the first step of a [`Pipeline`](@ref). Every fitted step after it then sees the training window only, and `fit_predict(pipe, data)` scores the pipeline on the test window.

A size is a number of rows, an `Integer`, or a fraction of the observations, an `AbstractFloat` in `(0, 1)`. If you give one size, the other window takes the remaining rows. If you give both, the rows between the two windows go to neither, which is an embargo.

Called with keywords, `train_test_split` returns a `(train, test)` tuple. Called with a `TrainTestSplit`, as `train_test_split(tts, data)`, it returns the [`TrainTestSplitResult`](@ref) that the split step of a pipeline makes. You can use one `TrainTestSplit` both inside and outside a pipeline.

## Types

```@docs
TrainTestSplit
TrainTestSplitResult
```

## Functions

```@docs
train_test_split
```
