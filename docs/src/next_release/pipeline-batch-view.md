```@meta
Description = "A Pipeline views each step at the assets that a selector or a price filter keeps, so a value for each asset of the input reaches the right asset."
```

# [A Pipeline views each step at the assets that a universe step keeps](@id migration-next-pipeline-batch-view)

## [Results that move](@id migration-next-pipeline-batch-view-numbers)

- **A step after an asset selector or a `MissingDataFilter` takes the values of the assets that the selector keeps.** A step states a value for each asset over the universe of the input data, because the selector picks its assets at each fit. The previous release gave the step to the fit as it was stated. A vector `WeightBounds`, a `NegativeSkewness` with its own coskewness, and every other value for each asset then met fewer assets than values, and the fit raised a `DimensionMismatch`, or read the values of the first assets. `fit` and `cross_val_predict` of a `Pipeline` now view each such step at the kept assets, in the order of the returns data, as `cross_val_predict` of an optimiser with `cols` does. A step with no value for each asset gives the same result as before.
- **A nested `Pipeline` after a universe step views each of its steps too.** Its steps state their values over the universe of the input data of the outer `Pipeline`. The previous release fitted them as they were stated, and a vector `WeightBounds` in a nested optimiser raised a `DimensionMismatch`.
- **`fit(pipe)` with no data views the steps before and after the row owner too**, so `cross_val_predict` with an online scheme gives the weights of the batch walk for a pipeline whose optimiser follows a prior step.
