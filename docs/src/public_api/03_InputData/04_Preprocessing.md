```@meta
Description = "Preprocessing, public API of PortfolioOptimisers.jl: fit_preprocessing, apply_preprocessing, AbstractPreprocessingEstimator, …"
```

# Preprocessing

## Preprocessing estimators

A preprocessing estimator transforms prices or returns in two steps. [`fit_preprocessing`](@ref) learns what the transform needs from a training window, such as the assets to keep, the values to impute or a threshold. [`apply_preprocessing`](@ref) applies that fitted result to another window, and learns nothing from it. The data of a test window never changes the transform.

A preprocessing estimator does not depend on a pipeline. A [`Pipeline`](@ref) calls these two functions on it, as it calls [`prior`](@ref) on a prior estimator and [`optimise`](@ref) on an optimiser.

```@docs
fit_preprocessing
apply_preprocessing
```

## Types

A new estimator subtypes `AbstractPricesPreprocessingEstimator` or `AbstractReturnsPreprocessingEstimator`. Its fitted result subtypes `AbstractPricesPreprocessingResult` or `AbstractReturnsPreprocessingResult`.

```@docs
PortfolioOptimisers.AbstractPreprocessingEstimator
PortfolioOptimisers.AbstractPricesPreprocessingEstimator
PortfolioOptimisers.AbstractReturnsPreprocessingEstimator
PortfolioOptimisers.AbstractPreprocessingResult
PortfolioOptimisers.AbstractPricesPreprocessingResult
PortfolioOptimisers.AbstractReturnsPreprocessingResult
```
