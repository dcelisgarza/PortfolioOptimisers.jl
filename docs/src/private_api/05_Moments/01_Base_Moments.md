```@meta
Description = "Base moments, private API of PortfolioOptimisers.jl: AbstractExpectedReturnsAlgorithm, AbstractMomentAlgorithm, gap_fill_value, densify, …"
```

# Base moments: private API

## Abstract moment types and fallbacks

Some optimisations and constraints make use of summary statistics. These types and functions form the base for moment estimation in `PortfolioOptimisers.jl`.

They also provide generic fallbacks for the various functionality in the library.

```@docs
AbstractExpectedReturnsAlgorithm
AbstractMomentAlgorithm
gap_fill_value(::StatsBase.CovarianceEstimator)
densify
robust_cov
robust_cor
compat_cov
compat_cor
moment_window_and_weights
windowed_preamble
weighted_centre
demean_returns
```

## Windowed estimator generation

The five windowed estimators — [`WindowedExpectedReturns`](@ref), [`WindowedVariance`](@ref), [`WindowedCovariance`](@ref), [`WindowedCoskewness`](@ref) and [`WindowedCokurtosis`](@ref) — share one shape: wrap an inner estimator, restrict it to a trailing window, and forward every moment call to it. Each is generated from a single declaration by [`@windowed_estimator`](@ref), so the struct, its constructor, its `factory`/`port_opt_view` methods, its forwarding methods and all of their docstrings cannot drift apart.

The entries below are the macro and its expansion-time machinery. They are internal: callers use the five estimators, not these.

```@docs
WINDOWED_ESTIMATOR_KEYS
WINDOWED_ESTIMATOR_INPUTS
@windowed_estimator
windowed_parse_field
windowed_parse_forward
windowed_estimator_check_key
windowed_estimator_suggest
windowed_estimator_error
windowed_type_doc
windowed_method_ref
windowed_method_doc
windowed_method_def
```

## FullMoment and semi moments

Moments other than the expected return can be estimated using the entire spectrum of deviations (full), or only the deviations below a target (semi/downside). These types allow us to provide such functionality.

```@docs
coverage_comoment_deviations
coverage_comoment_block
```
