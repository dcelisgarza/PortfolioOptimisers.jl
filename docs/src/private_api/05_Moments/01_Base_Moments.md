```@meta
Description = "Base moments, private API of PortfolioOptimisers.jl: AbstractExpectedReturnsAlgorithm, AbstractMomentAlgorithm, gap_fill_value, densify, robust_cov, …"
```

# Base moments: private API

## Abstract moment types and fallbacks

The abstract types below are the roots of the moment estimators. The functions are the steps that many moment estimators share, such as the calls to a `StatsBase` covariance estimator, the choice of the observation window and its weights, and the centring of the returns.

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

The five windowed estimators have one shape: [`WindowedExpectedReturns`](@ref), [`WindowedVariance`](@ref), [`WindowedCovariance`](@ref), [`WindowedCoskewness`](@ref) and [`WindowedCokurtosis`](@ref). Each wraps an inner estimator, keeps only the last observations of the sample, and passes every moment call to the inner estimator. The macro [`@windowed_estimator`](@ref) writes each of them from one declaration. It writes the struct, its constructor, its `factory` and `port_opt_view` methods, the methods that pass each moment call on, and their docstrings.

The entries below are the macro and the functions it calls when it expands. You use the five estimators, and never call these directly.

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

## Available-case co-moments

An available-case fit estimates each asset from the rows where its return is finite and the asset is active. The two functions below build the block that the available-case coskewness and cokurtosis read. A co-moment reads every deviation from the mean with [`FullMoment`](@ref), or only the negative deviations with [`SemiMoment`](@ref), and [`coverage_comoment_deviations`](@ref) is the one step where the two differ.

```@docs
coverage_comoment_deviations
coverage_comoment_block
```
