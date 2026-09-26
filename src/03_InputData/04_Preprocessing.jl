"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing estimator types.

A preprocessing estimator transforms price or returns data in two steps. [`fit_preprocessing`](@ref) fits it on a training window and returns a fitted object. The fitted object holds the state of the transform, such as the per-asset values of a gap fill, a row threshold or the selected asset universe. [`apply_preprocessing`](@ref) applies the fitted object to a window and fits nothing on that window, so the data of a test window never changes the transform. A stateless estimator is its own fitted object.

A preprocessing estimator does not depend on a pipeline. A [`Pipeline`](@ref) calls the same two functions that any other caller calls.

The data level of an estimator decides which slot of a pipeline the estimator reads and writes. A concrete estimator that reads and writes one level subtypes one of the two data-level subtypes:

  - [`AbstractPricesPreprocessingEstimator`](@ref): reads and writes a [`PricesResult`](@ref).
  - [`AbstractReturnsPreprocessingEstimator`](@ref): reads and writes a [`ReturnsResult`](@ref).

An estimator that changes the level, or that works at both levels, subtypes `AbstractPreprocessingEstimator` directly. [`PricesToReturns`](@ref) reads prices and writes returns, and [`TrainTestSplit`](@ref) splits a window of either level. Each has its own methods of [`run_step`](@ref) and [`apply_fitted_step`](@ref), because a pipeline refuses a direct subtype that has no `run_step` method.

# Interfaces

To implement a new preprocessing estimator, subtype `AbstractPricesPreprocessingEstimator` or `AbstractReturnsPreprocessingEstimator`, and implement the following methods:

## `fit_preprocessing`

  - `fit_preprocessing(est::MyPreprocessing, data) -> fitted`: Fit the state of the transform on the training window `data`. A stateless estimator returns `est`.

## `apply_preprocessing`

  - `apply_preprocessing(fitted, data) -> data′`: Transform the window `data` with `fitted`, the object that `fit_preprocessing` returned.

A subtype that does not implement a method gets the fallback of that method, which throws an `ArgumentError` that names the subtype.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`apply_preprocessing`](@ref)
"""
abstract type AbstractPreprocessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce price-level data.

A concrete subtype transforms a [`PricesResult`](@ref) into another [`PricesResult`](@ref). A [`Pipeline`](@ref) fits the estimator on the prices in its prices slot, and writes the transformed prices back to the same slot.

# Interfaces

To implement a new price-level estimator, subtype `AbstractPricesPreprocessingEstimator` and implement the following methods:

## `fit_preprocessing`

  - `fit_preprocessing(est::MyPricesPreprocessing, pr::PricesResult) -> MyPricesPreprocessingResult`: Fit the state of the transform on the training prices `pr`. A stateless estimator returns `est`.

## `apply_preprocessing`

  - `apply_preprocessing(res::MyPricesPreprocessingResult, pr::PricesResult) -> PricesResult`: Transform the price window `pr` with the fitted state of `res`.

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
  - [`PriceGapFill`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce returns-level data.

A concrete subtype transforms a [`ReturnsResult`](@ref) into another [`ReturnsResult`](@ref). A [`Pipeline`](@ref) fits the estimator on the returns in its returns slot, and writes the transformed returns back to the same slot.

# Interfaces

To implement a new returns-level estimator, subtype `AbstractReturnsPreprocessingEstimator` and implement the following methods:

## `fit_preprocessing`

  - `fit_preprocessing(est::MyReturnsPreprocessing, rd::AbstractReturnsResult) -> MyReturnsPreprocessingResult`: Fit the state of the transform on the training returns `rd`. A stateless estimator returns `est`.

## `apply_preprocessing`

  - `apply_preprocessing(res::MyReturnsPreprocessingResult, rd::AbstractReturnsResult) -> AbstractReturnsResult`: Transform the returns window `rd` with the fitted state of `res`.

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
  - [`AbstractAssetSelector`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing result types.

[`fit_preprocessing`](@ref) returns a preprocessing result from a training window. The result holds the fitted state that [`apply_preprocessing`](@ref) needs to do the same transform on another window, such as the per-asset values of a gap fill, a row threshold or the selected asset universe. A stateless estimator returns itself and makes no result. A [`TrainTestSplitResult`](@ref) is not a preprocessing result, because it holds the two windows of one split and applies to no other window.

A concrete result subtypes the data-level subtype that matches its estimator, [`AbstractPricesPreprocessingResult`](@ref) or [`AbstractReturnsPreprocessingResult`](@ref). A [`Pipeline`](@ref) reads that subtype to find the slot that the result transforms.

# Interfaces

To implement a new preprocessing result, subtype `AbstractPricesPreprocessingResult` or `AbstractReturnsPreprocessingResult`, and implement the following method:

  - `apply_preprocessing(res::MyPreprocessingResult, data) -> data′`: Transform the window `data` with the fitted state of `res`.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
"""
abstract type AbstractPreprocessingResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to price-level data ([`PricesResult`](@ref)).

A [`Pipeline`](@ref) applies a result of this type to the prices in its prices slot, and passes a returns window through with no change.

# Interfaces

To implement a new price-level result, subtype `AbstractPricesPreprocessingResult` and implement the following method:

  - `apply_preprocessing(res::MyPricesPreprocessingResult, pr::PricesResult) -> PricesResult`: Transform the price window `pr` with the fitted state of `res`.

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`PriceGapFillResult`](@ref)
  - [`MissingDataFilterResult`](@ref)
"""
abstract type AbstractPricesPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to returns-level data ([`ReturnsResult`](@ref)).

A [`Pipeline`](@ref) applies a result of this type to the returns in its returns slot, and passes a price window through with no change.

# Interfaces

To implement a new returns-level result, subtype `AbstractReturnsPreprocessingResult` and implement the following method:

  - `apply_preprocessing(res::MyReturnsPreprocessingResult, rd::AbstractReturnsResult) -> AbstractReturnsResult`: Transform the returns window `rd` with the fitted state of `res`.

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
  - [`AssetSelectorResult`](@ref)
"""
abstract type AbstractReturnsPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `x` counts as a missing observation in price-level data.

A price table marks an absent observation with `missing` or with `NaN`. [`unify_gaps`](@ref) writes every absence as `NaN`, but a [`PricesResult`](@ref) that a caller builds by hand can still hold `missing`, so this function accepts both.

# Algorithm

 1. Return `true` when `x` is `missing`.
 2. Return `true` when `x` is a `Number` and `isnan(x)` holds. The type test comes first because `isnan` has no method for a value that is not a number, such as a `String`.
 3. Return `false` otherwise.

# Arguments

  - `x`: The value to test.

# Returns

  - `flag::Bool`: `true` when `x` is `missing` or a `NaN` number. An infinite number is not missing.

# Related

  - [`unify_gaps`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`PriceGapFill`](@ref)
"""
function is_missing_value(x)::Bool
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
"""
    fit_preprocessing(est::AbstractPreprocessingEstimator, data) -> fitted

Fit a preprocessing estimator on a training window, and return the fitted object that [`apply_preprocessing`](@ref) reads.

The fitted object holds the state that the transform needs to apply to another window, such as the per-asset values of a gap fill, a row threshold or the selected asset universe. A stateless estimator, such as [`PricesToReturns`](@ref), returns itself.

The method for an `AbstractPreprocessingEstimator` is a fallback that throws. Each concrete estimator implements its own method, and [`AbstractPreprocessingEstimator`](@ref) states what that method must do.

# Arguments

  - `est`: The preprocessing estimator.
  - `data`: The training window, a [`PricesResult`](@ref) or a [`ReturnsResult`](@ref) as the data level of `est` needs.

# Validation

  - An estimator whose type does not implement this method throws an `ArgumentError` that names the type.

# Returns

  - `fitted`: The fitted object. It is an [`AbstractPreprocessingResult`](@ref), or the estimator itself when the estimator is stateless. A [`TrainTestSplit`](@ref) returns a [`TrainTestSplitResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`apply_preprocessing`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`partial_fit_transform`](@ref): the online form, whose data-less `fit_preprocessing(est)` reads the fitted object out of the estimator's state.
"""
function fit_preprocessing(est::AbstractPreprocessingEstimator, data)
    return throw(ArgumentError("$(typeof(est)) subtypes AbstractPreprocessingEstimator but does not implement fit_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′."))
end
"""
    apply_preprocessing(fitted, data) -> data′

Transform a data window with a fitted preprocessing object.

`fitted` is the object that [`fit_preprocessing`](@ref) returned for a training window. On another window it does the same transform, with the same asset universe and the same per-asset values, and it fits nothing on that window. So no data of a test window reaches the fit.

The method for an `AbstractPreprocessingEstimator` or an `AbstractPreprocessingResult` is a fallback that throws. Each concrete fitted object implements its own method.

# Arguments

  - `fitted`: The object that [`fit_preprocessing`](@ref) returned, an [`AbstractPreprocessingResult`](@ref) or a stateless estimator.
  - `data`: The data window to transform.

# Validation

  - A fitted object whose type does not implement this method throws an `ArgumentError` that names the type.

# Returns

  - `data′`: The transformed window, at the data level of `data`. [`PricesToReturns`](@ref) changes the level. It takes a [`PricesResult`](@ref) and returns a [`ReturnsResult`](@ref).

# Related

  - [`fit_preprocessing`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
function apply_preprocessing(fitted::Union{<:AbstractPreprocessingEstimator,
                                           <:AbstractPreprocessingResult}, data)
    return throw(ArgumentError("$(typeof(fitted)) subtypes AbstractPreprocessingEstimator or AbstractPreprocessingResult but does not implement apply_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′; a stateless estimator returns itself from fit_preprocessing and does the work here."))
end
export fit_preprocessing, apply_preprocessing
public AbstractPreprocessingEstimator, AbstractPricesPreprocessingEstimator,
       AbstractReturnsPreprocessingEstimator, AbstractPreprocessingResult,
       AbstractPricesPreprocessingResult, AbstractReturnsPreprocessingResult
