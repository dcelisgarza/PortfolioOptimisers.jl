"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing estimator types.

Preprocessing estimators transform price or returns data (prices-to-returns conversion, missing-data filtering, imputation) under a fit/apply contract. Fitting one on training data with [`fit_preprocessing`](@ref) produces a result carrying any fitted state — imputation parameters, thresholds, and the selected asset universe — which [`apply_preprocessing`](@ref) then replays on unseen data so train and test windows are transformed consistently. Stateless preprocessing estimators carry no state, and applying them is equivalent to running them.

They are ordinary estimators: they know nothing about pipelines. A `Pipeline` drives them through the same fit/apply verbs any other caller would use.

All concrete preprocessing estimators should subtype one of the two data-level subtypes:

  - [`AbstractPricesPreprocessingEstimator`](@ref): consumes and produces price-level data ([`PricesResult`](@ref)).
  - [`AbstractReturnsPreprocessingEstimator`](@ref): consumes and produces returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
abstract type AbstractPreprocessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce price-level data.

Concrete subtypes transform a [`PricesResult`](@ref) into another [`PricesResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce returns-level data.

Concrete subtypes transform a [`ReturnsResult`](@ref) into another [`ReturnsResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing result types.

Preprocessing results are produced by [`fit_preprocessing`](@ref) on training data. They carry the fitted state needed to apply the same transformation to unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators produce results that carry only their configuration.

All concrete preprocessing results should subtype one of the two data-level subtypes, [`AbstractPricesPreprocessingResult`](@ref) or [`AbstractReturnsPreprocessingResult`](@ref), so a caller can replay each fitted transformation at the data level it applies to.

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

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
"""
abstract type AbstractPricesPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
"""
abstract type AbstractReturnsPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `x` counts as a missing observation in price-level data.

Price-level data stores absent observations either as `missing` or as `NaN` (the two conventions [`prices_to_returns`](@ref) already unifies).

# Algorithm

 1. Return `true` when `x` is `missing`.
 2. Return `true` when `x` is a `Number` and `isnan(x)` holds. The type test guards the call, because `isnan` is not defined for every value a price table can carry.
 3. Return `false` otherwise.

# Arguments

  - `x`: The value to test.

# Returns

  - `flag::Bool`: `true` when `x` is `missing` or a `NaN` number.

# Related

  - [`MissingDataFilter`](@ref)
  - [`PriceGapFill`](@ref)
"""
function is_missing_value(x)::Bool
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
"""
    fit_preprocessing(est::AbstractPreprocessingEstimator, data) -> fitted

Fit a preprocessing estimator on a data window and return the fitted object consumed by [`apply_preprocessing`](@ref).

The fitted object carries whatever state the transformation needs to be replayed consistently on unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators return themselves.

# Interfaces

Concrete preprocessing estimators must implement:

  - `fit_preprocessing(est::MyPreprocessing, data) -> fitted`: Compute the fitted state from the training window.
  - `apply_preprocessing(fitted, data) -> data′`: Transform a data window with the fitted state.

# Arguments

  - `est`: The preprocessing estimator.
  - `data`: The training data window ([`PricesResult`](@ref) or [`ReturnsResult`](@ref) depending on the estimator's level).

# Returns

  - `fitted`: The fitted object, typically an [`AbstractPreprocessingResult`](@ref) or the estimator itself when stateless.

# Related

  - [`apply_preprocessing`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
"""
function fit_preprocessing(est::AbstractPreprocessingEstimator, data)
    return throw(ArgumentError("$(typeof(est)) subtypes AbstractPreprocessingEstimator but does not implement fit_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′."))
end
"""
    apply_preprocessing(fitted, data) -> data′

Transform a data window with a fitted preprocessing object.

Applying the fitted object produced by [`fit_preprocessing`](@ref) on the training window to an unseen (test) window replays the *same* transformation — the same asset universe, the same imputation parameters — so train and test data stay consistent and no information flows from test to train.

# Arguments

  - `fitted`: The fitted object returned by [`fit_preprocessing`](@ref) (an [`AbstractPreprocessingResult`](@ref), or a stateless estimator).
  - `data`: The data window to transform.

# Returns

  - `data′`: The transformed data window.

# Related

  - [`fit_preprocessing`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
function apply_preprocessing(fitted::Union{<:AbstractPreprocessingEstimator,
                                           <:AbstractPreprocessingResult}, data)
    return throw(ArgumentError("$(typeof(fitted)) subtypes the preprocessing interface but does not implement apply_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′; a stateless estimator returns itself from fit_preprocessing and does the work here."))
end
export fit_preprocessing, apply_preprocessing
