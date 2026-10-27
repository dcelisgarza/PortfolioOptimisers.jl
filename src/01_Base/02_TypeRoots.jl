"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator types.

All custom estimators must subtype `AbstractEstimator`.

An estimator reads data and estimates a parameter or a model from it. An estimator can hold an algorithm. One algorithm can differ from another only in its numerical behaviour, or it can use a different method that gives a different result.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all algorithm types.

All algorithms must subtype `AbstractAlgorithm`.

An estimator holds an algorithm to select how it does one task. The algorithm can change a numerical detail, or it can select a different procedure to estimate the quantity.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all result types.

All result objects must subtype `AbstractResult`.

A Result holds the output of an estimator. A function that reads the output dispatches on the type of the Result, and it reads each quantity from a named field.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the state of an incremental fit.

All partial-fit state objects must subtype `AbstractPartialFitState`.

A partial-fit state holds the quantities that an estimator needs to fold one more observation into an estimate, and the estimator does not read the sample again. The state subtypes [`AbstractResult`](@ref), so it gets the length-1 iteration protocol and the pretty `show`. No other part of the library reads a state. Each consumer reads an ordinary Result, and the batch verb called with no data first turns the state into one. A partial-fit state is the one kind of Result that an estimator holds.

# Interfaces

To implement a new partial-fit state, subtype `AbstractPartialFitState` and implement the two methods below.

## Merging two states

  - `merge_states(a::MyState, b::MyState) -> MyState`: Combines the states of two disjoint blocks of observations into the state of the concatenated block.

### Arguments

  - `a`: The state of the first block.
  - `b`: The state of the second block.

### Returns

  - `state::MyState`: The state that the two blocks give when the estimator fits them as one block.

### Algorithm

 1. Call [`assert_mergeable_states`](@ref) on the pair. It refuses two states of different types, and two states over different numbers of assets.
 2. Refuse each other mismatch that the family cannot merge. An exponentially weighted pair must also have the same decay, because two decays give the same observation two different weights.
 3. Fold the observation count, the mean and the second-moment accumulator with [`chan_merge`](@ref), and build the merged state from the folded values. A family whose state is not a sufficient statistic for its own block refuses the pair at this step, with a message that gives the reason. No fold can recover a quantity that the state did not record.

## Copying a state

  - `Base.copy(x::MyState) -> MyState`: Returns a fresh state whose arrays alias none of the arrays of `x`.

[`partial_fit`](@ref) calls this method, and the value semantics of that verb depend on it. The estimator that the caller gives keeps the state that it held, whatever [`partial_fit!`](@ref) then writes into the copy. A state that shares one array with its copy breaks this promise for the quantity in that array alone. No other quantity shows the defect, so it is hard to find.

### Arguments

  - `x`: The state to copy.

### Returns

  - `state::MyState`: A fresh state, equal to `x`, whose arrays are fresh.

### Algorithm

 1. Call the constructor of the state by name. Pass `copy` of each array field, and pass each scalar field unchanged. The method names its constructor and does not find it by reflection, as the prior results do. A family that the library does not know then gets a `MethodError` that names this method, and not a state that reflection built.

# Related

  - [`AbstractResult`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
  - [`chan_merge`](@ref)
"""
abstract type AbstractPartialFitState <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation estimators.

This file declares it, and not the file of the rest of its family, because an estimator of an earlier layer binds a slot to it. The Return Forecast family reads a fold split, and that family loads before the optimisation layer that holds the other cross-validation types.

# Related

  - [`CrossValidationResult`](@ref)
  - [`OptimisationCrossValidationEstimator`](@ref)
  - [`NonOptimisationCrossValidationEstimator`](@ref)
"""
abstract type CrossValidationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for dynamically computed observation weight estimators.

A subtype of `DynamicAbstractWeights` computes the observation weights from the data, and the caller does not give them as a numeric vector. The caller passes it to an estimator that takes an `ObsWeights` argument, and the estimator evaluates it when it fits.

# Interfaces

To implement a new dynamic observation weight estimator, subtype `DynamicAbstractWeights` with a struct that holds its parameters, and implement the two methods below.

  - `get_observation_weights(w::DynamicAbstractWeights, X::VecNum; kwargs...) -> StatsBase.AbstractWeights`: Returns the observation weights for a vector `X`.
  - `get_observation_weights(w::DynamicAbstractWeights, X::MatNum; dims::Int = 1, kwargs...) -> StatsBase.AbstractWeights`: Returns the observation weights for a matrix `X`. `dims` is the dimension that holds the observations.

## Arguments

  - `w`: The subtype of `DynamicAbstractWeights` that holds the parameters.
  - $(arg_dict[:X_Xv])
  - `dims`: The dimension of a matrix `X` that holds the observations.
  - `kwargs...`: More keyword arguments for the weight computation.

## Returns

  - `w::StatsBase.AbstractWeights`: The observation weights for the input data `X`.

# Examples

A minimal dynamic observation weight estimator:

```jldoctest
julia> struct MyWeights{T} <: PortfolioOptimisers.DynamicAbstractWeights
           half_life::T
           function MyWeights(half_life::Integer)
               if half_life < one(half_life)
                   throw(DomainError(half_life, \"half_life must be an integer greater than zero\"))
               end
               return new{typeof(half_life)}(half_life)
           end
       end

julia> function MyWeights(; half_life::Integer = 5)
           return MyWeights(half_life)
       end
MyWeights

julia> function PortfolioOptimisers.get_observation_weights(w::MyWeights,
                                                            X::PortfolioOptimisers.VecNum;
                                                            kwargs...)
           lambda = 2^(-inv(w.half_life))
           return eweights(1:length(X), lambda; scale = true)
       end

julia> function PortfolioOptimisers.get_observation_weights(w::MyWeights,
                                                            X::PortfolioOptimisers.MatNum;
                                                            dims::Int = 1, kwargs...)
           lambda = 2^(-inv(w.half_life))
           return eweights(1:size(X, dims), lambda; scale = true)
       end

julia> PortfolioOptimisers.get_observation_weights(MyWeights(), 1:10)
10-element Weights{Float64, Float64, Vector{Float64}}:
 1.0207079199119523e-8
 7.88499313633082e-8
 6.091176089370138e-7
 4.705448122809607e-6
 3.63496994859362e-5
 0.00028080229942667527
 0.002169204490777577
 0.016757156662950766
 0.12944943670387588
 1.0

julia> PortfolioOptimisers.get_observation_weights(MyWeights(), ones(3, 10); dims = 2)
10-element Weights{Float64, Float64, Vector{Float64}}:
 1.0207079199119523e-8
 7.88499313633082e-8
 6.091176089370138e-7
 4.705448122809607e-6
 3.63496994859362e-5
 0.00028080229942667527
 0.002169204490777577
 0.016757156662950766
 0.12944943670387588
 1.0
```

Both methods must dispatch on the concrete subtype, as above. A method on `DynamicAbstractWeights` itself catches every other subtype too.

An implementation must give both methods. A call on a shape that has no method does not compute an unweighted result. It raises [`ObservationWeightsError`](@ref), and the message names the methods to write.

```jldoctest
julia> struct PartialWeights <: PortfolioOptimisers.DynamicAbstractWeights end

julia> function PortfolioOptimisers.get_observation_weights(w::PartialWeights,
                                                            X::PortfolioOptimisers.VecNum;
                                                            kwargs...)
           return eweights(1:length(X), 0.5; scale = true)
       end

julia> PortfolioOptimisers.get_observation_weights(PartialWeights(), 1:3)
3-element Weights{Float64, Float64, Vector{Float64}}:
 0.25
 0.5
 1.0

julia> PortfolioOptimisers.get_observation_weights(PartialWeights(), ones(3, 10))
ERROR: ObservationWeightsError: PartialWeights is a DynamicAbstractWeights with no `get_observation_weights` method for a 2-dimensional input of size (3, 10). Implement `get_observation_weights(w::PartialWeights, X::VecNum; kwargs...)` and/or `get_observation_weights(w::PartialWeights, X::MatNum; dims::Int = 1, kwargs...)`, or pass a `StatsBase.AbstractWeights` instead (or `nothing` to compute unweighted). See the `DynamicAbstractWeights` docstring for a worked example.
Stacktrace:
[...]
```

# Related

  - [`ObsWeights`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`ObservationWeightsError`](@ref)
  - [`get_observation_weights`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
abstract type DynamicAbstractWeights <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all portfolio optimisation estimators.

All optimisers and optimisation components must subtype `AbstractOptimisationEstimator`, so that the optimisation methods dispatch on them.

# Interfaces

`AbstractOptimisationEstimator` declares no method of its own. It has the default method of [`port_opt_view`](@ref), which returns the estimator unchanged. It has two subtypes. Subtype [`BaseOptimisationEstimator`](@ref) for a configuration that an optimiser holds, and [`OptimisationEstimator`](@ref) for an estimator that [`optimise`](@ref) runs.

This file declares the root, and not the file of the optimisers, because a field bound earlier in the load order names it. `VarianceFraction.w0` accepts a [`NonFiniteAllocationOptimisationEstimator`](@ref) as its reference portfolio. `src/11_UncertaintySets/` declares `VarianceFraction`, and it loads before `src/17_Optimisation/`. The library prefers a type bound to a runtime check, so the types that the bound names moved to this file, and the bound stayed. [`CrossValidationEstimator`](@ref) is in this file for the same reason.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
"""
abstract type AbstractOptimisationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce portfolio weights.

All optimisers that `optimise` can run must subtype `OptimisationEstimator`.

# Interfaces

To implement a new optimiser, subtype `OptimisationEstimator`, give it an `fb` field, and implement the method below.

## `_optimise`

  - `_optimise(opt::MyOptimiser, rd::ReturnsResult, args...; kwargs...) -> OptimisationResult`: Solves the problem that `opt` states over the data in `rd`, and returns the result type of the optimiser.

### Arguments

  - `opt`: An instance of the concrete subtype.
  - `rd`: The returns data.
  - `args...`, `kwargs...`: The arguments that [`optimise`](@ref) forwards.

### Returns

  - `res::OptimisationResult`: The result. Its `retcode` decides whether [`optimise`](@ref) moves on to the fallback.

## The `fb` field

[`optimise`](@ref) reads `opt.fb` to go along the fallback chain, so every subtype has the field. It holds the next optimiser to try, or `nothing` to end the chain.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`AbstractOptimisationEstimator`](@ref)
"""
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce continuous (non-integer) portfolio weights.

# Interfaces

`NonFiniteAllocationOptimisationEstimator` adds no method to [`OptimisationEstimator`](@ref). It marks the optimisers whose weights are continuous. The cross-validation and meta-optimisation entry points accept an optimiser only when it has this mark. [`OptE_Opt`](@ref) is the union of these optimisers and of their results.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
"""
abstract type NonFiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Make estimators, algorithms, and results iterate as length-1 collections. The first iteration returns the object itself, and the next returns `nothing`.

A caller can then write one loop over a value that is a single estimator or a vector of estimators, with no branch on which of the two it got.

# Algorithm

 1. Return `nothing` when `state` is above `1`, which ends the iteration.
 2. Otherwise return the pair of `obj` and the next state.

# Arguments

  - `obj`: The estimator, algorithm or result to iterate.
  - `state = 1`: The iteration state. Only `1` gives a value.

# Returns

  - `nothing` after the first iteration.
  - `(obj, state + 1)` on the first iteration.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function Base.iterate(obj::Union{AbstractEstimator, AbstractAlgorithm, AbstractResult},
                      state = 1)
    return state > 1 ? nothing : (obj, state + 1)
end
Base.length(::Union{AbstractEstimator, AbstractAlgorithm, AbstractResult})::Int = 1
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Index estimators, algorithms, and results as length-1 containers. Only index `1` is valid, and each other index throws a `BoundsError`.

# Algorithm

 1. Return `obj` when `i` is `1`.
 2. Otherwise throw a `BoundsError` that names `obj` and `i`.

# Arguments

  - `obj`: The estimator, algorithm or result to index.
  - `i::Int`: The index. Only `1` is valid.

# Validation

  - `i == 1`. Any other index raises a `BoundsError`.

# Returns

  - `obj`: The object itself.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function Base.getindex(obj::Union{AbstractEstimator, AbstractAlgorithm, AbstractResult},
                       i::Int)
    return i == 1 ? obj : throw(BoundsError(obj, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a calibrated quantity from the data that a prior result holds, so that the quantity refits each time the sample changes.

The quantity is a tail probability, a deformation parameter, an ambiguity radius, an Esfahani-Kuhn tail weight, a norm ceiling, a Spanned Shrinkage or an Orthogonal Forecast Scale. This file declares the root with the other roots, because the slots of a [`CrossSectionalFactorPrior`](@ref) bind to it before the calibration rules load.

All concrete subtypes must subtype one of the families under this root, not the root itself. `# Related` names the families. A plain number in the place of a rule is the quantity itself, with no calibration.

The name of a rule is the name of the method that it uses. The name gets the quantity as a suffix only when another rule already has the method word alone. [`ScenarioCount`](@ref), [`EntropyBudget`](@ref), [`HillTailDecay`](@ref), [`RadialTailDecay`](@ref), [`TailTermParity`](@ref) and [`EffectiveAssetFloor`](@ref) name a method and no quantity. Five names contain the quantity, each for a reason. [`RateSignificance`](@ref) and [`RateRadius`](@ref) use one method for two quantities, so neither can have the word `Rate` alone. [`DimensionalRateRadius`](@ref) adds a prefix to the same stem. [`ConcentrationRadius`](@ref) and [`DualNormRadius`](@ref) each have the name of a mathematical object, and the method word alone names the object and not the rule.

A rule gives a default to each keyword that can have one, so a call with no arguments constructs the rule. Two rules give no default, because their keyword is the full content of the rule and no value fits every sample. [`ScenarioCount`](@ref) and [`EntropyBudget`](@ref) are those two. The keyword of each is `nothing` by default, which is not a value of the quantity. A call with no arguments then fails, and the message names the quantity, the reason that it has no default, and a value to start from.

A rule states the method and nothing more. The slot states the quantity: `alpha` is the lower tail, `kappa` is the deformation parameter, `r` is an ambiguity radius and `l2c` is a norm ceiling. The caller writes the rule alone. The `Num_` bound of each slot names the one family that has a meaning in that slot, so the bound refuses a rule of another family at construction.

A Calibration Rule is not a [`DeferredQuantity`](@ref), and the two mechanisms do not share code. The library fits a Deferred Quantity and reads the quantity from the fit. A rule fits nothing. It reads the sample size and the moments that the prior result already holds. A rule also reads the effective observation weights, which [`resolve_slot`](@ref) does not pass. A rule resolves through [`resolve_calibration_slot`](@ref), and [`calibration_slots`](@ref) declares it. [`assert_calibrated_slots`](@ref) refuses it at a value-level entry point.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`AbstractSpannedShrinkageCalibrationAlgorithm`](@ref)
  - [`AbstractOrthogonalForecastScaleCalibrationAlgorithm`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`DeferredQuantity`](@ref)
"""
abstract type AbstractCalibrationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the centring of an exponentially weighted moment. The centring selects the location that the estimator subtracts from each return to get a deviation.

The four exponentially weighted moment estimators hold a centring in their `centring` field: [`ExpWeightedVariance`](@ref), [`ExpWeightedCovariance`](@ref), [`RegimeAdjustedExpWeightedVariance`](@ref) and [`RegimeAdjustedExpWeightedCovariance`](@ref). [`EstimatedCentring`](@ref) estimates the location and corrects the bias that the estimate puts in each deviation. [`PreCentred`](@ref) takes the returns as deviations from a mean of zero. [`ZeroStartCentring`](@ref) takes the location of a recursion from zero that is not divided by its weight. This file declares the root with the other roots, because the estimators bind their field to it before the members load.

# Interfaces

A centring is a marker for dispatch, and it holds no data. A new centring needs a method of each verb that the members implement: [`centring_lag`](@ref), [`centring_deviation_mask`](@ref), [`centring_location!`](@ref), [`centring_report_location`](@ref), [`centring_factor`](@ref), [`centring_overlap`](@ref), [`centring_lag_records`](@ref), [`centring_lag_factor`](@ref) and [`centring_law`](@ref).

The bias of a regime statistic is a moment of the law of the deviations, so a new centring names its law with [`centring_law`](@ref). A centring that shares the law of [`PreCentred`](@ref) or of [`EstimatedCentring`](@ref) returns that centring, and the bias tables read its tables. A centring with a law of its own returns itself, and adds a method of [`regime_bias_table`](@ref) and of [`mahalanobis_level_bias`](@ref) for that law.

# Related

  - [`EstimatedCentring`](@ref)
  - [`PreCentred`](@ref)
  - [`ZeroStartCentring`](@ref)
"""
abstract type AbstractCentring <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the debias rule of a regime statistic. The rule selects whether the statistic corrects the bias of the estimated variance that it reads.

The two regime-adjusted estimators hold a debias rule in their `debias` field: [`RegimeAdjustedExpWeightedVariance`](@ref) and [`RegimeAdjustedExpWeightedCovariance`](@ref). This file declares the root with the other roots, because the estimators bind their field to it before the members load.

# Interfaces

A debias rule is a marker for dispatch, and it holds no data. A new rule needs a method of [`debiases`](@ref) and of [`reads_dependence`](@ref).

# Related

  - [`ExactDebias`](@ref)
  - [`LawDebias`](@ref)
  - [`RawStatistic`](@ref)
"""
abstract type AbstractRegimeDebias <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the floor of a HAC term. The rule selects whether the estimator floors each HAC-adjusted product at zero before the product enters the variance recursion.

The two regime-adjusted estimators hold a floor rule, or `nothing` for no floor, in their `hac_floor` field: [`RegimeAdjustedExpWeightedVariance`](@ref) and [`RegimeAdjustedExpWeightedCovariance`](@ref). The rule acts only where `hac_lags` is not `nothing`. On the covariance estimator it acts only on the variance of the separate correlation path, because the covariance of one decay floors no term. This file declares the root with the other roots, because the estimators bind their field to it before the members load.

# Interfaces

A floor rule is a marker for dispatch, and it holds no data. A new rule needs a method of [`hac_floor!`](@ref).

# Related

  - [`PerTermHacFloor`](@ref)
  - [`hac_floor!`](@ref)
"""
abstract type AbstractHacFloor <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the volatility that standardises a row of the separate correlation path under HAC. The rule selects whether the estimator divides the row by the volatility before or after the volatility updates on that row.

[`RegimeAdjustedExpWeightedCovariance`](@ref) holds the rule in its `hac_vol_before` field. The rule acts only where `hac_lags` is not `nothing` and `cor_decay` differs from `decay`. This file declares the root with the other roots, because the estimator binds its field to it before the members load.

# Interfaces

A rule is a marker for dispatch, and it holds no data. A new rule needs a method of [`volatility_before_update`](@ref).

# Related

  - [`VolatilityBeforeUpdate`](@ref)
  - [`VolatilityAfterUpdate`](@ref)
"""
abstract type AbstractHacVolatilityTiming <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the variance that standardises each product of the separate correlation path. The rule selects whether the estimator divides the variance state by its weight before the product divides by the variance.

[`RegimeAdjustedExpWeightedCovariance`](@ref) holds the rule in its `cor_vol` field. The rule acts only where `cor_decay` is not `nothing`, and only in the warm-up, where the weight of the variance state is below one. This file declares the root with the other roots, because the estimator binds its field to it before the members load.

# Interfaces

A rule is a marker for dispatch, and it holds no data. A new rule needs a method of [`correlation_variance`](@ref).

# Related

  - [`BiasCorrectedVolatility`](@ref)
  - [`RawVolatility`](@ref)
"""
abstract type AbstractCorrelationVolatility <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rule that says what the portfolio regime target does with a covariance block that is not positive definite.

[`PortfolioTarget`](@ref) holds the rule in its `indefinite` field. A HAC block in the warm-up, and a block of assets with different histories, can be indefinite. A direction with a small positive variance then gives a large statistic. This file declares the root with the other roots, because the target binds its field to it before the members load.

# Interfaces

A rule is a marker for dispatch, and it holds no data. A new rule needs a method of [`indefinite_block`](@ref).

# Related

  - [`RefuseIndefiniteWarmUp`](@ref)
  - [`ClipIndefiniteBlock`](@ref)
  - [`ScoreIndefiniteBlock`](@ref)
"""
abstract type AbstractIndefiniteBlockRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rule that computes the Mahalanobis bias factor from the level recursion over the directions of the estimate.

[`MahalanobisTarget`](@ref) holds the rule in its `recursion` field. The level recursion of [`mahalanobis_level_bias`](@ref) puts the mean log-determinant of each level inside the exponential. The mean leaves out the covariance of the log-determinants, and the rule selects whether the recursion carries that covariance. This file declares the root with the other roots, because the target binds its field to it before the members load.

# Interfaces

A rule is a marker for dispatch, and it holds no data. A new rule needs a method of [`mahalanobis_recursion_ratio`](@ref).

# Related

  - [`MeanLevelRecursion`](@ref)
  - [`GaussianLevelRecursion`](@ref)
"""
abstract type AbstractLevelRecursion <: AbstractAlgorithm end
public AbstractPartialFitState, DynamicAbstractWeights, AbstractOptimisationEstimator,
       OptimisationEstimator, NonFiniteAllocationOptimisationEstimator, AbstractCentring,
       AbstractRegimeDebias, AbstractHacFloor, AbstractHacVolatilityTiming,
       AbstractCorrelationVolatility, AbstractIndefiniteBlockRule, AbstractLevelRecursion
