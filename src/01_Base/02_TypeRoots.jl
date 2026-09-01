"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator types.

All custom estimators should subtype `AbstractEstimator`.

Estimators consume data to estimate parameters or models. Some estimators may utilise different algorithms. These can range from simple implementation details that don't change the result much but may have different numerical characteristics, to entirely different methodologies or algorithms yielding different results.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all algorithm types.

All algorithms should subtype `AbstractAlgorithm`.

Algorithms are often used by estimators to perform specific tasks. These can be in the form of simple implementation details to entirely different procedures for estimating a quantity.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all result types.

All result objects should subtype `AbstractResult`.

Result types encapsulate the outcomes of estimators. This makes dispatch and usage more straightforward, especially when the results encapsulate a wide range of information.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for dynamically computed observation weight estimators.

`DynamicAbstractWeights` subtypes are used when observation weights must be computed from data (rather than supplied directly as a numeric vector). They are passed to estimators that accept an `ObsWeights` argument and evaluated at fit time.

# Interfaces

In order to implement a new dynamic observation weight estimator which will work seamlessly with the library, subtype `DynamicAbstractWeights` with all necessary parameters struct, and implement the following methods:

  - `get_observation_weights(w::DynamicAbstractWeights, X::VecNum; kwargs...) -> StatsBase.AbstractWeights`: Returns observation weights for a 1D vector `X`.
  - `get_observation_weights(w::DynamicAbstractWeights, X::MatNum; dims::Int = 1, kwargs...) -> StatsBase.AbstractWeights`: Returns observation weights for a 2D matrix `X`, with `dims` specifying the dimension along which to compute weights.

## Arguments

  - `w`: Subtype of `DynamicAbstractWeights` with all necessary parameters.
  - $(arg_dict[:X_Xv])
  - `dims`: Dimension along which to compute weights for a 2D matrix `X`.
  - `kwargs...`: Additional keyword arguments passed to the weight computation function.

## Returns

  - `w::StatsBase.AbstractWeights`: Observation weights for the input data `X`.

# Examples

We can create a dummy dynamic observation weight estimator as follows:

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

Both methods must be dispatched on the concrete subtype, as above — never on `DynamicAbstractWeights` itself, which would capture every other subtype too.

Implementing only one of the two arities is the mistake to avoid. Rather than silently computing an unweighted result, the unimplemented shape raises [`ObservationWeightsError`](@ref) and names the methods to write:

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
$(DocStringExtensions.TYPEDSIGNATURES)

Make estimators, algorithms, and results behave as length-1 iterables, returning the object itself on the first iteration and `nothing` thereafter.

This is what lets a caller write one loop over a value that may be a single estimator or a vector of them, without a branch on which it received.

# Algorithm

 1. Return `nothing` when `state` is above `1`, which ends the iteration.
 2. Otherwise return the pair of `obj` and the next state.

# Arguments

  - `obj`: The estimator, algorithm or result to iterate.
  - `state = 1`: Iteration state. Only `1` yields a value.

# Returns

  - `nothing` after the first iteration.
  - `(obj, state + 1)` on the first iteration.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                 <:AbstractResult}, state = 1)
    return state > 1 ? nothing : (obj, state + 1)
end
Base.length(::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult})::Int = 1
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Index into estimators, algorithms, and results as length-1 containers. Only index `1` is valid; any other index throws `BoundsError`.

# Algorithm

 1. Return `obj` when `i` is `1`.
 2. Otherwise throw a `BoundsError` naming `obj` and `i`.

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
function Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                  <:AbstractResult}, i::Int)
    return i == 1 ? obj : throw(BoundsError(obj, i))
end
