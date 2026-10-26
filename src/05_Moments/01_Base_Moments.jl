"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the asset view of a covariance estimator, which is the estimator unchanged.

This is the fallback of [`port_opt_view`](@ref) for a covariance estimator that defines no method of its own. A covariance estimator of `StatsBase` holds no field that is indexed by asset, so its view is the estimator itself.

# Arguments

  - $(arg_dict[:ce])
  - `::Any`: The asset indices. The method ignores them.
  - `args...`: More arguments. The method ignores them.

# Returns

  - `ce::StatsBase.CovarianceEstimator`: The original covariance estimator.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function port_opt_view(ce::StatsBase.CovarianceEstimator, ::Any,
                       args...)::StatsBase.CovarianceEstimator
    return ce
end
"""
    factory(
        ce::StatsBase.CovarianceEstimator,
        args...;
        kwargs...
    ) -> StatsBase.CovarianceEstimator

Returns a covariance estimator unchanged when a factory call gives it new observation weights.

This is the fallback of [`factory`](@ref) for a covariance estimator that defines no method of its own. A covariance estimator of `StatsBase` holds no observation weights, so a factory call has nothing to replace.

# Arguments

  - $(arg_dict[:ce])
  - `args...`: More arguments. The method ignores them.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `ce::StatsBase.CovarianceEstimator`: The original covariance estimator.

# Related

  - [`factory`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function factory(ce::StatsBase.CovarianceEstimator, args...;
                 kwargs...)::StatsBase.CovarianceEstimator
    return ce
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all covariance estimator types.

Every concrete or abstract type that estimates a covariance matrix must subtype `AbstractCovarianceEstimator`.

# Interfaces

To implement a new covariance estimator, subtype `AbstractCovarianceEstimator`. Keep every parameter in a field of the struct, the observation weights among them. Then implement these methods:

## Covariance and correlation

  - `Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...) -> MatNum`: Estimates the covariance matrix.
  - `Statistics.cor(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...) -> MatNum`: Estimates the correlation matrix.

### Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - `kwargs...`: More keyword arguments, passed to the inner covariance estimator.

### Returns

  - $(ret_dict[:sigrho])

## Factory

  - `PortfolioOptimisers.factory(ce::AbstractCovarianceEstimator, w::PortfolioOptimisers.ObsWeights) -> AbstractCovarianceEstimator`: Builds a copy of the estimator that holds the new observation weights.

### Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

### Returns

  - $(ret_dict[:ce])

# Examples

This example defines a minimal covariance estimator.

```jldoctest
julia> struct MyCovarianceEstimator{T1} <: PortfolioOptimisers.AbstractCovarianceEstimator
           w::T1
           function MyCovarianceEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyCovarianceEstimator(;
                                      w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyCovarianceEstimator(w)
       end
MyCovarianceEstimator

julia> function PortfolioOptimisers.factory(::MyCovarianceEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyCovarianceEstimator(; w = w)
       end

julia> function Statistics.cov(est::MyCovarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           X = PortfolioOptimisers.dims_oriented(dims, X)
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           sigma = X' * X
           return sigma
       end

julia> function Statistics.cor(est::MyCovarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           X = PortfolioOptimisers.dims_oriented(dims, X)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           sigma = X' * X
           d = LinearAlgebra.diag(sigma)
           StatsBase.cov2cor!(sigma, sqrt.(d))
           return sigma
       end

julia> cov(MyCovarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
2×2 Matrix{Float64}:
 1.34  2.76
 2.76  5.7

julia> cor(MyCovarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
2×2 Matrix{Float64}:
 1.0       0.998664
 0.998664  1.0

julia> PortfolioOptimisers.factory(MyCovarianceEstimator(), StatsBase.Weights([1, 2, 3]))
MyCovarianceEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractMomentAlgorithm`](@ref)
"""
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all variance estimator types.

Every concrete or abstract type that estimates a variance must subtype `AbstractVarianceEstimator`.

# Interfaces

To implement a new variance estimator, subtype `AbstractVarianceEstimator`. Keep every parameter in a field of the struct, the observation weights among them. Then implement these methods:

## Variance and standard deviation

  - `Statistics.var(ve::AbstractVarianceEstimator, X::MatNum; kwargs...) -> ArrNum`: Estimates the variance of each asset.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::MatNum; kwargs...) -> ArrNum`: Estimates the standard deviation of each asset.
  - `Statistics.var(ve::AbstractVarianceEstimator, X::VecNum; kwargs...) -> Number`: Estimates the variance of the vector.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::VecNum; kwargs...) -> Number`: Estimates the standard deviation of the vector.

### Arguments

  - $(arg_dict[:ve])

  - `X`

      + $(arg_dict[:X])
      + $(arg_dict[:Xv])

  - `kwargs...`: More keyword arguments, passed to the expected returns estimator.

### Returns

  - $(arg_dict[:X])

      + $(ret_dict[:stdvar])

  - $(arg_dict[:Xv])

      + $(ret_dict[:stdvarnum])

## Covariance and correlation

`Statistics.cov(ve::AbstractVarianceEstimator, X::MatNum; kwargs...)` and `Statistics.cor(ve::AbstractVarianceEstimator, X::MatNum; kwargs...)` always throw a `MethodError`. A variance estimator computes one marginal variance for each asset and no relation between two assets, so it cannot compute either matrix.

## Factory

  - `PortfolioOptimisers.factory(ve::AbstractVarianceEstimator, w::PortfolioOptimisers.ObsWeights) -> AbstractVarianceEstimator`: Builds a copy of the estimator that holds the new observation weights.

### Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:ow])

### Returns

  - $(ret_dict[:ve])

# Examples

This example defines a minimal variance estimator.

```jldoctest
julia> struct MyVarianceEstimator{T1} <: PortfolioOptimisers.AbstractVarianceEstimator
           w::T1
           function MyVarianceEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyVarianceEstimator(;
                                    w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyVarianceEstimator(w)
       end
MyVarianceEstimator

julia> function PortfolioOptimisers.factory(::MyVarianceEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyVarianceEstimator(; w = w)
       end

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           X = PortfolioOptimisers.dims_oriented(dims, X)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           sigma = LinearAlgebra.diag(X' * X)
           return isone(dims) ? reshape(sigma, 1, :) : reshape(sigma, :, 1)
       end

julia> function Statistics.std(est::MyVarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           X = PortfolioOptimisers.dims_oriented(dims, X)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           sigma = sqrt.(LinearAlgebra.diag(X' * X))
           return isone(dims) ? reshape(sigma, 1, :) : reshape(sigma, :, 1)
       end

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           return sum(abs2, X)
       end

julia> function Statistics.std(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           return sqrt(sum(abs2, X))
       end

julia> var(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
1×2 Matrix{Float64}:
 1.34  5.7

julia> std(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
1×2 Matrix{Float64}:
 1.15758  2.38747

julia> var(MyVarianceEstimator(), [1.0 0.3 0.5; 2.0 0.7 1.1]; dims = 2)
2×1 Matrix{Float64}:
 1.34
 5.7

julia> var(MyVarianceEstimator(), [1.0, 0.3, 0.5])
1.34

julia> PortfolioOptimisers.factory(MyVarianceEstimator(), StatsBase.Weights([1, 2, 3]))
MyVarianceEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Statistics.cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`Statistics.cor(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
@define_pretty_show(AbstractCovarianceEstimator)
function factory_child(v::AbstractCovarianceEstimator, args...; kwargs...)
    return factory(v, args...; kwargs...)
end
function factory_child(v::AbstractArray{<:AbstractCovarianceEstimator}, args...; kwargs...)
    return [factory_child(vi, args...; kwargs...) for vi in v]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes a covariance matrix from the observations of a [`SampleBufferState`](@ref) with the batch verb.

An estimator in an [`Online`](@ref) wrapper holds a refit Sample Buffer and not a family state, so it has no exact fold. Its estimate is the batch fit over the rows of the buffer. An uncapped buffer holds every observation folded so far, and a capped buffer holds the last `max_history` of them.

The method does no [`assert_partial_fittable`](@ref) check of its own. The batch verb accepts configurations that an incremental fold refuses, and the wrapper gives a caller access to them.

The buffer also keeps the masks of each observation that it folded. The method gives them to the batch verb, so a wrapped estimator under a [`CoveragePolicy`](@ref) returns the same value as an unwrapped one. The buffer keeps the implied volatilities and the `ivpa` of each fold too, so a wrapped [`ImpliedVolatility`](@ref) reads its surface over the same rows.

# Arguments

  - $(arg_dict[:ce])
  - `state`: The buffer to read.
  - `kwargs...`: More keyword arguments, passed to the batch verb. `strict` is one of them.

# Returns

  - `sigma::MatNum`: Covariance matrix of the observations the buffer holds.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.cov(ce::AbstractCovarianceEstimator, state::SampleBufferState;
                        kwargs...)
    return Statistics.cov(ce, sample_buffer(state); sample_buffer_kwargs(state)...,
                          implied_volatility_buffer_kwargs(state)..., kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes a correlation matrix from the observations of a [`SampleBufferState`](@ref) with the batch verb.

An estimator in an [`Online`](@ref) wrapper holds a refit Sample Buffer and not a family state, so it has no exact fold. Its estimate is the batch fit over the rows of the buffer. An uncapped buffer holds every observation folded so far, and a capped buffer holds the last `max_history` of them.

The method does no [`assert_partial_fittable`](@ref) check of its own. The batch verb accepts configurations that an incremental fold refuses, and the wrapper gives a caller access to them.

The buffer also keeps the masks of each observation that it folded. The method gives them to the batch verb, so a wrapped estimator under a [`CoveragePolicy`](@ref) returns the same value as an unwrapped one. The buffer keeps the implied volatilities and the `ivpa` of each fold too, so a wrapped [`ImpliedVolatility`](@ref) reads its surface over the same rows.

# Arguments

  - $(arg_dict[:ce])
  - `state`: The buffer to read.
  - `kwargs...`: More keyword arguments, passed to the batch verb. `strict` is one of them.

# Returns

  - `rho::MatNum`: Correlation matrix of the observations the buffer holds.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.cor(ce::AbstractCovarianceEstimator, state::SampleBufferState;
                        kwargs...)
    return Statistics.cor(ce, sample_buffer(state); sample_buffer_kwargs(state)...,
                          implied_volatility_buffer_kwargs(state)..., kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the variance of each asset from the observations of a [`SampleBufferState`](@ref) with the batch verb.

An estimator in an [`Online`](@ref) wrapper holds a refit Sample Buffer and not a family state, so it has no exact fold. Its estimate is the batch fit over the rows of the buffer. An uncapped buffer holds every observation folded so far, and a capped buffer holds the last `max_history` of them.

The method does no [`assert_partial_fittable`](@ref) check of its own. The batch verb accepts configurations that an incremental fold refuses, and the wrapper gives a caller access to them.

The buffer also keeps the masks of each observation that it folded. The method gives them to the batch verb, so a wrapped estimator under a [`CoveragePolicy`](@ref) returns the same value as an unwrapped one.

# Arguments

  - $(arg_dict[:ve])
  - `state`: The buffer to read.

# Returns

  - `sigma2::ArrNum`: Variance of the observations the buffer holds, one entry per asset.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.var(ve::AbstractVarianceEstimator, state::SampleBufferState)
    return Statistics.var(ve, sample_buffer(state); sample_buffer_kwargs(state)...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the standard deviation of each asset from the observations of a [`SampleBufferState`](@ref) with the batch verb.

An estimator in an [`Online`](@ref) wrapper holds a refit Sample Buffer and not a family state, so it has no exact fold. Its estimate is the batch fit over the rows of the buffer. An uncapped buffer holds every observation folded so far, and a capped buffer holds the last `max_history` of them.

The method does no [`assert_partial_fittable`](@ref) check of its own. The batch verb accepts configurations that an incremental fold refuses, and the wrapper gives a caller access to them.

The buffer also keeps the masks of each observation that it folded. The method gives them to the batch verb, so a wrapped estimator under a [`CoveragePolicy`](@ref) returns the same value as an unwrapped one.

# Arguments

  - $(arg_dict[:ve])
  - `state`: The buffer to read.

# Returns

  - `sigma::ArrNum`: Standard deviation of the observations the buffer holds, one entry per asset.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.std(ve::AbstractVarianceEstimator, state::SampleBufferState)
    return Statistics.std(ve, sample_buffer(state); sample_buffer_kwargs(state)...)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all expected returns estimator types.

Every concrete or abstract type that estimates expected returns must subtype `AbstractExpectedReturnsEstimator`.

# Interfaces

To implement a new expected returns estimator, subtype `AbstractExpectedReturnsEstimator`. Keep every parameter in a field of the struct, the observation weights among them. Then implement these methods:

## Expected returns

  - `Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum; kwargs...) -> ArrNum`: Estimates the expected returns.

### Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:X])
  - `kwargs...`: More keyword arguments for the estimator.

### Returns

  - $(ret_dict[:mu])

## Factory

  - `PortfolioOptimisers.factory(me::AbstractExpectedReturnsEstimator, w::PortfolioOptimisers.ObsWeights) -> AbstractExpectedReturnsEstimator`: Builds a copy of the estimator that holds the new observation weights.

### Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:ow])

### Returns

  - $(ret_dict[:me])

# Examples

```jldoctest
julia> struct MyExpectedReturnsEstimator{T1} <:
              PortfolioOptimisers.AbstractExpectedReturnsEstimator
           w::T1
           function MyExpectedReturnsEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyExpectedReturnsEstimator(;
                                           w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyExpectedReturnsEstimator(w)
       end
MyExpectedReturnsEstimator

julia> function PortfolioOptimisers.factory(::MyExpectedReturnsEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyExpectedReturnsEstimator(; w = w)
       end

julia> function Statistics.mean(est::MyExpectedReturnsEstimator, X::PortfolioOptimisers.MatNum;
                                dims::Int = 1, kwargs...)
           X = PortfolioOptimisers.dims_oriented(dims, X)
           w = isnothing(est.w) ? fill(one(eltype(X)), size(X, 1)) : est.w
           X = X .* w
           mu = sum(X; dims = 1) / sum(w)
           return isone(dims) ? reshape(mu, 1, :) : reshape(mu, :, 1)
       end

julia> mean(MyExpectedReturnsEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1]; dims = 2)
3×1 Matrix{Float64}:
 1.5
 0.5
 0.8

julia> PortfolioOptimisers.factory(MyExpectedReturnsEstimator(), StatsBase.Weights([1, 2, 3]))
MyExpectedReturnsEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the expected returns from the observations of a [`SampleBufferState`](@ref) with the batch verb.

An estimator in an [`Online`](@ref) wrapper holds a refit Sample Buffer and not a family state, so it has no exact fold. Its estimate is the batch fit over the rows of the buffer. An uncapped buffer holds every observation folded so far, and a capped buffer holds the last `max_history` of them.

The method does no [`assert_partial_fittable`](@ref) check of its own. The batch verb accepts configurations that an incremental fold refuses, and the wrapper gives a caller access to them.

The buffer also keeps the masks of each observation that it folded. The method gives them to the batch verb, so a wrapped estimator under a [`CoveragePolicy`](@ref) returns the same value as an unwrapped one.

# Arguments

  - $(arg_dict[:me])
  - `state`: The buffer to read.

# Returns

  - `mu::VecNum`: Expected returns of the observations the buffer holds, `assets × 1`.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.mean(me::AbstractExpectedReturnsEstimator, state::SampleBufferState)
    return Statistics.mean(me, sample_buffer(state); sample_buffer_kwargs(state)...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the asset view of an expected returns estimator, which is the estimator unchanged.

This is the fallback of [`port_opt_view`](@ref) for an expected returns estimator that defines no method of its own. An estimator that holds a field indexed by asset needs a method of its own.

# Arguments

  - $(arg_dict[:me])
  - `::Any`: The asset indices. The method ignores them.
  - `args...`: More arguments. The method ignores them.

# Returns

  - `me::AbstractExpectedReturnsEstimator`: The original expected returns estimator.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
function port_opt_view(me::AbstractExpectedReturnsEstimator, ::Any,
                       args...)::AbstractExpectedReturnsEstimator
    return me
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Return Forecast Estimator types.

A Return Forecast Estimator makes a Return Forecast. A Return Forecast is the part of the expected return of an asset that the factor mean does not hold. The caller gives it, or the estimator fits it from Descriptors.

The estimator reads an Asset Panel and a fitted [`CrossSectionalFactorModel`](@ref). It implements [`return_forecast`](@ref) and never `mean(me, X)`. A Return Forecast Estimator is not an expected returns estimator. The two families have separate roots, so no `me` field accepts a Return Forecast Estimator.

Every concrete type that makes a Return Forecast must subtype `AbstractReturnForecastEstimator`.

# Interfaces

To implement a new concrete type, subtype `AbstractReturnForecastEstimator` and implement these methods:

## `return_forecast`

  - [`return_forecast(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult, csfm::CrossSectionalFactorModel)`](@ref): Computes the Return Forecast from `rd` and the fitted factor model.

### Arguments

  - `rfe`: The concrete subtype instance.
  - `rd`: The returns result that carries the Asset Panel.
  - `csfm`: The fitted factor-model block.

### Returns

  - `rf::AbstractReturnForecastResult`: The member's own Result.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`CustomValueReturnForecast`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`DescriptorScores`](@ref)
"""
abstract type AbstractReturnForecastEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Return Forecast Result types.

Each member of the Return Forecast family has its own fitted state, so each returns its own Result type. Every one of these Result types has these two fields:

  - `mu`: The latest Return Forecast in return units, with one entry for each asset of the Coverage Universe. It is `NaN` for an asset that the member gives no forecast for.
  - `hist`: The history of the Return Forecast in return units, `observations × assets`. It is `nothing` for a member that computes no history.

A consumer can read these two fields from any member. It reads the other fields of a member only when it knows the type of that member.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractReturnForecastEstimator`](@ref)
  - [`return_forecast`](@ref)
  - [`CustomValueReturnForecastResult`](@ref)
  - [`FixedWeightedReturnForecastResult`](@ref)
"""
abstract type AbstractReturnForecastResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all expected returns algorithm types.

Every concrete or abstract type that is an algorithm of an expected returns estimator must subtype `AbstractExpectedReturnsAlgorithm`.

# Interfaces

A subtype needs no method of its own. An expected returns estimator holds the algorithm in a field and dispatches on its type. Define the methods on the expected returns estimator that uses the algorithm.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
"""
abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the asset view of an expected returns algorithm, which is the algorithm unchanged.

This is the fallback of [`port_opt_view`](@ref) for an expected returns algorithm that defines no method of its own.

# Arguments

  - `alg`: The expected returns algorithm.
  - `::Any`: The asset indices. The method ignores them.
  - `args...`: More arguments. The method ignores them.

# Returns

  - `alg::AbstractExpectedReturnsAlgorithm`: The original expected returns algorithm.

# Related

  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
function port_opt_view(alg::AbstractExpectedReturnsAlgorithm, ::Any,
                       args...)::AbstractExpectedReturnsAlgorithm
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all moment algorithm types.

Every concrete or abstract type that is an algorithm of a moment estimator must subtype `AbstractMomentAlgorithm`.

# Interfaces

A subtype needs no method of its own. A moment estimator holds the algorithm in a field and dispatches on its type. Define the methods on the moment estimator that uses the algorithm.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractMomentAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps every deviation from the target, so the moment is two-sided.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{D} &= \\boldsymbol{X} - t
\\end{align}
```

Where:

  - $(math_dict[:Xv])
  - $(math_dict[:tgt])

# Constructors

    FullMoment() -> FullMoment

# Examples

```jldoctest
julia> FullMoment()
FullMoment()
```

# Related

  - [`AbstractMomentAlgorithm`](@ref)
  - [`SemiMoment`](@ref)
"""
struct FullMoment <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Clips every deviation above the target to zero, so the moment measures only the downside.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{D} &= \\min\\left(\\boldsymbol{X} - t,\\, 0\\right)
\\end{align}
```

Where:

  - $(math_dict[:Xv])
  - $(math_dict[:tgt])

# Constructors

    SemiMoment() -> SemiMoment

# Examples

```jldoctest
julia> SemiMoment()
SemiMoment()
```

# Related

  - [`AbstractMomentAlgorithm`](@ref)
  - [`FullMoment`](@ref)
"""
struct SemiMoment <: AbstractMomentAlgorithm end
"""
    coverage_comoment_deviations(alg::FullMoment, Xo::MatNum, mu::VecNum) -> MatNum
    coverage_comoment_deviations(alg::SemiMoment, Xo::MatNum, mu::VecNum) -> MatNum

Centres an available-case block on the mean of each asset, and clips the deviations when the moment algorithm is a semi-moment.

The two moment algorithms change only this step of a higher-order available-case fit, so [`coverage_comoment_block`](@ref) has one body for both. [`FullMoment`](@ref) keeps every deviation, and [`SemiMoment`](@ref) clips every positive deviation to zero. The plain path without a Coverage Policy does the same.

# Arguments

  - `alg`: Moment algorithm of the estimator.
  - `Xo`: The oriented block, `observations × assets`, whose invalid entries are still non-finite.
  - `mu`: Each asset's available-case mean, over its own finite and active observations.

# Returns

  - `Y::MatNum`: The deviations, non-finite wherever `Xo` is.

# Related

  - [`coverage_comoment_block`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
"""
function coverage_comoment_deviations(::FullMoment, Xo::MatNum, mu::VecNum)
    return Xo .- transpose(mu)
end
function coverage_comoment_deviations(::SemiMoment, Xo::MatNum, mu::VecNum)
    return min.(Xo .- transpose(mu), zero(eltype(mu)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the pairwise expansions and the admission mask a higher-order available-case fit reads.

[`coverage_coskewness`](@ref) and [`coverage_cokurtosis`](@ref) both call this function. The third and the fourth co-moment differ only in the pair of expansions that they multiply. This function returns both expansions, and each order takes its own product. Neither order folds, so the function reads the whole block and keeps no state.

The function centres the deviations on the available-case mean of each asset, and then sets the invalid entries to zero. After this step, an observation at which an asset has no quote adds nothing to the numerator or to the denominator of a cell of that asset. The mask expansion `zc` is the same product over the mask, so the denominator of a cell is the number of observations at which every asset of that cell is valid.

# Algorithm

 1. Read the valid entries, the per-asset available-case mean and the per-asset bookkeeping of the block with [`coverage_valid_block`](@ref).
 2. Centre and clip the block with [`coverage_comoment_deviations`](@ref), and zero the invalid entries.
 3. Take the valid mask as integers, `Mi`.
 4. Build the pairwise expansions `z` of the deviations and `zc` of the mask, both `observations × assets²`.
 5. Admit the assets with [`coverage_admission`](@ref), against each asset's own observation count.

# Arguments

  - `alg`: Moment algorithm of the estimator.
  - `cvg`: The policy the estimator carries.
  - `X`: Data matrix.
  - `active_mask`: The active mask of the Asset Panel over the window, or `nothing`.
  - $(arg_dict[:dims])

# Returns

  - `(Xo, Y, Mi, z, zc, cmsk)::Tuple`: The oriented block, the zeroed deviations, the valid mask as integers, the pairwise expansion of each, and the admitted assets or `nothing`.

# Related

  - [`coverage_comoment_deviations`](@ref)
  - [`coverage_valid_block`](@ref)
  - [`coverage_admission`](@ref)
  - [`coverage_coskewness`](@ref)
  - [`coverage_cokurtosis`](@ref)
"""
function coverage_comoment_block(alg::AbstractMomentAlgorithm, cvg::CoveragePolicy,
                                 X::MatNum, active_mask::Option{<:AbstractMatrix{<:Bool}},
                                 dims::Int)
    Xo, msk, mu, active, stale = coverage_valid_block(X, active_mask, cvg.alg; dims = dims)
    Y = coverage_comoment_deviations(alg, Xo, mu)
    Y[.!msk] .= zero(eltype(Y))
    Mi = Int.(msk)
    o = transpose(ones(Int, size(Xo, 2)))
    z = kron(o, Y) ⊙ kron(Y, o)
    zc = kron(o, Mi) ⊙ kron(Mi, o)
    counts = CoverageCounts(vec(sum(Mi; dims = 1)), nothing, active, stale)
    return Xo, Y, Mi, z, zc, coverage_admission(cvg, counts, size(Xo, 1))
end
"""
    Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Computes the covariance matrix from the correlation matrix of the estimator and the standard deviations of its variance estimator `ce.ve`.

With this fallback, a covariance estimator that computes a correlation needs only its own [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/) method. The fallback scales the correlation matrix by the standard deviations from `std(ce.ve, X)`. An estimator whose covariance is not a scaled correlation defines its own method. [`ImpliedVolatility`](@ref) is one such estimator.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\hat{\\rho}_{ij}\\,\\hat{\\sigma}_i\\,\\hat{\\sigma}_j\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``\\hat{\\rho}_{ij}``: Correlation between assets ``i`` and ``j``.
  - ``\\hat{\\sigma}_i``: Standard deviation of asset ``i``.

# Algorithm

 1. Estimate the correlation matrix with `Statistics.cor(ce, X)`.
 2. Estimate the marginal standard deviations with `Statistics.std(ce.ve, X)`.
 3. Scale the correlation matrix into a covariance matrix with `StatsBase.cor2cov!`. The call writes in place, over the matrix that step 1 returned.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, passed to `cor` and to the variance estimator.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
    return StatsBase.cor2cov!(Statistics.cor(ce, X; dims = dims, kwargs...),
                              Statistics.std(ce.ve, X; dims = dims, kwargs...))
end
"""
    gap_fill_value(ce::StatsBase.CovarianceEstimator) -> Number

Return the value that replaces a gapped cell of a sample before the sample goes to `ce`.

A gapped sample has a non-finite cell because the asset was absent at that observation, and not because a computation failed. A consumer that holds such a sample and a covariance estimator cannot know which value the estimator needs. The consumer calls this function, and the value depends on the estimator alone.

When the value is finite, the consumer writes it over every gapped cell before it gives the sample to the estimator. The estimator then gets a complete sample and no mask. When the value is not finite, the consumer keeps the gaps. It gives the active mask that explains them to `Statistics.cov(ce, X; dims, active_mask)`, and an estimator that returns a non-finite value must implement that method.

The fallback value is zero. A plain moment estimator refuses a gapped sample, and zero is the neutral value of a standardised series. An estimator that accepts gaps returns `NaN`. That one method is all that such an estimator adds to this function.

A covariance estimator can hold another covariance estimator, so the value is recursive:

  - An estimator that gives the sample and its keywords to its inner estimator with no change returns the value of its inner estimator. [`PortfolioOptimisersCovariance`](@ref) and [`CorrelationCovariance`](@ref) are two of them. So a wrapper around an estimator that accepts gaps keeps the gaps.
  - An estimator that reads the sample before it calls its inner estimator keeps the fallback. [`Covariance`](@ref) is one of them, because its expected returns estimator reads the sample. It refuses the gap itself, whatever its inner estimator returns.
  - A [`Covariance`](@ref) with a [`CoveragePolicy`](@ref) reads the gaps itself, over the observations of each pair, so it returns `NaN`. A wrapper around it keeps the gaps.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `fv::Number`: The value of a gapped cell. It is zero for a plain estimator and `NaN` for an estimator that accepts gaps.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`cross_sectional_idiosyncratic_covariance`](@ref)
"""
function gap_fill_value(::StatsBase.CovarianceEstimator)
    return 0
end
"""
    densify(X::MatNum) -> MatNum

Materialise a lazy or sparse observation matrix as a dense `Matrix`.

The weighted moment methods of `StatsBase` take a `DenseMatrix`. A `Transpose`, an `Adjoint`, a `SubArray` or a sparse matrix does not match them. Without a `mean`, the call raises a `MethodError`, and the caller can recover from it. With a `mean`, the caller cannot recover. `cov(::SimpleCovariance, X, w; mean = mu)` forwards four positional arguments as `covm(X, mu, w, dims)`. When the `DenseMatrix` method does not match, that call resolves to `Statistics.covm(x, xmean, y, ymean, vardim)`, which is the cross-covariance of `X` and the weight vector. It returns an `N × 1` matrix in place of an `N × N` matrix, and raises no error. [`robust_cov`](@ref) and [`robust_cor`](@ref) densify `X` before every weighted call, so neither failure can happen.

# Algorithm

 1. When `X` is a `DenseMatrix` of numbers or of `JuMP` scalars, return it unchanged.
 2. Otherwise return `Matrix(X)`, a dense copy.

# Arguments

  - $(arg_dict[:X])

# Returns

  - `X::MatNum`: `X` itself when it is already a dense `Matrix`, and `Matrix(X)` otherwise.

# Related

  - [`MatNum`](@ref)
  - [`robust_cov`](@ref)
  - [`robust_cor`](@ref)
"""
function densify(X::DenseMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}})
    return X
end
function densify(X::MatNum)
    return Matrix(X)
end
"""
    compat_cov(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix with the estimator `ce`, and pass the extra keyword arguments only when the method of `ce` accepts them.

# Algorithm

 1. When the caller passed extra keyword arguments, and `hasmethod` reports that `Statistics.cov` takes `dims`, `mean` and those keys for this estimator and these arguments, call `Statistics.cov` with all of them and return the result.
 2. When step 1 raises a `MethodError`, drop the extra keyword arguments and continue. A method whose signature ends in a `kwargs...` slurp satisfies `hasmethod` and can still reject a key further down its call chain. Any other error propagates to the caller.
 3. Call `Statistics.cov(ce, X, args...; dims = dims, mean = mean)`, and return the result.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for `cov`.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`MatNum`](@ref)
  - [`robust_cov`](@ref)
  - [`compat_cor`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function compat_cov(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix, args...;
                    dims::Int = 1, mean = nothing, kwargs...)
    if !isempty(kwargs) &&
       hasmethod(Statistics.cov, Tuple{typeof(ce), typeof(X), typeof.(args)...},
                 (:dims, :mean, keys(kwargs)...))
        try
            return Statistics.cov(ce, X, args...; dims = dims, mean = mean, kwargs...)
        catch err
            # A method with a `kwargs...` slurp satisfies `hasmethod` but can still
            # reject the keyword arguments further down its call chain.
            if !(err isa MethodError)
                rethrow()
            end
        end
    end
    return Statistics.cov(ce, X, args...; dims = dims, mean = mean)
end
"""
    robust_cov(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Computes the weighted or unweighted covariance with [`compat_cov`](@ref) on dense observations. The unweighted method retries once with a densified `Matrix` after a `MethodError`. The weighted method calls [`densify`](@ref) before the estimator.

# Algorithm

 1. Without `w`, call [`compat_cov`](@ref) on `X` as the caller gave it. When that call raises a `MethodError`, call [`compat_cov`](@ref) once more on `Matrix(X)`. Any other error propagates to the caller.
 2. With `w`, densify `X` with [`densify`](@ref), then call [`compat_cov`](@ref) once. The method does no retry, because [`densify`](@ref) already removes the only failure that the retry handles.
 3. Return the matrix that [`compat_cov`](@ref) returned.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for `compat_cov`.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`MatNum`](@ref)
  - [`densify`](@ref)
  - [`compat_cov`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    assert_dims(dims)
    return try
        compat_cov(ce, X; dims = dims, mean = mean, kwargs...)
    catch err
        if !(err isa MethodError)
            rethrow()
        end
        compat_cov(ce, Matrix(X); dims = dims, mean = mean, kwargs...)
    end
end
function robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum,
                    w::StatsBase.AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    assert_dims(dims)
    return compat_cov(ce, densify(X), w; dims = dims, mean = mean, kwargs...)
end
"""
    compat_cor(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix with the estimator `ce`, and fall back to its covariance matrix when `ce` has no `cor` method for these arguments.

# Algorithm

 1. When `hasmethod` reports that `Statistics.cor` takes `dims` and `mean` for this estimator and these arguments, take steps 2 and 3. Otherwise take step 4.
 2. When the caller passed extra keyword arguments, and `hasmethod` reports that `Statistics.cor` takes those keys too, call `Statistics.cor` with all of them and return the result. When that call raises a `MethodError`, drop the extra keyword arguments and continue. A method whose signature ends in a `kwargs...` slurp satisfies `hasmethod` and can still reject a key further down its call chain.
 3. Call `Statistics.cor(ce, X, args...; dims = dims, mean = mean)`, and return the result. When that call raises a `MethodError`, continue to step 4. Any other error propagates to the caller.
 4. The estimator has no `cor` method for these arguments, so compute the covariance matrix `sigma` with [`robust_cov`](@ref).
 5. When `sigma` is mutable, convert it to a correlation matrix in place with `StatsBase.cov2cor!` and the square roots of its own diagonal. Otherwise convert a dense copy with `StatsBase.cov2cor`.
 6. Return `sigma`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for `cor`.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`MatNum`](@ref)
  - [`robust_cov`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function compat_cor(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix, args...;
                    dims::Int = 1, mean = nothing, kwargs...)
    if hasmethod(Statistics.cor, Tuple{typeof(ce), typeof(X), typeof.(args)...},
                 (:dims, :mean))
        if !isempty(kwargs) &&
           hasmethod(Statistics.cor, Tuple{typeof(ce), typeof(X), typeof.(args)...},
                     (:dims, :mean, keys(kwargs)...))
            try
                return Statistics.cor(ce, X, args...; dims = dims, mean = mean, kwargs...)
            catch err
                # A method with a `kwargs...` slurp satisfies `hasmethod` but can still
                # reject the keyword arguments further down its call chain.
                if !(err isa MethodError)
                    rethrow()
                end
            end
        end
        try
            return Statistics.cor(ce, X, args...; dims = dims, mean = mean)
        catch err
            if !(err isa MethodError)
                rethrow()
            end
        end
    end
    sigma = robust_cov(ce, X, args...; dims = dims, mean = mean, kwargs...)
    if ismutable(sigma)
        StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
    else
        sigma = StatsBase.cov2cor(Matrix(sigma))
    end
    return sigma
end
"""
    robust_cor(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Computes the weighted or unweighted correlation with [`compat_cor`](@ref) on dense observations. The unweighted method retries once with a densified `Matrix` after a `MethodError`. The weighted method calls [`densify`](@ref) before the estimator.

# Algorithm

 1. Without `w`, call [`compat_cor`](@ref) on `X` as the caller gave it. When that call raises a `MethodError`, call [`compat_cor`](@ref) once more on `Matrix(X)`. Any other error propagates to the caller.
 2. With `w`, densify `X` with [`densify`](@ref), then call [`compat_cor`](@ref) once. The method does no retry, because [`densify`](@ref) already removes the only failure that the retry handles.
 3. Return the matrix that [`compat_cor`](@ref) returned.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for `compat_cor`.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`MatNum`](@ref)
  - [`densify`](@ref)
  - [`compat_cor`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    assert_dims(dims)
    return try
        compat_cor(ce, X; dims = dims, mean = mean, kwargs...)
    catch err
        if !(err isa MethodError)
            rethrow()
        end
        compat_cor(ce, Matrix(X); dims = dims, mean = mean, kwargs...)
    end
end
function robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum,
                    w::StatsBase.AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    assert_dims(dims)
    return compat_cor(ce, densify(X), w; dims = dims, mean = mean, kwargs...)
end
"""
    moment_window_and_weights(
        X::VecNum_MatNum,
        w::Option{<:ObsWeights},
        args...;
        dims::Int = 1,
        kwargs...
    ) -> (VecNum_MatNum, Option{<:StatsBase.AbstractWeights})
    moment_window_and_weights(
        X::VecNum_MatNum,
        w::Option{<:ObsWeights},
        window::VecInt;
        dims::Int = 1,
        kwargs...
    ) -> (VecNum_MatNum, Option{<:StatsBase.AbstractWeights})

Apply the observation window to the data, and resolve the observation weights over the windowed data.

The function takes the view of `X` over the observations that `window` indexes, and resolves the observation weights over that view. The caller resolves `window` with [`get_window`](@ref) before the call, so an `Int` window is already a range here.

# Algorithm

 1. Without a `window`, `X` passes through unchanged, and step 3 resolves the weights over the whole of it.
 2. With a `window`, take the view of `X` over those observations. For a matrix that is `view(X, window, :)` when `dims == 1`, and `view(X, :, window)` when `dims == 2`. For a vector it is `view(X, window)`. Index `w` to the same observations with [`nothing_scalar_array_getindex`](@ref).
 3. Resolve the observation weights with [`get_observation_weights`](@ref), over `w` and the `X` of the step above.
 4. Return that `X` and the resolved `w`.

# Arguments

  - $(arg_dict[:X_Xv])
  - $(arg_dict[:oow])
  - Either:
      + $(arg_dict[:ignargs])
      + $(arg_dict[:window])
  - $(arg_dict[:dims]) The function ignores it when `X` is a vector.
  - $(arg_dict[:ignkwargs])

# Returns

  - `X::VecNum_MatNum`: The data over the window, or all of the data when there is no window.
  - `w::Option{<:StatsBase.AbstractWeights}`: The resolved observation weights over the same observations.

# Related

  - [`get_window`](@ref)
  - [`get_observation_weights`](@ref)
  - [`nothing_scalar_array_getindex`](@ref)
"""
function moment_window_and_weights(X::MatNum, w::Option{<:ObsWeights}, args...;
                                   dims::Int = 1, kwargs...)
    w = get_observation_weights(w, X; dims = dims, kwargs...)
    return X, w
end
function moment_window_and_weights(X::VecNum, w::Option{<:ObsWeights}, args...; kwargs...)
    w = get_observation_weights(w, X; kwargs...)
    return X, w
end
function moment_window_and_weights(X::MatNum, w::Option{<:ObsWeights}, window::VecInt;
                                   dims::Int = 1, kwargs...)
    X = isone(dims) ? view(X, window, :) : view(X, :, window)
    w = nothing_scalar_array_getindex(w, window)
    w = get_observation_weights(w, X; dims = dims, kwargs...)
    return X, w
end
function moment_window_and_weights(X::VecNum, w::Option{<:ObsWeights}, window::VecInt;
                                   kwargs...)
    X = view(X, window)
    w = nothing_scalar_array_getindex(w, window)
    w = get_observation_weights(w, X; kwargs...)
    return X, w
end
"""
    weighted_centre(X::MatNum, me::AbstractExpectedReturnsEstimator,
                    w::Option{<:ObsWeights}; dims::Int = 1, mean = nothing,
                    kwargs...) -> Union{<:Number, <:ArrNum}

Resolve the centre of a moment estimator with the same observation weights as its deviations.

The observation weights of a moment estimator apply to its centre, and not only to its deviations. This function holds that rule, so [`SimpleVariance`](@ref), [`Covariance`](@ref), [`Coskewness`](@ref) and [`Cokurtosis`](@ref) compute their centre in the same way.

# Algorithm

 1. When `mean` is not `nothing`, return it unchanged. Use the keyword for a centre that `w` does not describe.
 2. When `w` is `nothing`, return `Statistics.mean(me, X; dims = dims, kwargs...)`.
 3. Otherwise, rebuild `me` with `w` through [`factory`](@ref), so that the centre has the weights of the deviations. Return the mean of the rebuilt estimator.

Step 2 makes the function faster and does not change the value. `w` comes from a field, so its type selects the branch. Without step 2, a windowed loop rebuilds the estimator tree of `me` once for each window.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:me])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for the expected returns estimator.

# Returns

  - `mu::Union{<:Number, <:ArrNum}`: The centre, weighted by `w` when `w` is not `nothing`.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`demean_returns`](@ref)
  - [`covariance_centre_and_estimator`](@ref)
  - [`factory`](@ref)
"""
function weighted_centre(X::MatNum, me::AbstractExpectedReturnsEstimator,
                         w::Option{<:ObsWeights}; dims::Int = 1, mean = nothing, kwargs...)
    return if !isnothing(mean)
        mean
    elseif isnothing(w)
        Statistics.mean(me, X; dims = dims, kwargs...)
    else
        Statistics.mean(factory(me, w), X; dims = dims, kwargs...)
    end
end
"""
    demean_returns(X::MatNum, me::AbstractExpectedReturnsEstimator; dims::Int = 1, mean = nothing,
                   kwargs...) -> MatNum

Subtracts the mean of each asset from the returns in `X`.

The expected returns estimator `me` computes the mean, unless the caller gives `mean`.

# Algorithm

 1. Resolve the centre `mu` with [`weighted_centre`](@ref). When `mean` is `nothing`, the estimator computes `mu`. Otherwise `mu` is `mean`. The estimator uses its own weights, and this function adds no weights.
 2. Subtract `mu` from `X` by broadcast, and return the result. `dims` is the axis of the observations, and the estimator shapes `mu` along the other axis. The broadcast then subtracts one value for each asset from every observation of that asset.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:me])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for the expected returns estimator.

# Returns

  - `X::MatNum`: The demeaned returns matrix.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`weighted_centre`](@ref)
"""
function demean_returns(X::MatNum, me::AbstractExpectedReturnsEstimator; dims::Int = 1,
                        mean = nothing, kwargs...)
    return X .- weighted_centre(X, me, nothing; dims = dims, mean = mean, kwargs...)
end
# ---------------------------------------------------------------------------
# @windowed_estimator — one declaration per windowed moment estimator (ADR 0039)
# ---------------------------------------------------------------------------
"""
    WINDOWED_ESTIMATOR_KEYS

The assignment keys that a [`@windowed_estimator`](@ref) body accepts, in addition to the one
`field::Type = default` line. The macro refuses every other key at macro-expansion time, with a
[`did_you_mean`](@ref) suggestion. So a mistyped key raises an error, and cannot make a wrong
docstring or omit a forwarding method.

# Related

  - [`@windowed_estimator`](@ref)
"""
const WINDOWED_ESTIMATOR_KEYS = (:noun, :forward, :doctest)
"""
    WINDOWED_ESTIMATOR_INPUTS

The input types that a `forward` entry of [`@windowed_estimator`](@ref) can declare.
`MatNum` makes the matrix forwarding method and the Asset Panel forwarding method of the same
generic. The matrix method passes `dims` and `iv` to [`windowed_preamble`](@ref). `VecNum`
makes the vector forwarding method.

# Related

  - [`@windowed_estimator`](@ref)
"""
const WINDOWED_ESTIMATOR_INPUTS = (:MatNum, :VecNum)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Throw an [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError) at macro-expansion time for a wrong [`@windowed_estimator`](@ref) declaration.

# Arguments

  - `msg::AbstractString`: The body of the message. The function puts `@windowed_estimator: ` before it, so every message of the macro starts with the same prefix.

# Validation

  - The function always raises, so it never returns to its caller.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_estimator_error(msg::AbstractString)
    return throw(ArgumentError("@windowed_estimator: " * msg))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that `key` names an entry of `dict`. When it does not, raise an error that ends with a
[`did_you_mean`](@ref) suggestion. The message names the table `what`.

# Arguments

  - `key::Symbol`: The key the declaration wrote.
  - `dict::AbstractDict`: The table the key must name an entry of.
  - `what::AbstractString`: The name of that table. The message contains it.

# Validation

  - `haskey(dict, key)`. For a key that names no entry, the function raises through [`windowed_estimator_error`](@ref), and the message ends with the suffix from [`windowed_estimator_suggest`](@ref).

# Returns

  - `key::Symbol`: The key, unchanged, so a caller can check the key and bind it in one expression.

# Related

  - [`@windowed_estimator`](@ref)
  - [`did_you_mean`](@ref)
"""
function windowed_estimator_check_key(key::Symbol, dict::AbstractDict, what::AbstractString)
    if !haskey(dict, key)
        windowed_estimator_error("`$(key)` is not a `$(what)` key" *
                                 windowed_estimator_suggest(key, keys(dict)) *
                                 ".")
    end
    return key
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Suggest the nearest `candidates` entry to a mistyped [`@windowed_estimator`](@ref) key.

The function calls [`suggest_declared_key`](@ref), which every suggestion for a declaration key
uses. That function scores the candidates with the Damerau-Levenshtein distance at
`min_score = 0.5`, which accepts more matches than the global default. This is safe here,
because the candidates are constants of the source code: the block keys and the keys of
`field_dict` and `ret_dict`. A suggestion cannot show the name of a real asset to a caller who
guesses.

# Arguments

  - `key`: The mistyped key.
  - `candidates`: The keys that the declaration accepts.

# Returns

  - `msg::String`: The suffix of the suggestion, to put at the end of a message. It is empty when no candidate gets a score that is high enough.

# Related

  - [`@windowed_estimator`](@ref)
  - [`suggest_declared_key`](@ref)
  - [`did_you_mean`](@ref)
"""
function windowed_estimator_suggest(key, candidates)
    return suggest_declared_key(key, candidates)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Parse the `field::Type = default` line of a [`@windowed_estimator`](@ref) body into the
field name of the inner estimator, its declared type, and its default in the keyword constructor.

The field name is also the [`field_dict`](@ref) key of the generated field docstring, and the
argument name of every generated forwarding method. So it must follow the naming convention of
the library, which uses `me`, `ce`, `ve`, `ske` and `kte`.

# Algorithm

 1. Read the field name and its declared type from the left of the `=`, giving `name` and `type`.
 2. Check `name` against [`field_dict`](@ref) with [`windowed_estimator_check_key`](@ref).
 3. Return `name`, `type`, and the right of the `=`, which is the keyword-constructor default.

# Arguments

  - `ex`: The one `field::Type = default` line of the declaration body.

# Validation

  - `ex` reads `field::Type = default`. Any other shape raises.
  - The field name is a `Symbol`. Any other name raises.
  - The field name names an entry of [`field_dict`](@ref).

# Returns

  - `(name, type, default)`: The field name, its declared type, and its keyword-constructor default.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_parse_field(ex)
    if !(Meta.isexpr(ex, :(=)) &&
         Meta.isexpr(ex.args[1], :(::)) &&
         length(ex.args[1].args) == 2)
        windowed_estimator_error("the inner estimator must be declared as `field::Type = default`; got `$(ex)`.")
    end
    name, type = ex.args[1].args
    if !isa(name, Symbol)
        windowed_estimator_error("the inner estimator's field name must be a symbol; got `$(name)`.")
    end
    windowed_estimator_check_key(name, field_dict, "field_dict")
    return name, type, ex.args[2]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Parse one `forward` entry of a [`@windowed_estimator`](@ref) body, such as
`generic(::MatNum; mean) => :ret_key`. The parts are the forwarded generic, its input type, a
flag for a `mean` keyword, and the [`ret_dict`](@ref) keys of its return values.

A `mean` in the mini-signature keeps `mean` out of the forwarded `kwargs...`. Without it,
`mean` goes into `kwargs...` and from there to [`windowed_preamble`](@ref).

# Algorithm

 1. Split `ex` at the `=>`, giving the mini-signature `sig` and the return keys `rets`.
 2. Read the forwarded generic from the head of `sig`, giving `gen`.
 3. Read each argument of `sig`. A `mean` keyword sets `has_mean`. The one positional argument type sets `input`.
 4. Check `input` against [`WINDOWED_ESTIMATOR_INPUTS`](@ref).
 5. Check every entry of `rets` against [`ret_dict`](@ref) with [`windowed_estimator_check_key`](@ref), and collect the entries into `keys_`.
 6. Return `gen`, `input`, `has_mean` and `keys_`.

# Arguments

  - `ex`: One entry of the `forward` vector of the declaration body.

# Validation

  - `ex` reads `generic(::Input[; mean]) => :ret_key`, or `=> (:k1, :k2)` for a tuple return. Any other shape raises.
  - The left of the `=>` is a call.
  - `mean` is the only keyword that the mini-signature can name.
  - The mini-signature declares exactly one positional argument type.
  - That type names an entry of [`WINDOWED_ESTIMATOR_INPUTS`](@ref).
  - Every return key is a quoted symbol, and names an entry of [`ret_dict`](@ref).

# Returns

  - `(gen, input, has_mean, keys_)`: The forwarded generic, its input type, whether the mini-signature names `mean`, and the [`ret_dict`](@ref) keys of its return values.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_parse_forward(ex)
    if !(Meta.isexpr(ex, :call) && length(ex.args) == 3 && ex.args[1] === :(=>))
        windowed_estimator_error("each `forward` entry must read `generic(::MatNum[; mean]) => :ret_key` (or `=> (:k1, :k2)` for a tuple return); got `$(ex)`.")
    end
    sig, rets = ex.args[2], ex.args[3]
    if !Meta.isexpr(sig, :call)
        windowed_estimator_error("the left of a `forward` entry must be a call; got `$(sig)`.")
    end
    gen = sig.args[1]
    input, has_mean = nothing, false
    for arg in sig.args[2:end]
        if Meta.isexpr(arg, :parameters)
            for kw in arg.args
                if kw !== :mean
                    windowed_estimator_error("`mean` is the only keyword a `forward` entry may name; got `$(kw)`.")
                end
                has_mean = true
            end
        elseif Meta.isexpr(arg, :(::)) && length(arg.args) == 1
            if !isnothing(input)
                windowed_estimator_error("a `forward` entry takes exactly one positional argument type; got `$(sig)`.")
            end
            input = arg.args[1]
        else
            windowed_estimator_error("unexpected argument `$(arg)` in `forward` entry `$(sig)`.")
        end
    end
    if !(input in WINDOWED_ESTIMATOR_INPUTS)
        windowed_estimator_error("a `forward` entry's input type must be one of $(WINDOWED_ESTIMATOR_INPUTS); got `$(input)`.")
    end
    raw = Meta.isexpr(rets, :tuple) ? rets.args : Any[rets]
    keys_ = Symbol[]
    for r in raw
        if !(isa(r, QuoteNode) && isa(r.value, Symbol))
            windowed_estimator_error("`ret_dict` keys must be quoted symbols; got `$(r)`.")
        end
        push!(keys_, windowed_estimator_check_key(r.value, ret_dict, "ret_dict"))
    end
    return gen, input, has_mean, keys_
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render the `generic(field::Name, X::Input)` reference to one generated forwarding method. The
`# Related` sections of the type and of the other generated methods use it.

The reference leaves out the keyword arguments. Documenter resolves an `@ref` by the positional
signature of the method, and the positional types are enough to find one method. The panel
method has a third positional argument, `pnl::Option{<:AssetPanel}`, after `X::MatNum`.

# Arguments

  - `gen`: The forwarded generic.
  - `field::Symbol`: Name of the inner estimator's field, which is also the argument name of the generated method.
  - `name::Symbol`: Name of the windowed estimator type.
  - `input::Symbol`: Input kind of the generated method, `:MatNum`, `:VecNum` or `:AssetPanel`.

# Returns

  - `ref::String`: A Documenter cross-reference to the generated method. It links the code span `gen(field::Name, X::Input)` to that method.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_method_ref(gen, field::Symbol, name::Symbol, input::Symbol)
    args = input === :AssetPanel ? "X::MatNum, pnl::Option{<:AssetPanel}" : "X::$(input)"
    return "[`$(gen)($(field)::$(name), $(args))`](@ref)"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render the signature line of the docstring of one generated forwarding method.

# Algorithm

 1. For the vector input, return `gen(field::Name, X::VecNum)`, with `; mean = nothing` when `has_mean` is set.
 2. Otherwise open `gen(field::Name, X::MatNum`, add `pnl::Option{<:AssetPanel}` for the panel input, and add the keywords: `dims`, then `mean` when `has_mean` is set, then `iv` for the matrix input alone, then `kwargs...`. The panel method takes `iv` in `kwargs...`.

# Arguments

  - `gen`: The forwarded generic.
  - `field::Symbol`: Name of the inner estimator's field, which is also the argument name of the generated method.
  - `name::Symbol`: Name of the windowed estimator type.
  - `input::Symbol`: Input kind of the generated method, `:MatNum`, `:VecNum` or `:AssetPanel`.
  - `has_mean::Bool`: `true` when the method declares a `mean` keyword.

# Returns

  - `sig::String`: The indented signature line.

# Related

  - [`windowed_method_doc`](@ref)
  - [`windowed_method_def`](@ref)
"""
function windowed_method_sig(gen, field::Symbol, name::Symbol, input::Symbol,
                             has_mean::Bool)
    if input === :VecNum
        return "    $(gen)($(field)::$(name), X::VecNum" *
               (has_mean ? "; mean = nothing" : "") *
               ")"
    end
    pnl = input === :AssetPanel
    return "    $(gen)($(field)::$(name), X::MatNum" *
           (pnl ? ", pnl::Option{<:AssetPanel}" : "") *
           "; dims::Int = 1, " *
           (has_mean ? "mean = nothing, " : "") *
           (pnl ? "" : "iv::Option{<:MatNum} = nothing, ") *
           "kwargs...)"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the docstring for one generated forwarding method as an interpolation AST.

The function returns an `Expr(:string, ...)` and not a `String`, so each lookup of
[`arg_dict`](@ref) and [`ret_dict`](@ref) stays a live part of the `DocStr`, as a hand-written
`\$(arg_dict[:dims])` does.

# Algorithm

 1. Build the signature line `sig` with [`windowed_method_sig`](@ref). The matrix signature has `dims`, `iv` and `kwargs...`, and the vector signature has none of them. The panel signature has `pnl` after `X`, and takes `iv` in `kwargs...`. Each signature also has `mean` when `has_mean` is set.
 2. Start `parts` with `sig`, the summary sentences, the steps of the algorithm of the generated method, and the arguments heading with its two prose bullets. The panel method has one more step, the cut of the panel. The summary names the generic and not the noun of the type, because `std` on a `WindowedVariance` computes a standard deviation and not a variance.
 3. For the panel input, push the `arg_dict[:pnl_moment]` lookup as a live expression.
 4. For the matrix and panel inputs, push the `arg_dict[:dims]` lookup.
 5. When `has_mean` is set, push the `mean` bullet.
 6. For the matrix and panel inputs, push the `arg_dict[:oiv]` lookup and the `kwargs...` bullet.
 7. Push the returns heading, then one live `ret_dict` lookup per entry of `ret_keys`.
 8. Push the related heading, the type, every entry of `siblings`, and [`windowed_preamble`](@ref).
 9. Return `parts` wrapped in `Expr(:string, ...)`, so every lookup stays live.

# Arguments

  - `gen`: The forwarded generic.
  - `field::Symbol`: Name of the inner estimator's field, which is also the argument name of the generated method.
  - `name::Symbol`: Name of the windowed estimator type.
  - `input::Symbol`: Input kind of the generated method, `:MatNum`, `:VecNum` or `:AssetPanel`.
  - `has_mean::Bool`: `true` when the method declares a `mean` keyword.
  - `ret_keys::Vector{Symbol}`: The [`ret_dict`](@ref) keys documenting the return values.
  - `noun::AbstractString`: The capitalised noun phrase of the moment. The generated prose uses it.
  - `siblings::Vector{String}`: Cross-references to the other generated methods of the type, from [`windowed_method_ref`](@ref).

# Returns

  - `doc::Expr`: An `Expr(:string, ...)` holding the docstring, with every dictionary lookup left unevaluated.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_method_doc(gen, field::Symbol, name::Symbol, input::Symbol,
                             has_mean::Bool, ret_keys::Vector{Symbol}, noun::AbstractString,
                             siblings::Vector{String})
    lc = lowercasefirst(noun)
    is_mat = input !== :VecNum
    is_pnl = input === :AssetPanel
    n0 = Int(is_pnl)
    sig = windowed_method_sig(gen, field, name, input, has_mean)
    # The summary names the *generic*, not the type's noun: `std` on a WindowedVariance
    # computes a standard deviation, not a variance.
    parts = Any["\n", sig,
                "\n\nCompute `$(gen)` over a rolling or indexed window of observations, for $(is_pnl ? "an Asset Panel" : is_mat ? "a matrix" : "a vector") input.\n\nThe method selects a window of observations from `X`, and applies the observation weights when the estimator holds them. Then it calls the inner $(lc) estimator.$(is_pnl ? " It cuts the Asset Panel to the same observations and gives it to the inner estimator. A mask-aware inner estimator then reads the active mask of the window, and a plain one reduces to the Coverage Universe of the window." : "")\n\n# Algorithm\n\n$(is_pnl ? " 1. Cut `pnl` to the observations of the window with [`windowed_panel`](@ref).\n" : "") $(1 + n0). Resolve the window and the observation weights with [`windowed_preamble`](@ref). It returns `inner`, a copy of `$(field).$(field)` that holds the windowed weights, and the windowed `X`$(is_mat ? ", the windowed `iv`, and the `active_mask` and `estimation_mask` keywords cut to the window" : "").\n $(2 + n0). Call `$(gen)` on `inner` and the windowed `X`$(is_pnl ? " and `pnl`" : ""), and return its result. The inner estimator alone computes the value. This method changes only the observations and the weights that the inner estimator gets.\n\n# Arguments\n\n  - `$(field)`: Windowed $(lc) estimator.\n  - `X`: Data $(is_mat ? "matrix of asset returns, `observations × assets`" : "vector of returns").\n"]
    if is_pnl
        push!(parts, "  - ", :(arg_dict[:pnl_moment]), "\n")
    end
    if is_mat
        push!(parts, "  - ", :(arg_dict[:dims]), "\n")
    end
    if has_mean
        push!(parts,
              "  - `mean`: Optional precomputed mean, passed to the inner estimator.\n")
    end
    if is_pnl
        push!(parts,
              "  - `kwargs...`: More keyword arguments, passed to the inner estimator. The method cuts the `iv`, `active_mask` and `estimation_mask` keywords to the window.\n")
    elseif is_mat
        push!(parts, "  - ", :(arg_dict[:oiv]),
              "\n  - `kwargs...`: More keyword arguments, passed to the inner estimator. The method cuts the `active_mask` and `estimation_mask` keywords to the window.\n")
    end
    push!(parts, "\n# Returns\n")
    for k in ret_keys
        push!(parts, "\n  - ", :(ret_dict[$(QuoteNode(k))]))
    end
    push!(parts, "\n\n# Related\n\n  - [`$(name)`](@ref)\n")
    for s in siblings
        push!(parts, "  - ", s, "\n")
    end
    push!(parts, "  - [`windowed_preamble`](@ref)\n")
    return Expr(:string, parts...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build one generated forwarding method. The method resolves the window with
[`windowed_preamble`](@ref), and then calls the method of the inner estimator.

# Algorithm

 1. Build the three field accesses that the body reads. `field.field` is the inner estimator, `field.w` holds the observation weights, and `field.window` holds the window.
 2. For the vector input, build a method whose only keyword is `mean`, and only when `has_mean` is set. Its body binds `inner` and the windowed `X` with [`windowed_preamble`](@ref), and returns `gen` of the two. Return this method.
 3. Build the keyword list of the signature. It holds `dims`, then `mean` when `has_mean` is set, then `iv` and `kwargs...`. The panel method names no `iv`. As in the panel methods of the moment verbs, `iv` comes in `kwargs...`, and the preamble binds it from there.
 4. Build the matching keyword list of the inner call. It has the same names, and passes each value on. It passes the windowed `iv`, and the keywords `kw`, in which the two universe masks are cut to the window.
 5. For the panel input, the positional arguments are `field`, `X` and `pnl`, and the body opens with the cut of `pnl` to the window by [`windowed_panel`](@ref). For the matrix input, they are `field` and `X`.
 6. Build the body: one call to [`windowed_preamble`](@ref) that binds `inner`, the windowed `X`, `iv` and `kw`, then a `return` of `gen` applied to `inner`, that `X`, and the cut `pnl` for the panel input.
 7. Return the whole `Expr(:function, ...)`.

# Arguments

  - `gen`: The forwarded generic.
  - `field::Symbol`: Name of the inner estimator's field, which is also the argument name of the generated method.
  - `name::Symbol`: Name of the windowed estimator type.
  - `input::Symbol`: Input kind of the generated method, `:MatNum`, `:VecNum` or `:AssetPanel`.
  - `has_mean::Bool`: `true` when the method declares a `mean` keyword.

# Returns

  - `def::Expr`: The `Expr(:function, ...)` of the forwarding method.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_method_def(gen, field::Symbol, name::Symbol, input::Symbol,
                             has_mean::Bool)
    self = Expr(:(::), field, name)
    inner = Expr(:., field, QuoteNode(field))
    w = Expr(:., field, QuoteNode(:w))
    window = Expr(:., field, QuoteNode(:window))
    if input === :VecNum
        kws = has_mean ? Any[Expr(:kw, :mean, nothing)] : Any[]
        call = has_mean ? Any[:(mean = mean)] : Any[]
        return Expr(:function,
                    Expr(:call, gen, Expr(:parameters, kws...), self, :(X::VecNum)),
                    quote
                        inner, X = windowed_preamble($inner, $w, $window, X)
                        return $(Expr(:call, gen, Expr(:parameters, call...), :inner, :X))
                    end)
    end
    kws = Any[Expr(:kw, :(dims::Int), 1)]
    call = Any[:(dims = dims)]
    if has_mean
        push!(kws, Expr(:kw, :mean, nothing))
        push!(call, :(mean = mean))
    end
    push!(call, :(iv = iv), :(kw...))
    # The panel method takes `iv` in `kwargs...`, as the panel methods of the moment verbs do,
    # and cuts the panel before `X` is rebound to the window.
    args, cargs, cut, pre = if input === :AssetPanel
        push!(kws, :(kwargs...))
        Any[self, :(X::MatNum), :(pnl::Option{<:AssetPanel})], Any[:inner, :X, :pnl],
        :(pnl = windowed_panel(pnl, get_window($window, X, dims))),
        :(windowed_preamble($inner, $w, $window, X; dims = dims, kwargs...))
    else
        push!(kws, Expr(:kw, :(iv::Option{<:MatNum}), nothing), :(kwargs...))
        Any[self, :(X::MatNum)], Any[:inner, :X], nothing,
        :(windowed_preamble($inner, $w, $window, X; iv = iv, dims = dims, kwargs...))
    end
    return Expr(:function, Expr(:call, gen, Expr(:parameters, kws...), args...),
                quote
                    $cut
                    inner, X, iv, kw = $pre
                    return $(Expr(:call, gen, Expr(:parameters, call...), cargs...))
                end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the type docstring of a generated windowed estimator as an interpolation AST.

The `DocStringExtensions` abbreviations and the dictionary lookups stay live, as in
[`windowed_method_doc`](@ref).

# Algorithm

 1. Read the name of the inner estimator from `default` into `inner_ref`. For a call such as `SimpleVariance()`, the name is the head of the call. For a bare name, it is the name itself.
 2. Start `parts` with the live `DocStringExtensions.TYPEDEF` abbreviation, the two summary sentences, the `# Fields` heading, and the live `DocStringExtensions.FIELDS` abbreviation.
 3. Push the `# Constructors` section and the keyword signature built from `name`, `field`, `ftype` and `default`. Then push the `## Validation` subsection, with the live `val_dict[:oow]` lookup and the window rule.
 4. Push the three propagation subsections, `## Propagated parameters`, `## View parameters` and `## Observation weight parameters`. Each one names `field`, `w` and `cache` as the tags of the generated struct declare them.
 5. Push the `# Examples` section, fencing `doctest` as a `jldoctest` block.
 6. Push the `# Related` heading, the supertype, `inner_ref`, every entry of `methods`, the two window rules, and the four shared functions that the type implements: `factory`, `port_opt_view`, `obs_weights_view` and `windowed_preamble`.
 7. Return `parts` wrapped in `Expr(:string, ...)`, so every lookup stays live.

# Arguments

  - `name::Symbol`: Name of the windowed estimator type.
  - `super`: Its supertype.
  - `field::Symbol`: Name of the inner estimator's field.
  - `ftype`: Declared type of that field.
  - `default`: Its keyword-constructor default.
  - `noun::AbstractString`: The capitalised noun phrase of the moment. The generated prose uses it.
  - `doctest::AbstractString`: The body of the `jldoctest` block of the `# Examples` section, without its fences.
  - `methods::Vector{String}`: Cross-references to the generated methods of the type, from [`windowed_method_ref`](@ref).

# Returns

  - `doc::Expr`: An `Expr(:string, ...)` holding the docstring, with every abbreviation and lookup left unevaluated.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_type_doc(name::Symbol, super, field::Symbol, ftype, default,
                           noun::AbstractString, doctest::AbstractString,
                           methods::Vector{String})
    lc = lowercasefirst(noun)
    inner_ref = Meta.isexpr(default, :call) ? default.args[1] : default
    parts = Any["\n", :(DocStringExtensions.TYPEDEF),
                "\n\nFits an inner $(lc) estimator on a rolling or indexed window of observations.\n\n`$(name)` holds another $(lc) estimator, and applies it to the observations of a window, with optional observation weights. So the estimate can follow the most recent observations, or give them more weight.\n\n# Fields\n\n",
                :(DocStringExtensions.FIELDS),
                "\n\n# Constructors\n\n    $(name)(;\n        $(field)::$(ftype) = $(default),\n        w::Option{<:ObsWeights} = nothing,\n        window::Option{<:Int_VecInt} = nothing,\n        rule::AbstractWindowRule = RollingWindow(),\n        cache::Option{<:AbstractPartialFitState} = nothing\n    ) -> $(name)\n\nEach keyword sets the struct field of the same name.\n\n## Validation\n\n  - ",
                :(val_dict[:oow]),
                "\n  - If `window` is not `nothing`, it must be nonempty, nonnegative and finite.\n\n## Propagated parameters\n\n[`factory`](@ref) passes its arguments on to these `@fprop` fields:\n\n  - `$(field)`: Recursively updated through [`factory`](@ref).\n  - `w`: Replaced with the incoming [`ObsWeights`](@ref).\n  - `cache`: Carried unchanged through [`factory`](@ref).\n\n## View parameters\n\n[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:\n\n  - `$(field)`: Recursively viewed through [`port_opt_view`](@ref).\n  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).\n\n## Observation weight parameters\n\n[`obs_weights_view`](@ref) indexes these fields to the selected observations:\n\n  - `$(field)`: Recursively indexed through [`obs_weights_view`](@ref).\n  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).\n\n# Examples\n\n```jldoctest\n$(strip(doctest))\n```\n\n# Related\n\n  - [`$(super)`](@ref)\n  - [`$(inner_ref)`](@ref)\n"]
    for m in methods
        push!(parts, "  - ", m, "\n")
    end
    push!(parts,
          "  - [`RollingWindow`](@ref)\n  - [`SeedWindow`](@ref)\n  - [`factory`](@ref)\n  - [`port_opt_view`](@ref)\n  - [`obs_weights_view`](@ref)\n  - [`windowed_preamble`](@ref)\n")
    return Expr(:string, parts...)
end
"""
    @windowed_estimator Name <: Super begin
        field::FieldType = Default()
        noun    = "Noun"
        forward = [generic(::MatNum; mean) => :ret_key, ...]
        doctest = \"\"\"...\"\"\"
    end

Declare a windowed moment estimator, which applies an inner moment estimator to a window of
observations.

The windowed estimator binds the observation weights to the window. The inner estimator
computes the moment in the same way as without a window.

One call of the macro declares all the parts of the windowed estimator:

  - The [`@propagatable`](@ref) `@concrete` struct. Its fields are the inner estimator tagged
    `@fprop @vprop`, `w` tagged `@wprop`, `window`, `rule`, and a `cache` tagged
    `@fprop @vprop` that holds a partial-fit state.
  - The two constructors with their validation.
  - One forwarding method for each `forward` entry, and one Asset Panel method for each matrix
    entry.
  - The `export` of the type, and every docstring.

The library has five nominal types and not one parametric `Windowed{E}`. Each type implements a
different generic, and must subtype a different abstract estimator, such as
`AbstractCovarianceEstimator` or `CoskewnessEstimator`. Dispatch across the library reads these
supertypes, and a Julia struct cannot take its supertype from a type parameter. This macro keeps
the five types the same in every other part.

# Body

  - `field::FieldType = Default()`: The inner estimator. The field name is also its
    [`field_dict`](@ref) key and the argument name of every generated method. So it must
    follow the naming convention of the library, which uses `me`, `ce`, `ve`, `ske` and `kte`.
  - `noun`: The capitalised noun phrase of the moment, for example `"Expected returns"`. All
    the generated prose uses it.
  - `forward`: One mini-signature for each generic to forward, with the [`ret_dict`](@ref)
    keys of its return values. A `mean` in the mini-signature makes `mean` a named keyword of
    the generated method. Without it, `mean` goes into `kwargs...` and from there to
    [`windowed_preamble`](@ref).
  - `doctest`: The body of the `jldoctest` block of the `# Examples` section, without its
    fences.

At macro-expansion time, the macro refuses an unknown key, a wrong `forward` entry, and an
unknown key of `field_dict` or `ret_dict`. The error for an unknown key ends with a
[`did_you_mean`](@ref) suggestion.

# Algorithm

 1. Read `name` and `super` from the header.
 2. Read the body once. [`windowed_parse_field`](@ref) parses the one `field::Type = default` line, and returns `field`, `ftype` and `default`. The `noun`, `forward` and `doctest` lines bind their values. Any other key raises an error.
 3. Parse every entry of `forward` with [`windowed_parse_forward`](@ref), giving `specs`. Append one `:AssetPanel` copy of each `MatNum` entry, so every matrix generic also gets its Asset Panel method.
 4. Render one cross-reference per entry of `specs` with [`windowed_method_ref`](@ref), giving `refs`.
 5. For each entry of `specs`, build one documented forwarding method: [`windowed_method_doc`](@ref) writes its docstring, and [`windowed_method_def`](@ref) writes its body. Each method's `# Related` section lists the `refs` of its siblings and not its own.
 6. Build `structexpr`, the `@concrete` struct. It declares the inner estimator tagged `@fprop @vprop`, `w` tagged `@wprop`, `window` and `rule` with no tag, and `cache` tagged `@fprop @vprop`, each with its live [`field_dict`](@ref) lookup, and the inner constructor that validates `w` and `window`.
 7. Build `kwctor`, the keyword constructor, whose defaults are `default`, `nothing`, `nothing`, [`RollingWindow`](@ref)`()` and `nothing`.
 8. Write the type's docstring with [`windowed_type_doc`](@ref), and attach it to `structexpr` wrapped in `@propagatable @concrete`.
 9. Return the escaped block: the documented struct, `kwctor`, the forwarding methods, and the `export` of `name`.

# Validation

  - The header reads `Name <: Super`, and `Name` is a `Symbol`.
  - The declaration body is a `begin ... end` block, and every line of it is an assignment.
  - At most one `field::Type = default` line appears.
  - Every other key names an entry of [`WINDOWED_ESTIMATOR_KEYS`](@ref). An unknown key raises with a [`windowed_estimator_suggest`](@ref) suffix.
  - All four of `field::Type = default`, `noun`, `forward` and `doctest` are present.
  - `noun` and `doctest` are string literals.
  - `forward` is a vector, and it declares at least one generic.
  - Every failure above raises an `ArgumentError` through [`windowed_estimator_error`](@ref), at macro-expansion time.

# Returns

  - `ex::Expr`: The escaped block that declares the whole family member.

# Examples

```julia
@windowed_estimator WindowedVariance <: AbstractVarianceEstimator begin
    ve::AbstractVarianceEstimator = SimpleVariance()
    noun = \"Variance\"
    forward = [Statistics.var(::MatNum; mean) => :vararr,
               Statistics.var(::VecNum; mean) => :varnum]
    doctest = \"""
    julia> WindowedVariance()
    ...
    \"""
end
```

# Related

  - [`windowed_preamble`](@ref)
  - [`@propagatable`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
macro windowed_estimator(head, body)
    if !(Meta.isexpr(head, :<:) && isa(head.args[1], Symbol))
        windowed_estimator_error("the header must read `Name <: Super`; got `$(head)`.")
    end
    name, super = head.args
    if !Meta.isexpr(body, :block)
        windowed_estimator_error("the declaration body must be a `begin ... end` block.")
    end
    field, ftype, default = nothing, nothing, nothing
    noun, forward, doctest = nothing, nothing, nothing
    for stmt in body.args
        if isa(stmt, LineNumberNode)
            continue
        end
        if !Meta.isexpr(stmt, :(=))
            windowed_estimator_error("every line of the body must be an assignment; got `$(stmt)`.")
        end
        lhs = stmt.args[1]
        if Meta.isexpr(lhs, :(::))
            if !isnothing(field)
                windowed_estimator_error("exactly one `field::Type = default` line is allowed.")
            end
            field, ftype, default = windowed_parse_field(stmt)
        elseif lhs === :noun
            noun = stmt.args[2]
        elseif lhs === :forward
            if !Meta.isexpr(stmt.args[2], :vect)
                windowed_estimator_error("`forward` must be a vector of `generic(::Input) => :ret_key` entries.")
            end
            forward = stmt.args[2].args
        elseif lhs === :doctest
            doctest = stmt.args[2]
        else
            windowed_estimator_error("`$(lhs)` is not a recognised key" *
                                     windowed_estimator_suggest(lhs,
                                                                WINDOWED_ESTIMATOR_KEYS) *
                                     ".")
        end
    end
    for (val, key) in
        ((field, "field::Type = default"), (noun, "noun"), (forward, "forward"),
         (doctest, "doctest"))
        isnothing(val) && windowed_estimator_error("missing required `$(key)` declaration.")
    end
    if !isa(noun, AbstractString) || !isa(doctest, AbstractString)
        windowed_estimator_error("`noun` and `doctest` must be string literals.")
    end
    if isempty(forward)
        windowed_estimator_error("`forward` must declare at least one generic.")
    end
    specs = [windowed_parse_forward(f) for f in forward]
    append!(specs, [(g, :AssetPanel, m, k) for (g, i, m, k) in specs if i === :MatNum])
    refs = [windowed_method_ref(gen, field, name, input) for (gen, input, _, _) in specs]
    defs = Any[]
    siblings = String[]
    for (i, (gen, input, has_mean, ret_keys)) in pairs(specs)
        append!(empty!(siblings), (r for (j, r) in pairs(refs) if j != i))
        push!(defs,
              Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), LineNumberNode(@__LINE__),
                   windowed_method_doc(gen, field, name, input, has_mean, ret_keys, noun,
                                       siblings),
                   windowed_method_def(gen, field, name, input, has_mean)))
    end
    # The inner estimator and `cache` are both tagged `@fprop @vprop`.
    tagged(f) = Expr(:macrocall, Symbol("@fprop"), LineNumberNode(@__LINE__),
                     Expr(:macrocall, Symbol("@vprop"), LineNumberNode(@__LINE__), f))
    structexpr = Expr(:struct, false, Expr(:<:, name, super),
                      Expr(:block, Expr(:string, :(field_dict[$(QuoteNode(field))])),
                           tagged(field), Expr(:string, :(field_dict[:oow])),
                           Expr(:macrocall, Symbol("@wprop"), LineNumberNode(@__LINE__),
                                :w), Expr(:string, :(field_dict[:window])), :window,
                           Expr(:string, :(field_dict[:window_rule])), :rule,
                           Expr(:string, :(field_dict[:pfcache])), tagged(:cache),
                           Expr(:function,
                                Expr(:call, name, Expr(:(::), field, ftype),
                                     :(w::Option{<:ObsWeights}),
                                     :(window::Option{<:Int_VecInt}),
                                     :(rule::AbstractWindowRule),
                                     :(cache::Option{<:AbstractPartialFitState})),
                                quote
                                    assert_nonempty_nonneg_finite_val(w, :w)
                                    assert_nonempty_nonneg_finite_val(window, :window)
                                    return $(Expr(:call,
                                                  Expr(:curly, :new,
                                                       Expr(:call, :typeof, field),
                                                       :(typeof(w)), :(typeof(window)),
                                                       :(typeof(rule)), :(typeof(cache))),
                                                  field, :w, :window, :rule, :cache))
                                end)))
    kwctor = Expr(:function,
                  Expr(:(::),
                       Expr(:call, name,
                            Expr(:parameters, Expr(:kw, Expr(:(::), field, ftype), default),
                                 Expr(:kw, :(w::Option{<:ObsWeights}), nothing),
                                 Expr(:kw, :(window::Option{<:Int_VecInt}), nothing),
                                 Expr(:kw, :(rule::AbstractWindowRule), :(RollingWindow())),
                                 Expr(:kw, :(cache::Option{<:AbstractPartialFitState}),
                                      nothing))), name),
                  Expr(:block, :(return $name($field, w, window, rule, cache))))
    return esc(Expr(:block,
                    Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
                         LineNumberNode(@__LINE__),
                         windowed_type_doc(name, super, field, ftype, default, noun,
                                           doctest, refs),
                         Expr(:macrocall, Symbol("@propagatable"),
                              LineNumberNode(@__LINE__),
                              Expr(:macrocall, Symbol("@concrete"),
                                   LineNumberNode(@__LINE__), structexpr))), kwctor,
                    defs..., Expr(:export, name)))
end

export FullMoment, SemiMoment
public AbstractCovarianceEstimator, AbstractVarianceEstimator,
       AbstractExpectedReturnsEstimator, AbstractReturnForecastEstimator
