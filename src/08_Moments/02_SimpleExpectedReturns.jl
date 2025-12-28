"""
    struct SimpleExpectedReturns{T1} <: AbstractExpectedReturnsEstimator
        w::T1
    end

A simple expected returns estimator for PortfolioOptimisers.jl, representing the sample mean with optional observation weights.

`SimpleExpectedReturns` is the standard estimator for computing expected returns as the (possibly weighted) mean of asset returns. It supports both unweighted and weighted mean estimation by storing an optional weights vector.

# Fields

  - `w`: Optional weights for each observation. If `nothing`, the unweighted mean is computed.

# Constructor

    SimpleExpectedReturns(; w::Option{<:StatsBase.AbstractWeights} = nothing)

Keyword arguments correspond to the fields above.

## Validation

    - If `w` is not `nothing`, `!isempty(w)`.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`Option`](@ref)
  - [`StatsBase.StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
struct SimpleExpectedReturns{T1} <: AbstractExpectedReturnsEstimator
    w::T1
    function SimpleExpectedReturns(w::Option{<:StatsBase.AbstractWeights})
        assert_nonempty_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function SimpleExpectedReturns(; w::Option{<:StatsBase.AbstractWeights} = nothing)
    return SimpleExpectedReturns(w)
end
"""
    Statistics.mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute the mean of asset returns using a [`SimpleExpectedReturns`](@ref) estimator.

This method computes the expected returns as the sample mean of the input data `X`, optionally using observation weights stored in the estimator. If no weights are provided, the unweighted mean is computed.

# Arguments

  - `me`: The expected returns estimator.
  - `X`: Data array of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `kwargs...`: Additional keyword arguments passed to [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean).

# Returns

  - `mu::VecNum`: The expected returns vector.

# Examples

```jldoctest
julia> using StatsBase

julia> X = [0.01 0.02; 0.03 0.04];

julia> ser = SimpleExpectedReturns()
SimpleExpectedReturns
  w ┴ nothing

julia> Statistics.mean(ser, X)
1×2 Matrix{Float64}:
 0.02  0.03

julia> w = Weights([0.2, 0.8]);

julia> serw = SimpleExpectedReturns(; w = w)
SimpleExpectedReturns
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.8]

julia> Statistics.mean(serw, X)
1×2 Matrix{Float64}:
 0.026  0.036
```

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
  - [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean)
"""
function Statistics.mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    return if isnothing(me.w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, me.w; dims = dims)
    end
end
"""
    factory(me::SimpleExpectedReturns, w::Option{<:StatsBase.AbstractWeights} = nothing)

Create a new `SimpleExpectedReturns` estimator with updated observation weights.

This function constructs a new [`SimpleExpectedReturns`](@ref) object, optionally replacing the weights stored in the input estimator with the provided weights. If `w` is `nothing`, the weights from `me` are used.

# Arguments

  - `me`: Existing `SimpleExpectedReturns` estimator.
  - `w`: Optional observation weights to use in the new estimator. If `nothing`, uses `me.w`.

# Returns

  - `me::SimpleExpectedReturns`: New estimator with updated weights.

# Details

  - Returns a new estimator, preserving the type and updating weights as specified.
  - If `w` is not provided, the weights from `me` are used.
  - Validates that weights are non-empty and finite.

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function factory(me::SimpleExpectedReturns,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return SimpleExpectedReturns(; w = ifelse(isnothing(w), me.w, w))
end

export SimpleExpectedReturns
