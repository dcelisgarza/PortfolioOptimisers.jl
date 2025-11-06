"""
    struct SimpleExpectedReturns{T1} <: AbstractExpectedReturnsEstimator
        w::T1
    end

A simple expected returns estimator for PortfolioOptimisers.jl, representing the sample mean with optional observation weights.

`SimpleExpectedReturns` is the standard estimator for computing expected returns as the (possibly weighted) mean of asset returns. It supports both unweighted and weighted mean estimation by storing an optional weights vector.

# Fields

  - `w`: Optional weights for each observation. If `nothing`, the unweighted mean is computed.

# Constructor

    SimpleExpectedReturns(; w::WeightsType = nothing)

Keyword arguments correspond to the fields above.

## Validation

    - If `w` is provided, `!isempty(w)`.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::SimpleExpectedReturns, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
struct SimpleExpectedReturns{T1} <: AbstractExpectedReturnsEstimator
    w::T1
    function SimpleExpectedReturns(w::WeightsType)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function SimpleExpectedReturns(; w::WeightsType = nothing)
    return SimpleExpectedReturns(w)
end
"""
    mean(me::SimpleExpectedReturns, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the mean of asset returns using a [`SimpleExpectedReturns`](@ref) estimator.

This method computes the expected returns as the sample mean of the input data `X`, optionally using observation weights stored in the estimator. If no weights are provided, the unweighted mean is computed.

# Arguments

  - `me`: The expected returns estimator.
  - `X`: Data array of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `kwargs...`: Additional keyword arguments passed to [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean).

# Returns

  - `mu::Vector{<:Real}`: The expected returns vector.

# Examples

```jldoctest
julia> using StatsBase

julia> X = [0.01 0.02; 0.03 0.04];

julia> ser = SimpleExpectedReturns()
SimpleExpectedReturns
  w ┴ nothing

julia> mean(ser, X)
1×2 Matrix{Float64}:
 0.02  0.03

julia> w = Weights([0.2, 0.8]);

julia> serw = SimpleExpectedReturns(; w = w)
SimpleExpectedReturns
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.8]

julia> mean(serw, X)
1×2 Matrix{Float64}:
 0.026  0.036
```

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean)
"""
function Statistics.mean(me::SimpleExpectedReturns, X::AbstractMatrix; dims::Int = 1,
                         kwargs...)
    return isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
end
function factory(me::SimpleExpectedReturns, w::WeightsType = nothing)
    return SimpleExpectedReturns(; w = isnothing(w) ? me.w : w)
end

export SimpleExpectedReturns
