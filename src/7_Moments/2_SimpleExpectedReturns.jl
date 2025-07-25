"""
    struct SimpleExpectedReturns{T1 <: Union{Nothing, <:AbstractWeights}} <: AbstractExpectedReturnsEstimator
        w::T1
    end

A simple expected returns estimator for PortfolioOptimisers.jl, representing the sample mean with optional observation weights.

`SimpleExpectedReturns` is the standard estimator for computing expected returns as the (possibly weighted) mean of asset returns. It supports both unweighted and weighted mean estimation by storing an optional weights vector.

# Fields

  - `w::Union{Nothing, <:AbstractWeights}`: Optional weights for each observation. If `nothing`, the unweighted mean is computed.

# Constructor

    SimpleExpectedReturns(; w::Union{Nothing, <:AbstractWeights} = nothing)

Construct a [`SimpleExpectedReturns`](@ref) estimator with optional observation weights.

# Fields

  - `w::Union{Nothing, <:AbstractWeights}`: Optional weights for each observation.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::SimpleExpectedReturns, X::AbstractArray; dims::Int = 1, kwargs...)`](@ref)
"""
struct SimpleExpectedReturns{T1 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractExpectedReturnsEstimator
    w::T1
end
"""
    SimpleExpectedReturns(; w::Union{Nothing, <:AbstractWeights} = nothing)

Construct a [`SimpleExpectedReturns`](@ref) estimator for computing expected returns as the (optionally weighted) sample mean.

# Arguments

  - `w::Union{Nothing, <:AbstractWeights}`: Optional observation weights. If `nothing`, the unweighted mean is computed.

# Returns

  - `SimpleExpectedReturns`: A simple expected returns estimator configured with optional weights.

# Validation

  - If `w` is provided, it must not be empty.

# Returns

  - `SimpleExpectedReturns`: A simple expected returns estimator.

# Examples

```jldoctest
julia> using StatsBase

julia> ser = SimpleExpectedReturns()
SimpleExpectedReturns
  w | nothing

julia> w = Weights([0.2, 0.3, 0.5]);

julia> ser = SimpleExpectedReturns(; w = w)
SimpleExpectedReturns
  w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`SimpleExpectedReturns`](@ref)
  - [`mean(me::SimpleExpectedReturns, X::AbstractArray; dims::Int = 1, kwargs...)`](@ref)
"""
function SimpleExpectedReturns(; w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return SimpleExpectedReturns(w)
end

"""
    mean(me::SimpleExpectedReturns, X::AbstractArray; dims::Int = 1, kwargs...)

Compute the mean of asset returns using a [`SimpleExpectedReturns`](@ref) estimator.

This method computes the expected returns as the sample mean of the input data `X`, optionally using observation weights stored in the estimator. If no weights are provided, the unweighted mean is computed.

# Arguments

  - `me::SimpleExpectedReturns`: The expected returns estimator.
  - `X::AbstractArray`: Data array of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `kwargs...`: Additional keyword arguments passed to [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean).

# Returns

  - The mean of `X` along the specified dimension using the [`SimpleExpectedReturns`](@ref) estimator.

# Examples

```jldoctest
julia> using StatsBase

julia> X = [0.01 0.02; 0.03 0.04];

julia> ser = SimpleExpectedReturns()
SimpleExpectedReturns
  w | nothing

julia> mean(ser, X)
1×2 Matrix{Float64}:
 0.02  0.03

julia> w = Weights([0.2, 0.8]);

julia> serw = SimpleExpectedReturns(; w = w)
SimpleExpectedReturns
  w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.8]

julia> mean(serw, X)
1×2 Matrix{Float64}:
 0.026  0.036
```

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean)
"""
function Statistics.mean(me::SimpleExpectedReturns, X::AbstractArray; dims::Int = 1,
                         kwargs...)
    return isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
end

function factory(me::SimpleExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleExpectedReturns(; w = isnothing(w) ? me.w : w)
end

export SimpleExpectedReturns
