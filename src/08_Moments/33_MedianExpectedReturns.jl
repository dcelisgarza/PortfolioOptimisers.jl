"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the optionally weighted asset medians.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MedianExpectedReturns(;
        w::Option{<:ObsWeights} = nothing
    ) -> MedianExpectedReturns

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> me = MedianExpectedReturns()
MedianExpectedReturns
  w ┴ nothing

julia> factory(me, StatsBase.Weights([0.1, 0.2, 0.7]))
MedianExpectedReturns
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
"""
@concrete struct MedianExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:oow])"
    w
    function MedianExpectedReturns(w::Option{<:ObsWeights})
        return new{typeof(w)}(w)
    end
end
function MedianExpectedReturns(; w::Option{<:ObsWeights} = nothing)::MedianExpectedReturns
    return MedianExpectedReturns(w)
end
"""
    factory(ce::MedianExpectedReturns, w::ObsWeights) -> MedianExpectedReturns

Return a new [`MedianExpectedReturns`](@ref) estimator with observation weights `w`.

# Arguments

  - `ce`: Median expected returns estimator.
  - $(arg_dict[:ow])

# Returns

  - `me::MedianExpectedReturns`: Updated estimator with weights applied.

# Related

  - [`MedianExpectedReturns`](@ref)
  - [`factory`](@ref)
"""
function factory(::MedianExpectedReturns, w::ObsWeights)::MedianExpectedReturns
    return MedianExpectedReturns(; w = w)
end
"""
    Statistics.mean(me::MedianExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the median of each asset.

This method returns the median of each asset across observations in `X`. If `me.w` is `nothing`,
the median is computed directly with `Statistics.median(X; dims = dims)`. Otherwise, the method
computes a weighted median for each asset using the observation weights `w`.

# Summary Statistics

```math
\\hat{\\mu}_j = \\mathrm{median}(r_{1j}, r_{2j}, \\ldots, r_{Tj})
```

Where ``r_{tj}`` is the return of asset ``j`` at time ``t``.

# Arguments

  - `me`: Median expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `mu::Matrix{<:Number}`: Median vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`MedianExpectedReturns`](@ref)
"""
function Statistics.mean(me::MedianExpectedReturns{Nothing}, X::MatNum; dims::Int = 1,
                         kwargs...)
    return Statistics.median(X; dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Weighted-median overload of [`mean(me::MedianExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Computes per-asset weighted median using the [`ObsWeights`](@ref) stored in `me.w`.
"""
function Statistics.mean(me::MedianExpectedReturns{<:ObsWeights}, X::MatNum; dims::Int = 1,
                         kwargs...)
    @argcheck(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    w = get_observation_weights(me.w, X)
    Y = Vector{eltype(X)}(undef, size(X, 2))
    for i in axes(X, 2)
        Y[i] = Statistics.median(view(X, :, i), w)
    end
    return insertdims(Y; dims = dims)
end

export MedianExpectedReturns
