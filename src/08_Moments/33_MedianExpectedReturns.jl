"""
$(DocStringExtensions.TYPEDEF)

Computes the expected returns as the per-asset median of the asset returns.

`w` carries optional observation weights. If `w` is `nothing`, the median is unweighted. The median resists an outlier that would move the sample mean.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MedianExpectedReturns(;
        w::Option{<:ObsWeights} = nothing
    ) -> MedianExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
  - [`Option`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::MedianExpectedReturns{Nothing}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`mean(me::MedianExpectedReturns{<:ObsWeights}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct MedianExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:oow])
    """
    @wprop w
    function MedianExpectedReturns(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MedianExpectedReturns(; w::Option{<:ObsWeights} = nothing)::MedianExpectedReturns
    return MedianExpectedReturns(w)
end
"""
    Statistics.mean(me::MedianExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the median of each asset.

This method returns the median of each asset across observations in `X`. If `me.w` is `nothing`,
the median is computed directly with `Statistics.median(X; dims = dims)`. Otherwise, the method
computes a weighted median for each asset using the observation weights `w`.

# Mathematical definition

Unweighted:

```math
\\begin{align}
\\hat{\\mu}_j &= \\mathrm{median}(r_{1j}, r_{2j}, \\ldots, r_{Tj})\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: Median expected return of asset ``j``.
  - $(math_dict[:r_tj])
  - $(math_dict[:T])

Weighted. The weighted median is the `StatsBase` weighted quantile at probability ``1/2``, which **interpolates between two order statistics**. Order the returns of asset ``j`` so that ``r_{(1)j} \\leq \\ldots \\leq r_{(T)j}``, and let ``w_{(t)}`` be the weight that travels with each one:

```math
\\begin{align}
S_m &= \\sum_{t=1}^{m} w_{(t)}\\,, \\\\
h &= \\frac{1}{2} \\left( \\sum_{t=1}^{T} w_t - w_{(1)} \\right) + w_{(1)}\\,, \\\\
k &= \\max \\left\\lbrace m : S_m \\leq h \\right\\rbrace\\,, \\\\
\\hat{\\mu}_j &= r_{(k)j} + \\frac{h - S_k}{S_{k+1} - S_k} \\left( r_{(k+1)j} - r_{(k)j} \\right)\\,.
\\end{align}
```

Where:

  - ``w_t``: Observation weight at time ``t``.
  - ``w_{(t)}``: Weight of the ``t``-th smallest return, so the weights are permuted with the returns.
  - ``r_{(t)j}``: ``t``-th smallest return of asset ``j``.
  - ``S_m``: Cumulative weight of the ``m`` smallest returns.
  - ``h``: Cumulative weight that the probability ``1/2`` corresponds to.

The result is therefore not in general one of the observed returns. Under equal weights it reduces to the ordinary median.

# Arguments

  - `me`: Median expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::Matrix{<:Number}`: Median vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`MedianExpectedReturns`](@ref)
  - [`Statistics.median`](https://juliastats.org/StatsBase.jl/stable/robust/)
"""
function Statistics.mean(me::MedianExpectedReturns{Nothing}, X::MatNum; dims::Int = 1,
                         kwargs...)
    assert_dims(dims)
    return Statistics.median(X; dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Weighted-median overload of [`mean(me::MedianExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Computes per-asset weighted median using the [`ObsWeights`](@ref) stored in `me.w`.
"""
function Statistics.mean(me::MedianExpectedReturns{<:ObsWeights}, X::MatNum; dims::Int = 1,
                         kwargs...)
    X = dims_oriented(dims, X)
    w = get_observation_weights(me.w, X)
    Y = Vector{eltype(X)}(undef, size(X, 2))
    for i in axes(X, 2)
        Y[i] = Statistics.median(view(X, :, i), w)
    end
    return insertdims(Y; dims = dims)
end

export MedianExpectedReturns
