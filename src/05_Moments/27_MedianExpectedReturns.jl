"""
$(DocStringExtensions.TYPEDEF)

Computes the expected returns as the per-asset median of the asset returns.

`w` holds optional observation weights. If `w` is `nothing`, the median is unweighted. An outlier moves the sample mean in proportion to its size, and it moves the median only through its rank.

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

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

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

This method returns the median of each asset over the observations in `X`. If `me.w` is
`nothing`, the method calls `Statistics.median(X; dims = dims)`. Otherwise, the method takes the
weighted median of each asset under the observation weights `me.w`.

# Mathematical definition

Unweighted:

```math
\\begin{align}
\\hat{\\mu}_j &= \\mathrm{median}(r_{1j}, r_{2j}, \\ldots, r_{Tj})\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: The median expected return of asset ``j``.
  - $(math_dict[:r_tj])
  - $(math_dict[:T])

Weighted. The weighted median is the `StatsBase` weighted quantile at probability ``1/2``, which interpolates between two order statistics. Sort the returns of asset ``j`` so that ``r_{(1)j} \\leq \\ldots \\leq r_{(T)j}``, and let ``w_{(t)}`` be the weight of the observation that gives ``r_{(t)j}``.

```math
\\begin{align}
S_m &= \\sum_{t=1}^{m} w_{(t)}\\,, \\\\
h &= \\frac{1}{2} \\left( \\sum_{t=1}^{T} w_t - w_{(1)} \\right) + w_{(1)}\\,, \\\\
k &= \\max \\left\\lbrace m : S_m \\leq h \\right\\rbrace\\,, \\\\
\\hat{\\mu}_j &= r_{(k)j} + \\frac{h - S_k}{S_{k+1} - S_k} \\left( r_{(k+1)j} - r_{(k)j} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_obs])
  - ``w_{(t)}``: The weight of the ``t``-th smallest return. The sort moves each weight with its return.
  - ``r_{(t)j}``: The ``t``-th smallest return of asset ``j``.
  - ``S_m``: The cumulative weight of the ``m`` smallest returns.
  - ``h``: The cumulative weight at the probability ``1/2``.

The weighted median is not an order statistic, for two reasons.

  - The result is in general not one of the observed returns, because the last line interpolates between ``r_{(k)j}`` and ``r_{(k+1)j}``. Under equal weights it reduces to the ordinary median.
  - ``w_{(1)}`` in the second line is the weight of the smallest return, not the weight of the first observation. The sort moves the weights with the returns before the second line gives ``h``. A sample whose smallest return is the last observation therefore gives a different ``h`` from the one that `w[1]` gives, and a different result.

# Arguments

  - `me`: The median expected returns estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - `all(isfinite, X)`. Both methods refuse a gapped sample with [`assert_finite_sample`](@ref), which raises an [`IsNonFiniteError`](@ref).

# Returns

  - `mu::Matrix{<:Number}`: The medians, a `(1, N)` matrix if `dims == 1` or an `(N, 1)` matrix if `dims == 2`.

# Related

  - [`MedianExpectedReturns`](@ref)
  - [`Statistics.median`](https://juliastats.org/StatsBase.jl/stable/robust/)
"""
function Statistics.mean(me::MedianExpectedReturns{Nothing}, X::MatNum; dims::Int = 1,
                         kwargs...)
    assert_dims(dims)
    assert_finite_sample(X)
    return Statistics.median(X; dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted-median method of [`mean(me::MedianExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). It takes the weighted median of each asset under the [`ObsWeights`](@ref) in `me.w`.

`StatsBase` has a weighted median of a vector and none of a matrix, so the method takes one column at a time. The weighted-quantile expression of the method above gives the result of one column.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), so that the observations run down the columns. This step also checks `dims`.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 3. Resolve `me.w` against the oriented `X` with [`get_observation_weights`](@ref). The result is `w`.
 4. Allocate the result vector `Y`, of length `size(X, 2)`.
 5. For each column `i` of `X`, put the weighted median of that column under `w` in `Y[i]`.
 6. Insert the reduced dimension back into `Y` with `insertdims`. The result is a `(1, N)` matrix when `dims == 1` and an `(N, 1)` matrix when `dims == 2`.

# Validation

  - $(val_dict[:dims]) This method does not check `dims` itself. [`dims_oriented`](@ref) raises the `DomainError`.
  - `all(isfinite, X)`, which [`assert_finite_sample`](@ref) checks.
"""
function Statistics.mean(me::MedianExpectedReturns{<:ObsWeights}, X::MatNum; dims::Int = 1,
                         kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    w = get_observation_weights(me.w, X)
    Y = Vector{eltype(X)}(undef, size(X, 2))
    for i in axes(X, 2)
        Y[i] = Statistics.median(view(X, :, i), w)
    end
    return insertdims(Y; dims = dims)
end
function effective_observation_count(me::MedianExpectedReturns, X::MatNum; kwargs...)
    return observation_weight_count(get_observation_weights(me.w, X), size(X, 1))
end

export MedianExpectedReturns
