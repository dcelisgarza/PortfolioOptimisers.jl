"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for centering functions used in the Median Absolute Deviation risk measure.

# Related Types

  - [`MedianCentering`](@ref)
  - [`MeanCentering`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
abstract type MedianCenteringFunction <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Centres the returns series using the (weighted) median before computing the Median Absolute Deviation.

# Related Types

  - [`MedianCenteringFunction`](@ref)
  - [`MeanCentering`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
struct MedianCentering <: MedianCenteringFunction end
"""
$(DocStringExtensions.TYPEDEF)

Centres the returns series using the (weighted) mean before computing the Median Absolute Deviation.

# Related Types

  - [`MedianCenteringFunction`](@ref)
  - [`MedianCentering`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
struct MeanCentering <: MedianCenteringFunction end
"""
    const MedAbsDevMu = Union{<:Num_VecNum_VecScalar, <:MedianCenteringFunction}

Union of valid centring-target types for [`MedianAbsoluteDeviation`](@ref).

Accepts a numeric scalar/vector target or a [`MedianCenteringFunction`](@ref) (e.g. mean or median centering).

# Related

  - [`Num_VecNum_VecScalar`](@ref)
  - [`MedianCenteringFunction`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
const MedAbsDevMu = Union{<:Num_VecNum_VecScalar, <:MedianCenteringFunction}
"""
$(DocStringExtensions.TYPEDEF)

Represents the Median Absolute Deviation (MAD) risk measure for hierarchical portfolio optimisation.

`MedianAbsoluteDeviation` computes the median (or mean) absolute deviation of portfolio returns about a specified centre, providing a robust alternative to variance for measuring dispersion.

# Mathematical Definition

Let ``\\mu`` be the chosen centre (median or mean of returns, or a user-supplied value). Define the deviations ``\\delta_t = x_t - \\mu``. The MAD is:

```math
\\mathrm{MAD}(\\boldsymbol{x}) = \\mathrm{median}_{1 \\leq t \\leq T}(|\\delta_t|)\\,.
```

When `flag = true` (default), the result is scaled by a consistency factor ``(\\Phi^{-1}(3/4))^{-1} \\approx 1.4826`` so that the MAD is a consistent estimator of the standard deviation under normality.

# Fields

  - `settings`: Hierarchical risk measure configuration.
  - `w`: Optional observation weights.
  - `mu`: Centre of the deviation (centering function, scalar, vector, or `VecScalar`).
  - `flag`: If `true`, applies the consistency correction factor.

# Constructors

    MedianAbsoluteDeviation(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::MedAbsDevMu = MedianCentering(),
        flag::Bool = true
    ) -> MedianAbsoluteDeviation

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is a `VecNum`: `!isempty(mu)` and `all(isfinite, mu)`.
  - If `mu` is a `Number`: `isfinite(mu)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::MedianAbsoluteDeviation)(w::VecNum, X::MatNum, fees = nothing)

Computes the MAD of the portfolio returns series.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

# Examples

```jldoctest
julia> MedianAbsoluteDeviation()
MedianAbsoluteDeviation
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
         w ┼ nothing
        mu ┼ MedianCentering()
      flag ┴ Bool: true
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`MedianCentering`](@ref)
  - [`MeanCentering`](@ref)
"""
@concrete struct MedianAbsoluteDeviation <: HierarchicalRiskMeasure
    settings
    w
    mu
    flag
    function MedianAbsoluteDeviation(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:ObsWeights}, mu::MedAbsDevMu,
                                     flag::Bool = true)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
            @argcheck(all(isfinite, mu))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu))
        end
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(w), typeof(mu), typeof(flag)}(settings, w, mu,
                                                                          flag)
    end
end
function MedianAbsoluteDeviation(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Option{<:ObsWeights} = nothing,
                                 mu::MedAbsDevMu = MedianCentering(), flag::Bool = true)
    return MedianAbsoluteDeviation(settings, w, mu, flag)
end
function factory(r::MedianAbsoluteDeviation, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return MedianAbsoluteDeviation(; settings = r.settings, w = w, mu = r.mu, flag = r.flag)
end
function nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
    return x
end
function risk_measure_view(r::MedianAbsoluteDeviation, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return MedianAbsoluteDeviation(; settings = r.settings, w = r.w, mu = mu, flag = r.flag)
end
function calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering,
                                                      <:Any}, ::Any, x::VecNum)
    return Statistics.mean(x)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights, <:MeanCentering,
                                                       <:Any}, ::Any, x::VecNum)
    w = get_observation_weights(r.w, x)
    return Statistics.mean(x, w)
end
function calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MedianCentering,
                                                      <:Any}, ::Any, x::VecNum)
    return Statistics.median(x)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights,
                                                       <:MedianCentering, <:Any}, ::Any,
                            x::VecNum)
    w = get_observation_weights(r.w, x)
    return Statistics.median(x, w)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecNum, <:Any},
                            w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:Number, <:Any},
                            ::Any, ::Any)
    return r.mu
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecScalar, <:Any},
                            w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
function calc_deviations_vec(r::MedianAbsoluteDeviation, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
function (r::MedianAbsoluteDeviation)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return StatsBase.mad(val; center = zero(eltype(X)), normalize = r.flag)
end

export MedianAbsoluteDeviation, MedianCentering, MeanCentering
