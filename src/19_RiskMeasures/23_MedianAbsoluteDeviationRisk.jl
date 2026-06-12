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

# Mathematical definition

Let ``\\mu`` be the chosen centre (median or mean of returns, or a user-supplied value). Define the deviations ``\\delta_t = x_t - \\mu``. The MAD is:

```math
\\begin{align}
\\mathrm{MAD}(\\boldsymbol{x}) &= \\mathrm{median}_{1 \\leq t \\leq T}(|\\delta_t|)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MAD}(\\boldsymbol{x})``: Median absolute deviation of portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Chosen centre (median, mean, or user-supplied value).
  - ``\\delta_t = x_t - \\mu``: Deviation of return at period ``t`` from the centre.

When `flag = true` (default), the result is scaled by a consistency factor ``(\\Phi^{-1}(3/4))^{-1} \\approx 1.4826`` so that the MAD is a consistent estimator of the standard deviation under normality.

# Fields

$(DocStringExtensions.FIELDS)

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
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_rm])
    """
    w
    """
    $(field_dict[:mu_rm])
    """
    mu
    """
    $(field_dict[:flag])
    """
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
                                 mu::MedAbsDevMu = MedianCentering(),
                                 flag::Bool = true)::MedianAbsoluteDeviation
    return MedianAbsoluteDeviation(settings, w, mu, flag)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`MedianAbsoluteDeviation`](@ref) by selecting observation weights from the risk-measure instance or falling back to the prior result.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::MedianAbsoluteDeviation, pr::AbstractPriorResult, args...;
                 kwargs...)::MedianAbsoluteDeviation
    w = nothing_scalar_array_selector(r.w, pr.w)
    return MedianAbsoluteDeviation(; settings = r.settings, w = w, mu = r.mu, flag = r.flag)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`MedianCenteringFunction`](@ref) `x` unchanged.

Identity pass-through: centering functions are not sliced by asset index.

# Related

  - [`MedianCenteringFunction`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
    return x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`MedianAbsoluteDeviation`](@ref) `r` sliced to asset indices `i`.

Slices the centering target `mu` for cluster-based optimisation.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`risk_measure_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function risk_measure_view(r::MedianAbsoluteDeviation, i, args...)::MedianAbsoluteDeviation
    mu = nothing_scalar_array_view(r.mu, i)
    return MedianAbsoluteDeviation(; settings = r.settings, w = r.w, mu = mu, flag = r.flag)
end
"""
    calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights, <:MeanCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MedianCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights, <:MedianCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecNum, ...}, w::VecNum, ::Any)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:Number, ...}, ::Any, ::Any)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecScalar, ...}, w::VecNum, ::Any)

Compute the centering target for [`MedianAbsoluteDeviation`](@ref) risk measure calculations.

Dispatches on the type of `r.w` and `r.mu`:

  - No weights + `MeanCentering`: arithmetic mean of `x`.
  - `ObsWeights` + `MeanCentering`: weighted mean of `x`.
  - No weights + `MedianCentering`: median of `x`.
  - `ObsWeights` + `MedianCentering`: weighted median of `x`.
  - `VecNum` mu: dot product ``\\boldsymbol{w}^\\intercal \\boldsymbol{\\mu}``.
  - `Number` mu: the scalar `r.mu` directly.
  - `VecScalar` mu: ``\\boldsymbol{w}^\\intercal \\boldsymbol{\\mu}_v + \\mu_s``.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_deviations_vec`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the vector of deviations from the centering target for [`MedianAbsoluteDeviation`](@ref) risk measures.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::MedianAbsoluteDeviation, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the vector of deviations from the centering target for a precomputed returns series for [`MedianAbsoluteDeviation`](@ref).

Single-argument form used by the precomputed-returns functor `r(x::VecNum)` (ADR 0007).

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::MedianAbsoluteDeviation, x::VecNum)
    return x .- calc_moment_target(r, nothing, x)
end
function _moment_risk(r::MedianAbsoluteDeviation, val::VecNum)
    return StatsBase.mad(val; center = zero(eltype(val)), normalize = r.flag)
end
function (r::MedianAbsoluteDeviation)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    return _moment_risk(r, calc_deviations_vec(r, w, X, fees))
end
function (r::MedianAbsoluteDeviation)(x::VecNum)
    return _moment_risk(r, calc_deviations_vec(r, x))
end
# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::MedianAbsoluteDeviation) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`: [`MedianCenteringFunction`](@ref) targets are weight-independent centering
functions and can be evaluated on a bare return series.

# Related

  - [`weight_independent_target`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`MedianCenteringFunction`](@ref)
"""
weight_independent_target(::MedianCenteringFunction) = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`MedianAbsoluteDeviation`](@ref) `r` supports precomputed-return evaluation.

Delegates to [`weight_independent_target`](@ref) on `r.mu`: `true` iff the target is
`Nothing`, a `Number`, or a [`MedianCenteringFunction`](@ref); `false` for per-asset targets.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
supports_precomputed_returns(r::MedianAbsoluteDeviation) = weight_independent_target(r.mu)

export MedianAbsoluteDeviation, MedianCentering, MeanCentering
