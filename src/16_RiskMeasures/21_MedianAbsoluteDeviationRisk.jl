"""
$(DocStringExtensions.TYPEDEF)

Selects the centre about which a median absolute deviation takes the deviations of the portfolio returns.

A subtype centres the portfolio return series at the point of use, so the centre follows the weights of each call. [`MedianCentering`](@ref) takes the median of the series, and [`MeanCentering`](@ref) takes its mean. The two give different values on a skewed series. Each subtype takes the observation weights of the measure when the measure holds them.

# Related

  - [`MedianCentering`](@ref)
  - [`MeanCentering`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
abstract type MedianCenteringFunction <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Centres the portfolio returns on their median.

It is the default centre of [`MedianAbsoluteDeviation`](@ref). With this centre and `flag = true`, the measure is the median absolute deviation that estimates the standard deviation of a Gaussian series. When the measure holds observation weights, the centre is the weighted median `Statistics.median(x, w)`.

# Related

  - [`MedianCenteringFunction`](@ref)
  - [`MeanCentering`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
struct MedianCentering <: MedianCenteringFunction end
"""
$(DocStringExtensions.TYPEDEF)

Centres the portfolio returns on their mean.

[`MedianAbsoluteDeviation`](@ref) then gives the median of the absolute deviations about the mean. On a skewed series, this value is not the deviation about the median. When the measure holds observation weights, the centre is the weighted mean `Statistics.mean(x, w)`.

# Related

  - [`MedianCenteringFunction`](@ref)
  - [`MedianCentering`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
struct MeanCentering <: MedianCenteringFunction end
"""
    const MedAbsDevMu = Union{MuSlot, MedianCenteringFunction}

Union of valid centring-target types for [`MedianAbsoluteDeviation`](@ref).

The union admits a number, a vector of expected asset returns, an estimator that computes such a vector, or a [`MedianCenteringFunction`](@ref). The estimator is a **Deferred Quantity**, which [`MuSlot`](@ref) states.

The field has **two resolution points**, and the type of the value selects one. A centring function resolves in [`calc_moment_target`](@ref), at the point of use, because it centres the *portfolio* series and so needs the asset weights. A Deferred Quantity resolves in [`factory`](@ref), because it needs the returns matrix, which `calc_moment_target` never sees. The field has no `nothing` state, and the default is `MedianCentering()`.

The two resolutions give different quantities. The median is not linear, so `median(w'X) ≠ w' * median(X)`, and `MedianCentering()` does not agree with `mu = MedianExpectedReturns()`. A resolved vector is a gross expected return, and the series is net of the fee. So [`moment_target_fees`](@ref) subtracts the mean fee per period from the target, and the target is the net mean. `MeanCentering()` then agrees with a vector that is the mean of the same returns, with a fee or without one.

# Related

  - [`MuSlot`](@ref)
  - [`DeferredQuantity`](@ref)
  - [`MedianCenteringFunction`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
const MedAbsDevMu = Union{MuSlot, MedianCenteringFunction}
"""
$(DocStringExtensions.TYPEDEF)

Measures the dispersion of the portfolio returns as the median of their absolute deviations from a centre.

A few extreme returns move a median less than they move a mean of squares, so the measure reacts less to outliers than the standard deviation does. The alias [`MAD`](@ref) does not build it. `MAD` builds the mean absolute deviation, a [`LowOrderMoment`](@ref) with [`MeanAbsoluteDeviation`](@ref).

The measure is a [`HierarchicalRiskMeasure`](@ref). A hierarchical optimiser, such as [`HierarchicalRiskParity`](@ref) or [`HierarchicalEqualRiskContribution`](@ref), evaluates it on each cluster. The library has no `JuMP` model of a median, so the type bound of a `JuMP` optimiser, such as [`MeanRisk`](@ref), refuses it.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= x_t - \\mu\\,, \\\\
\\mathrm{MedAD}(\\boldsymbol{x}) &= c \\, \\underset{1 \\leq t \\leq T}{\\mathrm{median}} \\lvert \\delta_t \\rvert\\,, \\\\
c &= \\begin{cases}
  1 / \\Phi^{-1}(3/4) \\approx 1.4826 & \\texttt{flag} = \\texttt{true}\\,, \\\\
  1 & \\texttt{flag} = \\texttt{false}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathrm{MedAD}(\\boldsymbol{x})``: Median absolute deviation of the portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Centre of the deviations. [`calc_moment_target`](@ref) states it for each type of `mu`. It is the median of ``\\boldsymbol{x}`` under [`MedianCentering`](@ref) and its mean under [`MeanCentering`](@ref).
  - ``\\delta_t``: Deviation of the return at observation ``t`` from the centre.
  - ``c``: Consistency constant.
  - $(math_dict[:Phi_normal])

For a Gaussian series with standard deviation ``\\sigma``, the median of ``\\lvert x_t - \\mu \\rvert`` about the median is ``\\Phi^{-1}(3/4) \\, \\sigma``. So with `flag = true` and [`MedianCentering`](@ref), the measure is a consistent estimator of ``\\sigma``. When `w` holds observation weights, a weighted median replaces the median of the absolute deviations, and the centre takes the same weights. Equal weights give the unweighted value.

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
  - $(val_dict[:oow_rm])

!!! warning

    A stated `mu` is pinned. It crosses a Cross-Validation fold or a subset view as the value of the whole universe, so it does not follow the refit that the optimisation runs on. A caller who wants it to follow the fit names a **Deferred Quantity** in `mu`, or keeps a [`MedianCenteringFunction`](@ref), which recomputes the centre from the portfolio series at every call.

# Functor

    (r::MedianAbsoluteDeviation)(w::VecNum, X::MatNum, fees = nothing)
    (r::MedianAbsoluteDeviation)(x::VecNum)

Computes the median absolute deviation of the portfolio returns `X * w` net of `fees`, or of the net portfolio returns `x`. The second form needs a centre that does not read the weights, which [`supports_precomputed_returns`](@ref) reports.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.
  - `x::VecNum`: Net portfolio returns.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `mu`: A stated value is sliced to the selected indices through [`port_opt_view`](@ref). A **Deferred Quantity** passes through unsliced, and then fits on the subset.

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
  - [`calc_moment_target`](@ref)
  - [`StandardDeviation`](@ref): the dispersion that `flag = true` estimates on a Gaussian series.

# References

  - $(ref_dict[:hampel1974]) Pages 383–393. $(ref_dict[:no_original_source])
  - $(ref_dict[:rousseeuw1993])
"""
@propagatable @concrete struct MedianAbsoluteDeviation <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    """
    $(field_dict[:mu_mad_slot])
    """
    @vprop mu
    """
    If `true`, the measure multiplies the median of the absolute deviations by the consistency constant ``1 / \\Phi^{-1}(3/4)``, so that it estimates the standard deviation of a Gaussian series. If `false`, it returns the median itself.
    """
    flag
    function MedianAbsoluteDeviation(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:ObsWeights}, mu::MedAbsDevMu,
                                     flag::Bool = true)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu),
                      IsNonFiniteError("all elements of mu must be finite"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        assert_observation_weights(w, :w)
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

Resolve a **Deferred Quantity** in [`MedianAbsoluteDeviation`](@ref)'s `mu` slot against prior result `pr`.

This is the only change that `factory` makes to `mu`. The field is tagged [`@vprop`](@ref) and not [`@pprop`](@ref), so the prior never *fills* it. A bare `MedianAbsoluteDeviation()` keeps its [`MedianCentering`](@ref) after `factory`, and does not take `pr.mu`.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`MedAbsDevMu`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::MedianAbsoluteDeviation, pr::AbstractPriorResult,
                                     ::Any = nothing)
    return r
end
function resolve_deferred_quantities(r::MedianAbsoluteDeviation{<:Any, <:Any,
                                                                <:DeferredQuantity},
                                     pr::AbstractPriorResult, ::Any = nothing)
    return rebuild_with_slots(r, (; mu = resolve_slot(r.mu, :mu, pr)))
end
# Deferrable slots — see `deferred_slots`. A centring strategy in `mu` is not deferred: it
# resolves at the point of use, on the portfolio series.
deferred_slots(r::MedianAbsoluteDeviation) = (; mu = r.mu)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`MedianCenteringFunction`](@ref) `x` unchanged.

A centring function holds no value for each asset, so a view of a subset of the assets has nothing to slice.

# Related

  - [`MedianCenteringFunction`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
    return x
end
"""
    calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights, <:MeanCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MedianCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights, <:MedianCentering, ...}, ::Any, x::VecNum)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecNum, ...}, w::VecNum, ::Any)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:Number, ...}, ::Any, ::Any)
    calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:Any, <:VecScalar, ...}, w::VecNum, ::Any)

Compute the centre ``\\mu`` about which [`MedianAbsoluteDeviation`](@ref) takes the absolute deviations.

The method dispatches on the types of `r.w` and `r.mu`. A centring function reads the net series `x`, and a stated `mu` reads the asset weights. Observation weights in `r.w` change the centre of a centring function, and do not change a stated `mu`. A per-asset `mu` is gross, so [`calc_deviations_vec`](@ref) subtracts the mean fee per period from it.

# Mathematical definition

```math
\\begin{align}
\\mu &= \\begin{cases}
  \\underset{1 \\leq t \\leq T}{\\mathrm{median}} \\, x_t & \\texttt{MedianCentering}\\,, \\\\
  \\frac{1}{T} \\sum_{t=1}^{T} x_t & \\texttt{MeanCentering}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\boldsymbol{\\mu} & \\texttt{mu} \\text{ is a vector}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\boldsymbol{v} + s & \\texttt{mu} \\text{ is a } \\texttt{VecScalar}\\,, \\\\
  \\mu_0 & \\texttt{mu} \\text{ is a number}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\mu}``: Vector `mu`, one centre for each asset.
  - ``\\boldsymbol{v}``, ``s``: The fields `v` and `s` of a [`VecScalar`](@ref) `mu`.
  - ``\\mu_0``: Number `mu`.

When `r.w` holds observation weights ``\\omega_t``, the first case is the weighted median `Statistics.median(x, w)`, and the second case is ``\\sum_{t} \\omega_t x_t / \\sum_{t} \\omega_t``.

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
    w = checked_observation_weights(r.w, x)
    return Statistics.mean(x, w)
end
function calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MedianCentering,
                                                      <:Any}, ::Any, x::VecNum)
    return Statistics.median(x)
end
function calc_moment_target(r::MedianAbsoluteDeviation{<:Any, <:ObsWeights,
                                                       <:MedianCentering, <:Any}, ::Any,
                            x::VecNum)
    w = checked_observation_weights(r.w, x)
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

Compute the deviations ``\\delta_t`` of the net portfolio returns from the centre of [`MedianAbsoluteDeviation`](@ref) `r`.

# Algorithm

 1. Compute `x`, the portfolio returns net of `fees`, with [`calc_net_returns`](@ref).
 2. Compute `tgt`, the centre of [`calc_moment_target`](@ref) less the mean fee per observation of [`moment_target_fees`](@ref). The fee is not zero only for a per-asset `mu`, which states a gross expected return.
 3. Return `x .- tgt`.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::MedianAbsoluteDeviation, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x) - moment_target_fees(r.mu, w, fees, length(x))
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deviations ``\\delta_t`` of a precomputed series of portfolio returns `x` from the centre of [`MedianAbsoluteDeviation`](@ref) `r`.

The precomputed-returns functor `r(x::VecNum)` calls this method. The series carries no weights, so the centre is one that [`supports_precomputed_returns`](@ref) accepts: a [`MedianCenteringFunction`](@ref) or a number. A per-asset `mu` raises a `MethodError`.

# Related

  - [`MedianAbsoluteDeviation`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::MedianAbsoluteDeviation, x::VecNum)
    return x .- calc_moment_target(r, nothing, x)
end
function moment_risk(r::MedianAbsoluteDeviation, val::VecNum)
    w = checked_observation_weights(r.w, val)
    val = abs.(val)
    res = isnothing(w) ? Statistics.median(val) : Statistics.median(val, w)
    if r.flag
        res *= inv(Distributions.quantile(Distributions.Normal(zero(res), one(res)),
                                          3 // 4))
    end
    return res
end
function (r::MedianAbsoluteDeviation)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    return moment_risk(r, calc_deviations_vec(r, w, X, fees))
end
function (r::MedianAbsoluteDeviation)(x::VecNum)
    return moment_risk(r, calc_deviations_vec(r, x))
end
# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::MedianAbsoluteDeviation) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`. A [`MedianCenteringFunction`](@ref) reads the return series and no asset weights, so a bare return series is enough to evaluate it.

# Related

  - [`weight_independent_target`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`MedianCenteringFunction`](@ref)
"""
weight_independent_target(::MedianCenteringFunction)::Bool = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`MedianAbsoluteDeviation`](@ref) `r` supports precomputed-return evaluation.

The method returns [`weight_independent_target`](@ref) of `r.mu`. That is `true` for a [`MedianCenteringFunction`](@ref) and for a number, and `false` for a per-asset `mu`, which reads the asset weights.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`MedianAbsoluteDeviation`](@ref)
"""
supports_precomputed_returns(r::MedianAbsoluteDeviation) = weight_independent_target(r.mu)

export MedianAbsoluteDeviation, MedianCentering, MeanCentering
