"""
$(DocStringExtensions.TYPEDEF)

Measures the mean of the portfolio returns, or the mean of their logarithmic returns.

The measure is a [`NonOptimisationRiskMeasure`](@ref). It is a return and not a risk, so the type bound of every optimiser refuses it. An optimiser reads the expected return through its objective function, such as [`MaximumReturn`](@ref). A larger mean is better, so [`bigger_is_better`](@ref) returns `true`, and an asset selection or a parameter search keeps the largest mean. The measure is the numerator of [`MeanReturnRiskRatio`](@ref).

# Mathematical definition

For `flag = false`, the arithmetic mean:

```math
\\begin{align}
\\bar{x} &= \\frac{1}{T} \\sum_{t=1}^{T} x_t\\,.
\\end{align}
```

Where:

  - ``\\bar{x}``: Arithmetic mean portfolio return.
  - $(math_dict[:xret])
  - $(math_dict[:T])

For `flag = true`, the mean of the logarithmic returns:

```math
\\begin{align}
\\bar{x}_{\\log} &= \\frac{1}{T} \\sum_{t=1}^{T} \\log(1 + x_t)\\,.
\\end{align}
```

Where:

  - ``\\bar{x}_{\\log}``: Mean of the logarithmic portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])

When `w` holds observation weights ``\\omega_t``, each form takes the weighted mean. The first form is then ``\\sum_{t} \\omega_t x_t / \\sum_{t} \\omega_t``, and the second form puts ``\\log(1 + x_t)`` in place of ``x_t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanReturn(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        flag::Bool = false
    ) -> MeanReturn

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow_rm])

# Functor

    (r::MeanReturn)(x::VecNum)

Computes the mean return of the net portfolio returns `x`. [`expected_risk`](@ref) charges the fees before the call.

## Arguments

  - `x::VecNum`: Net portfolio returns.

# Examples

```jldoctest
julia> MeanReturn()
MeanReturn
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
         w ┼ nothing
      flag ┴ Bool: false
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`ExpectedReturn`](@ref): the expected return of a prior, which a `JuMP` optimiser reads.
"""
@propagatable @concrete struct MeanReturn <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    """
    If `true`, the measure takes the mean of `log1p.(x)`, the logarithmic returns. If `false`, it takes the mean of `x`.
    """
    flag
    function MeanReturn(settings::HierarchicalRiskMeasureSettings, w::Option{<:ObsWeights},
                        flag::Bool)
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w), typeof(flag)}(settings, w, flag)
    end
end
function MeanReturn(;
                    settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                    w::Option{<:ObsWeights} = nothing, flag::Bool = false)::MeanReturn
    return MeanReturn(settings, w, flag)
end
function (r::MeanReturn)(x::VecNum)
    if r.flag
        x = log1p.(x)
    end
    w = checked_observation_weights(r.w, x)
    return isnothing(w) ? Statistics.mean(x) : Statistics.mean(x, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the mean portfolio return in excess of a risk-free rate, per unit of risk.

The numerator is the mean of [`MeanReturn`](@ref) `rt` less the risk-free rate `rf`, and the denominator is the risk of `rk`. With a [`StandardDeviation`](@ref) in `rk` and the arithmetic mean in `rt`, the ratio is the reward-to-variability ratio of [sharpe1966](@cite), the Sharpe ratio. Another risk measure in `rk` gives a generalisation of that ratio.

A larger ratio is better, so [`bigger_is_better`](@ref) returns `true`. The ratio is a [`NonOptimisationRiskMeasure`](@ref), because it is not a risk, and the type bound of every optimiser refuses it. The objective [`MaximumRatio`](@ref) of a `JuMP` optimiser maximises a ratio of this kind. The quotient follows IEEE division. A zero risk gives `Inf` for a positive numerator, `NaN` for a zero numerator and `-Inf` for a negative numerator. [`needs_previous_weights`](@ref) answers as `rk` does.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MRRR}(\\boldsymbol{x}) &= \\frac{\\bar{x} - r_f}{\\rho(\\boldsymbol{x})}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MRRR}(\\boldsymbol{x})``: Mean return to risk ratio.
  - $(math_dict[:xret])
  - ``\\bar{x}``: Mean portfolio return of `rt`.
  - ``r_f``: Risk-free rate `rf`.
  - ``\\rho(\\boldsymbol{x})``: Risk of `rk`. A vector `rk` gives one number through `sca`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanReturnRiskRatio(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        rt::MeanReturn = MeanReturn(),
        rk::BaseRM_VecBaseRM = ConditionalValueatRisk(),
        sca::Scalariser = SumScalariser(),
        rf::Number = 0.0
    ) -> MeanReturnRiskRatio

Keywords correspond to the struct's fields.

## Multiplicity

`rk` takes one risk measure or a vector of them. `sca` scalarises a vector into **one** number, and weights each element by its own `settings.scale`. The field beats the `sca` keyword of a caller, so a value that this type reports always uses the scalariser that the type names.

## Validation

  - If `rk` is a vector: `!isempty(rk)`.
  - $(val_dict[:rf])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `rt`: Recursively updated through [`factory`](@ref).
  - `rk`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> MeanReturnRiskRatio()
MeanReturnRiskRatio
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
        rt ┼ MeanReturn
           │   settings ┼ HierarchicalRiskMeasureSettings
           │            │   scale ┴ Float64: 1.0
           │          w ┼ nothing
           │       flag ┴ Bool: false
        rk ┼ ConditionalValueatRisk
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: true
           │      alpha ┼ Float64: 0.05
           │          w ┴ nothing
       sca ┼ SumScalariser()
        rf ┴ Float64: 0.0
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MeanReturn`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`NonOptimisationRiskRatio`](@ref): a ratio of any two measures.
  - [`MaximumRatio`](@ref)

# References

  - $(ref_dict[:sharpe1966]) Pages 122–123.
"""
@propagatable @concrete struct MeanReturnRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:rt_mean])
    """
    @fprop rt
    """
    $(field_dict[:rk])
    """
    @fprop rk
    """
    $(field_dict[:sca_rk])
    """
    sca
    """
    $(field_dict[:rf])
    """
    rf
    function MeanReturnRiskRatio(settings::HierarchicalRiskMeasureSettings, rt::MeanReturn,
                                 rk::BaseRM_VecBaseRM, sca::Scalariser, rf::Number)
        if isa(rk, VecBaseRM)
            @argcheck(!isempty(rk), IsEmptyError("rk cannot be empty"))
        end
        @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
        return new{typeof(settings), typeof(rt), typeof(rk), typeof(sca), typeof(rf)}(settings,
                                                                                      rt,
                                                                                      rk,
                                                                                      sca,
                                                                                      rf)
    end
end
function MeanReturnRiskRatio(;
                             settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                             rt::MeanReturn = MeanReturn(),
                             rk::BaseRM_VecBaseRM = ConditionalValueatRisk(),
                             sca::Scalariser = SumScalariser(),
                             rf::Number = 0.0)::MeanReturnRiskRatio
    return MeanReturnRiskRatio(settings, rt, rk, sca, rf)
end
# Deferrable slots — see `deferred_slots`. `rt` is bounded to `MeanReturn`, which defers
# nothing, so only the risk side is declared.
deferred_slots(r::MeanReturnRiskRatio) = (; rk = r.rk)
"""
    reader_frames(r::MeanReturnRiskRatio, path::AbstractString,
                  refit::Bool) -> ReturnsFrames

Return the frames of the readers inside the risk side of a [`MeanReturnRiskRatio`](@ref), for the frame walk of a head.

The risk side `rk` can hold a risk measure with a reader, such as an [`UncertaintySetVariance`](@ref). Each of those readers belongs to the group of the head that holds the measure, so the method walks every field of `r` with [`field_frames`](@ref). A field that holds no reader adds nothing.

# Arguments

  - `r`: The risk measure.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the fields of `r`.

# Related

  - [`reader_frames`](@ref)
  - [`field_frames`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
"""
function reader_frames(r::MeanReturnRiskRatio, path::AbstractString, refit::Bool)
    return field_frames(r, path, refit)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`MeanReturnRiskRatio`](@ref) `r` reads the previous portfolio weights.

The method returns `needs_previous_weights(r.rk)`. The return side `rt` is a [`MeanReturn`](@ref), which never reads them. A vector `rk` reads them when one of its measures does, such as a [`TurnoverRiskMeasure`](@ref) whose `fixed` is `false`.

# Related

  - [`MeanReturnRiskRatio`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::MeanReturnRiskRatio)
    return needs_previous_weights(r.rk)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the third central moment of the portfolio returns.

The moment has the divisor ``T``. It is not divided by the cube of the standard deviation, so it is not the skewness coefficient of [`Skewness`](@ref). It measures the asymmetry of the returns, and a larger moment is better, so [`bigger_is_better`](@ref) returns `true`.

The measure is a [`NonOptimisationRiskMeasure`](@ref). The library has no `JuMP` model of it, so the type bound of every optimiser refuses it. It serves reports, and the scores of an asset selection or a parameter search.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= x_t - \\mu\\,, \\\\
m_3(\\boldsymbol{x}) &= \\frac{1}{T} \\sum_{t=1}^{T} \\delta_t^3\\,.
\\end{align}
```

Where:

  - ``m_3(\\boldsymbol{x})``: Third central moment of the portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Centre of the moment. [`calc_moment_target`](@ref) states it for each type of `mu`.
  - ``\\delta_t``: Centred portfolio return at observation ``t``.

When `w` holds observation weights ``\\omega_t``, the weighted mean ``\\sum_{t} \\omega_t \\delta_t^3 / \\sum_{t} \\omega_t`` replaces the mean, and the centre takes the same weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThirdCentralMoment(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:MuSlot} = nothing
    ) -> ThirdCentralMoment

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is a `VecNum`: `!isempty(mu)` and `all(isfinite, mu)`.
  - If `mu` is a `Number`: `isfinite(mu)`.
  - $(val_dict[:oow_rm])

!!! warning

    A stated `mu` is pinned. It crosses a Cross-Validation fold or a subset view as the value of the whole universe, so it does not follow the refit that the optimisation runs on. A caller who wants it to follow the fit names a **Deferred Quantity** in `mu`, or leaves the slot `nothing` and lets the prior supply it.

# Functor

    (r::ThirdCentralMoment)(w::VecNum, X::MatNum, fees = nothing)
    (r::ThirdCentralMoment)(x::VecNum)

Computes the third central moment of the portfolio returns `X * w` net of `fees`, or of the net portfolio returns `x`. The second form needs a centre that does not read the weights, which [`supports_precomputed_returns`](@ref) reports.

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
julia> ThirdCentralMoment()
ThirdCentralMoment
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
         w ┼ nothing
        mu ┴ nothing
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`Skewness`](@ref): the third moment divided by the cube of the standard deviation.
  - [`calc_moment_target`](@ref)
  - [`bigger_is_better`](@ref)
"""
@propagatable @concrete struct ThirdCentralMoment <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    """
    $(field_dict[:mu_slot])
    """
    @pprop @vprop mu
    function ThirdCentralMoment(settings::HierarchicalRiskMeasureSettings,
                                w::Option{<:ObsWeights}, mu::Option{<:MuSlot})
        assert_observation_weights(w, :w)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu),
                      IsNonFiniteError("all elements of mu must be finite"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        return new{typeof(settings), typeof(w), typeof(mu)}(settings, w, mu)
    end
end
function ThirdCentralMoment(;
                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                            w::Option{<:ObsWeights} = nothing,
                            mu::Option{<:MuSlot} = nothing)::ThirdCentralMoment
    return ThirdCentralMoment(settings, w, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in [`ThirdCentralMoment`](@ref)'s `mu` slot against prior result `pr`. The measure carries one prior-derived slot, so the slot itself admits the estimator and there is no fan-out to make.

# Related

  - [`ThirdCentralMoment`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::ThirdCentralMoment, pr::AbstractPriorResult,
                                     ::Any = nothing)
    return r
end
function resolve_deferred_quantities(r::ThirdCentralMoment{<:Any, <:Any,
                                                           <:DeferredQuantity},
                                     pr::AbstractPriorResult, ::Any = nothing)
    return rebuild_with_slots(r, (; mu = resolve_slot(r.mu, :mu, pr)))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::ThirdCentralMoment) = (; mu = r.mu)
"""
    const TCM_Sk{T1, T2} = Union{ThirdCentralMoment{<:Any, T1, T2}, Skewness{<:Any, <:Any, <:Any, T1, T2}}

Parameterised union of [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) sharing the same observation-weight (`T1`) and target-mean (`T2`) type parameters.

Used for unified dispatch on moment-target calculation methods.

# Related

  - [`ThirdCentralMoment`](@ref)
  - [`Skewness`](@ref)
  - [`calc_moment_target`](@ref)
"""
const TCM_Sk{T1, T2} = Union{ThirdCentralMoment{<:Any, T1, T2},
                             Skewness{<:Any, <:Any, <:Any, T1, T2}}
"""
    calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
    calc_moment_target(r::TCM_Sk{<:StatsBase.AbstractWeights, Nothing}, ::Any, x::VecNum)
    calc_moment_target(r::TCM_Sk{<:Any, <:VecNum}, w::VecNum, ::Any)
    calc_moment_target(r::TCM_Sk{<:Any, <:VecScalar}, w::VecNum, ::Any)
    calc_moment_target(r::TCM_Sk{<:Any, <:Number}, ::Any, ::Any)

Compute the centre ``\\mu`` about which [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) take their moment.

The method dispatches on the observation-weight type `T1` and the type `T2` of `mu` of [`TCM_Sk`](@ref). Observation weights in `r.w` change the mean of `x`, and do not change a stated `mu`. A per-asset `mu` is gross, so [`calc_deviations_vec`](@ref) subtracts the mean fee per period from it.

# Mathematical definition

```math
\\begin{align}
\\mu &= \\begin{cases}
  \\frac{1}{T} \\sum_{t=1}^{T} x_t & \\texttt{mu} \\text{ and } \\texttt{w} \\text{ are } \\texttt{nothing}\\,, \\\\
  \\sum_{t=1}^{T} \\omega_t x_t \\Big/ \\sum_{t=1}^{T} \\omega_t & \\texttt{mu} \\text{ is } \\texttt{nothing} \\text{ and } \\texttt{w} \\text{ holds weights}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\boldsymbol{\\mu} & \\texttt{mu} \\text{ is a vector}\\,, \\\\
  \\boldsymbol{w}^\\intercal \\boldsymbol{v} + s & \\texttt{mu} \\text{ is a } \\texttt{VecScalar}\\,, \\\\
  \\mu_0 & \\texttt{mu} \\text{ is a number}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\omega_t``: Observation weight of observation ``t``, from `r.w`.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\mu}``: Vector `mu`, one centre for each asset.
  - ``\\boldsymbol{v}``, ``s``: The fields `v` and `s` of a [`VecScalar`](@ref) `mu`.
  - ``\\mu_0``: Number `mu`.

# Related

  - [`TCM_Sk`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`Skewness`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_deviations_vec`](@ref)
"""
function calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
    return Statistics.mean(x)
end
function calc_moment_target(r::TCM_Sk{<:StatsBase.AbstractWeights, Nothing}, ::Any,
                            x::VecNum)
    return Statistics.mean(x, r.w)
end
function calc_moment_target(r::TCM_Sk{<:Any, <:VecNum}, w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_moment_target(r::TCM_Sk{<:Any, <:VecScalar}, w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
function calc_moment_target(r::TCM_Sk{<:Any, <:Number}, ::Any, ::Any)
    return r.mu
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deviations ``\\delta_t`` of the net portfolio returns from the centre of a [`ThirdCentralMoment`](@ref) or a [`Skewness`](@ref) `r`.

# Algorithm

 1. Compute `x`, the portfolio returns net of `fees`, with [`calc_net_returns`](@ref).
 2. Compute `tgt`, the centre of [`calc_moment_target`](@ref) less the mean fee per observation of [`moment_target_fees`](@ref). The fee is not zero only for a per-asset `mu`, which states a gross expected return.
 3. Return `x .- tgt`.

# Related

  - [`TCM_Sk`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x) - moment_target_fees(r.mu, w, fees, length(x))
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deviations ``\\delta_t`` of a precomputed series of portfolio returns `x` from the centre of a [`ThirdCentralMoment`](@ref) or a [`Skewness`](@ref) `r`.

The precomputed-returns functor `r(x::VecNum)` calls this method. The series carries no weights, so the centre is one that [`supports_precomputed_returns`](@ref) accepts: the mean of `x`, its weighted mean, or a number. A per-asset `mu` raises a `MethodError`.

# Related

  - [`TCM_Sk`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`Skewness`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::TCM_Sk, x::VecNum)
    return x .- calc_moment_target(r, nothing, x)
end
function moment_risk(r::ThirdCentralMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}},
                     val::VecNum)
    val .= val .^ 3
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
# Evaluation entry points — see `MomentRiskMeasures` in `25_RiskMeasureTools.jl`.

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::MeanReturn) = NetReturnsInput()
risk_input_kind(::ThirdCentralMoment) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`ThirdCentralMoment`](@ref) `r` supports precomputed-return evaluation.

The method returns [`weight_independent_target`](@ref) of `r.mu`. That is `true` when `mu` is `nothing` or a number, and `false` for a per-asset `mu`, which reads the asset weights.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`ThirdCentralMoment`](@ref)
"""
supports_precomputed_returns(r::ThirdCentralMoment) = weight_independent_target(r.mu)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`, because a larger third central moment is better.

The moment has the sign of the skewness, and [`bigger_is_better`](@ref) of [`Skewness`](@ref) also returns `true`. The generic method for an [`AbstractBaseRiskMeasure`](@ref) returns `false`. The asset selection, the parameter search and the ranking of cross-validation results read this trait to choose the direction of the score.

# Related

  - [`ThirdCentralMoment`](@ref)
  - [`bigger_is_better`](@ref)
"""
function bigger_is_better(::ThirdCentralMoment)::Bool
    return true
end

export MeanReturn, ThirdCentralMoment, MeanReturnRiskRatio
