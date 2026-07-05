"""
$(DocStringExtensions.TYPEDEF)

Represents a simple mean return measure for use in non-optimisation contexts.

`MeanReturn` computes the arithmetic (or geometric, when `flag = true`) mean of portfolio returns. It is used as the numerator in risk-adjusted performance ratios such as `MeanReturnRiskRatio`.

# Mathematical definition

For `flag = false` (arithmetic mean):

```math
\\begin{align}
\\bar{x} &= \\frac{1}{T} \\sum_{t=1}^{T} x_t\\,.
\\end{align}
```

Where:

  - ``\\bar{x}``: Arithmetic mean portfolio return.
  - $(math_dict[:xret])
  - $(math_dict[:T])

For `flag = true` (log-return mean):

```math
\\begin{align}
\\bar{x}_{\\log} &= \\frac{1}{T} \\sum_{t=1}^{T} \\log(1 + x_t)\\,.
\\end{align}
```

Where:

  - ``\\bar{x}_{\\log}``: Log-return mean portfolio return.
  - $(math_dict[:xret])
  - $(math_dict[:T])

For observation-weighted samples, the weighted mean is used instead.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanReturn(;
        w::Option{<:ObsWeights} = nothing,
        flag::Bool = false
    ) -> MeanReturn

Keywords correspond to the struct's fields.

## Validation

  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::MeanReturn)(x::VecNum)

Computes the mean return of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> MeanReturn()
MeanReturn
     w ┼ nothing
  flag ┴ Bool: false
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
"""
@propagatable @concrete struct MeanReturn <: NonOptimisationRiskMeasure
    """
    $(field_dict[:w_rm])
    """
    @pprop w
    """
    $(field_dict[:flag])
    """
    flag
    function MeanReturn(w::Option{<:ObsWeights}, flag::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(flag)}(w, flag)
    end
end
function MeanReturn(; w::Option{<:ObsWeights} = nothing, flag::Bool = false)::MeanReturn
    return MeanReturn(w, flag)
end
function (r::MeanReturn)(x::VecNum)
    if r.flag
        x = log1p.(x)
    end
    return isnothing(r.w) ? Statistics.mean(x) : Statistics.mean(x, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a mean return to risk ratio measure.

`MeanReturnRiskRatio` computes the ratio of the mean portfolio return (minus a risk-free rate) to a risk measure, used for performance analysis and comparison. It generalises the Sharpe ratio by allowing any risk measure in the denominator.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MRRR}(\\boldsymbol{x}) &= \\frac{\\bar{x} - r_f}{\\rho(\\boldsymbol{x})}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MRRR}(\\boldsymbol{x})``: Mean return to risk ratio.
  - $(math_dict[:xret])
  - ``\\bar{x}``: Mean portfolio return (computed by `rt`).
  - ``r_f``: Risk-free rate.
  - ``\\rho``: Base risk measure (computed by `rk`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanReturnRiskRatio(;
        rt::MeanReturn = MeanReturn(),
        rk::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        rf::Number = 0.0
    ) -> MeanReturnRiskRatio

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:rf])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `rt`: Recursively updated via [`factory`](@ref).
  - `rk`: Recursively updated via [`factory`](@ref).

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MeanReturn`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
"""
@propagatable @concrete struct MeanReturnRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:rt_mean])
    """
    @fprop rt
    """
    $(field_dict[:rk])
    """
    @fprop rk
    """
    $(field_dict[:rf])
    """
    rf
    function MeanReturnRiskRatio(rt::MeanReturn, rk::AbstractBaseRiskMeasure, rf::Number)
        @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
        return new{typeof(rt), typeof(rk), typeof(rf)}(rt, rk, rf)
    end
end
function MeanReturnRiskRatio(; rt::MeanReturn = MeanReturn(),
                             rk::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                             rf::Number = 0.0)::MeanReturnRiskRatio
    return MeanReturnRiskRatio(rt, rk, rf)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`MeanReturnRiskRatio`](@ref) `r` requires previous portfolio weights.

Delegates to the inner risk measure `r.rk`.

# Related

  - [`MeanReturnRiskRatio`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::MeanReturnRiskRatio)
    return needs_previous_weights(r.rk)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Third Central Moment risk measure.

`ThirdCentralMoment` computes the third central moment of portfolio returns about a specified centre. It is used as a measure of the asymmetry (skewness) of the return distribution in higher-order portfolio optimisation.

# Mathematical definition

Let ``\\mu`` be the specified centre and ``\\delta_t = x_t - \\mu`` the centred deviations. The third central moment is:

```math
\\begin{align}
m_3(\\boldsymbol{x}) &= \\frac{1}{T} \\sum_{t=1}^{T} \\delta_t^3\\,.
\\end{align}
```

Where:

  - ``m_3(\\boldsymbol{x})``: Third central moment of portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Specified centre of the distribution.
  - ``\\delta_t = x_t - \\mu``: Centred deviation at period ``t``.

For observation-weighted samples, the weighted mean is used.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThirdCentralMoment(;
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:Num_VecNum_VecScalar} = nothing
    ) -> ThirdCentralMoment

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is a `VecNum`: `!isempty(mu)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ThirdCentralMoment)(w::VecNum, X::MatNum, fees = nothing)

Computes the third central moment of the portfolio returns.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `mu`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ThirdCentralMoment()
ThirdCentralMoment
   w ┼ nothing
  mu ┴ nothing
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`Skewness`](@ref)
"""
@propagatable @concrete struct ThirdCentralMoment <: NonOptimisationRiskMeasure
    """
    $(field_dict[:w_rm])
    """
    @pprop w
    """
    $(field_dict[:mu_rm])
    """
    @pprop @vprop mu
    function ThirdCentralMoment(w::Option{<:ObsWeights}, mu::Option{<:Num_VecNum_VecScalar})
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        end
        return new{typeof(w), typeof(mu)}(w, mu)
    end
end
function ThirdCentralMoment(; w::Option{<:ObsWeights} = nothing,
                            mu::Option{<:Num_VecNum_VecScalar} = nothing)::ThirdCentralMoment
    return ThirdCentralMoment(w, mu)
end
"""
    const TCM_Sk{T1, T2} = Union{<:ThirdCentralMoment{T1, T2}, <:Skewness{<:Any, <:Any, T1, T2}}

Parameterised union of [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) sharing the same observation-weight (`T1`) and target-mean (`T2`) type parameters.

Used for unified dispatch on moment-target calculation methods.

# Related

  - [`ThirdCentralMoment`](@ref)
  - [`Skewness`](@ref)
  - [`calc_moment_target`](@ref)
"""
const TCM_Sk{T1, T2} = Union{<:ThirdCentralMoment{T1, T2},
                             <:Skewness{<:Any, <:Any, <:Any, T1, T2}}
"""
    calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
    calc_moment_target(r::TCM_Sk{<:StatsBase.AbstractWeights, Nothing}, ::Any, x::VecNum)
    calc_moment_target(r::TCM_Sk{<:Any, <:VecNum}, w::VecNum, ::Any)
    calc_moment_target(r::TCM_Sk{<:Any, <:VecScalar}, w::VecNum, ::Any)
    calc_moment_target(r::TCM_Sk{<:Any, <:Number}, ::Any, ::Any)

Compute the centering target for [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) risk measures.

Dispatches on the observation-weight type `T1` and mean type `T2` of [`TCM_Sk`](@ref):

  - No weights, no mu: arithmetic mean of `x`.
  - `AbstractWeights`, no mu: weighted mean of `x`.
  - `VecNum` mu: dot product ``\\boldsymbol{w}^\\intercal \\boldsymbol{\\mu}``.
  - `VecScalar` mu: ``\\boldsymbol{w}^\\intercal \\boldsymbol{\\mu}_v + \\mu_s``.
  - `Number` mu: the scalar `r.mu` directly.

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

Compute the vector of deviations from the centering target for [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) risk measures.

# Related

  - [`TCM_Sk`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the vector of deviations from the centering target for a precomputed returns series for [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) risk measures.

Single-argument form used by the precomputed-returns functor `r(x::VecNum)` (ADR 0007).

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
function moment_risk(r::ThirdCentralMoment{<:Option{<:StatsBase.AbstractWeights}},
                     val::VecNum)
    val .= val .^ 3
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function (r::ThirdCentralMoment{<:Option{<:StatsBase.AbstractWeights}})(w::VecNum,
                                                                        X::MatNum,
                                                                        fees::Option{<:Fees} = nothing)
    return moment_risk(r, calc_deviations_vec(r, w, X, fees))
end
function (r::ThirdCentralMoment{<:Option{<:StatsBase.AbstractWeights}})(x::VecNum)
    return moment_risk(r, calc_deviations_vec(r, x))
end
function (r::ThirdCentralMoment{<:DynamicAbstractWeights})(w::VecNum, X::MatNum,
                                                           fees::Option{<:Fees} = nothing)
    return ThirdCentralMoment(; w = get_observation_weights(r.w, X), mu = r.mu)(w, X, fees)
end
function (r::ThirdCentralMoment{<:DynamicAbstractWeights})(x::VecNum)
    return ThirdCentralMoment(; w = get_observation_weights(r.w, x), mu = r.mu)(x)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::MeanReturn) = NetReturnsInput()
risk_input_kind(::ThirdCentralMoment) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`ThirdCentralMoment`](@ref) `r` supports precomputed-return evaluation.

Delegates to [`weight_independent_target`](@ref) on `r.mu`: `true` iff the target is
`Nothing`, a `Number`, or a [`MedianCenteringFunction`](@ref); `false` for per-asset targets.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`ThirdCentralMoment`](@ref)
"""
supports_precomputed_returns(r::ThirdCentralMoment) = weight_independent_target(r.mu)

export MeanReturn, ThirdCentralMoment, MeanReturnRiskRatio
