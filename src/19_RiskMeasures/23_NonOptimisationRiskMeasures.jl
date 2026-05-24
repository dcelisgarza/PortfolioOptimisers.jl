"""
$(DocStringExtensions.TYPEDEF)

Represents a simple mean return measure for use in non-optimisation contexts.

`MeanReturn` computes the arithmetic (or geometric, when `flag = true`) mean of portfolio returns. It is used as the numerator in risk-adjusted performance ratios such as `MeanReturnRiskRatio`.

# Mathematical definition

For `flag = false` (arithmetic mean):

```math
\\bar{x} = \\frac{1}{T} \\sum_{t=1}^{T} x_t\\,.
```

For `flag = true` (log-return mean):

```math
\\bar{x}_{\\log} = \\frac{1}{T} \\sum_{t=1}^{T} \\log(1 + x_t)\\,.
```

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
@concrete struct MeanReturn <: NonOptimisationRiskMeasure
    "$(field_dict[:w_rm])"
    w
    "$(field_dict[:flag])"
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
function factory(r::MeanReturn, pr::AbstractPriorResult, args...)::MeanReturn
    w = nothing_scalar_array_selector(r.w, pr.w)
    return MeanReturn(; w = w, flag = r.flag)
end
function risk_measure_view(r::MeanReturn, ::Any, args...)::MeanReturn
    return r
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a mean return to risk ratio measure.

`MeanReturnRiskRatio` computes the ratio of the mean portfolio return (minus a risk-free rate) to a risk measure, used for performance analysis and comparison. It generalises the Sharpe ratio by allowing any risk measure in the denominator.

# Mathematical definition

```math
\\mathrm{MRRR}(\\boldsymbol{x}) = \\frac{\\bar{x} - r_f}{\\rho(\\boldsymbol{x})}\\,,
```

where ``\\bar{x}`` is the mean return (computed by `rt`), ``r_f`` is the risk-free rate, and ``\\rho`` is any base risk measure (computed by `rk`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanReturnRiskRatio(;
        rt::MeanReturn = MeanReturn(),
        rk::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
        rf::Number = 0.0
    ) -> MeanReturnRiskRatio

Keywords correspond to the struct's fields.

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MeanReturn`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
"""
@concrete struct MeanReturnRiskRatio <: NonOptimisationRiskMeasure
    "$(field_dict[:rt_mean])"
    rt
    "$(field_dict[:rk])"
    rk
    "$(field_dict[:rf])"
    rf
    function MeanReturnRiskRatio(rt::MeanReturn, rk::AbstractBaseRiskMeasure, rf::Number)
        return new{typeof(rt), typeof(rk), typeof(rf)}(rt, rk, rf)
    end
end
function MeanReturnRiskRatio(; rt::MeanReturn = MeanReturn(),
                             rk::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                             rf::Number = 0.0)::MeanReturnRiskRatio
    return MeanReturnRiskRatio(rt, rk, rf)
end
function factory(r::MeanReturnRiskRatio, args...; kwargs...)::MeanReturnRiskRatio
    rt = factory(r.rt, args...)
    rk = factory(r.rk, args...; kwargs...)
    return MeanReturnRiskRatio(; rt = rt, rk = rk, rf = r.rf)
end
function factory(r::MeanReturnRiskRatio, w::VecNum)::MeanReturnRiskRatio
    return MeanReturnRiskRatio(; rt = r.rt, rk = factory(r.rk, w), rf = r.rf)
end
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
m_3(\\boldsymbol{x}) = \\frac{1}{T} \\sum_{t=1}^{T} \\delta_t^3\\,.
```

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
@concrete struct ThirdCentralMoment <: NonOptimisationRiskMeasure
    "$(field_dict[:w_rm])"
    w
    "$(field_dict[:mu_rm])"
    mu
    function ThirdCentralMoment(w::Option{<:ObsWeights}, mu::Option{<:Num_VecNum_VecScalar})
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
        end
        return new{typeof(w), typeof(mu)}(w, mu)
    end
end
function ThirdCentralMoment(; w::Option{<:ObsWeights} = nothing,
                            mu::Option{<:Num_VecNum_VecScalar} = nothing)::ThirdCentralMoment
    return ThirdCentralMoment(w, mu)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the standardised Skewness risk measure.

`Skewness` computes the third standardised central moment (skewness) of portfolio returns. Negative skewness indicates a return distribution with a heavier left tail.

# Mathematical definition

Let ``\\mu`` be the specified centre, ``\\delta_t = x_t - \\mu``, and ``\\sigma`` the standard deviation of returns. The skewness is:

```math
\\mathrm{Skew}(\\boldsymbol{x}) = \\frac{1}{T \\sigma^3} \\sum_{t=1}^{T} \\delta_t^3\\,.
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Skewness(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:Num_VecNum_VecScalar} = nothing
    ) -> Skewness

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is a `VecNum`: `!isempty(mu)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::Skewness)(w::VecNum, X::MatNum, fees = nothing)

Computes the skewness of the portfolio returns.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

# Examples

```jldoctest
julia> Skewness()
Skewness
  ve ┼ SimpleVariance
     │          me ┼ SimpleExpectedReturns
     │             │   w ┴ nothing
     │           w ┼ nothing
     │   corrected ┴ Bool: true
   w ┼ nothing
  mu ┴ nothing
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
"""
@concrete struct Skewness <: NonOptimisationRiskMeasure
    "$(field_dict[:ve])"
    ve
    "$(field_dict[:w_rm])"
    w
    "$(field_dict[:mu_rm])"
    mu
    function Skewness(ve::AbstractVarianceEstimator, w::Option{<:ObsWeights},
                      mu::Option{<:Num_VecNum_VecScalar})
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
        end
        return new{typeof(ve), typeof(w), typeof(mu)}(ve, w, mu)
    end
end
function Skewness(; ve::AbstractVarianceEstimator = SimpleVariance(),
                  w::Option{<:ObsWeights} = nothing,
                  mu::Option{<:Num_VecNum_VecScalar} = nothing)::Skewness
    return Skewness(ve, w, mu)
end
"""
    const TCM_Sk{T1, T2} = Union{...}

Parameterised union of [`ThirdCentralMoment`](@ref) and [`Skewness`](@ref) sharing the same observation-weight (`T1`) and target-mean (`T2`) type parameters.

Used for unified dispatch on moment-target calculation methods.

# Related

  - [`ThirdCentralMoment`](@ref)
  - [`Skewness`](@ref)
  - [`calc_moment_target`](@ref)
"""
const TCM_Sk{T1, T2} = Union{<:ThirdCentralMoment{T1, T2}, <:Skewness{<:Any, T1, T2}}
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
function calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
function factory(r::ThirdCentralMoment, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return ThirdCentralMoment(; w = w, mu = mu)
end
function risk_measure_view(r::ThirdCentralMoment, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return ThirdCentralMoment(; w = r.w, mu = mu)
end
function (r::ThirdCentralMoment{<:Option{<:StatsBase.AbstractWeights}})(w::VecNum,
                                                                        X::MatNum,
                                                                        fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 3
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function (r::ThirdCentralMoment{<:DynamicAbstractWeights})(w::VecNum, X::MatNum,
                                                           fees::Option{<:Fees} = nothing)
    return ThirdCentralMoment(; w = get_observation_weights(r.w, X), mu = r.mu)(w, X, fees)
end
function factory(r::Skewness, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return Skewness(; ve = factory(r.ve, w), w = w, mu = mu)
end
function risk_measure_view(r::Skewness, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return Skewness(; ve = r.ve, w = r.w, mu = mu)
end
function (r::Skewness{<:Any, <:Option{<:StatsBase.AbstractWeights}})(w::VecNum, X::MatNum,
                                                                     fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.std(r.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^3
end
function (r::Skewness{<:Any, <:DynamicAbstractWeights})(w::VecNum, X::MatNum,
                                                        fees::Option{<:Fees} = nothing)
    return Skewness(; ve = r.ve, w = get_observation_weights(r.w, X), mu = r.mu)(w, X, fees)
end

export MeanReturn, ThirdCentralMoment, Skewness, MeanReturnRiskRatio
