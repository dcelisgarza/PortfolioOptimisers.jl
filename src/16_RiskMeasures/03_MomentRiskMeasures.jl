"""
$(DocStringExtensions.TYPEDEF)

Selects the reduction that turns the moment deviations of a portfolio into one risk value.

A [`LowOrderMoment`](@ref) holds a [`LowOrderMomentMeasureAlgorithm`](@ref), and a [`HighOrderMoment`](@ref) holds a [`HighOrderMomentMeasureAlgorithm`](@ref). The measure computes the deviations of the net portfolio returns from its Moment Target, and the algorithm states how [`moment_risk`](@ref) reduces them. Each concrete algorithm states its reduction under `# Mathematical definition`.

# Related

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`moment_risk`](@ref)
"""
abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects a moment of order one or two, or the root of an even moment, for a low-order moment risk measure.

[`LowOrderMoment`](@ref) holds it. The reduction of each member is homogeneous in the deviations, of degree one or two, so the model of each member is a linear, second-order cone or power cone programme. [`SecondMoment`](@ref) subtypes this type directly, and [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref) holds the other members.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`SecondMoment`](@ref)
  - [`LowOrderMoment`](@ref)
"""
abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Groups the low-order moments that hold no variance estimator.

The members are [`FirstLowerMoment`](@ref), [`MeanAbsoluteDeviation`](@ref) and [`EvenMoment`](@ref). Each reads only the deviations and the observation weights of the measure. No method of the library dispatches on the group. It is the low-order mirror of [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref), and the library has no standardised low-order moment.

# Related

  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`EvenMoment`](@ref)
"""
abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the moment measure algorithm `alg` unchanged.

A member that holds no variance estimator has nothing that a prior or observation weights can replace. [`factory`](@ref) on [`LowOrderMoment`](@ref) and [`HighOrderMoment`](@ref) calls this method for such a member. [`SecondMoment`](@ref) and [`StandardisedHighOrderMoment`](@ref) have methods of their own, which pass the weights to `ve`.

# Related

  - [`MomentMeasureAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::MomentMeasureAlgorithm, args...; kwargs...)::MomentMeasureAlgorithm
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the mean shortfall of the net portfolio returns below the Moment Target.

It is the lower partial moment of order one. Equation 7.45 of [cajas2025](@cite) states the lower partial moment, and it names [lpm](@cite) as the source. No copy of that work was found. A number in `mu` is a minimum acceptable return, and `mu = nothing` gives the mean shortfall below the mean, the semi-deviation of order one.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\mathbb{E}\\left[\\max(\\tau - \\hat{r}_t,\\, 0)\\right] = -\\mathbb{E}\\left[d_t^{-}\\right]\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:E_obs_mean])
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:tau_moment])
  - $(math_dict[:d_t_lower])

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`MeanAbsoluteDeviation`](@ref): the two-sided deviation from the same target.
  - [`LowOrderMoment`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.2, Equation 7.45 with ``q = 1``. $(ref_dict[:no_original_source])
  - $(ref_dict[:lpm])
"""
struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the mean absolute deviation of the net portfolio returns from the Moment Target.

Equation 7.31 of [cajas2025](@cite) states it with the mean as the target, and it names [mad](@cite) as the source. No copy of that work was found. The library adds the other targets of the `mu` slot.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\mathbb{E}\\left[\\left\\lvert d_t \\right\\rvert\\right] = 2\\, \\mathbb{E}\\left[\\max(\\tau - \\hat{r}_t,\\, 0)\\right] + \\mathbb{E}\\left[\\hat{r}_t\\right] - \\tau\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:E_obs_mean])
  - $(math_dict[:d_t_moment])
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:tau_moment])

The second form follows from ``\\lvert d \\rvert = 2 \\max(-d, 0) + d``. When ``\\tau`` is the observation mean of the net returns, the last two terms cancel, and the measure is twice the [`FirstLowerMoment`](@ref) of the same data. Against any other target, the measure differs from twice the first lower moment by ``\\mathbb{E}[\\hat{r}_t] - \\tau``.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`FirstLowerMoment`](@ref): the one-sided deviation from the same target.
  - [`LowOrderMoment`](@ref)
  - [`MedianAbsoluteDeviation`](@ref): a different risk measure, which centres at the median.

# References

  - $(ref_dict[:cajas2025]) Section 7.2.1.3, Equation 7.31. $(ref_dict[:no_original_source])
  - $(ref_dict[:mad])
"""
struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the second moment of the net portfolio returns about the Moment Target, or its square root.

`alg1` selects the deviations. [`FullMoment`](@ref) reads all of them, and [`SemiMoment`](@ref) reads the deviations below the target alone. `alg2` selects the units. [`SOCRiskExpr`](@ref) reports the square root, a standard deviation, and the three other formulations report the second moment itself.

With `mu = nothing` the measure is the variance or the semi-variance of the net portfolio returns. Against a stated target it is the second moment about that target, not a variance. `ve` reads the deviations at a mean of zero. So it supplies only its divisor and its observation weights, and its `me` has no effect. Page 81 of [markowitz1952](@cite) states the variance. Pages 188 and 189 of [markowitz1959](@cite) state the semi-variance about a target ``b`` and about the mean ``E``, with the divisor ``T``.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= f \\sum_{t=1}^{T} w_t\\, d_t^{2}\\,, \\\\
R(\\boldsymbol{w}) &= f \\sum_{t=1}^{T} w_t \\left(d_t^{-}\\right)^{2}\\,.
\\end{align}
```

The first line is the full moment and the second line is the semi moment. [`SOCRiskExpr`](@ref) reports ``\\sqrt{R(\\boldsymbol{w})}``.

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:f_ve_div])
  - $(math_dict[:w_t_obs]) It is ``1`` for every ``t`` when the measure holds no observation weights.
  - $(math_dict[:d_t_moment])
  - $(math_dict[:d_t_lower])
  - $(math_dict[:T])

The default `ve` sets `corrected = true`, so the default divisor is ``1/(T-1)``. Set `ve = SimpleVariance(; me = nothing, corrected = false)` for the divisor ``1/T`` of [markowitz1959](@cite).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SecondMoment(;
        ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
        alg1::AbstractMomentAlgorithm = FullMoment(),
        alg2::SecondMomentFormulation = SquaredSOCRiskExpr(),
    ) -> SecondMoment

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> SecondMoment()
SecondMoment
    ve ┼ SimpleVariance
       │          me ┼ nothing
       │           w ┼ nothing
       │   corrected ┴ Bool: true
  alg1 ┼ FullMoment()
  alg2 ┴ SquaredSOCRiskExpr()
```

# Related

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
  - [`SecondMomentFormulation`](@ref): each formulation states the cone or the quadratic form that the model builds on the deviations.
  - [`factory`](@ref)

# References

  - $(ref_dict[:markowitz1952]) Page 81.
  - $(ref_dict[:markowitz1959]) Chapter IX, pages 188 and 189.
"""
@propagatable @concrete struct SecondMoment <: LowOrderMomentMeasureAlgorithm
    """
    $(field_dict[:ve])
    """
    @fprop ve
    """
    $(field_dict[:alg1])
    """
    alg1
    """
    $(field_dict[:alg2])
    """
    alg2
    function SecondMoment(ve::AbstractVarianceEstimator, alg1::AbstractMomentAlgorithm,
                          alg2::SecondMomentFormulation)
        return new{typeof(ve), typeof(alg1), typeof(alg2)}(ve, alg1, alg2)
    end
end
function SecondMoment(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                      alg1::AbstractMomentAlgorithm = FullMoment(),
                      alg2::SecondMomentFormulation = SquaredSOCRiskExpr())::SecondMoment
    return SecondMoment(ve, alg1, alg2)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the ``p``-th root of the even moment of order ``2p`` of the net portfolio returns about the Moment Target.

The order is ``2p``, and `p` is an integer of at least ``2``, so the order is always even and at least ``4``. An odd order cannot be stated. `alg` selects all the deviations, [`FullMoment`](@ref), or the deviations below the target alone, [`SemiMoment`](@ref). The ``p``-th root is homogeneous of degree two in the weights, the degree of a variance. Equations 3 and 4 of [emom](@cite) state the full and the lower moment about the mean, with the divisor ``T``, and Formulations 5 and 8 state their power cone programmes. The divisor ``T_d``, the observation weights and the other targets are the library's own.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\left(\\frac{1}{T_d} \\sum_{t=1}^{T} w_t\\, d_t^{2p}\\right)^{1/p}\\,, \\\\
R(\\boldsymbol{w}) &= \\left(\\frac{1}{T_d} \\sum_{t=1}^{T} w_t \\left(d_t^{-}\\right)^{2p}\\right)^{1/p}\\,, \\\\
T_d &= \\frac{T - \\mathrm{ddof}}{T} \\sum_{t=1}^{T} w_t\\,.
\\end{align}
```

The first line is the full moment and the second line is the semi moment.

Where:

  - $(math_dict[:R_w])
  - ``p``: Order parameter `p`. The moment order is ``2p``.
  - ``T_d``: Effective sample size. It is ``T - \\mathrm{ddof}`` when the measure holds no observation weights.
  - ``\\mathrm{ddof}``: Degrees of freedom `ddof` that the divisor subtracts.
  - $(math_dict[:w_t_obs]) It is ``1`` for every ``t`` when the measure holds no observation weights.
  - $(math_dict[:d_t_moment])
  - $(math_dict[:d_t_lower])
  - $(math_dict[:T])

``T_d`` is zero at ``\\mathrm{ddof} = T``, and the value is then `Inf`. A larger `ddof` makes ``T_d`` negative, and the root raises a `DomainError`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EvenMoment(;
        p::Integer = 2,
        ddof::Integer = 0,
        alg::AbstractMomentAlgorithm = FullMoment(),
    ) -> EvenMoment

Keywords correspond to the struct's fields.

## Validation

  - `p >= 2`, else a `DomainError`.
  - `ddof >= 0`, else a `DomainError`.

# Examples

```jldoctest
julia> EvenMoment()
EvenMoment
     p ┼ Int64: 2
  ddof ┼ Int64: 0
   alg ┴ FullMoment()
```

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`FourthMoment`](@ref): the fourth moment itself, with no root, on a hierarchical measure.

# References

  - $(ref_dict[:emom]) Equations 3 and 4, Formulations 5 and 8.
"""
@concrete struct EvenMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:ddof])
    """
    ddof
    """
    $(field_dict[:malg])
    """
    alg
    function EvenMoment(p::Integer, ddof::Integer, alg::AbstractMomentAlgorithm)
        @argcheck(p >= 2,
                  DomainError(p,
                              "`EvenMoment.p` is $p, and the measure raises the deviations to the power `2p`, so the order `2p` must be at least `4`. State an integer greater than or equal to `2`."))
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(p), typeof(ddof), typeof(alg)}(p, ddof, alg)
    end
end
function EvenMoment(; p::Integer = 2, ddof::Integer = 0,
                    alg::AbstractMomentAlgorithm = FullMoment())::EvenMoment
    return EvenMoment(p, ddof, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects a moment of order three or four for a high-order moment risk measure.

[`HighOrderMoment`](@ref) holds it. [`HighOrderMoment`](@ref) is a hierarchical risk measure, and the library builds no `JuMP` model for it. [`StandardisedHighOrderMoment`](@ref) subtypes this type directly, and [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref) holds the other members.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
"""
abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Groups the high-order moments that the measure reports with no division by a variance.

The members are [`ThirdLowerMoment`](@ref) and [`FourthMoment`](@ref). The group is the type bound of the field `alg` of [`StandardisedHighOrderMoment`](@ref), which divides a member by a power of a variance. So a standardised moment cannot hold a second standardised moment.

# Related

  - [`ThirdLowerMoment`](@ref)
  - [`FourthMoment`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
"""
abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <:
              HighOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the negated third moment of the deviations below the Moment Target.

The cube of a deviation below the target is negative, so the measure negates the mean, and a larger value is more downside asymmetry. It is the lower partial moment of order three. Equation 7.45 of [cajas2025](@cite) states the lower partial moment, and it names [lpm](@cite) as the source. No copy of that work was found.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= -\\mathbb{E}\\left[\\left(d_t^{-}\\right)^{3}\\right] = \\mathbb{E}\\left[\\max(\\tau - \\hat{r}_t,\\, 0)^{3}\\right]\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:E_obs_mean])
  - $(math_dict[:d_t_lower])
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:tau_moment])

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`StandardisedHighOrderMoment`](@ref): divides the measure by the cube of the lower standard deviation.

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.2, Equation 7.45 with ``q = 3``. $(ref_dict[:no_original_source])
  - $(ref_dict[:lpm])
"""
struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the fourth moment of the net portfolio returns about the Moment Target.

`alg` selects all the deviations, [`FullMoment`](@ref), or the deviations below the target alone, [`SemiMoment`](@ref). The value has no root and no division by a variance, so it is in the units of a return to the fourth power. The semi moment is the lower partial moment of order four. Equation 7.45 of [cajas2025](@cite) states the lower partial moment, and it names [lpm](@cite) as the source. No copy of that work was found.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\mathbb{E}\\left[d_t^{4}\\right]\\,, \\\\
R(\\boldsymbol{w}) &= \\mathbb{E}\\left[\\left(d_t^{-}\\right)^{4}\\right]\\,.
\\end{align}
```

The first line is the full moment and the second line is the semi moment.

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:E_obs_mean])
  - $(math_dict[:d_t_moment])
  - $(math_dict[:d_t_lower])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FourthMoment(;
        alg::AbstractMomentAlgorithm = FullMoment(),
    ) -> FourthMoment

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> FourthMoment()
FourthMoment
  alg ┴ FullMoment()
```

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
  - [`EvenMoment`](@ref): at `p = 2` and `ddof = 0`, the square root of the same moment, on a measure that the optimiser can model.
  - [`Kurtosis`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.2, Equation 7.45 with ``q = 4``. $(ref_dict[:no_original_source])
  - $(ref_dict[:lpm])
"""
@concrete struct FourthMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm
    """
    $(field_dict[:malg])
    """
    alg
    function FourthMoment(alg::AbstractMomentAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function FourthMoment(; alg::AbstractMomentAlgorithm = FullMoment())::FourthMoment
    return FourthMoment(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Divides a high-order moment of the net portfolio returns by the matching power of their second moment.

`alg` selects the moment of order ``n``, three for [`ThirdLowerMoment`](@ref) and four for [`FourthMoment`](@ref). `ve` computes the second moment on the same deviations as `alg`, so a lower moment is divided by a lower second moment, and a full moment by a full one. The result has no units.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\frac{R_{n}(\\boldsymbol{w})}{\\left(\\sigma^{2}\\right)^{n/2}}\\,, \\\\
\\sigma^{2} &= f \\sum_{t=1}^{T} w_t\\, \\delta_t^{2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - ``R_{n}(\\boldsymbol{w})``: The value of `alg`, the moment of order ``n``.
  - ``n``: Order of the moment, ``3`` or ``4``.
  - ``\\sigma^{2}``: Second moment of the deviations that `alg` reads.
  - ``\\delta_t``: The deviation that `alg` reads, ``d_t^{-}`` for a lower moment and ``d_t`` for a full moment.
  - $(math_dict[:f_ve_div])
  - $(math_dict[:w_t_obs]) It is ``1`` for every ``t`` when the measure holds no observation weights.
  - $(math_dict[:d_t_moment])
  - $(math_dict[:d_t_lower])
  - $(math_dict[:T])

The numerator is an observation mean, and the denominator takes the divisor of `ve`. The default `ve` sets `corrected = true`, so the default denominator divides by ``T - 1`` and the numerator by ``T``. Set `ve = SimpleVariance(; me = nothing, corrected = false)` for the ratio of two observation means.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardisedHighOrderMoment(;
        ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
        alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
    ) -> StandardisedHighOrderMoment

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> StandardisedHighOrderMoment()
StandardisedHighOrderMoment
   ve ┼ SimpleVariance
      │          me ┼ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
  alg ┴ ThirdLowerMoment()
```

# Related

  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`Skewness`](@ref)
  - [`Kurtosis`](@ref)
  - [`factory`](@ref)
"""
@concrete struct StandardisedHighOrderMoment <: HighOrderMomentMeasureAlgorithm
    """
    $(field_dict[:ve])
    """
    ve
    """
    $(field_dict[:malg])
    """
    alg
    function StandardisedHighOrderMoment(ve::AbstractVarianceEstimator,
                                         alg::UnstandardisedHighOrderMomentMeasureAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedHighOrderMoment(;
                                     ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                    me = nothing),
                                     alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())::StandardisedHighOrderMoment
    return StandardisedHighOrderMoment(ve, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`StandardisedHighOrderMoment`](@ref) whose `ve` holds the observation weights `w`.

The keyword constructors of [`HighOrderMoment`](@ref) and [`factory`](@ref) against a prior call it, so the second moment of the denominator reads the same weights as the measure. `alg` holds no weights and passes through unchanged.

# Related

  - [`StandardisedHighOrderMoment`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::StandardisedHighOrderMoment,
                 w::ObsWeights)::StandardisedHighOrderMoment
    return StandardisedHighOrderMoment(; ve = factory(alg.ve, w), alg = alg.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures a low-order moment of the deviations of the net portfolio returns from the Moment Target.

`alg` selects the moment: [`FirstLowerMoment`](@ref), the default, [`MeanAbsoluteDeviation`](@ref), [`SecondMoment`](@ref) or [`EvenMoment`](@ref). Each states its closed form. `mu` selects the target, and the target alone. A stated target shifts every deviation by one constant, and `alg` reduces the shifted deviations with the same formula. The measure is an optimisation risk measure, and [`set_risk_constraints!`](@ref) builds its model. Pages 188 and 189 of [markowitz1959](@cite) measure a downside deviation from a target ``b`` or from the mean ``E``.

# Mathematical definition

```math
\\begin{align}
d_t &= \\hat{r}_t - \\tau\\,,\\quad t = 1, \\ldots, T\\,, \\\\
R(\\boldsymbol{w}) &= \\rho\\left(d_1, \\ldots, d_T\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:d_t_moment])
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:tau_moment])
  - $(math_dict[:Fbar_fee_mean])
  - $(math_dict[:E_obs_mean])
  - $(math_dict[:T])
  - $(math_dict[:R_w])
  - ``\\rho``: The reduction that `alg` states.

A number in `mu` is a threshold on the net series, so it takes no fee. A per-asset target is net of the mean fee per period, so a fee that every period pays in full cancels from every deviation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowOrderMoment(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:MuSlot} = nothing,
        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment(),
    ) -> LowOrderMoment

Keywords correspond to the struct's fields. The keyword constructor passes a `w` that is not `nothing` to `alg` with [`factory`](@ref), so a variance estimator inside `alg` reads the same weights as the measure.

## Validation

  - If `mu` is a number, `isfinite(mu)`, else an [`IsNonFiniteError`](@ref).
  - If `mu` is a vector, `!isempty(mu)`, else an [`IsEmptyError`](@ref), and `all(isfinite, mu)`, else an [`IsNonFiniteError`](@ref).
  - $(val_dict[:oow_rm])

!!! warning

    A stated `mu` is fixed. It crosses a Cross-Validation fold or a subset view as the answer for the whole universe, so it does not follow the refit that the optimisation runs on. To make the target follow the fit, name a **Deferred Quantity** in `mu`, or leave the slot `nothing` and let the prior supply it.

# Functor

    (r::LowOrderMoment)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    (r::LowOrderMoment)(x::VecNum)

Computes the risk of the portfolio `w` on the returns `X` net of `fees`, or the risk of the net portfolio returns `x`. A [`DynamicAbstractWeights`](@ref) in `w` first resolves against the data with [`resolve_observation_weights`](@ref). [`calc_deviations_vec`](@ref) then gives the deviations from the Moment Target, and [`moment_risk`](@ref) reduces them. The second form reads `x` as a net series and charges no fee. It needs a target that does not read the asset weights, which [`supports_precomputed_returns`](@ref) reports. A `mu` that is `nothing` reads the observation mean of the series. So a measure that [`factory`](@ref) did not build against a prior measures the deviations from the mean of the data, not from the expected returns of a prior.

## Arguments

  - `w::VecNum`: Asset weights.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees::Option{<:Fees}`: Fees that the net returns pay, or `nothing`.
  - `x::VecNum`: Net portfolio returns `observations × 1`.

# Examples

```jldoctest
julia> LowOrderMoment()
LowOrderMoment
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
       alg ┴ FirstLowerMoment()

julia> x = [0.02, -0.01, 0.03, -0.04, 0.01];

julia> LowOrderMoment(; alg = FirstLowerMoment())(x)
0.0108

julia> LowOrderMoment(; alg = MeanAbsoluteDeviation())(x)
0.0216

julia> LowOrderMoment(; mu = 0.0, alg = MeanAbsoluteDeviation())(x)
0.022
```

The mean of `x` is `0.002`. Against the mean, the mean absolute deviation is twice the first lower moment. Against the target `0.0`, it is `0.0216 + 0.002 - 0.0`.

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`SecondMoment`](@ref)
  - [`EvenMoment`](@ref)
  - [`calc_moment_target`](@ref): selects the Moment Target from `mu`.
  - [`moment_risk`](@ref)
  - [`set_risk_constraints!`](@ref): builds the model of each `alg`.
  - [`HighOrderMoment`](@ref): the hierarchical measure of the moments of order three and four.

# References

  - $(ref_dict[:markowitz1959]) Chapter IX, pages 188 and 189.
"""
@concrete struct LowOrderMoment <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    w
    """
    $(field_dict[:mu_slot])
    """
    mu
    """
    $(field_dict[:malg])
    """
    alg
    function LowOrderMoment(settings::RiskMeasureSettings, w::Option{<:ObsWeights},
                            mu::Option{<:MuSlot}, alg::LowOrderMomentMeasureAlgorithm)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu), IsNonFiniteError("mu must be finite, got $mu"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        w::Option{<:ObsWeights} = nothing, mu::Option{<:MuSlot} = nothing,
                        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())::LowOrderMoment
    return LowOrderMoment(settings, w, mu, factory(alg, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Measures a moment of order three or four of the deviations of the net portfolio returns from the Moment Target.

`alg` selects the moment: [`ThirdLowerMoment`](@ref), the default, [`FourthMoment`](@ref), or [`StandardisedHighOrderMoment`](@ref), which divides one of the two by a power of a second moment. Each states its closed form. `mu` selects the target, as it does for [`LowOrderMoment`](@ref). The measure is a hierarchical risk measure. The hierarchical optimisers and the value level read it, and the library builds no `JuMP` model for it.

A larger value is always more risk. The odd moment is the negated mean of negative cubes, and the even moments are non-negative.

# Mathematical definition

```math
\\begin{align}
d_t &= \\hat{r}_t - \\tau\\,,\\quad t = 1, \\ldots, T\\,, \\\\
R(\\boldsymbol{w}) &= \\rho\\left(d_1, \\ldots, d_T\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:d_t_moment])
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:tau_moment])
  - $(math_dict[:Fbar_fee_mean])
  - $(math_dict[:E_obs_mean])
  - $(math_dict[:T])
  - $(math_dict[:R_w])
  - ``\\rho``: The reduction that `alg` states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderMoment(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:MuSlot} = nothing,
        alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
    ) -> HighOrderMoment

Keywords correspond to the struct's fields. The keyword constructor passes a `w` that is not `nothing` to `alg` with [`factory`](@ref), so a variance estimator inside `alg` reads the same weights as the measure.

## Validation

  - If `mu` is a number, `isfinite(mu)`, else an [`IsNonFiniteError`](@ref).
  - If `mu` is a vector, `!isempty(mu)`, else an [`IsEmptyError`](@ref), and `all(isfinite, mu)`, else an [`IsNonFiniteError`](@ref).
  - $(val_dict[:oow_rm])

!!! warning

    A stated `mu` is fixed. It crosses a Cross-Validation fold or a subset view as the answer for the whole universe, so it does not follow the refit that the optimisation runs on. To make the target follow the fit, name a **Deferred Quantity** in `mu`, or leave the slot `nothing` and let the prior supply it.

# Functor

    (r::HighOrderMoment)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    (r::HighOrderMoment)(x::VecNum)

Computes the risk of the portfolio `w` on the returns `X` net of `fees`, or the risk of the net portfolio returns `x`, as the functor of [`LowOrderMoment`](@ref) does. A [`DynamicAbstractWeights`](@ref) in `w` first resolves against the data with [`resolve_observation_weights`](@ref). [`calc_deviations_vec`](@ref) then gives the deviations from the Moment Target, and [`moment_risk`](@ref) reduces them. The second form needs a target that does not read the asset weights, which [`supports_precomputed_returns`](@ref) reports.

## Arguments

  - `w::VecNum`: Asset weights.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees::Option{<:Fees}`: Fees that the net returns pay, or `nothing`.
  - `x::VecNum`: Net portfolio returns `observations × 1`.

# Examples

```jldoctest
julia> HighOrderMoment()
HighOrderMoment
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
       alg ┴ ThirdLowerMoment()

julia> x = [0.02, -0.01, 0.03, -0.04, 0.01];

julia> HighOrderMoment(; alg = StandardisedHighOrderMoment())(x)
1.4555036361053537

julia> HighOrderMoment(;
                       alg = StandardisedHighOrderMoment(;
                                                         ve = SimpleVariance(; me = nothing,
                                                                             corrected = false)))(x)
2.034128169893555
```

The two values differ in the divisor of the lower second moment alone, ``T - 1`` under the default `ve` and ``T`` with `corrected = false`.

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`ThirdLowerMoment`](@ref)
  - [`FourthMoment`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref): selects the Moment Target from `mu`.
  - [`moment_risk`](@ref)
  - [`LowOrderMoment`](@ref): the optimisation measure of the moments of order one and two.
"""
@concrete struct HighOrderMoment <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    w
    """
    $(field_dict[:mu_slot])
    """
    mu
    """
    $(field_dict[:malg])
    """
    alg
    function HighOrderMoment(settings::RiskMeasureSettings, w::Option{<:ObsWeights},
                             mu::Option{<:MuSlot}, alg::HighOrderMomentMeasureAlgorithm)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu), IsNonFiniteError("mu must be finite, got $mu"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:ObsWeights} = nothing, mu::Option{<:MuSlot} = nothing,
                         alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())::HighOrderMoment
    return HighOrderMoment(settings, w, mu, factory(alg, w))
end
"""
    const LoHiOrderMoment{T1, T2, T3, T4} = Union{LowOrderMoment{T1, T2, T3, T4},
                                                  HighOrderMoment{T1, T2, T3, T4}}

Groups [`LowOrderMoment`](@ref) and [`HighOrderMoment`](@ref) under the same four type parameters.

The two measures have the same fields in the same order, `settings`, `w`, `mu` and `alg`, and they select the Moment Target and the deviations in the same way. So [`calc_moment_target`](@ref) dispatches on the second and the third parameter, the types of `w` and `mu`, once for both measures. [`calc_deviations_vec`](@ref) and [`supports_precomputed_returns`](@ref) also take the group.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_deviations_vec`](@ref)
"""
const LoHiOrderMoment{T1, T2, T3, T4} = Union{LowOrderMoment{T1, T2, T3, T4},
                                              HighOrderMoment{T1, T2, T3, T4}}
"""
    calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any},
                       ::Any, x::VecNum)

Return the Moment Target of a measure whose `w` and `mu` are both `nothing`, the mean of `x`.

# Arguments

  - `r`: A [`LowOrderMoment`](@ref) or a [`HighOrderMoment`](@ref) whose `w` and `mu` are `nothing`.
  - `::Any`: The asset weights. The method does not read them.
  - `x::VecNum`: Net portfolio returns.

# Returns

  - `tgt::Number`: `Statistics.mean(x)`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any}, ::Any,
                            x::VecNum)
    return Statistics.mean(x)
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing, <:Any},
                       ::Any, x::VecNum)

Return the Moment Target of a measure that holds observation weights and no `mu`, the weighted mean of `x`.

# Arguments

  - `r`: A [`LowOrderMoment`](@ref) or a [`HighOrderMoment`](@ref) whose `w` is a `StatsBase.AbstractWeights` and whose `mu` is `nothing`.
  - `::Any`: The asset weights. The method does not read them.
  - `x::VecNum`: Net portfolio returns.

# Returns

  - `tgt::Number`: `Statistics.mean(x, r.w)`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing,
                                               <:Any}, ::Any, x::VecNum)
    return Statistics.mean(x, r.w)
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any},
                       w::VecNum, ::Any)

Return the gross Moment Target of a measure whose `mu` holds expected asset returns, the expected portfolio return.

[`calc_deviations_vec`](@ref) subtracts the mean fee per period from it, so the target that the deviations read is net.

# Arguments

  - `r`: A [`LowOrderMoment`](@ref) or a [`HighOrderMoment`](@ref) whose `mu` is a vector.
  - `w::VecNum`: Asset weights.
  - `::Any`: The net portfolio returns. The method does not read them.

# Returns

  - `tgt::Number`: `LinearAlgebra.dot(w, r.mu)`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any}, w::VecNum,
                            ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                       w::VecNum, ::Any)

Return the gross Moment Target of a measure whose `mu` is a [`VecScalar`](@ref), the expected portfolio return of `mu.v` plus the offset `mu.s`.

[`calc_deviations_vec`](@ref) subtracts the mean fee per period from it, so the target that the deviations read is net.

# Arguments

  - `r`: A [`LowOrderMoment`](@ref) or a [`HighOrderMoment`](@ref) whose `mu` is a [`VecScalar`](@ref).
  - `w::VecNum`: Asset weights.
  - `::Any`: The net portfolio returns. The method does not read them.

# Returns

  - `tgt::Number`: `LinearAlgebra.dot(w, r.mu.v) + r.mu.s`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
  - [`dot`](https://docs.julialang.org/en/v1/base/math/#Base.dot)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}, w::VecNum,
                            ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any},
                       ::Any, ::Any)

Return the Moment Target of a measure whose `mu` is a number, the number itself.

The number is a minimum acceptable return on the net series, so no fee applies to it.

# Arguments

  - `r`: A [`LowOrderMoment`](@ref) or a [`HighOrderMoment`](@ref) whose `mu` is a number.
  - `::Any`: The asset weights. The method does not read them.
  - `::Any`: The net portfolio returns. The method does not read them.

# Returns

  - `tgt::Number`: `r.mu`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
    return r.mu
end
"""
    calc_deviations_vec(r::LoHiOrderMoment, w::VecNum, X::MatNum,
                        fees::Option{<:Fees} = nothing)

Return the deviations of the net portfolio returns from the net Moment Target of `r`.

A per-asset target is gross, and the series is net of the fee. So the method subtracts the mean fee per period from such a target, and a fee that every period pays in full cancels from every deviation. A target that is the mean of the series, or a number, takes no fee.

# Algorithm

 1. Compute the net portfolio returns `x` with [`calc_net_returns`](@ref).
 2. Compute the gross Moment Target with [`calc_moment_target`](@ref) from `r`, `w` and `x`.
 3. Subtract [`moment_target_fees`](@ref) from it, giving the net target `tgt`.
 4. Return `x .- tgt`.

# Arguments

  - `r::LoHiOrderMoment`: The moment risk measure.
  - `w::VecNum`: Asset weights.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees::Option{<:Fees}`: Fees that the net returns pay, or `nothing`.

# Returns

  - `val::VecNum`: The deviations, `observations × 1`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_net_returns`](@ref)
  - [`moment_target_fees`](@ref)
  - [`moment_risk`](@ref): reduces the deviations to the risk value.
"""
function calc_deviations_vec(r::LoHiOrderMoment, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x) - moment_target_fees(r.mu, w, fees, length(x))
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the deviations of the net portfolio returns `x` from the Moment Target of `r`.

The functor `r(x)` calls this form. It reads `x` as a net series, so no fee applies. It passes no asset weights to [`calc_moment_target`](@ref), so it needs a target that does not read them, a `mu` that is `nothing` or a number. A per-asset `mu` raises a `MethodError`. [`supports_precomputed_returns`](@ref) reports the condition before the call.

# Related

  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
  - [`LoHiOrderMoment`](@ref)
"""
function calc_deviations_vec(r::LoHiOrderMoment, x::VecNum)
    return x .- calc_moment_target(r, nothing, x)
end
"""
    moment_risk(r::LoHiOrderMoment, val::VecNum)
    moment_risk(r::Kurtosis, val::VecNum)
    moment_risk(r::Skewness, val::VecNum)
    moment_risk(r::MedianAbsoluteDeviation, val::VecNum)
    moment_risk(r::ThirdCentralMoment, val::VecNum)

Reduce the moment deviations `val` of a moment risk measure to its risk value.

`val` is the output of [`calc_deviations_vec`](@ref). Both forms of the functor call this function, so the value of `r(w, X, fees)` and of `r(x)` comes from one definition. Dispatch on `r.alg` selects the reduction, and the docstring of each algorithm states its closed form. The methods for a full [`FourthMoment`](@ref) overwrite `val` in place.

# Algorithm

The steps for [`LowOrderMoment`](@ref) and [`HighOrderMoment`](@ref):

 1. For a lower moment, replace `val` with `min.(val, 0)`. [`FirstLowerMoment`](@ref), [`ThirdLowerMoment`](@ref) and a [`SemiMoment`](@ref) are lower moments.
 2. For [`SecondMoment`](@ref), return `Statistics.var(r.alg.ve, val; mean = 0)`, or `Statistics.std` under [`SOCRiskExpr`](@ref). Stop.
 3. For [`StandardisedHighOrderMoment`](@ref), compute `sigma = Statistics.var(r.alg.ve, val; mean = 0)` before the power.
 4. Raise each entry of `val` to the power of the moment: `abs` for [`MeanAbsoluteDeviation`](@ref), the cube or the fourth power for a high-order moment.
 5. Take the mean of `val`, weighted by `r.w` when it is not `nothing`, giving `res`.
 6. Negate `res` for [`FirstLowerMoment`](@ref) and [`ThirdLowerMoment`](@ref), so that a larger value is more risk.
 7. For [`StandardisedHighOrderMoment`](@ref), divide `res` by `sigma^(n / 2)`, with `n` the order of `r.alg.alg`.

[`EvenMoment`](@ref) takes the place of steps 4 and 5. It computes the norm of order `2p` of `val`, each entry scaled by `w_t^(1 / 2p)` when `r.w` is not `nothing`, and returns its square divided by `T^(1 / p)`, with `T` the effective sample size.

# Related

  - [`calc_deviations_vec`](@ref)
  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`MomentRiskMeasures`](@ref): the measures whose functor calls this function.
"""
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:FirstLowerMoment}, val::VecNum)
    val = min.(val, zero(eltype(val)))
    return isnothing(r.w) ? -Statistics.mean(val) : -Statistics.mean(val, r.w)
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:MeanAbsoluteDeviation}, val::VecNum)
    val = abs.(val)
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:SecondMoment{<:Any, <:FullMoment, <:SOCRiskExpr}},
                     val::VecNum)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:SecondMoment{<:Any, <:FullMoment,
                                                      <:QuadSecondMomentFormulations}},
                     val::VecNum)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:SecondMoment{<:Any, <:SemiMoment, <:SOCRiskExpr}},
                     val::VecNum)
    val = min.(val, zero(eltype(val)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:SecondMoment{<:Any, <:SemiMoment,
                                                      <:QuadSecondMomentFormulations}},
                     val::VecNum)
    val = min.(val, zero(eltype(val)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:EvenMoment{<:Any, <:Any, <:FullMoment}},
                     val::VecNum)
    T = length(val) - r.alg.ddof
    val = if isnothing(r.w)
        LinearAlgebra.norm(val, 2 * r.alg.p)
    else
        T = T / length(val) * sum(r.w)
        LinearAlgebra.norm(val .* (r.w .^ inv(2 * r.alg.p)), 2 * r.alg.p)
    end
    return val^2 / T^inv(r.alg.p)
end
function moment_risk(r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                       <:EvenMoment{<:Any, <:Any, <:SemiMoment}},
                     val::VecNum)
    T = length(val) - r.alg.ddof
    val = min.(val, zero(eltype(val)))
    val = if isnothing(r.w)
        LinearAlgebra.norm(val, 2 * r.alg.p)
    else
        T = T / length(val) * sum(r.w)
        LinearAlgebra.norm(val .* (r.w .^ inv(2 * r.alg.p)), 2 * r.alg.p)
    end
    return val^2 / T^inv(r.alg.p)
end
function moment_risk(r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                        <:ThirdLowerMoment}, val::VecNum)
    val = min.(val, zero(eltype(val)))
    val .= val .^ 3
    return isnothing(r.w) ? -Statistics.mean(val) : -Statistics.mean(val, r.w)
end
function moment_risk(r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                        <:StandardisedHighOrderMoment{<:Any,
                                                                      <:ThirdLowerMoment}},
                     val::VecNum)
    val = min.(val, zero(eltype(val)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -Statistics.mean(val) : -Statistics.mean(val, r.w)
    return res / (sigma * sqrt(sigma))
end
function moment_risk(r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                        <:FourthMoment{<:FullMoment}}, val::VecNum)
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function moment_risk(r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                        <:FourthMoment{<:SemiMoment}}, val::VecNum)
    val = min.(val, zero(eltype(val)))
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function moment_risk(r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                        <:StandardisedHighOrderMoment{<:Any,
                                                                      <:FourthMoment{<:FullMoment}}},
                     val::VecNum)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^2
end
function moment_risk(r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                                        <:StandardisedHighOrderMoment{<:Any,
                                                                      <:FourthMoment{<:SemiMoment}}},
                     val::VecNum)
    val = min.(val, zero(eltype(val)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^2
end
# Evaluation entry points — see `MomentRiskMeasures` in `25_RiskMeasureTools.jl`.
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create a [`LowOrderMoment`](@ref) whose observation weights and `mu` come from the measure, or from the prior result `pr` where the measure states none.

A measure whose `mu` is `nothing` takes `pr.mu`, so after this method its Moment Target is the expected portfolio return of the prior, net of the mean fee per period. Before it, the target is the mean of the data that the functor reads. The two agree when `pr.mu` is the observation mean of `pr.X` under the same weights.

# Algorithm

 1. Resolve a **Deferred Quantity** in `mu` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `w`, the weights of `r` or else `pr.w`, with [`nothing_scalar_array_selector`](@ref).
 3. Select `mu`, the target of `r` or else `pr.mu`, with [`nothing_scalar_array_selector`](@ref).
 4. Pass `w` to `r.alg` with [`factory`](@ref), giving `alg`.
 5. Build a new `LowOrderMoment` from `r.settings`, `w`, `mu` and `alg`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::LowOrderMoment, pr::AbstractPriorResult, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    alg = factory(r.alg, w)
    return LowOrderMoment(; settings = r.settings, alg = alg, w = w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `mu` slot of a [`LowOrderMoment`](@ref) against the prior result `pr`.

`mu` is the one slot of the measure that a prior can supply, so the slot holds the estimator itself, and no other slot reads its fit. A measure whose `mu` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`LowOrderMoment`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::LowOrderMoment, pr::AbstractPriorResult,
                                     ::Any = nothing)
    return r
end
function resolve_deferred_quantities(r::LowOrderMoment{<:Any, <:Any, <:DeferredQuantity},
                                     pr::AbstractPriorResult, ::Any = nothing)
    return rebuild_with_slots(r, (; mu = resolve_slot(r.mu, :mu, pr)))
end
# Deferrable slots — see `deferred_slots`. `alg` holds a variance estimator by design, not a
# Deferred Quantity, so it is not declared here.
deferred_slots(r::LowOrderMoment) = (; mu = r.mu)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`LowOrderMoment`](@ref) restricted to the assets `i`.

The method slices a per-asset `mu` to the assets `i` with [`nothing_scalar_array_view`](@ref). A number and `nothing` pass through unchanged, and so does a **Deferred Quantity**, which resolves on the subset. `w` weighs observations, not assets, so it passes through unchanged, and so do `settings` and `alg`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function port_opt_view(r::LowOrderMoment, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return LowOrderMoment(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create a [`HighOrderMoment`](@ref) whose observation weights and `mu` come from the measure, or from the prior result `pr` where the measure states none.

A measure whose `mu` is `nothing` takes `pr.mu`, so after this method its Moment Target is the expected portfolio return of the prior, net of the mean fee per period.

# Algorithm

 1. Resolve a **Deferred Quantity** in `mu` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `w`, the weights of `r` or else `pr.w`, with [`nothing_scalar_array_selector`](@ref).
 3. Select `mu`, the target of `r` or else `pr.mu`, with [`nothing_scalar_array_selector`](@ref).
 4. Pass `w` to `r.alg` with [`factory`](@ref), giving `alg`.
 5. Build a new `HighOrderMoment` from `r.settings`, `w`, `mu` and `alg`.

# Related

  - [`HighOrderMoment`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::HighOrderMoment, pr::AbstractPriorResult, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    alg = factory(r.alg, w)
    return HighOrderMoment(; settings = r.settings, alg = alg, w = w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `mu` slot of a [`HighOrderMoment`](@ref) against the prior result `pr`.

`mu` is the one slot of the measure that a prior can supply, so the slot holds the estimator itself, and no other slot reads its fit. A measure whose `mu` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`HighOrderMoment`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::HighOrderMoment, pr::AbstractPriorResult,
                                     ::Any = nothing)
    return r
end
function resolve_deferred_quantities(r::HighOrderMoment{<:Any, <:Any, <:DeferredQuantity},
                                     pr::AbstractPriorResult, ::Any = nothing)
    return rebuild_with_slots(r, (; mu = resolve_slot(r.mu, :mu, pr)))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::HighOrderMoment) = (; mu = r.mu)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`HighOrderMoment`](@ref) restricted to the assets `i`.

The method slices a per-asset `mu` to the assets `i` with [`nothing_scalar_array_view`](@ref). A number and `nothing` pass through unchanged, and so does a **Deferred Quantity**, which resolves on the subset. `w` weighs observations, not assets, so it passes through unchanged, and so do `settings` and `alg`.

# Related

  - [`HighOrderMoment`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function port_opt_view(r::HighOrderMoment, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return HighOrderMoment(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::LoHiOrderMoment) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the functor `r(x)` of a [`LoHiOrderMoment`](@ref) can evaluate the net portfolio returns `x` alone.

The answer is [`weight_independent_target`](@ref) of `r.mu`. It is `true` when `mu` is `nothing` or a number, because such a target does not read the asset weights. It is `false` for a vector, a [`VecScalar`](@ref) and an unresolved **Deferred Quantity**.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`LoHiOrderMoment`](@ref)
"""
supports_precomputed_returns(r::LoHiOrderMoment) = weight_independent_target(r.mu)

export FirstLowerMoment, SecondMoment, MeanAbsoluteDeviation, ThirdLowerMoment,
       FourthMoment, StandardisedHighOrderMoment, LowOrderMoment, HighOrderMoment,
       EvenMoment
