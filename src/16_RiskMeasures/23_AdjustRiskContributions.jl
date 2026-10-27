"""
    const SquaredRiskMeasures = Union{Variance, BrownianDistanceVariance, UncertaintySetVariance,
                                      LowOrderMoment{<:Any, <:Any, <:Any,
                                                     <:SecondMoment{<:Any, <:Any,
                                                                    <:QuadSecondMomentFormulations}},
                                      LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment},
                                      Kurtosis{<:Any, <:Any, <:Any, <:SOCRiskExpr},
                                      NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                       <:NSkeQuadFormulations},
                                      TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Norm}}

Union of the risk measures that are homogeneous of degree 2 in the portfolio weights. [`homogeneity_degree`](@ref) returns `2` for each member.

A member reports a squared quantity: a variance, a second moment, the square root of a fourth moment, or the square of a tracking error. [`EvenMoment`](@ref) of order ``2p`` reports the ``p``-th root of the moment, so it is a member for every ``p``.

# Related

  - [`Variance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`LowOrderMoment`](@ref)
  - [`Kurtosis`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const SquaredRiskMeasures = Union{Variance, BrownianDistanceVariance,
                                  UncertaintySetVariance,
                                  LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:SecondMoment{<:Any, <:Any,
                                                                <:QuadSecondMomentFormulations}},
                                  LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment},
                                  Kurtosis{<:Any, <:Any, <:Any, <:SOCRiskExpr},
                                  NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                   <:NSkeQuadFormulations},
                                  TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Norm}}
"""
    const QuadExpressionRiskMeasures = Union{Variance, BrownianDistanceVariance,
                                             LowOrderMoment{<:Any, <:Any, <:Any,
                                                            <:SecondMoment{<:Any, <:Any,
                                                                           <:QuadSecondMomentFormulations}},
                                             NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                              <:NSkeQuadFormulations},
                                             Kurtosis{<:Any, <:Any, <:Any,
                                                      <:QuadSecondMomentFormulations},
                                             TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Norm}}

Union of the risk measures whose risk expression in a JuMP model is quadratic in the weights.

The tracking constraint of a dependent variable and the near-optimal centering read the union, because a quadratic expression needs a different encoding there than an affine one.

# Related

  - [`Variance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`LowOrderMoment`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`Kurtosis`](@ref)
  - [`TrackingRiskMeasure`](@ref)
"""
const QuadExpressionRiskMeasures = Union{Variance, BrownianDistanceVariance,
                                         LowOrderMoment{<:Any, <:Any, <:Any,
                                                        <:SecondMoment{<:Any, <:Any,
                                                                       <:QuadSecondMomentFormulations}},
                                         NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                          <:NSkeQuadFormulations},
                                         Kurtosis{<:Any, <:Any, <:Any,
                                                  <:QuadSecondMomentFormulations},
                                         TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Norm}}
"""
    const CubedRiskMeasures = Union{Skewness, ThirdCentralMoment{<:Any, <:Any, <:Any, Nothing},
                                    HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment}}

Union of the risk measures that are homogeneous of degree 3 in the portfolio weights. [`homogeneity_degree`](@ref) returns `3` for each member.

A member reports a raw third moment. The standardised third moments, of [`StandardisedHighOrderMoment`](@ref) and of a [`ThirdCentralMoment`](@ref) with a variance estimator in `ve`, divide by the cube of a standard deviation, so they are members of [`ScaleFreeRiskMeasures`](@ref) and not of this union.

# Related

  - [`Skewness`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`ThirdLowerMoment`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const CubedRiskMeasures = Union{Skewness, ThirdCentralMoment{<:Any, <:Any, <:Any, Nothing},
                                HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment}}
"""
    const FourthPowerRiskMeasures = Union{HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment},
                                          Kurtosis{<:Any, <:Any, <:Any,
                                                   <:QuadSecondMomentFormulations}}

Union of the risk measures that are homogeneous of degree 4 in the portfolio weights. [`homogeneity_degree`](@ref) returns `4` for each member.

A member reports a raw fourth moment. [`Kurtosis`](@ref) under [`SOCRiskExpr`](@ref) reports the square root of the moment, so it is a member of [`SquaredRiskMeasures`](@ref). The standardised fourth moment is a member of [`ScaleFreeRiskMeasures`](@ref).

# Related

  - [`HighOrderMoment`](@ref)
  - [`FourthMoment`](@ref)
  - [`Kurtosis`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const FourthPowerRiskMeasures = Union{HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment},
                                      Kurtosis{<:Any, <:Any, <:Any,
                                               <:QuadSecondMomentFormulations}}
"""
    const ScaleFreeRiskMeasures = Union{EqualRisk,
                                        ThirdCentralMoment{<:Any, <:Any, <:Any,
                                                           <:AbstractVarianceEstimator},
                                        HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:StandardisedHighOrderMoment}}

Union of the risk measures that are homogeneous of degree 0 in the portfolio weights. [`homogeneity_degree`](@ref) returns `0` for each member.

A member does not change when every weight is multiplied by the same positive number. [`EqualRisk`](@ref) reads only the length of the weights, and a standardised moment divides by a power of the standard deviation of the same degree. So the risk contributions of a member sum to zero, and not to its risk. The contributions of [`EqualRisk`](@ref) are each zero, because its risk does not depend on the weights at all.

# Related

  - [`EqualRisk`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const ScaleFreeRiskMeasures = Union{EqualRisk,
                                    ThirdCentralMoment{<:Any, <:Any, <:Any,
                                                       <:AbstractVarianceEstimator},
                                    HighOrderMoment{<:Any, <:Any, <:Any,
                                                    <:StandardisedHighOrderMoment}}
"""
    homogeneity_degree(r::AbstractBaseRiskMeasure) -> Option{<:Integer}

Return the degree of homogeneity of the risk measure `r` in the portfolio weights, or `nothing` when `r` has no degree.

[`adjust_risk_contribution`](@ref) divides a risk by this degree, so that the risk contributions of [`risk_contribution`](@ref) sum to the risk. A measure that no other method names has degree 1.

# Mathematical definition

```math
\\begin{align}
R(c \\boldsymbol{w}) &= c^{d} R(\\boldsymbol{w}) \\quad \\forall c > 0\\,, \\\\
d\\left(\\frac{R_1}{R_2}\\right) &= d(R_1) - d(R_2)\\,.
\\end{align}
```

The second line gives the degree of a ratio of two homogeneous measures. A scalarised vector of measures has the degree of its elements when every element has the same degree, and the sum, the maximum and the minimum keep that degree.

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:w_port])
  - $(math_dict[:d_homog])
  - ``R_1``, ``R_2``: The risk measures of the numerator and of the denominator of a ratio.

# Algorithm

 1. Return `2` for a member of [`SquaredRiskMeasures`](@ref), `3` for a member of [`CubedRiskMeasures`](@ref), `4` for a member of [`FourthPowerRiskMeasures`](@ref), and `0` for a member of [`ScaleFreeRiskMeasures`](@ref).
 2. For [`RiskRatio`](@ref) and [`NonOptimisationRiskRatio`](@ref), get the degree of each side with [`scalarised_degree`](@ref), and subtract the degree of `r2` from the degree of `r1` with [`degree_difference`](@ref).
 3. For [`MeanReturnRiskRatio`](@ref), return `nothing` when `rf` is not zero. Otherwise subtract the degree of `rk` from the degree of `rt`, in the same way.
 4. Return `nothing` for [`VarianceSkewKurtosis`](@ref), whose three terms have the degrees 2, 3 and 4.
 5. Return `1` for every other measure.

# Arguments

  - $(arg_dict[:r])

# Returns

  - `d::Option{<:Integer}`: The degree of homogeneity, or `nothing` when `r` is not homogeneous.

# Examples

```jldoctest
julia> PortfolioOptimisers.homogeneity_degree(Variance())
2

julia> PortfolioOptimisers.homogeneity_degree(ConditionalValueatRisk())
1

julia> PortfolioOptimisers.homogeneity_degree(NonOptimisationRiskRatio(;
                                                                       r1 = ConditionalValueatRisk(),
                                                                       r2 = Variance()))
-1

julia> PortfolioOptimisers.homogeneity_degree(EqualRisk())
0
```

# Related

  - [`adjust_risk_contribution`](@ref)
  - [`risk_contribution`](@ref)
  - [`SquaredRiskMeasures`](@ref)
  - [`CubedRiskMeasures`](@ref)
  - [`FourthPowerRiskMeasures`](@ref)
  - [`ScaleFreeRiskMeasures`](@ref)
"""
function homogeneity_degree(::AbstractBaseRiskMeasure)
    return 1
end
function homogeneity_degree(::SquaredRiskMeasures)
    return 2
end
function homogeneity_degree(::CubedRiskMeasures)
    return 3
end
function homogeneity_degree(::FourthPowerRiskMeasures)
    return 4
end
function homogeneity_degree(::ScaleFreeRiskMeasures)
    return 0
end
function homogeneity_degree(::VarianceSkewKurtosis)
    return nothing
end
function homogeneity_degree(r::RiskRatio)
    return degree_difference(homogeneity_degree(r.r1), homogeneity_degree(r.r2))
end
function homogeneity_degree(r::NonOptimisationRiskRatio)
    return degree_difference(scalarised_degree(r.r1, r.sca1),
                             scalarised_degree(r.r2, r.sca2))
end
function homogeneity_degree(r::MeanReturnRiskRatio)
    if !iszero(r.rf)
        return nothing
    end
    return degree_difference(homogeneity_degree(r.rt), scalarised_degree(r.rk, r.sca))
end
"""
    scalarised_degree(r::BaseRM_VecBaseRM, sca::Scalariser) -> Option{<:Integer}

Return the degree of homogeneity of one side of a ratio, which is one measure or a vector of measures that `sca` combines.

A single measure ignores `sca`. A vector has the common degree of its elements under [`SumScalariser`](@ref), [`MaxScalariser`](@ref) and [`MinScalariser`](@ref). It has no degree when two elements have different degrees, or under [`LogSumExpScalariser`](@ref), which is not homogeneous.

# Arguments

  - $(arg_dict[:r])
  - `sca::Scalariser`: Scalariser of the side.

# Returns

  - `d::Option{<:Integer}`: The degree of the side, or `nothing`.

# Related

  - [`homogeneity_degree`](@ref)
  - [`degree_difference`](@ref)
"""
function scalarised_degree(r::AbstractBaseRiskMeasure, ::Scalariser)
    return homogeneity_degree(r)
end
function scalarised_degree(::VecBaseRM, ::LogSumExpScalariser)
    return nothing
end
function scalarised_degree(rs::VecBaseRM, ::Scalariser)
    d = homogeneity_degree(first(rs))
    return all(r -> homogeneity_degree(r) == d, rs) ? d : nothing
end
"""
    degree_difference(d1, d2) -> Option{<:Integer}

Return the degree of homogeneity of a ratio from the degrees `d1` of its numerator and `d2` of its denominator.

The ratio has degree `d1 - d2` when both sides have a degree, and no degree when either side has none.

# Arguments

  - `d1`: Degree of the numerator, or `nothing`.
  - `d2`: Degree of the denominator, or `nothing`.

# Returns

  - `d::Option{<:Integer}`: The degree of the ratio, or `nothing`.

# Related

  - [`homogeneity_degree`](@ref)
  - [`scalarised_degree`](@ref)
"""
function degree_difference(d1::Integer, d2::Integer)
    return d1 - d2
end
function degree_difference(::Any, ::Any)
    return nothing
end
"""
    adjust_risk_contribution(r::AbstractBaseRiskMeasure, val::Number) -> Number

Divide the risk `val` of the measure `r` by the degree of homogeneity of `r`.

By Euler's theorem, the gradient of the divided risk, multiplied by the weights, sums to the risk. [`adjusted_risk`](@ref) calls this function, and [`risk_contribution`](@ref) differentiates the result. A measure of degree 0, and a measure with no degree, keep `val` unchanged.

# Mathematical definition

```math
\\begin{align}
A(\\boldsymbol{w}) &= \\frac{R(\\boldsymbol{w})}{d}\\,, \\\\
\\sum_{i=1}^{N} w_i \\frac{\\partial A(\\boldsymbol{w})}{\\partial w_i} &= R(\\boldsymbol{w})\\,.
\\end{align}
```

The second line holds when ``R`` is homogeneous of a degree ``d \\neq 0``. At ``d = 0`` the sum is zero for every ``R``, so no division gives back the risk, and the contributions of a measure of [`ScaleFreeRiskMeasures`](@ref) sum to zero. A measure that is not homogeneous, such as a tracking error or a turnover, has contributions that do not sum to its risk in general.

Where:

  - ``A(\\boldsymbol{w})``: Homogeneity-adjusted risk.
  - $(math_dict[:R_w])
  - $(math_dict[:d_homog])
  - $(math_dict[:w_i_asset])
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:r])
  - `val::Number`: Risk of `r` at the weights.

# Returns

  - `Number`: `val / d`, or `val` when the degree `d` is `0` or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.adjust_risk_contribution(Variance(), 0.5)
0.25

julia> PortfolioOptimisers.adjust_risk_contribution(EqualRisk(), 0.5)
0.5
```

# Related

  - [`homogeneity_degree`](@ref)
  - [`adjusted_risk`](@ref)
  - [`risk_contribution`](@ref)
"""
function adjust_risk_contribution(r::AbstractBaseRiskMeasure, val::Number)
    d = homogeneity_degree(r)
    if isnothing(d) || iszero(d)
        return val
    end
    return val / d
end
