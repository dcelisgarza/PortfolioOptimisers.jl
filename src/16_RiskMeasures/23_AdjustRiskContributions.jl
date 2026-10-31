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
                                    HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment,
                                                    Nothing}}

Union of the risk measures that are homogeneous of degree 3 in the portfolio weights. [`homogeneity_degree`](@ref) returns `3` for each member.

A member reports a raw third moment. The standardised third moments, of a [`HighOrderMoment`](@ref) and of a [`ThirdCentralMoment`](@ref) with a variance estimator in `ve`, divide by the cube of a standard deviation, so they are members of [`ScaleFreeRiskMeasures`](@ref) and not of this union.

# Related

  - [`Skewness`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`ThirdLowerMoment`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const CubedRiskMeasures = Union{Skewness, ThirdCentralMoment{<:Any, <:Any, <:Any, Nothing},
                                HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment,
                                                Nothing}}
"""
    const FourthPowerRiskMeasures = Union{HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment,
                                                          Nothing},
                                          Kurtosis{<:Any, <:Any, <:Any,
                                                   <:QuadSecondMomentFormulations}}

Union of the risk measures that are homogeneous of degree 4 in the portfolio weights. [`homogeneity_degree`](@ref) returns `4` for each member.

A member reports a raw fourth moment. [`Kurtosis`](@ref) under [`SOCRiskExpr`](@ref) reports the square root of the moment, so it is a member of [`SquaredRiskMeasures`](@ref). A [`HighOrderMoment`](@ref) with a variance estimator in `ve` reports the standardised fourth moment, so it is a member of [`ScaleFreeRiskMeasures`](@ref).

# Related

  - [`HighOrderMoment`](@ref)
  - [`FourthMoment`](@ref)
  - [`Kurtosis`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const FourthPowerRiskMeasures = Union{HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment,
                                                      Nothing},
                                      Kurtosis{<:Any, <:Any, <:Any,
                                               <:QuadSecondMomentFormulations}}
"""
    const ScaleFreeRiskMeasures = Union{EqualRisk,
                                        ThirdCentralMoment{<:Any, <:Any, <:Any,
                                                           <:AbstractVarianceEstimator},
                                        HighOrderMoment{<:Any, <:Any, <:Any, <:Any,
                                                        <:AbstractVarianceEstimator}}

Union of the risk measures that are homogeneous of degree 0 in the portfolio weights. [`homogeneity_degree`](@ref) returns `0` for each member.

A member does not change when every weight is multiplied by the same positive number. [`EqualRisk`](@ref) reads only the length of the weights, and a standardised moment divides by a power of the standard deviation of the same degree. So the risk contributions of a member sum to zero, and not to its risk. The contributions of [`EqualRisk`](@ref) are each zero, because its risk does not depend on the weights at all.

# Related

  - [`EqualRisk`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const ScaleFreeRiskMeasures = Union{EqualRisk,
                                    ThirdCentralMoment{<:Any, <:Any, <:Any,
                                                       <:AbstractVarianceEstimator},
                                    HighOrderMoment{<:Any, <:Any, <:Any, <:Any,
                                                    <:AbstractVarianceEstimator}}
"""
    const RelativeDrawdownRiskMeasures = Union{RelativeAverageDrawdown,
                                               RelativeConditionalDrawdownatRisk,
                                               RelativeDrawdownatRisk,
                                               RelativeEntropicDrawdownatRisk,
                                               RelativeMaximumDrawdown,
                                               RelativePowerNormDrawdownatRisk,
                                               RelativeRelativisticDrawdownatRisk,
                                               RelativeUlcerIndex}

Union of the risk measures of the drawdowns of the compounded cumulative returns. [`homogeneity_degree`](@ref) returns `nothing` for each member.

The compounded wealth ``\\prod_{t} (1 + \\boldsymbol{x}_t^{\\intercal} \\boldsymbol{w})`` is not homogeneous in the weights, so no member has a degree. The measure of the drawdowns of the uncompounded cumulative returns, such as [`MaximumDrawdown`](@ref), has degree 1.

# Related

  - [`RelativeAverageDrawdown`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
  - [`RelativeEntropicDrawdownatRisk`](@ref)
  - [`RelativeMaximumDrawdown`](@ref)
  - [`RelativePowerNormDrawdownatRisk`](@ref)
  - [`RelativeRelativisticDrawdownatRisk`](@ref)
  - [`RelativeUlcerIndex`](@ref)
  - [`homogeneity_degree`](@ref)
"""
const RelativeDrawdownRiskMeasures = Union{RelativeAverageDrawdown,
                                           RelativeConditionalDrawdownatRisk,
                                           RelativeDrawdownatRisk,
                                           RelativeEntropicDrawdownatRisk,
                                           RelativeMaximumDrawdown,
                                           RelativePowerNormDrawdownatRisk,
                                           RelativeRelativisticDrawdownatRisk,
                                           RelativeUlcerIndex}
"""
    const ShiftInvariantRiskMeasures = Union{Range, BrownianDistanceVariance,
                                             ValueatRiskRange, ConditionalValueatRiskRange,
                                             DistributionallyRobustConditionalValueatRiskRange,
                                             EntropicValueatRiskRange,
                                             RelativisticValueatRiskRange,
                                             PowerNormValueatRiskRange}

Union of the risk measures whose risk does not change when the same number is added to each net return. [`cancels_constant_fee`](@ref) returns `true` for each member.

A range is the sum of the risk of the losses and the risk of the gains. Each tail moves by the number with an opposite sign, so the sum does not move. The distance variance of [`BrownianDistanceVariance`](@ref) reads only the differences between two returns. A fee that every period pays in the same amount is such a number, so it cancels from the risk of a member.

# Related

  - [`Range`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`EntropicValueatRiskRange`](@ref)
  - [`RelativisticValueatRiskRange`](@ref)
  - [`PowerNormValueatRiskRange`](@ref)
  - [`cancels_constant_fee`](@ref)
"""
const ShiftInvariantRiskMeasures = Union{Range, BrownianDistanceVariance, ValueatRiskRange,
                                         ConditionalValueatRiskRange,
                                         DistributionallyRobustConditionalValueatRiskRange,
                                         EntropicValueatRiskRange,
                                         RelativisticValueatRiskRange,
                                         PowerNormValueatRiskRange}
"""
    homogeneity_degree(r::AbstractBaseRiskMeasure, fees::Option{<:Fees} = nothing,
                       T::Option{<:Integer} = nothing) -> Option{<:Integer}

Return the degree of homogeneity of the risk measure `r` in the portfolio weights, under the fees `fees` on `T` observations, or `nothing` when `r` has no degree.

[`adjust_risk_contribution`](@ref) divides a risk by this degree, so that the risk contributions of [`risk_contribution`](@ref) sum to the risk. A measure that no other method names has degree 1, unless it measures a distance to a fixed target that is not zero. [`has_fixed_target`](@ref) finds such a target, and the measure then has no degree, because ``|x - c|`` with a fixed ``c \\neq 0`` is not homogeneous in ``x``.

The degree is a property of the risk as a function of the weights, so it reads the fees that the risk pays. A fee that does not scale with the weights is a fixed target on the net returns. A turnover fee against previous weights that are not zero, a liquidation charge and a fixed fee are such fees, and [`has_fixed_fee`](@ref) finds them. A measure that reads the net returns then has no degree, unless the fee cancels from its risk. A proportional fee, and a turnover fee against zero weights, scale with the weights and keep the degree.

Whether a fee cancels from the risk of an OWA measure depends on the sum of its weights, and a weight builder gives the weights of the observation count. [`adjusted_risk`](@ref) passes the count `T` of its returns matrix. Without `T`, the sum of a builder is not known, and the fee does not cancel.

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

 1. Return `nothing` when [`has_fixed_target`](@ref) of `r`, `fees` and `T` is `true`. This applies to steps 2 and 7.
 2. Return `2` for a member of [`SquaredRiskMeasures`](@ref), `3` for a member of [`CubedRiskMeasures`](@ref), `4` for a member of [`FourthPowerRiskMeasures`](@ref), and `0` for a member of [`ScaleFreeRiskMeasures`](@ref).
 3. For [`RiskTrackingRiskMeasure`](@ref), return `nothing` when its benchmark is a fixed target. Otherwise return the degree of its measure `r` under `fees` and `T`. The benchmark weights are zero at this step, so the measure tracks `r` itself on the same net returns.
 4. For [`RiskRatio`](@ref) and [`NonOptimisationRiskRatio`](@ref), get the degree of each side under `fees` and `T` with [`scalarised_degree`](@ref), and subtract the degree of `r2` from the degree of `r1` with [`degree_difference`](@ref).
 5. For [`MeanReturnRiskRatio`](@ref) and [`ExpectedReturnRiskRatio`](@ref), return `nothing` when `rf` is not zero. Otherwise subtract the degree of `rk` from the degree of `rt`, in the same way. A return estimator does not read `T`. [`ExpectedReturn`](@ref) has the degree of its return estimator `rt`.
 6. Return `nothing` for [`VarianceSkewKurtosis`](@ref), whose three terms have the degrees 2, 3 and 4, and for a member of [`RelativeDrawdownRiskMeasures`](@ref).
 7. Return `1` for every other measure.

# Arguments

  - $(arg_dict[:r])
  - `fees`: The fees that the risk pays, or `nothing`.
  - `T`: The number of observations of the net returns, or `nothing` when it is not known.

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

julia> PortfolioOptimisers.homogeneity_degree(LowOrderMoment(; mu = 0.01))

julia> PortfolioOptimisers.homogeneity_degree(TurnoverRiskMeasure(; w = [0.5, 0.5]))

julia> PortfolioOptimisers.homogeneity_degree(TurnoverRiskMeasure(; w = [0.0, 0.0]))
1

julia> fees = Fees(; tn = Turnover(; w = [0.5, 0.5], val = 0.001));

julia> PortfolioOptimisers.homogeneity_degree(LowOrderMoment(; mu = 0.0), fees)

julia> PortfolioOptimisers.homogeneity_degree(LowOrderMoment(), fees)
1

julia> r = RiskRatio(; r1 = OrderedWeightsArray(), r2 = Variance());

julia> PortfolioOptimisers.homogeneity_degree(r, fees),
       PortfolioOptimisers.homogeneity_degree(r, fees, 200)
(nothing, -1)
```

# Related

  - [`adjust_risk_contribution`](@ref)
  - [`risk_contribution`](@ref)
  - [`has_fixed_target`](@ref)
  - [`has_fixed_fee`](@ref)
  - [`SquaredRiskMeasures`](@ref)
  - [`CubedRiskMeasures`](@ref)
  - [`FourthPowerRiskMeasures`](@ref)
  - [`ScaleFreeRiskMeasures`](@ref)
  - [`RelativeDrawdownRiskMeasures`](@ref)
"""
function homogeneity_degree(r::AbstractBaseRiskMeasure, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return has_fixed_target(r, fees, T) ? nothing : 1
end
function homogeneity_degree(r::SquaredRiskMeasures, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return has_fixed_target(r, fees, T) ? nothing : 2
end
function homogeneity_degree(r::CubedRiskMeasures, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return has_fixed_target(r, fees, T) ? nothing : 3
end
function homogeneity_degree(r::FourthPowerRiskMeasures, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return has_fixed_target(r, fees, T) ? nothing : 4
end
function homogeneity_degree(r::ScaleFreeRiskMeasures, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return has_fixed_target(r, fees, T) ? nothing : 0
end
function homogeneity_degree(r::RiskTrackingRiskMeasure, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    if has_fixed_target(r)
        return nothing
    end
    return homogeneity_degree(r.r, fees, T)
end
function homogeneity_degree(::Union{VarianceSkewKurtosis, RelativeDrawdownRiskMeasures},
                            ::Option{<:Fees} = nothing, ::Option{<:Integer} = nothing)
    return nothing
end
function homogeneity_degree(r::RiskRatio, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return degree_difference(homogeneity_degree(r.r1, fees, T),
                             homogeneity_degree(r.r2, fees, T))
end
function homogeneity_degree(r::NonOptimisationRiskRatio, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    return degree_difference(scalarised_degree(r.r1, r.sca1, fees, T),
                             scalarised_degree(r.r2, r.sca2, fees, T))
end
function homogeneity_degree(r::MeanReturnRiskRatio, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    if !iszero(r.rf)
        return nothing
    end
    return degree_difference(homogeneity_degree(r.rt, fees),
                             scalarised_degree(r.rk, r.sca, fees, T))
end
"""
    has_fixed_target(r::AbstractBaseRiskMeasure) -> Bool
    has_fixed_target(r::AbstractBaseRiskMeasure, fees::Option{<:Fees},
                     T::Option{<:Integer} = nothing) -> Bool
    has_fixed_target(target) -> Bool

Return `true` when the risk measure `r` measures a distance to a target that is not zero and does not scale with the portfolio weights.

Such a measure is not homogeneous in the weights, so [`homogeneity_degree`](@ref) gives it no degree. A target that scales with the weights, such as the expected portfolio return ``\\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w}``, keeps the degree of the measure. The second form also reads the fees: a fee that does not scale with the weights moves the net returns by a fixed amount, so it is a fixed target of a measure that reads the net returns. The third form reads one target value. A [`Turnover`](@ref) fee is a distance to its previous weights.

# Algorithm

 1. For a [`LowOrderMoment`](@ref), a [`HighOrderMoment`](@ref), a [`ThirdCentralMoment`](@ref) and a [`MedianAbsoluteDeviation`](@ref), read the Moment Target `mu`. A number is fixed when it is not zero. A [`VecScalar`](@ref) is fixed when its offset `s` is not zero.
 2. For a [`TrackingRiskMeasure`](@ref) and a [`RiskTrackingRiskMeasure`](@ref), read the benchmark `tr`. The weights of a [`WeightsTracking`](@ref) and the series of a [`ReturnsTracking`](@ref) are fixed when they are not zero. A vector `B` of a [`DataBenchmarkTracking`](@ref) is fixed when it is not zero, and so is an unbound `B`, whose target is not known. A matrix `B` gives the series ``\\mathbf{B} \\boldsymbol{w}``, which scales with the weights.
 3. For a [`TurnoverRiskMeasure`](@ref), the reference weights `w` are fixed when they are not zero.
 4. For a [`Turnover`](@ref) fee, the previous weights `w` are fixed when an asset whose rate is not zero has a previous weight that is not zero.
 5. With `fees`, return `true` when step 1, 2 or 3 is `true`. Otherwise return `true` when `r` reads the fees, [`reads_fees`](@ref), and [`has_fixed_fee`](@ref) finds a fee that does not cancel from the risk of `r` on `T` observations, [`cancels_constant_fee`](@ref).
 6. Return `false` for every other measure and every other target value.

# Arguments

  - $(arg_dict[:r])
  - `fees`: The fees that the risk pays, or `nothing`.
  - `T`: The number of observations of the net returns, or `nothing` when it is not known.
  - `target`: The target value of a measure.

# Returns

  - `flag::Bool`: `true` when the target is fixed and not zero.

# Examples

```jldoctest
julia> PortfolioOptimisers.has_fixed_target(LowOrderMoment(; mu = 0.01))
true

julia> PortfolioOptimisers.has_fixed_target(LowOrderMoment(; mu = [0.01, 0.02]))
false

julia> PortfolioOptimisers.has_fixed_target(TrackingRiskMeasure(;
                                                                tr = WeightsTracking(;
                                                                                     w = [0.5,
                                                                                          0.5])))
true

julia> PortfolioOptimisers.has_fixed_target(ConditionalValueatRisk(), Fees(; fl = 0.001))
true

julia> PortfolioOptimisers.has_fixed_target(Variance(), Fees(; fl = 0.001))
false
```

# Related

  - [`homogeneity_degree`](@ref)
  - [`calc_moment_target`](@ref)
  - [`tracking_benchmark`](@ref)
  - [`has_fixed_fee`](@ref)
  - [`reads_fees`](@ref)
  - [`cancels_constant_fee`](@ref)
"""
function has_fixed_target(r::Union{LoHiOrderMoment, ThirdCentralMoment,
                                   MedianAbsoluteDeviation})
    return has_fixed_target(r.mu)
end
function has_fixed_target(r::Union{TrackingRiskMeasure, RiskTrackingRiskMeasure})
    return has_fixed_target(r.tr)
end
function has_fixed_target(r::TurnoverRiskMeasure)
    return !iszero(r.w)
end
function has_fixed_target(r::AbstractBaseRiskMeasure, ::Nothing,
                          ::Option{<:Integer} = nothing)
    return has_fixed_target(r)
end
function has_fixed_target(r::AbstractBaseRiskMeasure, fees::Fees,
                          T::Option{<:Integer} = nothing)
    return has_fixed_target(r) ||
           (reads_fees(r) && has_fixed_fee(fees, cancels_constant_fee(r, T)))
end
function has_fixed_target(tn::Turnover)
    return !iszero(tn.val .* tn.w)
end
function has_fixed_target(mu::Number)
    return !iszero(mu)
end
function has_fixed_target(mu::VecScalar)
    return !iszero(mu.s)
end
function has_fixed_target(tr::Union{WeightsTracking, ReturnsTracking})
    return !iszero(tr.w)
end
function has_fixed_target(tr::DataBenchmarkTracking)
    return isnothing(tr.B) || !iszero(tr.B)
end
function has_fixed_target(::DataBenchmarkTracking{<:MatNum})
    return false
end
function has_fixed_target(::Any)
    return false
end
"""
    has_fixed_fee(fees::Option{<:Fees}, cancels::Bool) -> Bool

Return `true` when `fees` holds a fee that does not scale with the portfolio weights and that does not cancel from the risk of a measure.

A periodic fee falls on each observation. The term of a [`Turnover`](@ref) fee whose previous weights are not zero is such a fee, and so is the proportional liquidation charge `lq`, which does not read the weights at all. A one-off fee is a fixed fee `fl` or `fs`, or the fixed liquidation charge `flq`. It counts the positions, so it does not scale with the weights. The Fee Amortisation `fa` names its clock. [`AmortisedFees`](@ref) spreads it over each observation, so it is then a periodic fee. Otherwise it falls on the first observation alone.

A periodic fee moves each net return by the same amount, so it cancels from the risk of a measure that a constant shift does not change. `cancels` states that property of the measure, from [`cancels_constant_fee`](@ref). A one-off fee on the first observation moves one return, and it cancels from no measure. A proportional fee `l` or `s`, and a turnover fee against zero weights, scale with the weights, so they are not fixed.

# Algorithm

 1. Return `false` for a `nothing` `fees`.
 2. Get the flags `periodic` and `one_off` on the clock `fees.fa` with [`fixed_fee_terms`](@ref).
 3. Return `true` when `one_off` is `true`, or when `periodic` is `true` and `cancels` is `false`.

# Arguments

  - `fees`: The fees that the risk pays, or `nothing`.
  - `cancels::Bool`: Whether a fee that every observation pays in the same amount cancels from the risk.

# Returns

  - `flag::Bool`: `true` when a fee that does not scale with the weights reaches the risk.

# Examples

```jldoctest
julia> fees = Fees(; tn = Turnover(; w = [0.5, 0.5], val = 0.001));

julia> PortfolioOptimisers.has_fixed_fee(fees, false),
       PortfolioOptimisers.has_fixed_fee(fees, true)
(true, false)

julia> PortfolioOptimisers.has_fixed_fee(Fees(; fl = 0.001), true)
true

julia> PortfolioOptimisers.has_fixed_fee(Fees(; fl = 0.001, fa = AmortisedFees()), true)
false

julia> PortfolioOptimisers.has_fixed_fee(Fees(; l = 0.001), false)
false
```

# Related

  - [`has_fixed_target`](@ref)
  - [`cancels_constant_fee`](@ref)
  - [`fixed_fee_terms`](@ref)
  - [`Fees`](@ref)
  - [`AmortisedFees`](@ref)
  - [`charge_fees`](@ref)
"""
function has_fixed_fee(::Nothing, ::Bool)
    return false
end
function has_fixed_fee(fees::Fees, cancels::Bool)
    periodic, one_off = fixed_fee_terms(fees, fees.fa)
    return one_off || (!cancels && periodic)
end
"""
    fixed_fee_terms(fees::Fees, fa::Option{<:AbstractFeeAmortisation}) -> Tuple{Bool, Bool}

Return the pair `(periodic, one_off)`: whether `fees` holds a fee that does not scale with the portfolio weights and falls on each observation, and whether it holds one that falls on the first observation alone.

[`has_fixed_fee`](@ref) reads the pair. The Fee Amortisation `fa` is the clock of the one-off terms, so it decides which flag they set.

# Algorithm

 1. Set `periodic` when [`has_fixed_target`](@ref) of `fees.tn` is `true`, or when the charge of `fees.lq` from [`calc_liquidation_fees`](@ref) is not zero.
 2. Set `one_off` when a rate `fees.fl` or `fees.fs` is not zero, or when the charge of `fees.flq` from [`calc_fixed_liquidation_fees`](@ref) is not zero. A `nothing` rate is zero.
 3. Under [`AmortisedFees`](@ref), the one-off terms fall on each observation, so return `(periodic || one_off, false)`. Otherwise return `(periodic, one_off)`.

# Arguments

  - `fees::Fees`: The fees that the risk pays.
  - `fa`: The Fee Amortisation of `fees`.

# Returns

  - `(periodic, one_off)::Tuple{Bool, Bool}`: The two flags.

# Related

  - [`has_fixed_fee`](@ref)
  - [`AmortisedFees`](@ref)
  - [`FirstObservationFees`](@ref)
"""
function fixed_fee_terms(fees::Fees, ::Union{Nothing, FirstObservationFees})
    periodic = has_fixed_target(fees.tn) || !iszero(calc_liquidation_fees(fees.lq))
    one_off = any(!iszero,
                  (something(fees.fl, false), something(fees.fs, false),
                   calc_fixed_liquidation_fees(fees.flq, fees.kwargs)))
    return (periodic, one_off)
end
function fixed_fee_terms(fees::Fees, ::AmortisedFees)
    periodic, one_off = fixed_fee_terms(fees, nothing)
    return (periodic || one_off, false)
end
"""
    reads_fees(r::AbstractBaseRiskMeasure) -> Bool
    reads_fees(kind::RiskInputKind) -> Bool

Return `true` when the risk of the measure `r` reads the fees.

The input kind of `r`, from [`risk_input_kind`](@ref), gives the default. A [`WeightsInput`](@ref) measure reads the weights alone, so it does not read the fees. A [`NetReturnsInput`](@ref) and a [`WeightsReturnsFeesInput`](@ref) measure read them. Three measures take the fees and do not read them, because their risk reads the moments of a prior: [`VarianceSkewKurtosis`](@ref), and a [`ValueatRisk`](@ref) and a [`ValueatRiskRange`](@ref) with [`DistributionValueatRisk`](@ref). A [`GenericValueatRiskRange`](@ref) reads the fees when one of its tails reads them.

# Arguments

  - $(arg_dict[:r])
  - `kind::RiskInputKind`: The input kind of a measure.

# Returns

  - `flag::Bool`: `true` when the risk reads the fees.

# Examples

```jldoctest
julia> PortfolioOptimisers.reads_fees(ConditionalValueatRisk())
true

julia> PortfolioOptimisers.reads_fees(Variance())
false
```

# Related

  - [`has_fixed_target`](@ref)
  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
"""
function reads_fees(r::AbstractBaseRiskMeasure)
    return reads_fees(risk_input_kind(r))
end
function reads_fees(::RiskInputKind)
    return true
end
function reads_fees(::WeightsInput)
    return false
end
function reads_fees(::Union{VarianceSkewKurtosis,
                            ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk},
                            ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                             <:DistributionValueatRisk}})
    return false
end
function reads_fees(r::GenericValueatRiskRange)
    return reads_fees(r.loss) || reads_fees(r.gain)
end
"""
    cancels_constant_fee(r::AbstractBaseRiskMeasure, T::Option{<:Integer} = nothing) -> Bool

Return `true` when the risk of the measure `r` on `T` observations does not change when the same number is added to each net return.

A fee that every observation pays in the same amount is such a number, so it cancels from the risk, and [`has_fixed_fee`](@ref) does not count it. A measure that no other method names returns `false`. A tail measure, a drawdown and a tracking error move with the number.

# Algorithm

 1. Return `true` for a member of [`ShiftInvariantRiskMeasures`](@ref).
 2. For a [`LowOrderMoment`](@ref), a [`HighOrderMoment`](@ref), a [`ThirdCentralMoment`](@ref) and a [`MedianAbsoluteDeviation`](@ref), return `true` when the Moment Target `mu` is not a number. The mean or the median of the net returns moves with the number. A per-asset target is net of the mean fee per period, [`moment_target_fees`](@ref), so it moves too. A number is a threshold on the net returns, and it does not move.
 3. For a [`GenericValueatRiskRange`](@ref), return `true` when both tails read the fees, [`reads_fees`](@ref). A tail with [`DistributionValueatRisk`](@ref) does not move, so the other tail does not cancel.
 4. For an [`OrderedWeightsArray`](@ref), return `true` when its weights on `T` observations sum to zero, from [`resolve_owa_weights`](@ref) and [`sums_to_zero`](@ref). A number ``c`` moves the risk by ``c`` times the sum of the weights. For an [`OrderedWeightsArrayRange`](@ref), apply the same rule to `w1 - w2`, so the two tails must hold the same sum. A weight builder without `T` has no known sum, so it returns `false`.
 5. Return `false` for every other measure.

# Arguments

  - $(arg_dict[:r])
  - `T`: The number of observations of the net returns, or `nothing` when it is not known.

# Returns

  - `flag::Bool`: `true` when a constant shift of the net returns cancels from the risk.

# Examples

```jldoctest
julia> PortfolioOptimisers.cancels_constant_fee(LowOrderMoment())
true

julia> PortfolioOptimisers.cancels_constant_fee(LowOrderMoment(; mu = 0.0))
false

julia> PortfolioOptimisers.cancels_constant_fee(ConditionalValueatRiskRange())
true

julia> PortfolioOptimisers.cancels_constant_fee(ConditionalValueatRisk())
false

julia> PortfolioOptimisers.cancels_constant_fee(OrderedWeightsArray(), 200)
true

julia> PortfolioOptimisers.cancels_constant_fee(OrderedWeightsArray())
false
```

# Related

  - [`has_fixed_fee`](@ref)
  - [`ShiftInvariantRiskMeasures`](@ref)
  - [`moment_target_fees`](@ref)
  - [`resolve_owa_weights`](@ref)
  - [`sums_to_zero`](@ref)
"""
function cancels_constant_fee(::AbstractBaseRiskMeasure, ::Option{<:Integer} = nothing)
    return false
end
function cancels_constant_fee(::ShiftInvariantRiskMeasures, ::Option{<:Integer} = nothing)
    return true
end
function cancels_constant_fee(r::Union{LoHiOrderMoment, ThirdCentralMoment,
                                       MedianAbsoluteDeviation},
                              ::Option{<:Integer} = nothing)
    return !isa(r.mu, Number)
end
function cancels_constant_fee(r::GenericValueatRiskRange, ::Option{<:Integer} = nothing)
    return reads_fees(r.loss) && reads_fees(r.gain)
end
function cancels_constant_fee(r::OrderedWeightsArray, T::Option{<:Integer} = nothing)
    return sums_to_zero(resolve_owa_weights(r.w, T))
end
function cancels_constant_fee(r::OrderedWeightsArrayRange, T::Option{<:Integer} = nothing)
    return sums_to_zero(resolve_owa_weights(r.w1, T), resolve_owa_weights(r.w2, T))
end
"""
    sums_to_zero(w::Option{<:VecNum}) -> Bool
    sums_to_zero(w1::Option{<:VecNum}, w2::Option{<:VecNum}) -> Bool

Return `true` when the OWA weights `w` sum to zero. The second form tests `w1 - w2`. A `nothing` stands for weights whose sum is not known, from [`resolve_owa_weights`](@ref), and returns `false`.

The vector test is `isapprox(s + sum(w), s)` with `s = sum(abs, w)`, so it takes the default tolerance of the element type. A floating point vector passes at a sum near the rounding error of `s`, and an exact type, such as a `Rational`, passes only at an exact zero.

# Arguments

  - `w::Option{<:VecNum}`: The weights, or `nothing`.
  - `w1`, `w2`: The weights of the two tails of a range, or `nothing`.

# Returns

  - `flag::Bool`: `true` when the weights sum to zero.

# Related

  - [`cancels_constant_fee`](@ref)
  - [`resolve_owa_weights`](@ref)
"""
function sums_to_zero(w::VecNum)
    s = sum(abs, w)
    return isapprox(s + sum(w), s)
end
function sums_to_zero(::Nothing)
    return false
end
function sums_to_zero(w1::VecNum, w2::VecNum)
    return sums_to_zero(w1 - w2)
end
function sums_to_zero(::Option{<:VecNum}, ::Option{<:VecNum})
    return false
end
"""
    resolve_owa_weights(w::OWA_Func_VecNum, T::Option{<:Integer}) -> Option{<:VecNum}

Return the OWA weights of the weight slot `w` on `T` observations, or `nothing` when they are not known.

A weight vector is its own weights. A weight builder gives the weights of the observation count, so it returns `w(T)`, as the risk of [`OrderedWeightsArray`](@ref) does. Without `T`, a builder returns `nothing`.

# Arguments

  - `w::OWA_Func_VecNum`: A weight vector or a weight builder.
  - `T`: The number of observations, or `nothing` when it is not known.

# Returns

  - `w::Option{<:VecNum}`: The weights, or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.resolve_owa_weights([0.5, -0.5], nothing)
2-element Vector{Float64}:
  0.5
 -0.5

julia> PortfolioOptimisers.resolve_owa_weights(owa_gmd, 3) == owa_gmd(3)
true

julia> isnothing(PortfolioOptimisers.resolve_owa_weights(owa_gmd, nothing))
true
```

# Related

  - [`sums_to_zero`](@ref)
  - [`cancels_constant_fee`](@ref)
  - [`OWA_Func_VecNum`](@ref)
"""
function resolve_owa_weights(w::VecNum, ::Option{<:Integer})
    return w
end
function resolve_owa_weights(w::Union{Function, AbstractOrderedWeightsArrayFunction},
                             T::Integer)
    return w(T)
end
function resolve_owa_weights(::Union{Function, AbstractOrderedWeightsArrayFunction},
                             ::Nothing)
    return nothing
end
"""
    scalarised_degree(r::BaseRM_VecBaseRM, sca::Scalariser, fees::Option{<:Fees} = nothing,
                      T::Option{<:Integer} = nothing) -> Option{<:Integer}

Return the degree of homogeneity of one side of a ratio, which is one measure or a vector of measures that `sca` combines, under the fees `fees` on `T` observations.

A single measure ignores `sca`. A vector has the common degree of its elements under [`SumScalariser`](@ref), [`MaxScalariser`](@ref) and [`MinScalariser`](@ref). It has no degree when two elements have different degrees, or under [`LogSumExpScalariser`](@ref), which is not homogeneous.

# Arguments

  - $(arg_dict[:r])
  - `sca::Scalariser`: Scalariser of the side.
  - `fees`: The fees that the risk pays, or `nothing`.
  - `T`: The number of observations of the net returns, or `nothing` when it is not known.

# Returns

  - `d::Option{<:Integer}`: The degree of the side, or `nothing`.

# Related

  - [`homogeneity_degree`](@ref)
  - [`degree_difference`](@ref)
"""
function scalarised_degree(r::AbstractBaseRiskMeasure, ::Scalariser,
                           fees::Option{<:Fees} = nothing, T::Option{<:Integer} = nothing)
    return homogeneity_degree(r, fees, T)
end
function scalarised_degree(::VecBaseRM, ::LogSumExpScalariser, ::Option{<:Fees} = nothing,
                           ::Option{<:Integer} = nothing)
    return nothing
end
function scalarised_degree(rs::VecBaseRM, ::Scalariser, fees::Option{<:Fees} = nothing,
                           T::Option{<:Integer} = nothing)
    d = homogeneity_degree(first(rs), fees, T)
    return all(r -> homogeneity_degree(r, fees, T) == d, rs) ? d : nothing
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
    adjust_risk_contribution(r::AbstractBaseRiskMeasure, val::Number,
                             fees::Option{<:Fees} = nothing,
                             T::Option{<:Integer} = nothing) -> Number

Divide the risk `val` of the measure `r` by the degree of homogeneity of `r` under the fees `fees` on `T` observations.

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
  - `fees`: The fees that the risk pays, or `nothing`.
  - `T`: The number of observations of the net returns, or `nothing` when it is not known.

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
function adjust_risk_contribution(r::AbstractBaseRiskMeasure, val::Number,
                                  fees::Option{<:Fees} = nothing,
                                  T::Option{<:Integer} = nothing)
    d = homogeneity_degree(r, fees, T)
    if isnothing(d) || iszero(d)
        return val
    end
    return val / d
end
