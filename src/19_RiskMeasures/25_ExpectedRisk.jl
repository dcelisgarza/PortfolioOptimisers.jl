const MatNum_Pr = Union{<:MatNum, <:AbstractPriorResult, <:ReturnsResult}
const ERkNetRet = Union{<:WorstRealisation, <:ValueatRisk, <:ValueatRiskRange,
                        <:ConditionalValueatRisk,
                        <:DistributionallyRobustConditionalValueatRisk,
                        <:DistributionallyRobustConditionalValueatRiskRange,
                        <:EntropicValueatRisk, <:EntropicValueatRiskRange,
                        <:RelativisticValueatRisk, <:RelativisticValueatRiskRange,
                        <:DrawdownatRisk, <:MaximumDrawdown, <:AverageDrawdown,
                        <:ConditionalDrawdownatRisk,
                        <:DistributionallyRobustConditionalDrawdownatRisk, <:UlcerIndex,
                        <:EntropicDrawdownatRisk, <:RelativisticDrawdownatRisk,
                        <:RelativeDrawdownatRisk, <:RelativeMaximumDrawdown,
                        <:RelativeAverageDrawdown, <:RelativeConditionalDrawdownatRisk,
                        <:RelativeUlcerIndex, <:RelativeEntropicDrawdownatRisk,
                        <:RelativeRelativisticDrawdownatRisk, <:Range,
                        <:ConditionalValueatRiskRange, <:OrderedWeightsArray,
                        <:OrderedWeightsArrayRange, <:BrownianDistanceVariance,
                        <:MeanReturn, <:PowerNormValueatRisk, <:PowerNormValueatRiskRange,
                        <:PowerNormDrawdownatRisk, <:RelativePowerNormDrawdownatRisk}
const ERkwXFees = Union{<:LowOrderMoment, <:HighOrderMoment, <:TrackingRiskMeasure,
                        <:RiskTrackingRiskMeasure, <:Kurtosis, <:ThirdCentralMoment,
                        <:Skewness, <:MedianAbsoluteDeviation}
const ERkX = Union{<:ERkNetRet, <:ERkwXFees}
const ERkw = Union{<:StandardDeviation, <:NegativeSkewness, <:TurnoverRiskMeasure,
                   <:Variance, <:UncertaintySetVariance, <:EqualRiskMeasure}
const TnTrRM = Union{<:TurnoverRiskMeasure, <:TrRM}
const SlvRM = Union{<:EntropicValueatRisk, <:EntropicValueatRiskRange,
                    <:EntropicDrawdownatRisk, <:RelativeEntropicDrawdownatRisk,
                    <:RelativisticValueatRisk, <:RelativisticValueatRiskRange,
                    <:RelativisticDrawdownatRisk, <:RelativeRelativisticDrawdownatRisk}
const RkRatioRM = Union{<:RiskRatioRiskMeasure, <:NonOptimisationRiskRatioRiskMeasure}
function expected_risk(r::ERkNetRet, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing;
                       kwargs...)
    return r(calc_net_returns(w, X, fees))
end
function expected_risk(r::ERkwXFees, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing;
                       kwargs...)
    return r(w, X, fees)
end
function expected_risk(r::ERkw, w::VecNum, args...; kwargs...)
    return r(w)
end
#! Start: Only exists to avoid ambiguities.
function expected_risk(r::ERkw, w::VecNum, ::Pr_RR, args...; kwargs...)
    return r(w)
end
#! End: Only exists to avoid ambiguities.
function expected_risk(r::RkRatioRM, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing;
                       kwargs...)
    return expected_risk(r.r1, w, X, fees; kwargs...) /
           expected_risk(r.r2, w, X, fees; kwargs...)
end
function expected_risk(r::MeanReturnRiskRatio, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return (expected_risk(r.rt, w, X, fees; kwargs...) - r.rf) /
           expected_risk(r.rk, w, X, fees; kwargs...)
end
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, args...; kwargs...)
    return expected_risk(r, w, pr.X, args...; kwargs...)
end
function expected_risk(r::AbstractBaseRiskMeasure, w::VecVecNum, args...; kwargs...)
    return [expected_risk(r, wi, args...; kwargs...) for wi in w]
end
"""
    number_effective_assets(w::VecNum)

Compute the effective number of assets (Herfindahl-Hirschman inverse index):

```math
N_{\\mathrm{eff}} = \\frac{1}{\\sum_i w_i^2}
```

Returns the number of equally-weighted assets that would produce the same level of
concentration as the given weight vector `w`.

# Arguments

  - `w::VecNum`: Portfolio weight vector.

# Returns

  - `Number`: Effective number of assets.

# Related

  - [`set_number_effective_assets!`](@ref)
  - [`EqualRiskMeasure`](@ref)
  - [`risk_contribution`](@ref)
  - [`EqualRiskMeasure`](@ref)
"""
function number_effective_assets(w::VecNum)
    return inv(LinearAlgebra.dot(w, w))
end
"""
    risk_contribution(
        r::AbstractBaseRiskMeasure,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        delta::Number = 1e-6,
        marginal::Bool = false,
        kwargs...
    ) -> Vector

Compute the risk contribution of each asset to the total portfolio risk using numerical differentiation.

The risk contribution of asset ``i`` is defined as:

```math
\\mathrm{RC}_i = w_i \\cdot \\frac{\\partial \\rho(\\boldsymbol{w})}{\\partial w_i}\\,,
```

where the partial derivative is approximated by a two-sided finite difference with step size `delta`. When `marginal = true`, the weighting by ``w_i`` is omitted (i.e., only the marginal risk ``\\partial \\rho / \\partial w_i`` is returned).

# Arguments

  - `r::AbstractBaseRiskMeasure`: Risk measure to differentiate.
  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum_Pr`: Asset returns matrix or prior result.
  - `fees::Option{<:Fees}`: Optional fee structure.

# Keyword Arguments

  - `delta::Number = 1e-6`: Finite difference step size.
  - `marginal::Bool = false`: If `true`, returns marginal risk contributions (without ``w_i`` weighting).

# Returns

  - `Vector`: Risk contributions (or marginal risks) for each asset.

# Related

  - [`expected_risk`](@ref)
  - [`factor_risk_contribution`](@ref)
"""
function risk_contribution(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum_Pr,
                           fees::Option{<:Fees} = nothing; delta::Number = 1e-6,
                           marginal::Bool = false, kwargs...)
    N = length(w)
    rc = Vector{eltype(w)}(undef, N)
    ws = Matrix{eltype(w)}(undef, N, 2)
    ws .= w
    id2 = inv(2 * delta)
    for i in eachindex(w)
        ws[i, 1] += delta
        ws[i, 2] -= delta
        r1 = expected_risk(r, view(ws, :, 1), X, fees; kwargs...)
        r2 = expected_risk(r, view(ws, :, 2), X, fees; kwargs...)
        r1 = adjust_risk_contribution(r, r1, delta)
        r2 = adjust_risk_contribution(r, r2, delta)
        rci = (r1 - r2) * id2
        rc[i] = rci
        ws[i, 1] = w[i]
        ws[i, 2] = w[i]
    end
    if !marginal
        rc .*= w
    end
    return rc
end
"""
    factor_risk_contribution(
        r::AbstractBaseRiskMeasure,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        re::RegE_Reg = StepwiseRegression(),
        rd::ReturnsResult = ReturnsResult(),
        delta::Number = 1e-6,
        kwargs...
    ) -> Vector

Compute the risk contribution of each factor (and the idiosyncratic component) to the total portfolio risk using a factor regression.

The factor risk contributions partition total portfolio risk into factor-specific components using the Brinson attribution framework:

```math
\\mathrm{FRC}_k = (\\mathbf{B}^\\intercal \\boldsymbol{w})_k \\cdot (\\mathbf{B}^{-\\intercal} \\nabla \\rho)_k\\,,
```

where ``\\mathbf{B}`` is the factor loading matrix estimated by regression.

# Arguments

  - `r::AbstractBaseRiskMeasure`: Risk measure to decompose.
  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum_Pr`: Asset returns matrix or prior result.
  - `fees::Option{<:Fees}`: Optional fee structure.

# Keyword Arguments

  - `re::RegE_Reg = StepwiseRegression()`: Regression estimator for factor loadings.
  - `rd::ReturnsResult = ReturnsResult()`: Returns result providing factor data.
  - `delta::Number = 1e-6`: Finite difference step size.

# Returns

  - `Vector`: Risk contributions for each factor, with the last element being the idiosyncratic (off-factor) contribution.

# Related

  - [`risk_contribution`](@ref)
  - [`expected_risk`](@ref)
"""
function factor_risk_contribution(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum_Pr,
                                  fees::Option{<:Fees} = nothing;
                                  re::RegE_Reg = StepwiseRegression(),
                                  rd::ReturnsResult = ReturnsResult(), delta::Number = 1e-6,
                                  kwargs...)
    rr = regression(re, rd)
    Bt = transpose(rr.L)
    b2t = transpose(LinearAlgebra.pinv(transpose(LinearAlgebra.nullspace(Bt))))
    b3t = transpose(LinearAlgebra.pinv(b2t))
    mr = risk_contribution(r, w, X, fees; delta = delta, marginal = true, kwargs...)
    rc_f = (Bt * w) .* (transpose(LinearAlgebra.pinv(Bt)) * mr)
    rc_of = sum((b2t * w) .* (b3t * mr))
    rc_f = [rc_f; rc_of]
    return rc_f
end

export RiskRatioRiskMeasure, number_effective_assets, risk_contribution,
       factor_risk_contribution
