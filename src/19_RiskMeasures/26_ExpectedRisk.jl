"""
    const MatNum_Pr = Union{<:MatNum, <:AbstractPriorResult, <:ReturnsResult}

Union of matrix-like types accepted as the data argument in [`risk_contribution`](@ref) and related functions.

# Related

  - [`MatNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
const MatNum_Pr = Union{<:MatNum, <:AbstractPriorResult, <:ReturnsResult}
"""
    const ERkNetRet = Union{...}

Union of risk measures whose expected risk is computed on net returns (returns after fees).

These risk measures receive the net-returns vector `calc_net_returns(w, X, fees)` as their sole argument.

# Related

  - [`expected_risk`](@ref)
  - [`calc_net_returns`](@ref)
"""
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
"""
    const ERkwXFees = Union{...}

Union of risk measures whose expected risk depends on both weights, returns matrix, and fees.

These risk measures are called as `r(w, X, fees)`.

# Related

  - [`expected_risk`](@ref)
  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`TrackingRiskMeasure`](@ref)
"""
const ERkwXFees = Union{<:LowOrderMoment, <:HighOrderMoment, <:TrackingRiskMeasure,
                        <:RiskTrackingRiskMeasure, <:Kurtosis, <:ThirdCentralMoment,
                        <:Skewness, <:MedianAbsoluteDeviation, <:VarianceSkewKurtosis}
"""
    const ERkX = Union{<:ERkNetRet, <:ERkwXFees}

Union of all risk measures that require the returns matrix `X` (and optionally fees) for expected risk computation.

# Related

  - [`ERkNetRet`](@ref)
  - [`ERkwXFees`](@ref)
  - [`expected_risk`](@ref)
"""
const ERkX = Union{<:ERkNetRet, <:ERkwXFees}
"""
    const ERkw = Union{...}

Union of risk measures whose expected risk depends only on portfolio weights.

These risk measures are called as `r(w)`.

# Related

  - [`expected_risk`](@ref)
  - [`StandardDeviation`](@ref)
  - [`Variance`](@ref)
"""
const ERkw = Union{<:StandardDeviation, <:NegativeSkewness, <:TurnoverRiskMeasure,
                   <:Variance, <:UncertaintySetVariance, <:EqualRiskMeasure}
"""
    const TnTrRM = Union{<:TurnoverRiskMeasure, <:TrRM}

Union of turnover and tracking risk measures used to update previous-weight dependent factories.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`TrRM`](@ref)
"""
const TnTrRM = Union{<:TurnoverRiskMeasure, <:TrRM}
"""
    const SlvRM = Union{...}

Union of solver-based risk measures (entropic and relativistic families) that require an iterative solver for expected risk computation.

# Related

  - [`expected_risk`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticValueatRisk`](@ref)
"""
const SlvRM = Union{<:EntropicValueatRisk, <:EntropicValueatRiskRange,
                    <:EntropicDrawdownatRisk, <:RelativeEntropicDrawdownatRisk,
                    <:RelativisticValueatRisk, <:RelativisticValueatRiskRange,
                    <:RelativisticDrawdownatRisk, <:RelativeRelativisticDrawdownatRisk}
"""
    const RkRatioRM = Union{<:RiskRatioRiskMeasure, <:NonOptimisationRiskRatioRiskMeasure}

Union of all risk-ratio risk measures, where the expected risk is defined as the ratio of two component risk values.

# Related

  - [`RiskRatioRiskMeasure`](@ref)
  - [`NonOptimisationRiskRatioRiskMeasure`](@ref)
  - [`expected_risk`](@ref)
"""
const RkRatioRM = Union{<:RiskRatioRiskMeasure, <:NonOptimisationRiskRatioRiskMeasure}
"""
    expected_risk(r::ERkNetRet, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::ERkwXFees, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::ERkw, w::VecNum, args...; kwargs...)
    expected_risk(r::RkRatioRM, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::MeanReturnRiskRatio, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, args...; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, w::VecVecNum, args...; kwargs...)

Compute the expected value of a risk measure for a portfolio.

Dispatches on the type of `r` to select the appropriate computation:

  - [`ERkNetRet`](@ref): calls `r(calc_net_returns(w, X, fees))`.
  - [`ERkwXFees`](@ref): calls `r(w, X, fees)`.
  - [`ERkw`](@ref): calls `r(w)` (ignores `X` and `fees`).
  - [`RkRatioRM`](@ref): computes `expected_risk(r.r1, ...) / expected_risk(r.r2, ...)`.
  - [`MeanReturnRiskRatio`](@ref): `(expected_risk(r.rt, ...) - r.rf) / expected_risk(r.rk, ...)`.
  - `AbstractBaseRiskMeasure` with [`Pr_RR`](@ref): extracts `X` from the prior result and recurses.
  - `AbstractBaseRiskMeasure` with `VecVecNum`: maps over each weight vector in `w`.

# Related

  - [`expected_risk`](@ref)
  - [`ERkNetRet`](@ref)
  - [`ERkwXFees`](@ref)
  - [`ERkw`](@ref)
  - [`RkRatioRM`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`calc_net_returns`](@ref)
"""
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

Compute the effective number of assets (Herfindahl-Hirschman inverse index).

# Mathematical definition

```math
\\begin{align}
N_{\\mathrm{eff}} &= \\frac{1}{\\sum_i w_i^2}\\,.
\\end{align}
```

Where:

  - ``N_{\\mathrm{eff}}``: Effective number of assets.
  - $(math_dict[:w_port])

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

# Mathematical definition

The risk contribution of asset ``i`` is defined as:

```math
\\begin{align}
\\mathrm{RC}_i &= w_i \\cdot \\frac{\\partial \\rho(\\boldsymbol{w})}{\\partial w_i}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RC}_i``: Risk contribution of asset ``i``.
  - $(math_dict[:w_port])
  - ``\\rho``: Portfolio risk measure.
  - ``w_i``: Weight of asset ``i``.

The partial derivative is approximated using a two-sided finite difference with step size `delta`. When `marginal = true`, the function omits the weighting by ``w_i`` (i.e., only the marginal risk ``\\partial \\rho / \\partial w_i`` is returned).

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

# Mathematical definition

The factor risk contributions partition total portfolio risk into factor-specific components using the Brinson attribution framework:

```math
\\begin{align}
\\mathrm{FRC}_k &= (\\mathbf{B}^\\intercal \\boldsymbol{w})_k \\cdot (\\mathbf{B}^{-\\intercal} \\nabla \\rho)_k\\,.
\\end{align}
```

Where:

  - ``\\mathrm{FRC}_k``: Risk contribution of factor ``k``.
  - $(math_dict[:w_port])
  - ``\\mathbf{B}``: Factor loading matrix ``N \\times K``, estimated by regression.
  - ``\\nabla \\rho``: Gradient of the risk measure with respect to the weights.

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
