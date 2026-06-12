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
    const RkRatioRM = Union{<:RiskRatioRiskMeasure, <:NonOptimisationRiskRatioRiskMeasure}

Union of all risk-ratio risk measures, where the expected risk is defined as the ratio of two component risk values.

# Related

  - [`RiskRatioRiskMeasure`](@ref)
  - [`NonOptimisationRiskRatioRiskMeasure`](@ref)
  - [`expected_risk`](@ref)
"""
const RkRatioRM = Union{<:RiskRatioRiskMeasure, <:NonOptimisationRiskRatioRiskMeasure}
"""
    expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, args...; kwargs...)
    expected_risk(kind::RiskInputKind, r, w::VecNum, args...; kwargs...)
    expected_risk(r::RkRatioRM, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::MeanReturnRiskRatio, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, args...; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, w::VecVecNum, args...; kwargs...)

Compute the expected value of a risk measure for a portfolio.

For a leaf measure, the generic entry consults [`risk_input_kind`](@ref) and dispatches on the returned [`RiskInputKind`](@ref):

  - [`NetReturnsInput`](@ref): calls `r(calc_net_returns(w, X, fees))`.
  - [`WeightsReturnsFeesInput`](@ref): calls `r(w, X, fees)`.
  - [`WeightsInput`](@ref): calls `r(w)` (ignores `X` and `fees`).

Composite and container forms keep explicit methods:

  - [`RkRatioRM`](@ref): computes `expected_risk(r.r1, ...) / expected_risk(r.r2, ...)`.
  - [`MeanReturnRiskRatio`](@ref): `(expected_risk(r.rt, ...) - r.rf) / expected_risk(r.rk, ...)`.
  - `AbstractBaseRiskMeasure` with [`Pr_RR`](@ref): extracts `X` from the prior result and recurses.
  - `AbstractBaseRiskMeasure` with `VecVecNum`: maps over each weight vector in `w`.

# Related

  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
  - [`RkRatioRM`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`calc_net_returns`](@ref)
"""
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, args...; kwargs...)
    return expected_risk(risk_input_kind(r), r, w, args...; kwargs...)
end
function expected_risk(::NetReturnsInput, r, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return r(calc_net_returns(w, X, fees))
end
function expected_risk(::WeightsReturnsFeesInput, r, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return r(w, X, fees)
end
function expected_risk(::WeightsInput, r, w::VecNum, args...; kwargs...)
    return r(w)
end
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
# Precomputed-returns contract for the ratio composites: decompose onto the series, mirroring
# the `(w, X, fees)` decomposition above. Each component is evaluated via its own single-vector
# functor, so a ratio whose parts all support the contract works; one with a part that does not
# (e.g. a weights-only risk) surfaces the fallback error from that part.
function (r::RkRatioRM)(x::VecNum)
    return r.r1(x) / r.r2(x)
end
function (r::MeanReturnRiskRatio)(x::VecNum)
    return (r.rt(x) - r.rf) / r.rk(x)
end
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, args...; kwargs...)
    return expected_risk(r, w, pr.X, args...; kwargs...)
end
function expected_risk(r::AbstractBaseRiskMeasure, w::VecVecNum, args...; kwargs...)
    return [expected_risk(r, wi, args...; kwargs...) for wi in w]
end
"""
    (r::AbstractBaseRiskMeasure)(::VecNum)

Backstop for the single-argument *precomputed-returns* functor contract `r(x::VecNum)`
(ADR 0007).

This method is only ever reached by a measure that defines **no** `VecNum` functor of its
own — e.g. a composite carrying a weights-only variance term such as `VarianceSkewKurtosis`.
For such a measure the precomputed-returns form is undefined, so this throws.

It is *not* the primary safety mechanism. A [`WeightsInput`](@ref) measure's own functor
`r(w)` shares this `r(::VecNum)` signature and would otherwise silently consume a return
series *as weights*; dispatch alone cannot tell the two apart. Eligibility is therefore
decided up front by [`supports_precomputed_returns`](@ref), which the contract entry
[`_expected_risk_from_returns`](@ref) consults before ever calling the functor.
"""
function (r::AbstractBaseRiskMeasure)(::VecNum)
    throw(ArgumentError("`$(typeof(r))` has no precomputed-return-series form `r(x::VecNum)`: its risk depends on portfolio weights and/or per-asset data (e.g. a variance-carrying composite such as `VarianceSkewKurtosis`). Evaluate it through `expected_risk(r, w, X, fees)` with explicit weights instead."))
end
"""
    supports_precomputed_returns(r::AbstractBaseRiskMeasure) -> Bool

Whether risk measure `r` has a well-defined *precomputed-returns* form — i.e. whether its
expected risk can be evaluated on an already-reduced net-return series `x` alone, via the
functor `r(x::VecNum)` (ADR 0007).

The contract is well-defined exactly when the measure's result is a function of the series
alone:

  - [`NetReturnsInput`](@ref) measures (quantile / drawdown families): always `true` — their
    functor *is* the net-returns functor.
  - The moment family ([`LowOrderMoment`](@ref), [`HighOrderMoment`](@ref), [`Skewness`](@ref),
    [`Kurtosis`](@ref), [`MedianAbsoluteDeviation`](@ref), [`ThirdCentralMoment`](@ref)):
    `true` iff its target is weight-independent (`mu` is `nothing`, a scalar, or a centering
    function); a per-asset `mu` (`VecNum`/`VecScalar`) reduces as `dot(w, mu)` and needs the
    weights the series no longer carries, so `false`.
  - [`WeightsInput`](@ref) measures, tracking measures, and variance-carrying composites
    (`VarianceSkewKurtosis`): `false` — "risk of a bare return series" is undefined for them.

This predicate is what makes the precomputed-returns contract *safe*. Because a
`WeightsInput` measure's functor `r(w)` shares the `r(::VecNum)` signature with the contract,
dispatch alone cannot distinguish weights from returns; [`_expected_risk_from_returns`](@ref)
consults this predicate and throws an explanatory error for ineligible measures rather than
silently consuming the series as weights.

# Related

  - [`_expected_risk_from_returns`](@ref)
  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
"""
function supports_precomputed_returns(r::AbstractBaseRiskMeasure)
    return _supports_precomputed_returns(risk_input_kind(r), r)
end
"""
    const PrecomputedMomentRM = Union{...}

The moment-family risk measures that gain a single-argument precomputed-returns functor
`r(x::VecNum)` under ADR 0007. Their eligibility is instance-dependent (it turns on the
`mu` target), so [`supports_precomputed_returns`](@ref) inspects `r.mu` for these.

# Related

  - [`supports_precomputed_returns`](@ref)
"""
const PrecomputedMomentRM = Union{<:LowOrderMoment, <:HighOrderMoment, <:Skewness,
                                  <:Kurtosis, <:MedianAbsoluteDeviation,
                                  <:ThirdCentralMoment}
_supports_precomputed_returns(::NetReturnsInput, ::Any) = true
_supports_precomputed_returns(::WeightsInput, ::Any) = false
_supports_precomputed_returns(::WeightsReturnsFeesInput, ::Any) = false
function _supports_precomputed_returns(::WeightsReturnsFeesInput, r::PrecomputedMomentRM)
    return _weight_independent_target(r.mu)
end
_weight_independent_target(::Nothing) = true
_weight_independent_target(::Number) = true
_weight_independent_target(::MedianCenteringFunction) = true
_weight_independent_target(::Any) = false
# Ratio composites evaluate on a series through their own `r(x)` functors (defined above),
# not the kind trait. They are series-eligible exactly when both constituents are.
function supports_precomputed_returns(r::RkRatioRM)
    return supports_precomputed_returns(r.r1) && supports_precomputed_returns(r.r2)
end
function supports_precomputed_returns(r::MeanReturnRiskRatio)
    return supports_precomputed_returns(r.rt) && supports_precomputed_returns(r.rk)
end
"""
    _expected_risk_from_returns(r::AbstractBaseRiskMeasure, x::VecNum)

Contract entry for evaluating a risk measure on an already-reduced net-return series `x`
(ADR 0007). Consults [`supports_precomputed_returns`](@ref): for an eligible measure it
returns `r(x)`; for an ineligible one it throws an explanatory `ArgumentError` instead of
silently consuming `x` as weights (a [`WeightsInput`](@ref) measure) or hitting an opaque
`MethodError` (a moment measure with a per-asset `mu`).

Internal call sites that hold a precomputed series — cross-validation prediction scoring —
route through here rather than calling the functor directly.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`expected_risk`](@ref)
"""
function _expected_risk_from_returns(r::AbstractBaseRiskMeasure, x::VecNum)
    if !supports_precomputed_returns(r)
        throw(ArgumentError("`$(typeof(r))` cannot be evaluated on a precomputed return series: it requires portfolio weights and/or per-asset data (e.g. a weights-only measure such as `TurnoverRiskMeasure`/`EqualRiskMeasure`, a tracking measure, a variance-carrying composite such as `VarianceSkewKurtosis`, or a moment measure with a per-asset `mu`). Evaluate it through `expected_risk(r, w, X, fees)` with explicit weights instead."))
    end
    return r(x)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the expected risk of a risk measure over rolling windows of the returns data.

# Arguments

  - `r::AbstractBaseRiskMeasure`: Risk measure to evaluate.
  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix.
  - `fees::Option{<:Fees}`: Optional fee structure.
  - `window::Integer`: Size of the rolling window (number of periods).

# Returns

  - `risks::VecNum`: Expected risk values for each rolling window.

# Related

  - [`expected_risk`](@ref)
"""
function rolling_window_measure(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                                fees::Option{<:Fees}, window::Integer)
    T = size(X, 1)
    return [expected_risk(r, w, view(X, (t - window + 1):t, :), fees) for t in window:T]
end

export RiskRatioRiskMeasure, number_effective_assets, risk_contribution,
       factor_risk_contribution, rolling_window_measure
