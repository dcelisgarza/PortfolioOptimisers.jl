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
function expected_risk(::NetReturnsInput, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return r(calc_net_returns(w, X, fees))
end
function expected_risk(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure, w::VecNum,
                       X::MatNum, fees::Option{<:Fees} = nothing; kwargs...)
    return r(w, X, fees)
end
function expected_risk(::WeightsInput, r::AbstractBaseRiskMeasure, w::VecNum, args...;
                       kwargs...)
    return r(w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`RkRatioRM`](@ref) `r` supports evaluation on a precomputed return series.

Returns `true` only when both constituent risk measures support precomputed returns.

# Related

  - [`RkRatioRM`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`expected_risk_from_returns`](@ref)
"""
function supports_precomputed_returns(r::RkRatioRM)
    return supports_precomputed_returns(r.r1) && supports_precomputed_returns(r.r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`MeanReturnRiskRatio`](@ref) `r` supports evaluation on a precomputed return series.

Returns `true` only when both the return measure `rt` and the risk measure `rk` support precomputed returns.

# Related

  - [`MeanReturnRiskRatio`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`expected_risk_from_returns`](@ref)
"""
function supports_precomputed_returns(r::MeanReturnRiskRatio)
    return supports_precomputed_returns(r.rt) && supports_precomputed_returns(r.rk)
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
    expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...) -> Number

Contract entry for evaluating a risk measure on an already-reduced net-return series `X`
(ADR 0007). Consults [`supports_precomputed_returns`](@ref): for an eligible measure it
returns `r(X)`; for an ineligible one it throws an explanatory `ArgumentError` instead of
silently consuming `X` as weights (a [`WeightsInput`](@ref) measure) or hitting an opaque
`MethodError` (a moment measure with a per-asset `mu`).

Internal call sites that hold a precomputed series — cross-validation prediction scoring —
route through here rather than calling the functor directly.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`expected_risk`](@ref)
"""
function expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    if !supports_precomputed_returns(r)
        throw(ArgumentError("`$(typeof(r))` cannot be evaluated on a precomputed return series: it requires portfolio weights and/or per-asset data (e.g. a weights-only measure such as `TurnoverRiskMeasure`/`EqualRiskMeasure`, a tracking measure, a variance-carrying composite such as `VarianceSkewKurtosis`, or a moment measure with a per-asset `mu`). Evaluate it through `expected_risk(r, w, X, fees)` with explicit weights instead."))
    end
    return r(X)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluate a risk measure on each element of a vector of precomputed return series.

Maps [`expected_risk_from_returns`](@ref) over each `Xi` in `X`.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecVecNum; kwargs...)
    return [expected_risk_from_returns(r, Xi; kwargs...) for Xi in X]
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
