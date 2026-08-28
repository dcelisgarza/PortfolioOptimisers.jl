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
    const RkRatioRM = Union{<:RiskRatio, <:NonOptimisationRiskRatio}

Union of all risk-ratio risk measures, where the expected risk is defined as the ratio of two component risk values.

# Related

  - [`RiskRatio`](@ref)
  - [`NonOptimisationRiskRatio`](@ref)
  - [`expected_risk`](@ref)
"""
const RkRatioRM = Union{<:RiskRatio, <:NonOptimisationRiskRatio}
"""
    expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, args...; kwargs...)
    expected_risk(rs::VecBaseRM, w::VecNum, args...; sca = SumScalariser(), kwargs...)
    expected_risk(kind::RiskInputKind, r, w::VecNum, args...; kwargs...)
    expected_risk(r::RiskRatio, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::NonOptimisationRiskRatio, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::MeanReturnRiskRatio, w::VecNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, args...; kwargs...)
    expected_risk(rs::VecBaseRM, w::VecNum, pr::Pr_RR, args...; kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, args...; kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, pr::Pr_RR, args...; kwargs...)

Compute the expected value of a risk measure for a portfolio.

For a leaf measure, the generic entry consults [`risk_input_kind`](@ref) and dispatches on the returned [`RiskInputKind`](@ref):

  - [`NetReturnsInput`](@ref): calls `r(calc_net_returns(w, X, fees))`.
  - [`WeightsReturnsFeesInput`](@ref): calls `r(w, X, fees)`.
  - [`WeightsInput`](@ref): calls `r(w)` (ignores `X` and `fees`).

Composite and container forms keep explicit methods:

  - [`RiskRatio`](@ref): computes `expected_risk(r.r1, ...) / expected_risk(r.r2, ...)`.
  - [`NonOptimisationRiskRatio`](@ref): the same ratio, with each axis scalarised by its own `sca1` or `sca2`.
  - [`MeanReturnRiskRatio`](@ref): `(expected_risk(r.rt, ...) - r.rf) / expected_risk(r.rk, ...)`, with `r.sca` on the risk axis.
  - `AbstractBaseRiskMeasure` with [`Pr_RR`](@ref): unwraps the returns matrix and recurses ([`resolve_risk_inputs`](@ref)).
  - [`BaseRM_VecBaseRM`](@ref) with `VecVecNum`: maps over each weight vector in `w`, resolving a [`Pr_RR`](@ref) once for the whole vector.

## One measure or several

Every public entry is bounded [`BaseRM_VecBaseRM`](@ref), so it takes one risk measure or a vector of them. A vector is scalarised into **one** number — see the vector method's own docstring for the three rules it inherits.

## The prior and the matrix are not interchangeable

Given a prior result the measure is put through [`factory`](@ref) first, so the number reported here is the one the optimiser optimises: a **Deferred Quantity** resolves against that prior, and a slot the measure leaves unstated falls back to the prior's own field.

Given a bare returns matrix neither can happen. That call has no `pr.w` to thread and no factor returns to reach, so an unresolved slot is refused by name rather than several frames down ([`assert_resolved_slots`](@ref)), and an unstated slot keeps whatever the measure holds.

A **Calibration Rule** is refused on the same terms, by [`assert_calibrated_slots`](@ref). A rule reads the sample size, the moments and the effective observation weights that a prior result carries, so a bare matrix cannot run one either. The two refusals are separate verbs because the two mechanisms are, and both are taken here.

# Related

  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
  - [`RkRatioRM`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`calc_net_returns`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`assert_calibrated_slots`](@ref)
"""
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, args...; kwargs...)
    assert_resolved_slots(r)
    assert_calibrated_slots(r)
    return expected_risk(risk_input_kind(r), r, w, args...; kwargs...)
end
"""
    expected_risk(rs::VecBaseRM, w::VecNum, args...; sca::Scalariser = SumScalariser(),
                  kwargs...)

Scalarise several risk measures into **one** number.

Each element is evaluated through the singular public entry above and weighted by its own `settings.scale`, then the vector is combined by `sca`. So the vector crosses **no** seam as a unit: [`risk_input_kind`](@ref), [`assert_resolved_slots`](@ref) and [`assert_calibrated_slots`](@ref) are taken by each element itself, and an unresolved slot in element three is refused by name rather than several frames down.

## Three rules the vector inherits

  - **`scale` is a combination weight.** It is inert on a single measure and applied on a vector, under **every** scalariser. This mirrors the model, where `set_risk_expression!` pushes `scale * r_expr` before any scalariser runs. So `expected_risk([r], w, pr)` is `r.settings.scale * expected_risk(r, w, pr)`.
  - **`sca` on a singular call is silently inert**, swallowed by `kwargs...` as any undeclared keyword is. A scalariser over one element returns that element, so the answer is right rather than merely tolerated.
  - **`rke` is ignored.** The model skips an element whose `settings.rke` is `false`; the value level does not. The caller named the vector, so every element is reported.

!!! warning

    The last rule means a figure taken over an optimiser's own `opt.r` can include a measure the objective never saw. A caller who wants the objective's aggregate must filter the vector first.

    The figure matches the optimisation only when the caller names the same measures **and** the same scalariser. `sca` defaults to `SumScalariser()`, which agrees with [`JuMPOptimiser`](@ref)'s own default, so a caller who names no scalariser anywhere gets a matched figure. `fees` and `rf` carry the same responsibility.

# Related

  - [`VecBaseRM`](@ref)
  - [`BaseRM_VecBaseRM`](@ref)
  - [`Scalariser`](@ref)
  - [`scalarise`](@ref)
  - [`expected_risk`](@ref)
"""
function expected_risk(rs::VecBaseRM, w::VecNum, args...; sca::Scalariser = SumScalariser(),
                       kwargs...)
    return scalarise(sca, rs) do r
        return expected_risk(r, w, args...; kwargs...) * r.settings.scale
    end
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
# The two ratio composites split by type rather than sharing a body on `RkRatioRM`.
# `NonOptimisationRiskRatio` names `sca1` and `sca2`; `RiskRatio` carries neither, so a shared
# body naming them would not resolve on it. Each composite pins its own scalariser **last**, so
# a `sca` supplied at the call site cannot beat the field.
function expected_risk(r::RiskRatio, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing;
                       kwargs...)
    return expected_risk(r.r1, w, X, fees; kwargs...) /
           expected_risk(r.r2, w, X, fees; kwargs...)
end
function expected_risk(r::NonOptimisationRiskRatio, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return expected_risk(r.r1, w, X, fees; kwargs..., sca = r.sca1) /
           expected_risk(r.r2, w, X, fees; kwargs..., sca = r.sca2)
end
function expected_risk(r::MeanReturnRiskRatio, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    # `rt` is a `MeanReturn` and stays singular, so only the risk axis takes a scalariser.
    return (expected_risk(r.rt, w, X, fees; kwargs...) - r.rf) /
           expected_risk(r.rk, w, X, fees; kwargs..., sca = r.sca)
end
# Precomputed-returns contract for the ratio composites: decompose onto the series, mirroring
# the `(w, X, fees)` decomposition above. Each component routes through
# `expected_risk_from_returns` rather than being called as a function, because a widened field
# may hold a **vector** — a vector is not callable, and a call method on `AbstractVector` would
# be piracy on `Base`. The detour also upgrades a raw `MethodError` to the guard's named
# refusal. On a single supported measure the guard returns `r(x)`, so the number is unchanged.
#
# `RkRatioRM` splits for the same reason its `expected_risk` methods do: `RiskRatio` has no
# `sca1`/`sca2` to name.
function (r::RiskRatio)(x::VecNum)
    return expected_risk_from_returns(r.r1, x) / expected_risk_from_returns(r.r2, x)
end
function (r::NonOptimisationRiskRatio)(x::VecNum)
    return expected_risk_from_returns(r.r1, x; sca = r.sca1) /
           expected_risk_from_returns(r.r2, x; sca = r.sca2)
end
function (r::MeanReturnRiskRatio)(x::VecNum)
    return (expected_risk_from_returns(r.rt, x) - r.rf) /
           expected_risk_from_returns(r.rk, x; sca = r.sca)
end
"""
    resolve_risk_inputs(r::BaseRM_VecBaseRM, X::MatNum_Pr)

Turn a value-level data argument into the pair a kernel takes: the measure to evaluate, and the returns matrix to evaluate it on.

A prior result resolves the measure through [`factory`](@ref) — a **Deferred Quantity** becomes a value, an unstated slot takes the prior's field — and hands back `pr.X`. A [`ReturnsResult`](@ref) carries no moments, so it only unwraps its `X`. A matrix is already the pair.

Resolution happens **once** per entry point rather than once per evaluation, which is what keeps [`risk_contribution`](@ref) from refitting a deferred covariance `2N` times.

# Related

  - [`expected_risk`](@ref)
  - [`risk_contribution`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`factory`](@ref)
"""
function resolve_risk_inputs(r::BaseRM_VecBaseRM, X::MatNum)
    return r, X
end
function resolve_risk_inputs(r::BaseRM_VecBaseRM, pr::AbstractPriorResult)
    return factory(r, pr), pr.X
end
function resolve_risk_inputs(r::BaseRM_VecBaseRM, rd::ReturnsResult)
    return r, rd.X
end
"""
    original_returns(X::MatNum_Pr)

Take the returns matrix the **caller** supplied out of whichever carrier holds it.

A prior result answers `pr.original_X`, a [`ReturnsResult`](@ref) answers its `X`, and a matrix answers itself. The three arms agree off a factor route, where `pr.original_X === pr.X`, and differ on one, where `pr.X` is the reconstruction `F * transpose(M) .+ transpose(b)`.

This is the read [`resolve_factor_risk_inputs`](@ref) takes, and it is deliberately **not** the read [`resolve_risk_inputs`](@ref) takes. `expected_risk` evaluates the return distribution the prior asserts, which is `pr.X`. A factor attribution partitions risk into a factor part and a residual part, and the reconstruction has no residual, so it can only attribute noise to the second.

# Arguments

  - `X::MatNum_Pr`: Returns matrix, prior result, or returns result.

# Returns

  - `X::MatNum`: The returns matrix the caller supplied.

# Related

  - [`resolve_factor_risk_inputs`](@ref)
  - [`factor_risk_contribution`](@ref)
"""
function original_returns(X::MatNum)
    return X
end
function original_returns(pr::AbstractPriorResult)
    return pr.original_X
end
function original_returns(rd::ReturnsResult)
    return rd.X
end
"""
    resolve_factor_risk_inputs(r::BaseRM_VecBaseRM, X::MatNum_Pr)

Turn a value-level data argument into the pair a **factor attribution** takes: the measure to evaluate, and the returns matrix to evaluate it on.

The sibling of [`resolve_risk_inputs`](@ref), and it differs in the second half only. The measure resolves the same way, so a **Deferred Quantity** is still fitted once rather than once per finite difference. The matrix is [`original_returns`](@ref) rather than `pr.X`.

Two seams and not one argument, because the two answers are both correct and neither is a default of the other. Every other caller of [`resolve_risk_inputs`](@ref) wants the distribution the prior asserts.

# Related

  - [`resolve_risk_inputs`](@ref)
  - [`original_returns`](@ref)
  - [`factor_risk_contribution`](@ref)
"""
function resolve_factor_risk_inputs(r::BaseRM_VecBaseRM, X::MatNum_Pr)
    return first(resolve_risk_inputs(r, X)), original_returns(X)
end
"""
    resolve_factor_regression(re::RegE_Reg, rd::ReturnsResult,
                              pr::Option{<:AbstractPriorResult} = nothing)

Pick the factor loadings a factor attribution decomposes against, from the three carriers that can supply them.

The precedence is fixed, and it is not a source selector:

 1. `re` when it is already a [`Regression`](@ref) result. A precomputed result is the caller stating the answer, and it needs no data.
 2. `pr.rr` when the prior carries a factor block. The loadings are then the ones fitted on `pr.original_X`, which is the matrix the risk is measured on, so the pair is matched by construction.
 3. `regression(re, rd)` otherwise, which needs `rd.X` and `rd.F`.

!!! warning

    A stated regression **estimator** loses to a prior that carries loadings. `re` is honoured only when the prior has none. Pass the loadings as a precomputed [`Regression`](@ref) to override a factor prior, or pass the returns matrix rather than the prior to keep the refit.

# Arguments

  - `re::RegE_Reg`: Regression result or estimator.
  - `rd::ReturnsResult`: Returns result carrying `X` and `F`.
  - `pr::Option{<:AbstractPriorResult}`: Prior result, or `nothing` when the caller passed a bare matrix.

# Validation

  - When none of the three arms applies, throws an [`IsNothingError`](@ref) naming all three.

# Returns

  - `rr::AbstractRegressionResult`: The factor loadings.

# Related

  - [`factor_risk_contribution`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`regression`](@ref)
"""
function resolve_factor_regression(re::RegE_Reg, rd::ReturnsResult,
                                   pr::Option{<:AbstractPriorResult} = nothing)
    if isa(re, AbstractRegressionResult)
        return re
    end
    if !isnothing(pr) && !isnothing(pr.rr)
        return pr.rr
    end
    @argcheck(!isnothing(rd.X) && !isnothing(rd.F),
              IsNothingError("a factor decomposition needs loadings, and none of the three carriers holds any. `re` is an estimator (`$(nameof(typeof(re)))`), so it must fit them from `rd.X` and `rd.F`; the prior carries no factor block to read them from instead.\nSupply the data as `rd`, or pass a precomputed `Regression` as `re`, or pass a prior fitted through a factor model (e.g. `FactorPrior`), which carries its own loadings in `rr`.\nGot\nisnothing(rd.X) => $(isnothing(rd.X))\nisnothing(rd.F) => $(isnothing(rd.F))\nisnothing(pr) => $(isnothing(pr))"))
    return regression(re, rd)
end
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, args...; kwargs...)
    r, X = resolve_risk_inputs(r, pr)
    return expected_risk(r, w, X, args...; kwargs...)
end
# The vector's own prior route. It resolves the whole vector **once** through
# `resolve_risk_inputs`, so a Deferred Quantity is fitted once per measure rather than once per
# element evaluation. Without it the generic vector twin above would resolve the prior inside the
# scalarise loop.
function expected_risk(rs::VecBaseRM, w::VecNum, pr::Pr_RR, args...; kwargs...)
    rs, X = resolve_risk_inputs(rs, pr)
    return expected_risk(rs, w, X, args...; kwargs...)
end
function expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, args...; kwargs...)
    return [expected_risk(r, wi, args...; kwargs...) for wi in w]
end
function expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, pr::Pr_RR, args...; kwargs...)
    r, X = resolve_risk_inputs(r, pr)
    return [expected_risk(r, wi, X, args...; kwargs...) for wi in w]
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
        throw(ArgumentError("`$(typeof(r))` cannot be evaluated on a precomputed return series: it requires portfolio weights and/or per-asset data (e.g. a weights-only measure such as `TurnoverRiskMeasure`/`EqualRisk`, a tracking measure, a variance-carrying composite such as `VarianceSkewKurtosis`, or a moment measure with a per-asset `mu`). Evaluate it through `expected_risk(r, w, X, fees)` with explicit weights instead."))
    end
    return r(X)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Scalarise several risk measures on one precomputed net-return series `X`.

The twin of the singular contract entry above, and it inherits the same three rules as [`expected_risk`](@ref) on a vector: `scale` weights each element, `sca` combines them, and `rke` is ignored. An ineligible element raises that element's own named refusal rather than an opaque `MethodError`.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`VecBaseRM`](@ref)
"""
function expected_risk_from_returns(rs::VecBaseRM, X::VecNum;
                                    sca::Scalariser = SumScalariser(), kwargs...)
    return scalarise(sca, rs) do r
        return expected_risk_from_returns(r, X; kwargs...) * r.settings.scale
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Evaluate a risk measure on each element of a vector of precomputed return series.

Maps [`expected_risk_from_returns`](@ref) over each `Xi` in `X`.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function expected_risk_from_returns(r::BaseRM_VecBaseRM, X::VecVecNum; kwargs...)
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

  - [`set_weight_norm_2_constraints!`](@ref)
  - [`EqualRisk`](@ref)
  - [`risk_contribution`](@ref)
  - [`EqualRisk`](@ref)
"""
function number_effective_assets(w::VecNum)
    return inv(LinearAlgebra.dot(w, w))
end
"""
    adjusted_risk(sca::Scalariser, r::BaseRM_VecBaseRM, w::VecNum, X::MatNum,
                  fees::Option{<:Fees}, delta::Number; kwargs...)

Evaluate the risk at `w` with the homogeneity correction already applied, for one measure or several.

The internal seam that lets [`risk_contribution`](@ref) keep **one** body across the multiplicity. It exists because the correction cannot be applied to the aggregate.

[`adjust_risk_contribution`](@ref) is a homogeneity correction: it divides by the measure's own degree, so that `Σᵢ wᵢ·rcᵢ` recovers the measure's value by Euler's identity. A mixed vector such as `[Variance(), ConditionalValueatRisk()]` is a sum of a degree-2 and a degree-1 function, so it has **no single degree** and adjusting the aggregate is not merely awkward but impossible.

Adjusting each element **before** the scalariser restores the identity exactly:

```text
Σᵢ wᵢ·rcᵢ  =  Σₖ sₖ·aₖ·Σᵢ wᵢ·∂ρₖ/∂wᵢ  =  Σₖ sₖ·aₖ·degₖ·ρₖ  =  Σₖ sₖ·ρₖ
```

which is `expected_risk(rs, w, X, fees)` under [`SumScalariser`](@ref). The invariant survives an arbitrary mixture of homogeneity degrees, and it survives **only** because the correction sits inside the loop.

`sca` is inert on a single measure and `scale` is inert on a single measure, exactly as they are in [`expected_risk`](@ref).

# Related

  - [`risk_contribution`](@ref)
  - [`adjust_risk_contribution`](@ref)
  - [`expected_risk`](@ref)
  - [`scalarise`](@ref)
"""
function adjusted_risk(::Scalariser, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                       fees::Option{<:Fees}, delta::Number; kwargs...)
    return adjust_risk_contribution(r, expected_risk(r, w, X, fees; kwargs...), delta)
end
function adjusted_risk(sca::Scalariser, rs::VecBaseRM, w::VecNum, X::MatNum,
                       fees::Option{<:Fees}, delta::Number; kwargs...)
    return scalarise(sca, rs) do r
        return adjust_risk_contribution(r, expected_risk(r, w, X, fees; kwargs...), delta) *
               r.settings.scale
    end
end
"""
    risk_contribution(
        r::BaseRM_VecBaseRM,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        delta::Number = 1e-6,
        marginal::Bool = false,
        sca::Scalariser = SumScalariser(),
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

  - `r::BaseRM_VecBaseRM`: Risk measure to differentiate, or a vector of them.
  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum_Pr`: Asset returns matrix or prior result.
  - `fees::Option{<:Fees}`: Optional fee structure.

# Keyword Arguments

  - `delta::Number = 1e-6`: Finite difference step size.
  - `marginal::Bool = false`: If `true`, returns marginal risk contributions (without ``w_i`` weighting).
  - `sca::Scalariser = SumScalariser()`: Scalariser combining a vector `r`. Inert on a single measure.

# Returns

  - `Vector`: Risk contributions (or marginal risks) for each asset.

# Details

  - A prior result resolves the measure **once**, before the loop ([`resolve_risk_inputs`](@ref)), so a **Deferred Quantity** is fitted once rather than once per finite difference.
  - A vector of measures differentiates the **aggregate**, which is the figure [`expected_risk`](@ref) reports. The homogeneity correction is applied per element inside the loop ([`adjusted_risk`](@ref)), so `Σᵢ wᵢ·rcᵢ` recovers the aggregate exactly even when the elements have different homogeneity degrees.

!!! warning

    All four scalarisers are admitted, and [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) are exact **almost everywhere**. Wherever the argmax is unique the aggregate's decomposition is the winning element's own, scaled. The figure degrades only at a near-exact tie between two scaled measures, where the argmax can flip between the `w+δ` and `w−δ` points and an asset's figure becomes a chord across the kink. At such a point the subgradient is genuinely a **set**, so no answer is uniquely correct.

# Related

  - [`expected_risk`](@ref)
  - [`adjusted_risk`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`resolve_risk_inputs`](@ref)
"""
function risk_contribution(r::BaseRM_VecBaseRM, w::VecNum, X::MatNum_Pr,
                           fees::Option{<:Fees} = nothing; delta::Number = 1e-6,
                           marginal::Bool = false, sca::Scalariser = SumScalariser(),
                           kwargs...)
    r, X = resolve_risk_inputs(r, X)
    N = length(w)
    rc = Vector{eltype(w)}(undef, N)
    ws = Matrix{eltype(w)}(undef, N, 2)
    ws .= w
    id2 = inv(2 * delta)
    for i in eachindex(w)
        ws[i, 1] += delta
        ws[i, 2] -= delta
        r1 = adjusted_risk(sca, r, view(ws, :, 1), X, fees, delta; kwargs...)
        r2 = adjusted_risk(sca, r, view(ws, :, 2), X, fees, delta; kwargs...)
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
        r::BaseRM_VecBaseRM,
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

  - `r::BaseRM_VecBaseRM`: Risk measure to decompose, or a vector of them.
  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum_Pr`: Asset returns matrix or prior result.
  - `fees::Option{<:Fees}`: Optional fee structure.

# Keyword Arguments

  - `re::RegE_Reg = StepwiseRegression()`: Regression estimator for factor loadings.
  - `rd::ReturnsResult = ReturnsResult()`: Returns result providing factor data.
  - `delta::Number = 1e-6`: Finite difference step size.

# Returns

  - `Vector`: Risk contributions for each factor, with the last element being the idiosyncratic (off-factor) contribution.

# Details

  - The gradient is taken on [`original_returns`](@ref) — the returns the **caller** supplied — and not on `pr.X` ([`resolve_factor_risk_inputs`](@ref)). The two differ under a factor prior, where `pr.X` is the reconstruction `F * transpose(M) .+ transpose(b)`. That matrix has rank `size(F, 2)` and carries **no residual**, so the off-factor term of a series-reducing measure reports the intercept's share rather than idiosyncratic risk, and can come out negative.
  - A consequence: under a factor prior the parts sum to the risk on the caller's returns, and **not** to `expected_risk(r, w, pr)`. A measure whose kernel reads a moment rather than the series — [`Variance`](@ref), [`StandardDeviation`](@ref), [`DistributionValueatRisk`](@ref) — is unaffected either way, because it never reduces the returns matrix.
  - The loadings come from [`resolve_factor_regression`](@ref), which prefers the prior's own `rr` over a refit, so the loadings and the returns are the pair the prior was fitted on. A stated regression **estimator** therefore loses to a prior that carries a factor block.
  - A prior result resolves the measure **once**, before the loop ([`resolve_factor_risk_inputs`](@ref)), so a **Deferred Quantity** is fitted once rather than once per finite difference.

# Related

  - [`risk_contribution`](@ref)
  - [`expected_risk`](@ref)
  - [`resolve_factor_risk_inputs`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`original_returns`](@ref)
"""
function factor_risk_contribution(r::BaseRM_VecBaseRM, w::VecNum, X::MatNum_Pr,
                                  fees::Option{<:Fees} = nothing;
                                  re::RegE_Reg = StepwiseRegression(),
                                  rd::ReturnsResult = ReturnsResult(), delta::Number = 1e-6,
                                  sca::Scalariser = SumScalariser(), kwargs...)
    rr = resolve_factor_regression(re, rd, isa(X, AbstractPriorResult) ? X : nothing)
    r, X = resolve_factor_risk_inputs(r, X)
    Bt = transpose(rr.L)
    b2t = transpose(LinearAlgebra.pinv(transpose(LinearAlgebra.nullspace(Bt))))
    b3t = transpose(LinearAlgebra.pinv(b2t))
    mr = risk_contribution(r, w, X, fees; delta = delta, marginal = true, sca = sca,
                           kwargs...)
    rc_f = (Bt * w) .* (transpose(LinearAlgebra.pinv(Bt)) * mr)
    rc_of = sum((b2t * w) .* (b3t * mr))
    rc_f = [rc_f; rc_of]
    return rc_f
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the expected risk of a risk measure over rolling windows of the returns data.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix.
  - `fees::Option{<:Fees}`: Optional fee structure.
  - `window::Integer`: Size of the rolling window (number of periods).

# Keyword Arguments

  - `sca::Scalariser = SumScalariser()`: Scalariser combining a vector `r`. Inert on a single measure.

# Validation

  - `1 <= window <= size(X, 1)`, else a `DomainError` naming `window`.

The window is checked here rather than left to the risk kernel. A non-positive window indexes `X` out of bounds and surfaces as a bare `BoundsError` from inside whichever measure `r` names, and a window longer than the sample produces an empty vector of risks that reads as a legitimate result. Both are caller errors, so both are refused at the boundary — the same discipline [`plot_rolling_measure`](@ref) applies to its own `rolling` keyword.

# Returns

  - `risks::VecNum`: Expected risk values for each rolling window.

# Related

  - [`expected_risk`](@ref)
  - [`plot_rolling_measure`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, w::VecNum, X::MatNum,
                                fees::Option{<:Fees}, window::Integer;
                                sca::Scalariser = SumScalariser(), kwargs...)
    T = size(X, 1)
    @argcheck(1 <= window <= T,
              DomainError(window,
                          "window must be in 1:$(T), the number of observations in X; got window => $window"))
    return [expected_risk(r, w, view(X, (t - window + 1):t, :), fees; sca = sca, kwargs...)
            for t in window:T]
end

export RiskRatio, number_effective_assets, risk_contribution, factor_risk_contribution,
       rolling_window_measure
