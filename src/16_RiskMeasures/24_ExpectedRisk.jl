"""
    const MatNum_Pr = Union{<:MatNum, <:AbstractPriorResult, <:ReturnsResult}

Groups the three carriers of a returns matrix that the value-level risk functions accept as their data argument.

A caller can hold a bare matrix, a prior result or a returns result, and each one carries the matrix that a risk measure is evaluated on. [`resolve_risk_inputs`](@ref) turns any of the three into the pair of a measure and a matrix, so one method of a function such as [`risk_contribution`](@ref) serves all three.

# Related

  - [`MatNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`resolve_risk_inputs`](@ref): Resolves each member to a measure and a matrix.
  - [`risk_contribution`](@ref)
"""
const MatNum_Pr = Union{<:MatNum, <:AbstractPriorResult, <:ReturnsResult}
"""
    const RkRatioRM = Union{<:RiskRatio, <:NonOptimisationRiskRatio}

Groups the two ratio measures whose value is one risk divided by another risk.

Both members hold their two risks in the fields `r1` and `r2`, so one method of [`supports_precomputed_returns`](@ref) answers for both. Their [`expected_risk`](@ref) methods stay separate, because only [`NonOptimisationRiskRatio`](@ref) carries a scalariser for each field.

# Related

  - [`RiskRatio`](@ref)
  - [`NonOptimisationRiskRatio`](@ref)
  - [`supports_precomputed_returns`](@ref)
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
    expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR, fees = nothing; strict = false, kwargs...)
    expected_risk(rs::VecBaseRM, w::VecNum, pr::Pr_RR, fees = nothing; strict = false, kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, args...; kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, pr::Pr_RR, fees = nothing; strict = false, kwargs...)

Compute the risk of a portfolio under one risk measure, or under a vector of them.

A risk measure declares the input that it reads through [`risk_input_kind`](@ref), and the generic entry dispatches on the returned [`RiskInputKind`](@ref):

  - [`NetReturnsInput`](@ref) calls `r(calc_net_returns(w, X, fees))`.
  - [`WeightsReturnsFeesInput`](@ref) calls `r(w, X, fees)`.
  - [`WeightsInput`](@ref) calls `r(w)` and ignores `X` and `fees`.

These are the constant-weight methods, in which the one vector `w` weighs every observation. The `w::MatNum` methods read a weight path instead, with one row of weights for each observation, which is what a fold held under a Weight Drift. The type of the weight argument picks the method.

The composite measures keep methods of their own. [`RiskRatio`](@ref) returns `expected_risk(r.r1, ...) / expected_risk(r.r2, ...)`. [`NonOptimisationRiskRatio`](@ref) returns the same ratio, and scalarises each field with its own `sca1` or `sca2`. [`MeanReturnRiskRatio`](@ref) returns `(expected_risk(r.rt, ...) - r.rf) / expected_risk(r.rk, ...)`, and scalarises the risk with `r.sca`. A [`VecVecNum`](@ref) of weights returns one risk for each weight vector, and the methods resolve a prior once for all of them.

Every public method takes one risk measure or a vector of them, which is the bound [`BaseRM_VecBaseRM`](@ref). A vector gives one number, by the three rules that the docstring of the vector method states.

A prior result and a bare returns matrix give different answers for one measure. The prior route puts the measure through [`factory`](@ref), so the figure is the one that the optimiser optimises. A Deferred Quantity resolves against the prior, and a slot that the measure leaves unstated takes the field of the prior. The matrix route has no prior to read, so it leaves every slot as the measure holds it. To fill one, it would have to pick an estimator that the caller never named. It refuses a Deferred Quantity by name through [`assert_resolved_slots`](@ref), and an empty slot that the functor reads through [`functor_slots`](@ref). `Variance()` holds `sigma = nothing` and its functor is `dot(w, r.sigma, w)`, so the matrix route refuses it by name and `LinearAlgebra` never raises a `MethodError`. [`assert_calibrated_slots`](@ref) refuses a Calibration Rule on the same terms, because a rule reads the sample size, the moments and the observation weights of a prior result.

The prior route reduces to the Investable Mask. A prior result holds every asset of the universe, and an asset that it could not estimate carries `NaN` in `mu`, on the diagonal of `sigma` and down its column of `pr.X`. On the whole universe, `dot(w, pr.sigma, w)` and `pr.X * w` are `NaN` at every weight, a zero weight included. The prior route therefore reduces the prior, the weights and the fees before it evaluates the measure. A bare matrix and a [`ReturnsResult`](@ref) carry no moments, so no mask exists and they pass through unchanged.

# Algorithm

The generic entry, `expected_risk(r, w, args...)`:

 1. Refuse a Deferred Quantity that is not resolved, with [`assert_resolved_slots`](@ref).
 2. Refuse a Calibration Rule that is not calibrated, with [`assert_calibrated_slots`](@ref).
 3. Read the input kind of `r` with [`risk_input_kind`](@ref), and call the method of that kind.

The prior route, `expected_risk(r, w, pr, fees)`:

 1. Reduce `pr`, `w` and `fees` to the Investable Mask with [`investable_reduction`](@ref).
 2. Resolve `r` against the reduced prior with [`resolve_risk_inputs`](@ref), which gives the resolved measure `r` and the returns matrix `X`.
 3. Evaluate `expected_risk(r, w, X, fees)` through the generic entry.

# Arguments

  - $(arg_dict[:r])
  - `w`: Portfolio weights vector `assets × 1`, or a vector of them.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `pr::Pr_RR`: Prior result, or [`ReturnsResult`](@ref) that carries the returns matrix.
  - `fees`: Optional [`Fees`](@ref) structure.

# Keyword Arguments

  - `strict::Bool = false`: Whether a held non-investable asset raises an error. When it is `false`, the asset warns and its weight is dropped. Only the methods that take a prior read it.

# Validation

  - A measure that reads a return series and gets no returns carrier, `X = nothing`, raises an [`IsNothingError`](@ref). The message names the measure and the four calls that carry one. A caller meets it when it scores a result that carries no carrier of its own, which is every optimiser whose fit keeps no returns.
  - A Deferred Quantity that is not resolved raises through [`assert_resolved_slots`](@ref), and an empty slot that the functor reads raises through [`functor_slots`](@ref).
  - A Calibration Rule that is not calibrated raises through [`assert_calibrated_slots`](@ref).
  - Under `strict = true`, a held non-investable asset raises through [`investable_reduction`](@ref).

# Returns

  - `rk::Number`: The risk of the portfolio. A [`VecVecNum`](@ref) of weights returns a vector with one risk for each weight vector.

# Related

  - [`investable_reduction`](@ref)
  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
  - [`RkRatioRM`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`calc_net_returns`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`assert_calibrated_slots`](@ref)
  - [`functor_slots`](@ref)
  - [`IsNothingError`](@ref)
"""
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, args...; kwargs...)
    assert_resolved_slots(r)
    assert_calibrated_slots(r)
    return expected_risk(risk_input_kind(r), r, w, args...; kwargs...)
end
"""
    missing_returns_carrier_message(r::AbstractBaseRiskMeasure)

Build the error message that refuses a `nothing` returns carrier to a measure that reads a return series.

Two methods raise this refusal, one for a weight vector and one for a weight path, and both take the text from here. The message names the measure and the four calls that carry a returns carrier, two on a weight vector and two on an [`OptimisationResult`](@ref).

# Arguments

  - `r`: The measure that was given no carrier.

# Returns

  - `msg::String`: The refusal message.

# Related

  - [`expected_risk`](@ref)
  - [`IsNothingError`](@ref)
  - [`risk_input_kind`](@ref)
"""
function missing_returns_carrier_message(r::AbstractBaseRiskMeasure)
    return "`$(nameof(typeof(r)))` is evaluated on a return series, and no returns carrier was given. Either the call named none, or the result it was taken from carries none of its own. Pass one: `expected_risk(r, w, X)` or `expected_risk(r, res, X)` for a returns matrix, `expected_risk(r, w, pr)` or `expected_risk(r, res, pr)` for a prior result."
end
"""
    expected_risk(rs::VecBaseRM, w::VecNum, args...; sca::Scalariser = SumScalariser(),
                  kwargs...)

Combine the risks of several risk measures into one number.

The method evaluates each element through the single-measure entry, multiplies it by the `settings.scale` of that element, and combines the products with `sca`. Each element passes through [`risk_input_kind`](@ref), [`assert_resolved_slots`](@ref) and [`assert_calibrated_slots`](@ref) by itself, so an unresolved slot in the third element raises an error that names that element.

The vector follows three rules.

  - `scale` is a combination weight. A single measure ignores it, and a vector applies it under every scalariser. The model does the same, because `set_risk_expression!` adds `scale * r_expr` before a scalariser runs. So `expected_risk([r], w, pr)` equals `r.settings.scale * expected_risk(r, w, pr)`.
  - A single-measure call ignores `sca`, because `kwargs...` takes it as it takes any keyword that the method does not declare. A scalariser over one element returns that element, so the answer is correct.
  - The method ignores `rke`. The model skips an element whose `settings.rke` is `false`, but the value level reports every element that the caller names.

!!! warning

    Because of the last rule, a figure over the `opt.r` of an optimiser can include a measure that the objective did not read. Filter the vector first to get the aggregate of the objective.

    The figure matches the optimisation only when the caller names the same measures and the same scalariser. `sca` defaults to `SumScalariser()`, which is also the default of [`JuMPOptimiser`](@ref), so a caller who names no scalariser gets a matched figure. The same holds for `fees` and `rf`.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= S\\left( s_1 R_1(\\boldsymbol{w}),\\, \\ldots,\\, s_n R_n(\\boldsymbol{w}) \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:S_sca_vec])
  - $(math_dict[:s_k_scale])
  - $(math_dict[:R_k_vec])
  - $(math_dict[:n_rm_vec])

# Arguments

  - `rs::VecBaseRM`: Vector of risk measures.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `args...`: The returns carrier and the fees, forwarded to each element.

# Keyword Arguments

  - `sca::Scalariser = SumScalariser()`: Scalariser that combines the scaled risks.

# Returns

  - `rk::Number`: The combined risk.

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
# The two kinds above read a return series, so a `nothing` carrier reaches no method of
# theirs. The call that lands here is the documented `expected_risk(r, res)` on a result
# that carries no carrier of its own: `result_investable_view` falls back to `res.pr`, and
# `nothing` arrives as `X`. Without this method that call raised a `MethodError` naming the
# kind singleton, which names neither the measure nor the missing argument. It is refused by
# name on the same terms as an unstated slot, and it states the ways out. `WeightsInput` is
# absent: it reads no series, so its `args...` method already answers a carrier-free call.
function expected_risk(::Union{<:NetReturnsInput, <:WeightsReturnsFeesInput},
                       r::AbstractBaseRiskMeasure, w::VecNum, X::Nothing = nothing,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return throw(IsNothingError(missing_returns_carrier_message(r)))
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

Turn the data argument of a value-level function into the measure to evaluate and the returns matrix to evaluate it on.

The method depends on the carrier.

  - A prior result resolves the measure through [`factory`](@ref) and returns `pr.X`. A Deferred Quantity becomes a value, and a slot that the measure leaves unstated takes the field of the prior.
  - A [`ReturnsResult`](@ref) carries no moments, so the measure stays as it is and the method returns `rd.X`.
  - A matrix returns the measure and the matrix unchanged.

Each entry point resolves once, not once for each evaluation. So [`risk_contribution`](@ref) fits a deferred covariance once, and not once for each of its `2N` evaluations.

# Arguments

  - $(arg_dict[:r])
  - `X::MatNum_Pr`: Returns matrix, prior result or returns result.

# Returns

  - `(r, X)::Tuple`: The resolved measure and the returns matrix.

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

Return the returns matrix that the caller supplied, from the carrier that holds it.

A prior result returns `pr.original_X`, a [`ReturnsResult`](@ref) returns its `X`, and a matrix returns itself. The three agree when the prior did not come from a factor model, because then `pr.original_X === pr.X`. A factor prior sets `pr.X` to the reconstruction `F * transpose(M) .+ transpose(b)`, and the two differ.

[`resolve_factor_risk_inputs`](@ref) reads this matrix, and [`resolve_risk_inputs`](@ref) does not. `expected_risk` evaluates the return distribution that the prior states, which is `pr.X`. A factor attribution splits the risk into a factor part and a residual part. The reconstruction has no residual, so on it the residual part reports the share of the intercept and not the idiosyncratic risk.

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

Turn the data argument of a factor attribution into the measure to evaluate and the returns matrix to evaluate it on.

The measure resolves as in [`resolve_risk_inputs`](@ref), so a Deferred Quantity is fitted once and not once for each finite difference. The matrix is [`original_returns`](@ref), not `pr.X`.

The two functions are separate because each answer is correct for its caller. A factor attribution needs the residual that the caller's returns carry, and every other caller of [`resolve_risk_inputs`](@ref) needs the distribution that the prior states.

# Arguments

  - $(arg_dict[:r])
  - `X::MatNum_Pr`: Returns matrix, prior result or returns result.

# Returns

  - `(r, X)::Tuple`: The resolved measure and the returns matrix that the caller supplied.

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

Pick the factor loadings that a factor attribution decomposes against, from the three carriers that can supply them.

The order of precedence is fixed. A precomputed result is an answer that the caller states, so it needs no data. The loadings of a prior were fitted on `pr.original_X`, which is the matrix that the attribution measures the risk on, so the loadings and the returns agree.

!!! warning

    A prior that carries loadings takes precedence over a regression estimator in `re`. The function fits `re` only when the prior carries no loadings. To replace the loadings of a factor prior, pass them as a precomputed [`Regression`](@ref). To fit `re`, pass the returns matrix in place of the prior.

# Algorithm

 1. When `re` is an `AbstractLoadingsRegressionResult`, return `re`.
 2. When `pr` is a prior result and `pr.rr` is not `nothing`, return `pr.rr`.
 3. Check that `rd.X` and `rd.F` are present.
 4. Fit and return `regression(re, rd)`.

# Arguments

  - `re::RegE_Reg`: Regression result or estimator.
  - `rd::ReturnsResult`: Returns result carrying `X` and `F`.
  - `pr::Option{<:AbstractPriorResult}`: Prior result, or `nothing` when the caller passed a bare matrix.

# Validation

  - When none of the three arms applies, throws an [`IsNothingError`](@ref) naming all three.

# Returns

  - `rr::AbstractLoadingsRegressionResult`: The factor loadings.

# Related

  - [`factor_risk_contribution`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`regression`](@ref)
"""
function resolve_factor_regression(re::RegE_Reg, rd::ReturnsResult,
                                   pr::Option{<:AbstractPriorResult} = nothing)
    if isa(re, AbstractLoadingsRegressionResult)
        return re
    end
    if !isnothing(pr) && !isnothing(pr.rr)
        return pr.rr
    end
    @argcheck(!isnothing(rd.X) && !isnothing(rd.F),
              IsNothingError("a factor decomposition needs loadings, and none of the three carriers holds any. `re` is an estimator (`$(nameof(typeof(re)))`), so it must fit them from `rd.X` and `rd.F`; the prior carries no factor block to read them from instead.\nSupply the data as `rd`, or pass a precomputed `Regression` as `re`, or pass a prior fitted through a factor model (e.g. `FactorPrior`), which carries its own loadings in `rr`.\nGot\nisnothing(rd.X) => $(isnothing(rd.X))\nisnothing(rd.F) => $(isnothing(rd.F))\nisnothing(pr) => $(isnothing(pr))"))
    return regression(re, rd)
end
# The value-level door. A non-investable asset carries `NaN` in `mu`, on the diagonal of
# `sigma` and down its column of `pr.X`, so the figure is `NaN` at any weight, the
# optimiser's own zero included. `investable_reduction` reduces the prior, the weights and
# the fees once here, which is ADR 0115's rule at the door a caller reaches by hand. A bare
# matrix and a `ReturnsResult` carry no moments, so the same call passes them through.
function expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, pr::Pr_RR,
                       fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    r, X = resolve_risk_inputs(r, pr)
    return expected_risk(r, w, X, fees; kwargs...)
end
# The vector's own prior route. It resolves the whole vector **once** through
# `resolve_risk_inputs`, so a Deferred Quantity is fitted once per measure rather than once per
# element evaluation. Without it the generic vector twin above would resolve the prior inside the
# scalarise loop. It reduces once too, so the vector warns once and not once per element.
function expected_risk(rs::VecBaseRM, w::VecNum, pr::Pr_RR, fees::Option{<:Fees} = nothing;
                       strict::Bool = false, kwargs...)
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    rs, X = resolve_risk_inputs(rs, pr)
    return expected_risk(rs, w, X, fees; kwargs...)
end
"""
    expected_risk(r::AbstractBaseRiskMeasure, w::MatNum, args...; kwargs...)
    expected_risk(rs::VecBaseRM, w::MatNum, args...; sca = SumScalariser(), kwargs...)
    expected_risk(kind::RiskInputKind, r, w::MatNum, args...; kwargs...)
    expected_risk(r::RiskRatio, w::MatNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::NonOptimisationRiskRatio, w::MatNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::MeanReturnRiskRatio, w::MatNum, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, w::MatNum, pr::Pr_RR, args...; kwargs...)
    expected_risk(rs::VecBaseRM, w::MatNum, pr::Pr_RR, args...; kwargs...)

Compute the risk of a portfolio whose weights follow a weight path.

The type of the weight argument picks the method. A [`VecNum`](@ref) is one target weight vector, which weighs every observation. A [`MatNum`](@ref) `w` is a `T × N` weight path, whose row `t` holds the weights that the portfolio carried through observation `t`. A fold held under a Weight Drift holds such a path, and [`weight_path`](@ref) makes one. This family has one method for each method of the [`VecNum`](@ref) family, so the entry reads [`risk_input_kind`](@ref), and the composite measures keep methods of their own.

The three input kinds treat a path differently.

  - [`NetReturnsInput`](@ref) computes `r(calc_net_returns(w, X, fees))`. [`calc_net_returns(w::MatNum, X::MatNum, args...)`](@ref) reads the path, so the kernel of the measure does not change.
  - [`WeightsReturnsFeesInput`](@ref) and [`WeightsInput`](@ref) refuse a path by name. Their kernels read a weight vector as one cross-section, and a path gives one number for each observation, which is a different quantity.

The library can score a measure that refuses a path under a drift. The question of such a measure is about the weights and not about the series, so score it against the target weights.

# Algorithm

 1. On the prior route, reduce the prior, the path and the fees to the Investable Mask with [`investable_reduction`](@ref). The mask selects the columns of the path, and each row keeps its observation. Then resolve `r` against the reduced prior with [`resolve_risk_inputs`](@ref).
 2. Refuse a Deferred Quantity that is not resolved, and a Calibration Rule that is not calibrated, as the [`VecNum`](@ref) entry does.
 3. Read the input kind of `r`, and call the method of that kind.

# Arguments

  - $(arg_dict[:r])
  - `w::MatNum`: Weight path `observations × assets`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `pr::Pr_RR`: Prior result, or [`ReturnsResult`](@ref) that carries the returns matrix.
  - `fees`: Optional [`Fees`](@ref) structure.
  - `args...`: The returns carrier and the fees, forwarded to the method of the input kind.

# Keyword Arguments

  - `strict::Bool = false`: Whether a held non-investable asset raises an error. Only the methods that take a prior read it.
  - `sca::Scalariser = SumScalariser()`: Scalariser that combines a vector of measures.

# Validation

  - A measure that declares [`WeightsReturnsFeesInput`](@ref) or [`WeightsInput`](@ref) raises an `ArgumentError` that names the measure and its kind.
  - A measure that reads a return series and gets no returns carrier raises an [`IsNothingError`](@ref).

# Returns

  - `rk::Number`: The risk of the portfolio over the path.

# Related

  - [`MatNum`](@ref)
  - [`weight_path`](@ref): Makes the path this family reads.
  - [`SelfFinancingDrift`](@ref)
  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
  - [`calc_net_returns`](@ref)
  - [`expected_risk`](@ref)
"""
function expected_risk(r::AbstractBaseRiskMeasure, w::MatNum, args...; kwargs...)
    assert_resolved_slots(r)
    assert_calibrated_slots(r)
    return expected_risk(risk_input_kind(r), r, w, args...; kwargs...)
end
function expected_risk(rs::VecBaseRM, w::MatNum, args...; sca::Scalariser = SumScalariser(),
                       kwargs...)
    return scalarise(sca, rs) do r
        return expected_risk(r, w, args...; kwargs...) * r.settings.scale
    end
end
function expected_risk(::NetReturnsInput, r::AbstractBaseRiskMeasure, w::MatNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return r(calc_net_returns(w, X, fees))
end
# The two weight-reading kinds refuse a path rather than falling through to a `MethodError`.
# Their kernels contract a weight vector across the assets, so a path is not a wider input to
# them but a different quantity. The message follows `risk_input_kind`'s own fallback.
function expected_risk(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure, w::MatNum,
                       args...; kwargs...)
    return throw(ArgumentError("`$(typeof(r))` declares `WeightsReturnsFeesInput`, so its kernel reads `w` as one cross-section of weights and is called as `r(w, X, fees)`. A `w::MatNum` is a weight path, one row of weights per observation, which is a different quantity and not a wider input to that kernel.\nScore this measure against the target weight vector, `w::VecNum`, which is the first row of the path.\nGot\nsize(w) => $(size(w))"))
end
function expected_risk(::WeightsInput, r::AbstractBaseRiskMeasure, w::MatNum, args...;
                       kwargs...)
    return throw(ArgumentError("`$(typeof(r))` declares `WeightsInput`, so its kernel reads `w` as one cross-section of weights and is called as `r(w)`. A `w::MatNum` is a weight path, one row of weights per observation, which is a different quantity and not a wider input to that kernel.\nScore this measure against the target weight vector, `w::VecNum`, which is the first row of the path.\nGot\nsize(w) => $(size(w))"))
end
# The carrier-free refusal of the `VecNum` block, for a weight path. Only `NetReturnsInput`
# needs it: the other two kinds refuse **any** path above, through an `args...` method that
# a carrier-free call already reaches, and that refusal is the more fundamental one.
function expected_risk(::NetReturnsInput, r::AbstractBaseRiskMeasure, w::MatNum,
                       X::Nothing = nothing, fees::Option{<:Fees} = nothing; kwargs...)
    return throw(IsNothingError(missing_returns_carrier_message(r)))
end
# The three ratio composites split by type for the same reason their `VecNum` twins do:
# `NonOptimisationRiskRatio` names `sca1` and `sca2`, and `RiskRatio` carries neither.
function expected_risk(r::RiskRatio, w::MatNum, X::MatNum, fees::Option{<:Fees} = nothing;
                       kwargs...)
    return expected_risk(r.r1, w, X, fees; kwargs...) /
           expected_risk(r.r2, w, X, fees; kwargs...)
end
function expected_risk(r::NonOptimisationRiskRatio, w::MatNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return expected_risk(r.r1, w, X, fees; kwargs..., sca = r.sca1) /
           expected_risk(r.r2, w, X, fees; kwargs..., sca = r.sca2)
end
function expected_risk(r::MeanReturnRiskRatio, w::MatNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return (expected_risk(r.rt, w, X, fees; kwargs...) - r.rf) /
           expected_risk(r.rk, w, X, fees; kwargs..., sca = r.sca)
end
# The path's value-level door, and it reduces by the rule the vector's door reduces by: the
# mask selects the columns of the path and every row keeps its own observation.
function expected_risk(r::AbstractBaseRiskMeasure, w::MatNum, pr::Pr_RR,
                       fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    r, X = resolve_risk_inputs(r, pr)
    return expected_risk(r, w, X, fees; kwargs...)
end
# The vector's own prior route, as above: it resolves the whole vector **once**, so a Deferred
# Quantity is fitted once per measure rather than once per element evaluation.
function expected_risk(rs::VecBaseRM, w::MatNum, pr::Pr_RR, fees::Option{<:Fees} = nothing;
                       strict::Bool = false, kwargs...)
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    rs, X = resolve_risk_inputs(rs, pr)
    return expected_risk(rs, w, X, fees; kwargs...)
end
function expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, args...; kwargs...)
    return [expected_risk(r, wi, args...; kwargs...) for wi in w]
end
# A population reduces once, on the mask a member holds an asset under. Reducing inside the
# comprehension would warn once per member of a population that all hold the same dead asset.
function expected_risk(r::BaseRM_VecBaseRM, w::VecVecNum, pr::Pr_RR,
                       fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    r, X = resolve_risk_inputs(r, pr)
    return [expected_risk(r, wi, X, fees; kwargs...) for wi in w]
end
"""
    difference_risk(r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum, X::MatNum,
                    fees::Option{<:Fees}) -> Number

Compute the risk of the weight difference `wd` on a return series that pays the fee of the portfolio `w`.

The independent mode of [`RiskTrackingRiskMeasure`](@ref) calls it with `wd = w - wb`. The model of that mode charges the fee once, on the portfolio, so the series that it tracks is the net return of the portfolio minus the gross return of the benchmark. This function reads the same series. [`expected_risk`](@ref) on `wd` charges the fee of `wd` instead, and `wd` is not a portfolio that pays a fee.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x} &= \\mathbf{X}(\\boldsymbol{w} - \\boldsymbol{w}_b) - F(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Return series ``T \\times 1`` that the tracked measure reads.
  - $(math_dict[:X_returns])
  - $(math_dict[:w_port])
  - $(math_dict[:w_b_track])
  - $(math_dict[:F_fee_series])

# Algorithm

 1. A `nothing` fee returns `expected_risk(r, wd, X)`, because the two series are equal without a fee.
 2. A measure that reads a return series, [`NetReturnsInput`](@ref), reads ``\\boldsymbol{x}``.
 3. A measure that reads the weights alone, [`WeightsInput`](@ref), reads `wd` and no fee.
 4. A moment measure resolves its observation weights against `X` with [`resolve_observation_weights`](@ref). It reads ``\\boldsymbol{x}`` and takes its target from `wd`. So a per-asset target is ``(\\boldsymbol{w} - \\boldsymbol{w}_b)^\\intercal \\boldsymbol{\\mu}``, as in the model.
 5. A [`TrackingRiskMeasure`](@ref) takes the norm of ``\\boldsymbol{x}`` minus its own benchmark series.
 6. A nested [`RiskTrackingRiskMeasure`](@ref) recurses. The independent mode subtracts its own benchmark weights from `wd`. The dependent mode subtracts the risk of its own benchmark weights, which pay their own fee, as in the model.
 7. A [`RiskRatio`](@ref) divides the results of its two measures. A [`VarianceSkewKurtosis`](@ref) reads `wd` and no fee.
 8. Any other [`WeightsReturnsFeesInput`](@ref) measure reads ``\\boldsymbol{x}`` through [`expected_risk_from_returns`](@ref), which refuses a measure that cannot read a series.

# Arguments

  - `r`: Tracked risk measure.
  - `wd`: Weight difference `assets × 1`.
  - $(arg_dict[:pw])
  - `X`: Asset returns matrix `observations × assets`.
  - `fees`: Fee of the portfolio `w`, or `nothing`.

# Returns

  - `rk::Number`: The risk of `wd` on the series ``\\boldsymbol{x}``.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`charge_fees`](@ref)
  - [`expected_risk`](@ref)
  - [`risk_input_kind`](@ref)
"""
function difference_risk(r::AbstractBaseRiskMeasure, wd::VecNum, ::VecNum, X::MatNum,
                         ::Nothing)
    return expected_risk(r, wd, X)
end
function difference_risk(r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum, X::MatNum,
                         fees::Fees)
    return difference_risk(risk_input_kind(r), r, wd, w, X, fees)
end
function difference_risk(::NetReturnsInput, r::AbstractBaseRiskMeasure, wd::VecNum,
                         w::VecNum, X::MatNum, fees::Fees)
    return r(charge_fees(X * wd, w, fees))
end
function difference_risk(::WeightsInput, r::AbstractBaseRiskMeasure, wd::VecNum, ::VecNum,
                         ::MatNum, ::Fees)
    return r(wd)
end
function difference_risk(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure, wd::VecNum,
                         w::VecNum, X::MatNum, fees::Fees)
    return expected_risk_from_returns(r, charge_fees(X * wd, w, fees))
end
function difference_risk(r::Union{<:LoHiOrderMoment, <:Kurtosis, <:TCM_Sk,
                                  <:MedianAbsoluteDeviation}, wd::VecNum, w::VecNum,
                         X::MatNum, fees::Fees)
    resolved = resolve_observation_weights(r, X)
    x = charge_fees(X * wd, w, fees)
    return moment_risk(resolved, x .- calc_moment_target(resolved, wd, x))
end
function difference_risk(r::TrackingRiskMeasure, wd::VecNum, w::VecNum, X::MatNum,
                         fees::Fees)
    return norm_error(r.alg, charge_fees(X * wd, w, fees), tracking_benchmark(r.tr, X),
                      size(X, 1))
end
function difference_risk(r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                    <:IndependentVariableTracking},
                         wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return difference_risk(r.r, wd - r.tr.w, w, X, fees)
end
function difference_risk(r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                    <:DependentVariableTracking},
                         wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return abs(difference_risk(r.r, wd, w, X, fees) - expected_risk(r.r, r.tr.w, X, fees))
end
function difference_risk(r::RiskRatio, wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return difference_risk(r.r1, wd, w, X, fees) / difference_risk(r.r2, wd, w, X, fees)
end
function difference_risk(r::VarianceSkewKurtosis, wd::VecNum, ::VecNum, X::MatNum, ::Fees)
    return r(wd, X)
end
"""
    expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...) -> Number

Evaluate a risk measure on a net return series that the caller already formed.

For a measure whose [`supports_precomputed_returns`](@ref) is `true`, the function returns `r(X)`. For any other measure it raises an `ArgumentError` that says why. Without the check, a [`WeightsInput`](@ref) measure would read `X` as weights, and a moment measure with a per-asset `mu` would raise a `MethodError` that does not say what is wrong. The internal callers that hold a series, such as the scoring of a cross-validation prediction, call this function and not the functor.

The series `X` must be finite. The function does not check it, because every internal caller gives it a finite series, and the three ratio kernels call it for each half at every evaluation. On a series with gaps, a tail measure returns a finite wrong number and not a `NaN`. `partialsort` puts a `NaN` after every real number, so a CVaR reads its order statistic from the finite values and divides by the length of the whole series. Remove the gaps first with `x[isfinite.(x)]`. To score a panel with gaps, use [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), which removes the Held Gaps once.

# Arguments

  - `r::AbstractBaseRiskMeasure`: Risk measure.
  - `X::VecNum`: Net portfolio return series `observations × 1`.

# Validation

  - A measure whose [`supports_precomputed_returns`](@ref) is `false` raises an `ArgumentError`.

# Returns

  - `rk::Number`: The risk of the series.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`expected_risk`](@ref)
  - [`filter_held_gaps`](@ref)
"""
function expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    if !supports_precomputed_returns(r)
        throw(ArgumentError("`$(typeof(r))` cannot be evaluated on a precomputed return series: it requires portfolio weights and/or per-asset data (e.g. a weights-only measure such as `TurnoverRiskMeasure`/`EqualRisk`, a tracking measure, a variance-carrying composite such as `VarianceSkewKurtosis`, or a moment measure with a per-asset `mu`). Evaluate it through `expected_risk(r, w, X, fees)` with explicit weights instead."))
    end
    return r(X)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Combine the risks of several risk measures on one net return series that the caller already formed.

The method follows the three rules of [`expected_risk`](@ref) on a vector. `scale` weighs each element, `sca` combines the products, and `rke` has no effect. An element that cannot read a series raises its own `ArgumentError`, which names that element.

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

Evaluate a risk measure on each series of a vector of net return series.

The method calls [`expected_risk_from_returns`](@ref) on each `Xi` in `X`, and returns one risk for each series.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function expected_risk_from_returns(r::BaseRM_VecBaseRM, X::VecVecNum; kwargs...)
    return [expected_risk_from_returns(r, Xi; kwargs...) for Xi in X]
end
"""
    number_effective_assets(w::VecNum)

Compute the effective number of assets of a portfolio, the inverse of the Herfindahl-Hirschman index of its weights.

The result is the number of equally weighted assets that have the same concentration as `w`. For a long-only portfolio whose weights sum to one, it lies between `1`, when one asset holds all of the weight, and `N`, at equal weights.

# Mathematical definition

```math
\\begin{align}
N_{\\mathrm{eff}} &= \\frac{1}{\\sum_{i=1}^{N} w_i^2}\\,.
\\end{align}
```

Where:

  - ``N_{\\mathrm{eff}}``: Effective number of assets.
  - $(math_dict[:w_i_asset])
  - $(math_dict[:N])

# Arguments

  - `w::VecNum`: Portfolio weights vector `assets × 1`.

# Returns

  - `neff::Number`: Effective number of assets.

# Examples

```jldoctest
julia> number_effective_assets([0.5, 0.25, 0.25])
2.6666666666666665
```

# Related

  - [`set_weight_norm_2_constraints!`](@ref)
  - [`EqualRisk`](@ref)
  - [`risk_contribution`](@ref)
"""
function number_effective_assets(w::VecNum)
    return inv(LinearAlgebra.dot(w, w))
end
"""
    adjusted_risk(ps::Nothing, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                  fees::Option{<:Fees}, delta::Number; kwargs...)
    adjusted_risk(ps::VecNum, rs::VecBaseRM, w::VecNum, X::MatNum,
                  fees::Option{<:Fees}, delta::Number; kwargs...)

Evaluate the risk at `w` with the homogeneity correction applied to each measure, which is the function that [`risk_contribution`](@ref) differentiates.

[`adjust_risk_contribution`](@ref) divides the risk of a measure by its degree of homogeneity. By Euler's theorem, the gradient of the divided risk, multiplied by the weights, then sums to the risk. A vector such as `[Variance(), ConditionalValueatRisk()]` holds a measure of degree 2 and a measure of degree 1, so the vector has no single degree. The function therefore corrects each element before it combines them.

The chain-rule weights ``p_k`` come from [`scalariser_element_weights`](@ref), which the caller evaluates once at the unperturbed weights. The finite difference moves the risk of each element and keeps the weights, so a maximum or a minimum keeps the element that it picks at `w`. Under the sum, the maximum and the minimum, the gradient of ``A`` times the weights then sums to ``\\sum_k p_k s_k R_k(\\boldsymbol{w})``, which is the aggregate that [`expected_risk`](@ref) reports. Under the log-sum-exp scalariser the same sum is the softmax-weighted mean of the scaled risks, which is below the aggregate, because the log-sum-exp is not homogeneous.

# Mathematical definition

```math
\\begin{align}
A(\\boldsymbol{w}) &= \\sum_{k=1}^{n} p_k \\, s_k \\, \\frac{R_k(\\boldsymbol{w})}{d_k}\\,.
\\end{align}
```

A single measure takes ``n = 1``, ``p_1 = 1`` and ``s_1 = 1``, because a single measure ignores its `scale`.

Where:

  - ``A(\\boldsymbol{w})``: Homogeneity-adjusted risk.
  - $(math_dict[:p_k_chain])
  - $(math_dict[:s_k_scale])
  - $(math_dict[:R_k_vec])
  - ``d_k``: Degree of homogeneity of the ``k``-th risk measure of a vector.
  - $(math_dict[:n_rm_vec])

# Arguments

  - `ps`: Chain-rule weights of a vector of measures, or `nothing` for a single measure.
  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.
  - `delta::Number`: Step of the finite difference, which [`adjust_risk_contribution`](@ref) reads for [`EqualRisk`](@ref).

# Returns

  - `rk::Number`: The homogeneity-adjusted risk.

# Related

  - [`risk_contribution`](@ref)
  - [`adjust_risk_contribution`](@ref)
  - [`scalariser_element_weights`](@ref)
  - [`expected_risk`](@ref)
"""
function adjusted_risk(::Nothing, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                       fees::Option{<:Fees}, delta::Number; kwargs...)
    return adjust_risk_contribution(r, expected_risk(r, w, X, fees; kwargs...), delta)
end
function adjusted_risk(ps::VecNum, rs::VecBaseRM, w::VecNum, X::MatNum,
                       fees::Option{<:Fees}, delta::Number; kwargs...)
    return sum(zip(ps, rs)) do (p, r)
        if iszero(p)
            return zero(p)
        end
        val = adjust_risk_contribution(r, expected_risk(r, w, X, fees; kwargs...), delta)
        return p * r.settings.scale * val
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
        strict::Bool = false,
        kwargs...
    ) -> Vector

Compute the contribution of each asset to the risk of a portfolio with a finite difference.

The contributions sum to the risk that [`expected_risk`](@ref) reports, because the function divides the gradient of each measure by its degree of homogeneity. A vector of measures decomposes its aggregate. The function corrects each element before the scalariser combines them, through [`adjusted_risk`](@ref), so the contributions sum to the aggregate when the elements have different degrees. Under [`MaxScalariser`](@ref) and [`MinScalariser`](@ref), the contributions are those of the element that the scalariser picks at `w`, and at a tie they are those of the earliest such element. Under [`LogSumExpScalariser`](@ref), the contributions sum to the softmax-weighted mean of the scaled risks, which is below the aggregate.

A prior result reduces to the Investable Mask before the finite difference, so no perturbed weight meets a `NaN` moment, and a non-investable asset reports exactly `0`. The prior also resolves the measure once, so a Deferred Quantity is fitted once and not once for each finite difference.

The contributions are those of the target weights. Under a Weight Drift the return series of a fold is not linear in one weight vector, so the contributions sum to the realised risk of the fold only to first order in the drift. A weight path holds no single vector for the finite difference to move.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RC}_i &= \\frac{w_i}{d} \\, \\frac{\\partial R(\\boldsymbol{w})}{\\partial w_i}\\,, \\\\
\\sum_{i=1}^{N} \\mathrm{RC}_i &= R(\\boldsymbol{w})\\,.
\\end{align}
```

The second line is Euler's theorem for a measure that is homogeneous of degree ``d``. With `marginal = true` the function returns ``\\partial R(\\boldsymbol{w}) / \\partial w_i`` divided by ``d``, without the factor ``w_i``.

Where:

  - ``\\mathrm{RC}_i``: Risk contribution of asset ``i``.
  - $(math_dict[:w_i_asset])
  - $(math_dict[:d_homog])
  - $(math_dict[:R_w])
  - $(math_dict[:N])

# Algorithm

 1. Reduce `X`, `w` and `fees` to the Investable Mask with [`investable_reduction`](@ref), which gives the mask `imsk`.
 2. Resolve `r` and the returns matrix `X` with [`resolve_risk_inputs`](@ref).
 3. Compute the chain-rule weights `ps` of a vector of measures at `w` with [`scalariser_element_weights`](@ref). A single measure gets `nothing`.
 4. Compute `rc`, the finite difference of [`adjusted_risk`](@ref) at `w` with the step `delta`, through [`finite_difference_gradient`](@ref).
 5. Unless `marginal` is `true`, multiply `rc` by `w` element by element.
 6. Expand `rc` to the full universe with [`expand_investable_weights`](@ref), with a zero for each non-investable asset.

# Arguments

  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum_Pr`: Returns matrix, prior result or returns result.
  - `fees`: Optional [`Fees`](@ref) structure.

# Keyword Arguments

  - `delta::Number = 1e-6`: Step of the finite difference.
  - `marginal::Bool = false`: Whether to return the marginal risks, without the factor ``w_i``.
  - `sca::Scalariser = SumScalariser()`: Scalariser that combines a vector of measures. A single measure ignores it.
  - `strict::Bool = false`: Whether a held non-investable asset raises an error. Only a prior result reads it.

# Validation

  - On a bare matrix, a Deferred Quantity that is not resolved, and an empty slot that the functor reads, raise through [`assert_resolved_slots`](@ref), as they do in [`expected_risk`](@ref).
  - Under `strict = true`, a held non-investable asset raises through [`investable_reduction`](@ref).

# Returns

  - `rc::Vector`: The risk contribution, or the marginal risk, of each asset of the full universe.

# Related

  - [`expected_risk`](@ref)
  - [`adjusted_risk`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`risk_gradient`](@ref): The gradient of the risk itself, without the homogeneity correction.
"""
function risk_contribution(r::BaseRM_VecBaseRM, w::VecNum, X::MatNum_Pr,
                           fees::Option{<:Fees} = nothing; delta::Number = 1e-6,
                           marginal::Bool = false, sca::Scalariser = SumScalariser(),
                           strict::Bool = false, kwargs...)
    # The value-level door reduces once, before the finite difference, so no perturbed
    # weight ever meets a `NaN` moment. The per asset answer expands back into a zero
    # vector of the full length, so a dead asset reports exactly `0`.
    imsk, X, w, fees = investable_reduction(X, w, fees, strict)
    r, X = resolve_risk_inputs(r, X)
    ps = scalariser_element_weights(sca, r, w, X, fees; kwargs...)
    rc = finite_difference_gradient(w, delta) do v
        return adjusted_risk(ps, r, v, X, fees, delta; kwargs...)
    end
    if !marginal
        rc .*= w
    end
    return expand_investable_weights(imsk, rc)
end
"""
    finite_difference_gradient(f, w::VecNum, delta::Number)

Compute the two-sided finite difference of a scalar function of the weights, with one entry for each asset.

[`risk_contribution`](@ref) and the fallback of [`risk_gradient`](@ref) share this function, and they give it different functions to differentiate. The contribution differentiates the homogeneity-adjusted risk, so that Euler's theorem gives the risk back. The gradient differentiates the risk itself.

# Mathematical definition

```math
\\begin{align}
g_i &= \\frac{f(\\boldsymbol{w} + \\delta \\boldsymbol{e}_i) - f(\\boldsymbol{w} - \\delta \\boldsymbol{e}_i)}{2 \\delta}\\,.
\\end{align}
```

Where:

  - ``g_i``: Entry ``i`` of the finite-difference gradient.
  - ``f``: Scalar function of the weights.
  - $(math_dict[:w_port])
  - ``\\delta``: Step of the finite difference.
  - ``\\boldsymbol{e}_i``: ``i``-th unit vector.

# Algorithm

 1. Copy `w` into both columns of the `N × 2` work matrix `ws`.
 2. For each asset `i`, add `delta` to entry `i` of the first column, and subtract `delta` from entry `i` of the second column.
 3. Set `g[i]` to the difference of `f` at the two columns, times `id2 = inv(2 * delta)`.
 4. Restore entry `i` of both columns to `w[i]`, and go to the next asset.

# Arguments

  - `f`: Scalar function of a weight vector.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `delta::Number`: Step of the finite difference.

# Returns

  - `g::Vector`: The finite-difference gradient, one entry for each asset.

# Related

  - [`risk_contribution`](@ref)
  - [`risk_gradient`](@ref)
"""
function finite_difference_gradient(f, w::VecNum, delta::Number)
    N = length(w)
    g = Vector{eltype(w)}(undef, N)
    ws = Matrix{eltype(w)}(undef, N, 2)
    ws .= w
    id2 = inv(2 * delta)
    for i in eachindex(w)
        ws[i, 1] += delta
        ws[i, 2] -= delta
        g[i] = (f(view(ws, :, 1)) - f(view(ws, :, 2))) * id2
        ws[i, 1] = w[i]
        ws[i, 2] = w[i]
    end
    return g
end
"""
    risk_gradient(
        r::BaseRM_VecBaseRM,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        delta::Number = 1e-6,
        sca::Scalariser = SumScalariser(),
        strict::Bool = false,
        kwargs...
    ) -> Vector

Compute the gradient of a risk measure with respect to the weights, with one entry for each asset.

The gradient is the derivative of the figure that [`expected_risk`](@ref) reports, at the weights that the caller gives, without a homogeneity correction. The marginal figure of [`risk_contribution`](@ref) divides by the degree of the measure, so that Euler's theorem gives the risk back. On a [`Variance`](@ref) the two differ by a factor of two.

Three measures have a closed form, and the function uses it. Every other measure, and a [`MeanReturn`](@ref) that is charged fees, takes the two-sided finite difference at the step `delta`, which costs `2N` evaluations of the measure. The fee is not linear in the weights, so a `MeanReturn` with fees has no closed form here. Where the gradient is not defined at the point, such as at a tie inside a maximum, at a scenario on a Value-at-Risk atom, or at a zero variance under the square root, the finite difference returns the chord across the kink. The closed form of the standard deviation falls back to the finite difference at a zero variance.

A vector of measures differentiates the scalarised aggregate by the chain rule, with each element at its own `settings.scale`. Under [`SumScalariser`](@ref) it sums the gradients of the elements. Under [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) it takes the gradient of the element that the scalariser picks, and at a tie the earliest such element, where the subgradient is a set. Under [`LogSumExpScalariser`](@ref) it takes the softmax-weighted sum, which is smooth everywhere. An element with a closed form keeps it inside a vector whose other elements use the finite difference.

# Mathematical definition

```math
\\begin{align}
\\nabla \\left( \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} \\right) &= 2 \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w}\\,, \\\\
\\nabla \\sqrt{\\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w}} &= \\frac{\\hat{\\mathbf{\\Sigma}} \\boldsymbol{w}}{\\sqrt{\\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w}}}\\,, \\\\
\\nabla \\sum_{t=1}^{T} \\omega_t \\, r_t(\\boldsymbol{w}) &= \\sum_{t=1}^{T} \\omega_t \\, \\nabla r_t(\\boldsymbol{w})\\,, \\\\
\\nabla S\\left( s_1 R_1(\\boldsymbol{w}),\\, \\ldots,\\, s_n R_n(\\boldsymbol{w}) \\right) &= \\sum_{k=1}^{n} p_k \\, s_k \\, \\nabla R_k(\\boldsymbol{w})\\,.
\\end{align}
```

The first line is the gradient of a [`Variance`](@ref), the second of a [`StandardDeviation`](@ref), and the third of a [`MeanReturn`](@ref). The return of observation ``t`` is ``r_t(\\boldsymbol{w}) = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``, with ``\\nabla r_t = \\boldsymbol{x}_t``. Under the log flag of the measure it is ``r_t(\\boldsymbol{w}) = \\log(1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w})``, with ``\\nabla r_t = \\boldsymbol{x}_t / (1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w})``. The observation weights are normalised, ``\\omega_t = w_t / \\sum_{s=1}^{T} w_s``, and they are ``\\omega_t = 1 / T`` when the measure states none. The last line is the chain rule for a vector of measures.

Where:

  - $(math_dict[:Sigma_hat])
  - $(math_dict[:w_port])
  - ``\\omega_t``: Normalised observation weight of observation ``t``.
  - ``r_t(\\boldsymbol{w})``: Portfolio return of observation ``t``.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])
  - $(math_dict[:S_sca_vec])
  - $(math_dict[:p_k_chain])
  - $(math_dict[:s_k_scale])
  - $(math_dict[:R_k_vec])
  - $(math_dict[:n_rm_vec])

# Algorithm

The vector route, `risk_gradient(rs::VecBaseRM, w, X, fees)`:

 1. Compute the chain-rule weights `ps` at `w` with [`scalariser_element_weights`](@ref).
 2. Set the gradient `g` to zero.
 3. For each element `r` whose weight `p` is not zero, add `p * r.settings.scale` times the gradient of `r` to `g`.

The prior route reduces the prior, the weights and the fees to the Investable Mask with [`investable_reduction`](@ref), resolves the measure once with [`resolve_risk_inputs`](@ref), and expands the gradient back to the full universe, as [`risk_contribution`](@ref) does.

# Arguments

  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum_Pr`: Returns matrix, prior result or returns result.
  - `fees`: Optional [`Fees`](@ref) structure.

# Keyword Arguments

  - `delta::Number = 1e-6`: Step of the finite difference.
  - `sca::Scalariser = SumScalariser()`: Scalariser that combines a vector of measures. A single measure ignores it.
  - `strict::Bool = false`: Whether a held non-investable asset raises an error. Only a prior result reads it.

# Validation

  - On a bare matrix, a Deferred Quantity that is not resolved, and an empty slot that the functor reads, raise through [`assert_resolved_slots`](@ref), as they do in [`expected_risk`](@ref).
  - A Calibration Rule that is not calibrated raises through [`assert_calibrated_slots`](@ref).
  - Under `strict = true`, a held non-investable asset raises through [`investable_reduction`](@ref).

# Returns

  - `g::Vector`: The gradient, one entry for each asset of the full universe. A non-investable asset reports exactly `0`.

# Examples

```jldoctest
julia> risk_gradient(Variance(; sigma = [0.04 0.02; 0.02 0.16]), [0.5, 0.5], zeros(2, 2))
2-element Vector{Float64}:
 0.06
 0.18
```

# Related

  - [`expected_risk`](@ref)
  - [`risk_contribution`](@ref)
  - [`finite_difference_gradient`](@ref)
  - [`measure_gradient`](@ref)
  - [`scalariser_gradient_weights`](@ref)
"""
function risk_gradient(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; delta::Number = 1e-6, kwargs...)
    assert_resolved_slots(r)
    assert_calibrated_slots(r)
    return measure_gradient(r, w, X, fees; delta = delta, kwargs...)
end
function risk_gradient(rs::VecBaseRM, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing;
                       delta::Number = 1e-6, sca::Scalariser = SumScalariser(), kwargs...)
    ps = scalariser_element_weights(sca, rs, w, X, fees; kwargs...)
    g = zeros(eltype(w), length(w))
    for (p, r) in zip(ps, rs)
        if iszero(p)
            continue
        end
        g .+= (p * r.settings.scale) .*
              risk_gradient(r, w, X, fees; delta = delta, kwargs...)
    end
    return g
end
# The prior route: reduce to the Investable Mask once, resolve the measure once, and expand
# the answer back to the full universe, exactly as `risk_contribution` does.
function risk_gradient(r::BaseRM_VecBaseRM, w::VecNum, pr::Pr_RR,
                       fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    imsk, pr, w, fees = investable_reduction(pr, w, fees, strict)
    r, X = resolve_risk_inputs(r, pr)
    return expand_investable_weights(imsk, risk_gradient(r, w, X, fees; kwargs...))
end
"""
    measure_gradient(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum, fees::Option{<:Fees}; delta::Number, kwargs...)
    measure_gradient(r::Variance, w::VecNum, X::MatNum, fees::Option{<:Fees}; kwargs...)
    measure_gradient(r::StandardDeviation, w::VecNum, X::MatNum, fees::Option{<:Fees}; kwargs...)
    measure_gradient(r::MeanReturn, w::VecNum, X::MatNum, fees::Nothing; kwargs...)

Compute the gradient of one resolved measure on a bare matrix.

The generic method takes the finite difference of [`expected_risk`](@ref). The methods for [`Variance`](@ref), [`StandardDeviation`](@ref) and [`MeanReturn`](@ref) use the closed forms that [`risk_gradient`](@ref) states. A `MeanReturn` that is charged fees takes the generic method, because the fee is not linear in the weights. A `StandardDeviation` at a zero variance also takes the finite difference.

# Arguments

  - `r::AbstractBaseRiskMeasure`: Resolved risk measure.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.
  - `delta::Number`: Step of the finite difference.

# Returns

  - `g::Vector`: The gradient, one entry for each asset.

# Related

  - [`risk_gradient`](@ref)
  - [`finite_difference_gradient`](@ref)
"""
function measure_gradient(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                          fees::Option{<:Fees}; delta::Number, kwargs...)
    return finite_difference_gradient(w, delta) do v
        return expected_risk(r, v, X, fees; kwargs...)
    end
end
function measure_gradient(r::Variance, w::VecNum, ::MatNum, ::Option{<:Fees}; kwargs...)
    return 2 .* (r.sigma * w)
end
function measure_gradient(r::StandardDeviation, w::VecNum, X::MatNum, fees::Option{<:Fees};
                          delta::Number, kwargs...)
    s = sqrt(LinearAlgebra.dot(w, r.sigma, w))
    if iszero(s)
        return finite_difference_gradient(w, delta) do v
            return expected_risk(r, v, X, fees; kwargs...)
        end
    end
    return (r.sigma * w) ./ s
end
function measure_gradient(r::MeanReturn, w::VecNum, X::MatNum, ::Nothing; kwargs...)
    x = X * w
    ow = get_observation_weights(r.w, x)
    omega = isnothing(ow) ? fill(inv(length(x)), length(x)) : ow ./ sum(ow)
    if r.flag
        omega = omega ./ (one(eltype(x)) .+ x)
    end
    return transpose(X) * omega
end
"""
    scalariser_gradient_weights(sca::SumScalariser, vals::VecNum)
    scalariser_gradient_weights(sca::MaxScalariser, vals::VecNum)
    scalariser_gradient_weights(sca::MinScalariser, vals::VecNum)
    scalariser_gradient_weights(sca::LogSumExpScalariser, vals::VecNum)

Compute the chain-rule weight of each element in the gradient of a scalarised aggregate, from the scaled risks of the elements.

The weight of an element is the partial derivative of the scalariser with respect to that element. [`SumScalariser`](@ref) gives a weight of one to every element. [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) give a weight of one to the earliest maximum or minimum, and zero to every other element. [`LogSumExpScalariser`](@ref) gives the softmax of ``\\gamma v_k``, which the method computes after it subtracts the maximum of `vals`, so that the exponentials do not overflow.

# Mathematical definition

```math
\\begin{align}
p_k^{\\mathrm{sum}} &= 1\\,, \\\\
p_k^{\\mathrm{max}} &= \\mathbf{1}\\left[ k = \\underset{j}{\\arg\\max} \\, v_j \\right]\\,, \\\\
p_k^{\\mathrm{min}} &= \\mathbf{1}\\left[ k = \\underset{j}{\\arg\\min} \\, v_j \\right]\\,, \\\\
p_k^{\\mathrm{lse}} &= \\frac{\\exp(\\gamma v_k)}{\\sum_{j=1}^{n} \\exp(\\gamma v_j)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:p_k_chain])
  - ``v_k``: Scaled risk ``s_k R_k(\\boldsymbol{w})`` of the ``k``-th risk measure of a vector.
  - ``\\gamma``: Smoothing parameter of the log-sum-exp scalariser.
  - $(math_dict[:n_rm_vec])

# Arguments

  - `sca`: Scalariser.
  - `vals::VecNum`: Scaled risks of the elements.

# Returns

  - `p::Vector`: The chain-rule weight of each element.

# Related

  - [`risk_gradient`](@ref)
  - [`Scalariser`](@ref)
"""
function scalariser_gradient_weights(::SumScalariser, vals::VecNum)
    return ones(eltype(vals), length(vals))
end
function scalariser_gradient_weights(::MaxScalariser, vals::VecNum)
    p = zeros(eltype(vals), length(vals))
    p[argmax(vals)] = one(eltype(vals))
    return p
end
function scalariser_gradient_weights(::MinScalariser, vals::VecNum)
    p = zeros(eltype(vals), length(vals))
    p[argmin(vals)] = one(eltype(vals))
    return p
end
function scalariser_gradient_weights(sca::LogSumExpScalariser, vals::VecNum)
    p = exp.(sca.gamma .* (vals .- maximum(vals)))
    return p ./ sum(p)
end
"""
    scalariser_element_weights(sca::Scalariser, r::AbstractBaseRiskMeasure, args...; kwargs...)
    scalariser_element_weights(sca::Scalariser, rs::VecBaseRM, w::VecNum, X::MatNum,
                               fees::Option{<:Fees}; kwargs...)

Return the chain-rule weight of each element of a vector of risk measures at `w`, or `nothing` for a single measure.

The weights are [`scalariser_gradient_weights`](@ref) of the scaled values ``s_k R_k(\\boldsymbol{w})``, the values that [`expected_risk`](@ref) scalarises. [`risk_gradient`](@ref) and [`risk_contribution`](@ref) take them once, at the weights they are asked at, so the finite difference never moves the element that a maximum or a minimum picks.

# Arguments

  - `sca`: Scalariser that combines the elements.
  - `r`, `rs`: One risk measure, or a vector of them.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Returns

  - `ps::Option{<:VecNum}`: One weight per element of `rs`, or `nothing` for a single measure.

# Related

  - [`scalariser_gradient_weights`](@ref)
  - [`adjusted_risk`](@ref)
"""
function scalariser_element_weights(::Scalariser, ::AbstractBaseRiskMeasure, args...;
                                    kwargs...)
    return nothing
end
function scalariser_element_weights(sca::Scalariser, rs::VecBaseRM, w::VecNum, X::MatNum,
                                    fees::Option{<:Fees}; kwargs...)
    vals = map(r -> expected_risk(r, w, X, fees; kwargs...) * r.settings.scale, rs)
    return scalariser_gradient_weights(sca, vals)
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
        strict::Bool = false,
        kwargs...
    ) -> Vector

Compute the contribution of each risk factor, and of the directions that no factor spans, to the risk of a portfolio.

The function splits the Euler decomposition of the risk between the factor exposures ``\\mathbf{B}^\\intercal \\boldsymbol{w}`` and the exposures to the null space of ``\\mathbf{B}^\\intercal``. The last entry sums the contributions of that null space, the off-factor part. The entries sum to the risk of the portfolio on the returns matrix that the function measures, because the marginal risks come from [`risk_contribution`](@ref) with its homogeneity correction.

A prior result reduces to the Investable Mask at the entry, through [`investable_reduction`](@ref), and the function reduces `rd` with it, so the regression fits the loadings over the investable assets alone. A held non-investable asset warns and its weight is dropped, or it raises under `strict`. The result has one entry for each factor and not one for each asset, so nothing expands.

The gradient is taken on [`original_returns`](@ref), the returns that the caller supplied, and not on `pr.X`. The two differ under a factor prior, where `pr.X` is the reconstruction `F * transpose(M) .+ transpose(b)`. That matrix has the rank of `F` and no residual, so on it the off-factor part of a measure that reads the series reports the share of the intercept and not the idiosyncratic risk, and it can be negative. So under a factor prior the entries sum to the risk on the returns of the caller, and not to `expected_risk(r, w, pr)`. A measure whose kernel reads a moment and not the series, such as [`Variance`](@ref), [`StandardDeviation`](@ref) or [`DistributionValueatRisk`](@ref), gives the same answer on both matrices.

The loadings come from [`resolve_factor_regression`](@ref), which takes the loadings of the prior before it fits `re`. So a regression estimator in `re` has no effect when the prior carries a factor block.

The contributions are those of the target weights. Under a Weight Drift the return series of a fold is not linear in one weight vector, so the contributions sum to the realised risk of the fold only to first order in the drift.

# Mathematical definition

```math
\\begin{align}
\\mathrm{FRC}_j &= \\left[ \\mathbf{B}^\\intercal \\boldsymbol{w} \\right]_j \\left[ \\mathbf{B}^{+} \\boldsymbol{g} \\right]_j\\,, \\quad j = 1, \\ldots, K\\,, \\\\
\\mathrm{FRC}_{K+1} &= \\left( \\tilde{\\mathbf{B}} \\boldsymbol{w} \\right)^\\intercal \\left( \\tilde{\\mathbf{B}}^{+} \\right)^\\intercal \\boldsymbol{g}\\,, \\\\
\\boldsymbol{g} &= \\frac{1}{d} \\nabla R(\\boldsymbol{w})\\,, \\\\
\\sum_{j=1}^{K+1} \\mathrm{FRC}_j &= R(\\boldsymbol{w})\\,.
\\end{align}
```

The rows of ``\\tilde{\\mathbf{B}}`` are an orthonormal basis of the null space of ``\\mathbf{B}^\\intercal``. So ``\\mathbf{B} \\mathbf{B}^{+} + \\tilde{\\mathbf{B}}^\\intercal \\tilde{\\mathbf{B}}`` is the identity, the entries sum to ``\\boldsymbol{w}^\\intercal \\boldsymbol{g}``, and Euler's theorem gives the last line.

Where:

  - ``\\mathrm{FRC}_j``: Risk contribution of factor ``j``, and of the off-factor part at ``j = K + 1``.
  - ``\\mathbf{B}``: Factor loading matrix ``N \\times K``.
  - ``\\tilde{\\mathbf{B}}``: Basis of the null space of ``\\mathbf{B}^\\intercal``, ``(N - K) \\times N``.
  - ``(\\cdot)^{+}``: Moore-Penrose pseudoinverse.
  - ``\\boldsymbol{g}``: Marginal risk of each asset, with the homogeneity correction.
  - $(math_dict[:w_port])
  - $(math_dict[:d_homog])
  - $(math_dict[:R_w])
  - $(math_dict[:N])
  - $(math_dict[:K])

# Algorithm

 1. Reduce `X`, `w` and `fees` to the Investable Mask with [`investable_reduction`](@ref), which gives the mask `imsk`, and reduce `rd` to the same assets with [`investable_returns_view`](@ref).
 2. Pick the loadings `rr` with [`resolve_factor_regression`](@ref).
 3. Resolve `r` and the returns that the caller supplied with [`resolve_factor_risk_inputs`](@ref).
 4. Form `Bt`, the transpose of the loadings, `b2t`, the basis of the null space of `Bt`, and `b3t`, the transpose of the pseudoinverse of `b2t`.
 5. Compute the marginal risks `mr` with [`risk_contribution`](@ref) and `marginal = true`.
 6. Compute the factor contributions `rc_f` and the off-factor contribution `rc_of`, and return `[rc_f; rc_of]`.

# Arguments

  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum_Pr`: Returns matrix, prior result or returns result.
  - `fees`: Optional [`Fees`](@ref) structure.

# Keyword Arguments

  - `re::RegE_Reg = StepwiseRegression()`: Regression estimator or result for the factor loadings.
  - `rd::ReturnsResult = ReturnsResult()`: Returns result that carries the factor returns `F`, which a regression estimator reads.
  - `delta::Number = 1e-6`: Step of the finite difference.
  - `sca::Scalariser = SumScalariser()`: Scalariser that combines a vector of measures. A single measure ignores it.
  - `strict::Bool = false`: Whether a held non-investable asset raises an error. Only a prior result reads it.

# Validation

  - When no carrier supplies the loadings, [`resolve_factor_regression`](@ref) raises an [`IsNothingError`](@ref).
  - Under `strict = true`, a held non-investable asset raises through [`investable_reduction`](@ref).

# Returns

  - `frc::Vector`: The contribution of each factor, with the off-factor contribution as the last entry.

# Related

  - [`risk_contribution`](@ref)
  - [`expected_risk`](@ref)
  - [`resolve_factor_risk_inputs`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`original_returns`](@ref)
  - [`FactorRiskContribution`](@ref): The optimiser that sets a budget on these contributions.

# References

  - $(ref_dict[:cajas2025]) Section 10.2.1.
  - $(ref_dict[:roncalliweisang2012])
  - $(ref_dict[:meucci2007])
"""
function factor_risk_contribution(r::BaseRM_VecBaseRM, w::VecNum, X::MatNum_Pr,
                                  fees::Option{<:Fees} = nothing;
                                  re::RegE_Reg = StepwiseRegression(),
                                  rd::ReturnsResult = ReturnsResult(), delta::Number = 1e-6,
                                  sca::Scalariser = SumScalariser(), strict::Bool = false,
                                  kwargs...)
    # The value-level door reduces once, and the loadings come from the reduced prior or
    # from the reduced data, so the regression is fitted over the live assets alone. The
    # answer is per factor rather than per asset, so nothing expands.
    imsk, X, w, fees = investable_reduction(X, w, fees, strict)
    rd = investable_returns_view(imsk, rd)
    rr = resolve_factor_regression(re, rd, isa(X, AbstractPriorResult) ? X : nothing)
    r, X = resolve_factor_risk_inputs(r, X)
    Bt = transpose(rr.L)
    b2t = transpose(LinearAlgebra.pinv(transpose(LinearAlgebra.nullspace(Bt))))
    b3t = transpose(LinearAlgebra.pinv(b2t))
    mr = risk_contribution(r, w, X, fees; delta = delta, marginal = true, sca = sca,
                           kwargs...)
    rc_f = (Bt * w) .* (transpose(LinearAlgebra.pinv(Bt)) * mr)
    rc_of = LinearAlgebra.dot(b2t * w, b3t * mr)
    rc_f = [rc_f; rc_of]
    return rc_f
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the risk of a portfolio over each rolling window of the returns matrix, at constant weights.

The method scores the one vector `w` on every window, so each number is a property of that weight vector and not of a history. The `(r, ret::VecNum, window)` method reads the realised history instead, because it rolls a net return series that the caller already formed. The two answer different questions, so they are two methods.

The `(r, w::MatNum, X, fees, window)` method scores each window against the weights of a weight path at the last row of the window. The type of the weight argument picks the method, so a vector reads one target and a matrix reads a path.

# Mathematical definition

```math
\\begin{align}
R_t &= R\\left( \\boldsymbol{w};\\, \\mathbf{X}_{t-W+1:t} \\right)\\,, \\quad t = W, \\ldots, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_t_roll])
  - $(math_dict[:R_w_rows])
  - $(math_dict[:w_port])
  - $(math_dict[:W_roll])
  - $(math_dict[:T])

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

The function checks the window itself. Without the check, a window that is not positive indexes `X` out of bounds and raises a `BoundsError` from inside the measure, and a window longer than the sample gives an empty vector that looks like a valid result. [`plot_rolling_measure`](@ref) checks its `rolling` keyword by the same rule.

# Returns

  - `risks::VecNum`: The risk of each rolling window, `T - window + 1` values.

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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the risk of a portfolio over each rolling window of the returns matrix, against a weight path.

`w` is a `T × N` weight path. The method scores the window that ends at row `t` against row `t` of the path, the weights that the portfolio held when that window closed. The rest of the library reads ending weights in the same way.

Each window gets one set of weights, so the result is not the realised risk of the window. The weights moved inside the window, and the method scores the window as if they did not. To score the realised history of each window, use the `(r, ret::VecNum, window)` method, which rolls the drifted series.

At constant weights every row of the path is the same vector, and this method gives the result of the `(r, w::VecNum, X, fees, window)` method.

# Mathematical definition

```math
\\begin{align}
R_t &= R\\left( \\boldsymbol{w}_t;\\, \\mathbf{X}_{t-W+1:t} \\right)\\,, \\quad t = W, \\ldots, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_t_roll])
  - $(math_dict[:R_w_rows])
  - ``\\boldsymbol{w}_t``: Row ``t`` of the weight path, the weights held at observation ``t``.
  - $(math_dict[:W_roll])
  - $(math_dict[:T])

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `w::MatNum`: Weight path (observations × assets), as [`weight_path`](@ref) makes one.
  - `X::MatNum`: Asset returns matrix.
  - `fees::Option{<:Fees}`: Optional fee structure.
  - `window::Integer`: Size of the rolling window (number of periods).

# Keyword Arguments

  - `sca::Scalariser = SumScalariser()`: Scalariser combining a vector `r`. Inert on a single measure.

# Validation

  - `1 <= window <= size(X, 1)`, else a `DomainError` naming `window`.
  - `size(w, 1) == size(X, 1)`, else a `DimensionMismatch` naming both. A path shorter than the sample indexes out of bounds inside whichever measure `r` names, which is the caller error the window check already refuses at the boundary.

# Returns

  - `risks::VecNum`: Expected risk values for each rolling window.

# Related

  - [`weight_path`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`expected_risk`](@ref)
  - [`MatNum`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, w::MatNum, X::MatNum,
                                fees::Option{<:Fees}, window::Integer;
                                sca::Scalariser = SumScalariser(), kwargs...)
    T = size(X, 1)
    @argcheck(1 <= window <= T,
              DomainError(window,
                          "window must be in 1:$(T), the number of observations in X; got window => $window"))
    @argcheck(size(w, 1) == T,
              DimensionMismatch("`size(w, 1) == size(X, 1)` must hold.\nsize(w, 1) => $(size(w, 1))\nsize(X, 1) => $(T)"))
    return [expected_risk(r, view(w, t, :), view(X, (t - window + 1):t, :), fees; sca = sca,
                          kwargs...) for t in window:T]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the risk of each rolling window of a net return series that the caller already formed.

The method reads the realised history. A window is a part of the series `ret`, so under a weight drift the drift does not start again at the first row of a window, and each row reads the weights held on the way to that row. The method takes no `fees`, because the series is already net of fees. The `(r, w::VecNum, X, fees, window)` method reads constant weights instead.

# Mathematical definition

```math
\\begin{align}
R_t &= R\\left( \\boldsymbol{r}_{t-W+1:t} \\right)\\,, \\quad t = W, \\ldots, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_t_roll])
  - ``\\boldsymbol{r}_{a:b}``: Entries ``a`` to ``b`` of the net return series.
  - $(math_dict[:W_roll])
  - $(math_dict[:T])

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `ret::VecNum`: Net portfolio return series.
  - `window::Integer`: Size of the rolling window (number of periods).

# Keyword Arguments

  - `sca::Scalariser = SumScalariser()`: Scalariser combining a vector `r`. Inert on a single measure.

# Validation

  - `1 <= window <= length(ret)`, else a `DomainError` naming `window`.
  - Each window is scored through [`expected_risk_from_returns`](@ref), so a measure whose [`supports_precomputed_returns`](@ref) is `false` raises that entry's own named `ArgumentError`.

The function checks the window for the reason that the constant-weight method states. A measure that reads weights raises the `ArgumentError` of [`expected_risk_from_returns`](@ref).

# Returns

  - `risks::VecNum`: Expected risk values for each rolling window.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`plot_rolling_measure`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, ret::VecNum, window::Integer;
                                sca::Scalariser = SumScalariser(), kwargs...)
    T = length(ret)
    @argcheck(1 <= window <= T,
              DomainError(window,
                          "window must be in 1:$(T), the number of observations in ret; got window => $window"))
    return [expected_risk_from_returns(r, view(ret, (t - window + 1):t); sca = sca,
                                       kwargs...) for t in window:T]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the risk of each rolling window of each series of a population of net return series.

A fold whose optimisation result carries a population of weight vectors forms one series for each member. The method rolls each series with the single-series method, as [`expected_risk_from_returns`](@ref) does on a [`VecVecNum`](@ref).

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `ret::VecVecNum`: Net portfolio return series, one per population member.
  - `window::Integer`: Size of the rolling window (number of periods).

# Returns

  - `risks::Vector{<:VecNum}`: Rolling risk values, one vector per population member.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`VecVecNum`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, ret::VecVecNum, window::Integer;
                                kwargs...)
    return [rolling_window_measure(r, reti, window; kwargs...) for reti in ret]
end

export RiskRatio, number_effective_assets, risk_contribution, risk_gradient,
       factor_risk_contribution, rolling_window_measure
