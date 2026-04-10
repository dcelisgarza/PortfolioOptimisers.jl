"""
    calc_risk_constraint_target(r::LoHiOrderMoment{<:Any,<:Any,Nothing,<:Any}, w, mu, args...)
    calc_risk_constraint_target(r::LoHiOrderMoment{<:Any,<:Any,<:VecNum,<:Any}, w, args...)
    calc_risk_constraint_target(r::LoHiOrderMoment{<:Any,<:Any,<:VecScalar,<:Any}, w, ::Any, k)
    calc_risk_constraint_target(r::LoHiOrderMoment{<:Any,<:Any,<:Number,<:Any}, ::Any, ::Any, k)

Compute the target return used as the reference level for lower/upper moment risk constraints.

Dispatches on the type of `r.mu`: when `nothing`, uses the prior mean vector `mu`; when a
`VecNum`, uses `r.mu`; when a `VecScalar`, combines the vector and scalar parts with `k`;
when a scalar `Number`, scales it by `k`.

# Arguments

  - `r::LoHiOrderMoment`: Risk measure carrying the target specification.
  - `w`: Portfolio weight vector.
  - `mu::VecNum`: Prior mean return vector.
  - `k`: Leverage/scale variable from the model.

# Related

  - [`set_risk_constraints!`](@ref)
"""
function calc_risk_constraint_target(::LoHiOrderMoment{<:Any, <:Any, Nothing, <:Any},
                                     w::VecNum, mu::VecNum, args...)
    return LinearAlgebra.dot(w, mu)
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any},
                                     w::VecNum, args...)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     w::VecNum, ::Any, k)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s * k
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any},
                                     ::Any, ::Any, k)
    return r.mu * k
end
"""
    set_risk_constraints!(model, i, r::LowOrderMoment{<:Any,<:Any,<:Any,<:FirstLowerMoment}, opt, pr, args...; kwargs...)
    set_risk_constraints!(model, i, r::LowOrderMoment{<:Any,<:Any,<:Any,<:MeanAbsoluteDeviation}, opt, pr, args...; kwargs...)
    set_risk_constraints!(model, i, r::LowOrderMoment{<:Any,<:Any,<:Any,<:SecondMoment}, opt, pr, args...; kwargs...)

Add first lower moment, mean absolute deviation, or second moment risk constraints to `model`.

Each overload introduces auxiliary non-negative variables (semi-deviations or lower
exceedances) for `T` observations, computes an observation-weighted mean, and adds an
inequality constraint linking the auxiliary variables to the portfolio returns minus the
target. The second-moment overload additionally supports full and lower-half formulations and
multiple variance encodings via [`set_second_moment_risk!`](@ref).

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `i`: Constraint index for unique naming.
  - `r::LowOrderMoment`: Risk measure instance.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result containing `X` (returns matrix) and `mu`.

# Related

  - [`calc_risk_constraint_target`](@ref)
  - [`set_second_moment_risk!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:flm_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    flm = model[Symbol(:flm_, i)] = JuMP.@variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    flm_risk = model[key] = if isnothing(wi)
        JuMP.@expression(model, Statistics.mean(flm))
    else
        JuMP.@expression(model, Statistics.mean(flm, wi))
    end
    model[Symbol(:cflm_mar_, i)] = JuMP.@constraint(model, sc * ((net_X + flm) .- tgt) >= 0)
    set_risk_bounds_and_expression!(model, opt, flm_risk, r.settings, key)
    return flm_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:MeanAbsoluteDeviation},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:mad_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    mad = model[Symbol(:mad_, i)] = JuMP.@variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    mad_risk = model[Symbol(:mad_risk_, i)] = if isnothing(wi)
        JuMP.@expression(model, 2 * Statistics.mean(mad))
    else
        JuMP.@expression(model, 2 * Statistics.mean(mad, wi))
    end
    model[Symbol(:cmar_mad_, i)] = JuMP.@constraint(model, sc * ((net_X + mad) .- tgt) >= 0)
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, key)
    return mad_risk
end
"""
    set_second_moment_risk!(model, ::QuadRiskExpr, ::Any, factor, second_moment, key, args...)
    set_second_moment_risk!(model, ::RSOCRiskExpr, i, factor, second_moment, key, keyt, keyc, args...)
    set_second_moment_risk!(model, ::SquaredSOCRiskExpr, i, factor, second_moment, key, keyt, keyc, tsecond_moment)
    set_second_moment_risk!(model, ::SOCRiskExpr, i, factor, second_moment, key, keyt, keyc, tsecond_moment)

Build the second-moment risk JuMP expression in one of four encodings.

The `QuadRiskExpr` overload encodes variance as a quadratic dot product. The `RSOCRiskExpr`
overload uses a rotated second-order cone to encode the squared norm. The `SquaredSOCRiskExpr`
overload squares an existing SOC variable. The `SOCRiskExpr` overload returns the SOC
variable directly (standard deviation form). All methods return the risk expression and a
scaling factor.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `i`: Constraint index for unique naming.
  - `factor::Number`: Variance correction factor (e.g. `1 / (T - 1)`).
  - `second_moment`: Return deviation vector or matrix.
  - `key::Symbol`: Symbol for storing the expression in the model.
  - `keyt`, `keyc`: Symbols for the auxiliary variable and its constraint.
  - `tsecond_moment`: Pre-existing SOC variable (used by SquaredSOC/SOC overloads).

# Related

  - [`set_risk_constraints!`](@ref)
  - [`second_moment_bound_val`](@ref)
"""
function set_second_moment_risk!(model::JuMP.Model, ::QuadRiskExpr, ::Any, factor::Number,
                                 second_moment, key::Symbol, args...)
    return model[key] = JuMP.@expression(model,
                                         factor *
                                         LinearAlgebra.dot(second_moment, second_moment)),
                        sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::RSOCRiskExpr, i::Any, factor::Number,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 args...)
    sc = model[:sc]
    tsecond_moment = model[Symbol(keyt, i)] = JuMP.@variable(model)
    model[Symbol(keyc, i)] = JuMP.@constraint(model,
                                              [sc * tsecond_moment;
                                               0.5;
                                               sc * second_moment] in
                                              JuMP.RotatedSecondOrderCone())
    return model[key] = JuMP.@expression(model, factor * tsecond_moment), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SquaredSOCRiskExpr, i::Any,
                                 factor::Number, second_moment, key::Symbol, keyt::Symbol,
                                 keyc::Symbol, tsecond_moment::JuMP.AbstractJuMPScalar)
    return model[key] = JuMP.@expression(model, factor * tsecond_moment^2), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SOCRiskExpr, i::Any, factor::Number,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 tsecond_moment::JuMP.AbstractJuMPScalar)
    factor = sqrt(factor)
    return model[key] = JuMP.@expression(model, factor * tsecond_moment), factor
end
"""
    second_moment_bound_val(alg::SecondMomentFormulation, ub::Frontier, factor)
    second_moment_bound_val(alg::SecondMomentFormulation, ub::VecNum, factor)
    second_moment_bound_val(alg::SecondMomentFormulation, ub::Number, factor)
    second_moment_bound_val(::Any, ::Nothing, ::Any)

Convert an upper-bound value to the appropriate scale for the second-moment bounding variable.

Scales `ub` by `inv(factor)` and, when the formulation is not `SOCRiskExpr`, applies a
square-root transformation to convert from variance to standard-deviation units. Returns
`nothing` when `ub` is `nothing`.

# Arguments

  - `alg::SecondMomentFormulation`: Second-moment risk formulation (e.g. `SOCRiskExpr`).
  - `ub`: Upper bound value (scalar, vector, `Frontier`, or `nothing`).
  - `factor::Number`: Variance correction factor.

# Related

  - [`set_second_moment_risk!`](@ref)
"""
function second_moment_bound_val(alg::SecondMomentFormulation, ub::Frontier, factor::Number)
    return _Frontier(; N = ub.N, factor = inv(factor), flag = isa(alg, SOCRiskExpr))
end
function second_moment_bound_val(alg::SecondMomentFormulation, ub::VecNum, factor::Number)
    return inv(factor) * (isa(alg, SOCRiskExpr) ? ub : sqrt.(ub))
end
function second_moment_bound_val(alg::SecondMomentFormulation, ub::Number, factor::Number)
    return inv(factor) * (isa(alg, SOCRiskExpr) ? ub : sqrt(ub))
end
function second_moment_bound_val(::Any, ::Nothing, ::Any)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:second_moment_risk_, i)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    bound_key = Symbol(:sqrt_second_moment_, i)
    sqrt_second_moment = model[bound_key] = JuMP.@variable(model)
    if isa(r.alg.alg1, Full)
        second_moment = model[Symbol(:second_moment_, i)] = JuMP.@expression(model,
                                                                             net_X .- tgt)
    else
        second_moment = model[Symbol(:second_lower_moment_, i)] = JuMP.@variable(model,
                                                                                 [1:T],
                                                                                 (lower_bound = 0))
        model[Symbol(:csecond_lower_moment_mar_, i)] = JuMP.@constraint(model,
                                                                        sc * ((net_X +
                                                                               second_moment) .-
                                                                              tgt) >= 0)
    end
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    second_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        set_second_moment_risk!(model, r.alg.alg2, i, factor, second_moment, key,
                                :tsecond_moment_risk_, :csecond_moment_rsoc_,
                                sqrt_second_moment)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        second_moment = model[Symbol(:scaled_second_moment_, i)] = JuMP.@expression(model,
                                                                                    wi .*
                                                                                    second_moment)
        set_second_moment_risk!(model, r.alg.alg2, i, factor, second_moment, key,
                                :tsecond_moment_risk_, :csecond_moment_rsoc_,
                                sqrt_second_moment)
    end
    model[Symbol(:csqrt_second_moment_soc_, i)] = JuMP.@constraint(model,
                                                                   [sc * sqrt_second_moment
                                                                    sc * second_moment] in
                                                                   JuMP.SecondOrderCone())
    ub = second_moment_bound_val(r.alg.alg2, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_moment, ub, bound_key,
                                             second_moment_risk, r.settings)
    return second_moment_risk
end
