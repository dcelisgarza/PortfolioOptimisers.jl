function calc_risk_constraint_target(::LoHiOrderMoment{<:Any, <:Any, Nothing, <:Any},
                                     w::VecNum, mu::VecNum, args...)
    return dot(w, mu)
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any},
                                     w::VecNum, args...)
    return dot(w, r.mu)
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     w::VecNum, ::Any, k)
    return dot(w, r.mu.v) + r.mu.s * k
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any},
                                     ::Any, ::Any, k)
    return r.mu * k
end
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
    flm_risk = model[key] = if isnothing(wi)
        JuMP.@expression(model, mean(flm))
    else
        JuMP.@expression(model, mean(flm, wi))
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
    mad_risk = model[Symbol(:mad_risk_, i)] = if isnothing(wi)
        JuMP.@expression(model, 2 * mean(mad))
    else
        JuMP.@expression(model, 2 * mean(mad, wi))
    end
    model[Symbol(:cmar_mad_, i)] = JuMP.@constraint(model, sc * ((net_X + mad) .- tgt) >= 0)
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, key)
    return mad_risk
end
function set_second_moment_risk!(model::JuMP.Model, ::QuadRiskExpr, ::Any, factor::Number,
                                 second_moment, key::Symbol, args...)
    return model[key] = JuMP.@expression(model, factor * dot(second_moment, second_moment)),
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
