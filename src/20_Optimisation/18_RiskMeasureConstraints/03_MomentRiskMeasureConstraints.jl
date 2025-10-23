function calc_risk_constraint_target(::LowOrderMoment{<:Any, <:Any, Nothing, <:Any},
                                     w::AbstractVector, mu::AbstractVector, args...)
    return dot(w, mu)
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}, w::AbstractVector, args...)
    return dot(w, r.mu)
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     w::AbstractVector, ::Any, k)
    return dot(w, r.mu.v) + r.mu.s * k
end
function calc_risk_constraint_target(r::LowOrderMoment{<:Any, <:Any, <:Real, <:Any}, ::Any,
                                     ::Any, k)
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
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    flm = model[Symbol(:flm_, i)] = @variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    flm_risk = model[key] = if isnothing(wi)
        @expression(model, mean(flm))
    else
        @expression(model, mean(flm, wi))
    end
    model[Symbol(:cflm_mar_, i)] = @constraint(model, sc * ((net_X + flm) .- target) >= 0)
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
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    mad = model[Symbol(:mad_, i)] = @variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    mad_risk = model[Symbol(:mad_risk_, i)] = if isnothing(wi)
        @expression(model, 2 * mean(mad))
    else
        @expression(model, 2 * mean(mad, wi))
    end
    model[Symbol(:cmar_mad_, i)] = @constraint(model, sc * ((net_X + mad) .- target) >= 0)
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, key)
    return mad_risk
end
function set_second_moment_risk!(model::JuMP.Model, ::QuadRiskExpr, ::Any, wi, factor::Real,
                                 second_moment, key::Symbol, args...)
    return model[key] = @expression(model, factor * dot(second_moment, second_moment)),
                        sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::RSOCRiskExpr, i::Any, wi,
                                 factor::Real, second_moment, key::Symbol, keyt::Symbol,
                                 keyc::Symbol, args...)
    if !isnothing(wi)
        second_moment = model[Symbol(:scaled_second_lower_moment_, i)] = @expression(model,
                                                                                     dot(wi,
                                                                                         second_lower_moment))
    end
    sc = model[:sc]
    tsecond_moment = model[Symbol(keyt, i)] = @variable(model)
    model[Symbol(keyc, i)] = @constraint(model,
                                         [sc * tsecond_moment;
                                          0.5;
                                          sc * second_moment] in RotatedSecondOrderCone())
    return model[key] = @expression(model, factor * tsecond_moment), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SquaredSOCRiskExpr, i::Any, ::Any,
                                 factor::Real, second_moment, key::Symbol, keyt::Symbol,
                                 keyc::Symbol, tsecond_moment::AbstractJuMPScalar)
    return model[key] = @expression(model, factor * tsecond_moment^2), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SOCRiskExpr, i::Any, ::Any,
                                 factor::Real, second_moment, key::Symbol, keyt::Symbol,
                                 keyc::Symbol, tsecond_moment::AbstractJuMPScalar)
    factor = sqrt(factor)
    return model[key] = @expression(model, factor * tsecond_moment), factor
end
"""
"""
function second_moment_bound_val(alg::SecondMomentFormulation, ub::Frontier, factor::Real)
    return _Frontier(; N = ub.N, factor = inv(factor), flag = isa(alg, SOCRiskExpr))
end
function second_moment_bound_val(alg::SecondMomentFormulation, ub::AbstractVector,
                                 factor::Real)
    return inv(factor) * (isa(alg, SOCRiskExpr) ? ub : sqrt.(ub))
end
function second_moment_bound_val(alg::SecondMomentFormulation, ub::Real, factor::Real)
    return inv(factor) * (isa(alg, SOCRiskExpr) ? ub : sqrt(ub))
end
function second_moment_bound_val(::Any, ::Nothing, ::Any)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:StandardisedLowOrderMoment{<:Any,
                                                                              <:SecondLowerMoment}},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:second_lower_moment_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    bound_key = Symbol(:sqrt_second_lower_moment_, i)
    sqrt_second_lower_moment, second_lower_moment = model[bound_key], model[Symbol(:second_lower_moment_, i)] = @variables(model,
                                                                                                                           begin
                                                                                                                               ()
                                                                                                                               [1:T],
                                                                                                                               (lower_bound = 0)
                                                                                                                           end)
    wi = nothing_scalar_array_factory(r.w, pr.w)
    second_lower_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        model[Symbol(:csqrt_second_lower_moment_soc_, i)] = @constraint(model,
                                                                        [sc *
                                                                         sqrt_second_lower_moment
                                                                         sc *
                                                                         second_lower_moment] in
                                                                        SecondOrderCone())
        set_second_moment_risk!(model, r.alg.alg.alg, i, nothing, factor,
                                second_lower_moment, key, :tsecond_lower_moment_,
                                :csecond_lower_moment_rsoc_, sqrt_second_lower_moment)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        model[Symbol(:csqrt_second_lower_moment_soc_, i)] = @constraints(model,
                                                                         [sc *
                                                                          sqrt_second_lower_moment
                                                                          sc * wi .*
                                                                          second_lower_moment] in
                                                                         SecondOrderCone())
        set_second_moment_risk!(model, r.alg.alg.alg, i, wi, factor, second_lower_moment,
                                key, :tsecond_lower_moment_, :csecond_lower_moment_rsoc_,
                                sqrt_second_lower_moment)
    end
    model[Symbol(:csecond_lower_moment_mar_, i)] = @constraint(model,
                                                               sc * ((net_X +
                                                                      second_lower_moment) .-
                                                                     target) >= 0)
    ub = second_moment_bound_val(r.alg.alg.alg, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_lower_moment, ub,
                                             bound_key, second_lower_moment_risk,
                                             r.settings)
    return second_lower_moment_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:StandardisedLowOrderMoment{<:Any,
                                                                              <:SecondCentralMoment}},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:second_central_moment_risk_, i)
    w = model[:w]
    k = model[:k]
    target = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    bound_key = Symbol(:sqrt_second_central_moment_, i)
    sqrt_second_central_moment = model[bound_key] = @variable(model)
    second_central_moment = model[Symbol(:second_central_moment_, i)] = @expression(model,
                                                                                    net_X .-
                                                                                    target)
    sc = model[:sc]
    wi = nothing_scalar_array_factory(r.w, pr.w)
    second_central_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        model[Symbol(:csqrt_second_central_moment_soc_, i)] = @constraint(model,
                                                                          [sc *
                                                                           sqrt_second_central_moment
                                                                           sc *
                                                                           second_central_moment] in
                                                                          SecondOrderCone())
        set_second_moment_risk!(model, r.alg.alg.alg, i, nothing, factor,
                                second_central_moment, key, :tsecond_central_moment_,
                                :csecond_central_moment_rsoc_, sqrt_second_central_moment)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        model[Symbol(:csqrt_second_central_moment_soc_, i)] = @constraint(model,
                                                                          [sc *
                                                                           sqrt_second_central_moment
                                                                           sc * wi .*
                                                                           second_central_moment] in
                                                                          SecondOrderCone())
        set_second_moment_risk!(model, r.alg.alg.alg, i, nothing, factor,
                                second_central_moment, key, :tsecond_central_moment_,
                                :csecond_central_moment_rsoc_, sqrt_second_central_moment)
    end

    ub = second_moment_bound_val(r.alg.alg.alg, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_central_moment, ub,
                                             bound_key, second_central_moment_risk,
                                             r.settings)
    return second_central_moment_risk
end
