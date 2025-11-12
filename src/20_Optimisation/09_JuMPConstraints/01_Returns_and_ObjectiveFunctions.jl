"""
"""
struct ArithmeticReturn{T1, T2, T3} <: JuMPReturnsEstimator
    ucs::T1
    lb::T2
    mu::T3
    function ArithmeticReturn(ucs::Option{<:UcSE_UcS}, lb::Option{<:RkRtBounds},
                              mu::Option{<:Num_VecNum})
        if isa(ucs, EllipseUncertaintySet)
            @argcheck(isa(ucs,
                          EllipseUncertaintySet{<:Any, <:Any, <:MuEllipseUncertaintySet}))
        end
        if isa(lb, Number)
            @argcheck(isfinite(lb))
        elseif isa(lb, VecNum)
            @argcheck(!isempty(lb))
            @argcheck(all(isfinite, lb))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
            @argcheck(all(isfinite, mu))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu))
        end
        return new{typeof(ucs), typeof(lb), typeof(mu)}(ucs, lb, mu)
    end
end
function ArithmeticReturn(; ucs::Option{<:UcSE_UcS} = nothing,
                          lb::Option{<:RkRtBounds} = nothing,
                          mu::Option{<:Num_VecNum} = nothing)
    return ArithmeticReturn(ucs, lb, mu)
end
function jump_returns_view(r::ArithmeticReturn, i, args...)
    uset = ucs_view(r.ucs, i)
    mu = nothing_scalar_array_view(r.mu, i)
    return ArithmeticReturn(; ucs = uset, lb = r.lb, mu = mu)
end
function no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool = true)
    return flag ? ArithmeticReturn(; ucs = r.ucs, mu = r.mu) : ArithmeticReturn()
end
"""
"""
struct KellyReturn{T1, T2} <: JuMPReturnsEstimator
    w::T1
    lb::T2
    function KellyReturn(w::Option{<:AbstractWeights}, lb::Option{<:RkRtBounds})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(lb, Number)
            @argcheck(isfinite(lb))
        elseif isa(lb, VecNum)
            @argcheck(!isempty(lb))
            @argcheck(all(isfinite, lb))
        end
        return new{typeof(w), typeof(lb)}(w, lb)
    end
end
function KellyReturn(; w::Option{<:AbstractWeights} = nothing,
                     lb::Option{<:RkRtBounds} = nothing)
    return KellyReturn(w, lb)
end
function no_bounds_returns_estimator(r::KellyReturn, args...)
    return KellyReturn(; w = r.w)
end
#=
mutable struct AKelly <: RetType
    formulation::VarianceFormulation
    a_rc::Union{<:MatNum, Nothing}
    b_rc::Union{<:AbstractVector, Nothing}
end
function AKelly(; formulation::VarianceFormulation = SOC(),
                a_rc::Union{<:MatNum, Nothing} = nothing,
                b_rc::Union{<:AbstractVector, Nothing} = nothing)
    if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
        @smart_assert(size(a_rc, 1) == length(b_rc))
    end
    return AKelly(formulation, a_rc, b_rc)
end
function Base.setproperty!(obj::AKelly, sym::Symbol, val)
    if sym == :a_rc
        if !isnothing(val) && !isnothing(obj.b_rc) && !isempty(val) && !isempty(obj.b_rc)
            @smart_assert(size(val, 1) == length(obj.b_rc))
        end
    elseif sym == :b_rc
        if !isnothing(val) && !isnothing(obj.a_rc) && !isempty(val) && !isempty(obj.a_rc)
            @smart_assert(size(obj.a_rc, 1) == length(val))
        end
    end
    return setfield!(obj, sym, val)
end
function set_objective_function(port, ::Sharpe, ::Union{AKelly, EKelly}, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = model[:ret]
    @expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(port, obj_func, -1, custom_obj)
    @objective(model, Max, scale_obj * obj_func)
    return nothing
end
function return_constraints(port, type, ::Any, kelly::AKelly, mu, sigma, returns,
                            kelly_approx_idx)
    if isempty(mu)
        return nothing
    end

    model = port.model
    get_fees(model)
    w = model[:w]
    fees = model[:fees]
    if isnothing(kelly_approx_idx) ||
       isempty(kelly_approx_idx) ||
       iszero(kelly_approx_idx[1])
        if !haskey(model, :variance_risk)
            a_rc = kelly.a_rc
            b_rc = kelly.b_rc
            sdp_rc_variance(model, type, a_rc, b_rc)
            calc_variance_risk(get_ntwk_clust_type(port, a_rc, b_rc), kelly.formulation,
                               model, mu, sigma, returns)
        end
        variance_risk = model[:variance_risk]
        @expression(model, ret, dot(mu, w) - fees - 0.5 * variance_risk)
    else
        variance_risk = model[:variance_risk]
        @expression(model, ret,
                    dot(mu, w) - fees - 0.5 * variance_risk[kelly_approx_idx[1]])
    end

    return_bounds(port)

    return nothing
end
function return_constraints(port, type, obj::Sharpe, kelly::AKelly, mu, sigma, returns,
                            kelly_approx_idx)
    a_rc = kelly.a_rc
    b_rc = kelly.b_rc
    sdp_rc_variance(port.model, type, a_rc, b_rc)
    return_sharpe_akelly_constraints(port, type, obj, kelly,
                                     get_ntwk_clust_type(port, a_rc, b_rc), mu, sigma,
                                     returns, kelly_approx_idx)
    return nothing
end
function return_sharpe_akelly_constraints(port, type, obj::Sharpe, kelly::AKelly,
                                          adjacency_constraint::Union{NoAdj, IP}, mu, sigma,
                                          returns, kelly_approx_idx)
    if isempty(mu)
        return nothing
    end

    model = port.model
    get_fees(model)
    scale_constr = model[:scale_constr]
    w = model[:w]
    k = model[:k]
    fees = model[:fees]
    ohf = model[:ohf]
    risk = model[:risk]
    rf = obj.rf
    @variable(model, tapprox_kelly)
    @constraint(model, constr_sr_akelly_risk, scale_constr * risk <= scale_constr * ohf)
    @expression(model, ret, dot(mu, w) - fees - 0.5 * tapprox_kelly - k * rf)
    if isnothing(kelly_approx_idx) ||
       isempty(kelly_approx_idx) ||
       iszero(kelly_approx_idx[1])
        if !haskey(model, :variance_risk)
            calc_variance_risk(adjacency_constraint, kelly.formulation, model, mu, sigma,
                               returns)
        end
        dev = model[:dev]
        @constraint(model, constr_sr_akelly_ret,
                    [scale_constr * (k + tapprox_kelly)
                     scale_constr * 2 * dev
                     scale_constr * (k - tapprox_kelly)] ∈ SecondOrderCone())
    else
        dev = model[:dev]
        @constraint(model, constr_sr_akelly_ret,
                    [scale_constr * (k + tapprox_kelly)
                     scale_constr * 2 * dev[kelly_approx_idx[1]]
                     scale_constr * (k - tapprox_kelly)] ∈ SecondOrderCone())
    end
    return_bounds(port)

    return nothing
end
function return_sharpe_akelly_constraints(port, type, obj::Sharpe, ::AKelly, ::SDP, ::Any,
                                          ::Any, returns, ::Any)
    return_constraints(port, type, obj, EKelly(), nothing, nothing, returns, nothing)
    return nothing
end
=#
for r in traverse_concrete_subtypes(JuMPReturnsEstimator)
    eval(quote
             function bounds_returns_estimator(r::$(r), lb::Number)
                 pnames = Tuple(setdiff(propertynames(r), (:lb,)))
                 return if isempty(pnames)
                     $(r)(; lb = lb)
                 else
                     $(r)(; lb = lb, NamedTuple{pnames}(getproperty.(r, pnames))...)
                 end
             end
         end)
end
function jump_returns_factory(r::KellyReturn, pr::AbstractPriorResult, args...; kwargs...)
    return KellyReturn(; w = nothing_scalar_array_factory(r.w, pr.w), lb = r.lb)
end
struct MinimumRisk <: ObjectiveFunction end
struct MaximumUtility{T1} <: ObjectiveFunction
    l::T1
    function MaximumUtility(l::Number)
        @argcheck(l >= zero(l))
        return new{typeof(l)}(l)
    end
end
function MaximumUtility(; l::Number = 2)
    return MaximumUtility(l)
end
struct MaximumRatio{T1, T2} <: ObjectiveFunction
    rf::T1
    ohf::T2
    function MaximumRatio(rf::Number, ohf::Option{<:Number})
        if !isnothing(ohf)
            @argcheck(ohf > zero(ohf))
        end
        return new{typeof(rf), typeof(ohf)}(rf, ohf)
    end
end
function MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing)
    return MaximumRatio(rf, ohf)
end
struct MaximumReturn <: ObjectiveFunction end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, mu::Num_VecNum,
                                             obj::MaximumRatio)
    ohf = if isnothing(obj.ohf)
        min(1e3, max(1e-3, mean(abs.(mu))))
    else
        @argcheck(obj.ohf > zero(obj.ohf))
        obj.ohf
    end
    @expression(model, ohf, ohf)
    @variable(model, k >= 0)
    return nothing
end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, args...)
    @expression(model, k, 1)
    return nothing
end
function set_return_bounds!(args...)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, lb::Number)
    sc = model[:sc]
    k = model[:k]
    ret = model[:ret]
    @constraint(model, ret_lb, sc * (ret - lb * k) >= 0)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, lb::Front_NumVec)
    @expression(model, ret_frontier, lb)
    return nothing
end
function set_max_ratio_return_constraints!(args...)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio,
                                           mu::Vec_VecNum)
    sc = model[:sc]
    k = model[:k]
    ohf = model[:ohf]
    ret = model[:ret]
    rf = obj.rf
    if haskey(model, :bucs_w) || haskey(model, :t_eucs_gw) || all(x -> x <= rf, mu)
        risk = model[:risk]
        @constraint(model, sr_risk, sc * (risk - ohf) <= 0)
    else
        @constraint(model, sr_ret, sc * (ret - rf * k - ohf) == 0)
    end
    return nothing
end
function add_fees_to_ret!(model::JuMP.Model, ret)
    if !haskey(model, :fees)
        return nothing
    end
    fees = model[:fees]
    add_to_expression!(ret, -fees)
    return nothing
end
function add_market_impact_cost!(model::JuMP.Model, ret)
    if !haskey(model, :wip)
        return nothing
    end
    cost_bgt_expr = model[:cost_bgt_expr]
    add_to_expression!(ret, -cost_bgt_expr)
    return nothing
end
function set_return_constraints!(model::JuMP.Model,
                                 pret::ArithmeticReturn{Nothing, <:Any, <:Any},
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    w = model[:w]
    lb = pret.lb
    mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    @expression(model, ret, dot_scalar(mu, w))
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    set_max_ratio_return_constraints!(model, obj, mu)
    set_return_bounds!(model, lb)
    return nothing
end
function set_ucs_return_constraints!(model::JuMP.Model, ucs::BoxUncertaintySet,
                                     mu::Num_VecNum)
    sc = model[:sc]
    w = model[:w]
    N = length(w)
    d_mu = (ucs.ub - ucs.lb) * 0.5
    @variable(model, bucs_w[1:N])
    @constraint(model, bucs_ret[i = 1:N], [sc * bucs_w[i]; sc * w[i]] in MOI.NormOneCone(2))
    @expression(model, ret, dot_scalar(mu, w) - dot(d_mu, bucs_w))
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
function set_ucs_return_constraints!(model::JuMP.Model, ucs::EllipseUncertaintySet,
                                     mu::Num_VecNum)
    sc = model[:sc]
    w = model[:w]
    G = cholesky(ucs.sigma).U
    k = ucs.k
    @expression(model, x_eucs_w, G * w)
    @variable(model, t_eucs_gw)
    @constraint(model, eucs_ret, [sc * t_eucs_gw; sc * x_eucs_w] in SecondOrderCone())
    @expression(model, ret, dot_scalar(mu, w) - k * t_eucs_gw)
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
function set_return_constraints!(model::JuMP.Model,
                                 pret::ArithmeticReturn{<:UcSE_UcS, <:Any, <:Any},
                                 obj::ObjectiveFunction, pr::AbstractPriorResult;
                                 rd::ReturnsResult, kwargs...)
    lb = pret.lb
    ucs = pret.ucs
    mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    set_ucs_return_constraints!(model, mu_ucs(ucs, rd; kwargs...), mu)
    set_max_ratio_return_constraints!(model, obj, mu)
    set_return_bounds!(model, lb)
    return nothing
end
function set_max_ratio_kelly_return_constraints!(args...)
    return nothing
end
function set_max_ratio_kelly_return_constraints!(model::JuMP.Model, ::MaximumRatio)
    sc = model[:sc]
    ohf = model[:ohf]
    risk = model[:risk]
    @constraint(model, sr_ekelly_risk, sc * (risk - ohf) <= 0)
end
function set_return_constraints!(model::JuMP.Model, pret::KellyReturn,
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    k = model[:k]
    sc = model[:sc]
    lb = pret.lb
    X = pr.X
    X = set_portfolio_returns!(model, X)
    T = length(X)
    @variable(model, t_ekelly[1:T])
    wi = nothing_scalar_array_factory(pret.w, pr.w)
    if isnothing(wi)
        @expression(model, ret, mean(t_ekelly))
    else
        @expression(model, ret, mean(t_ekelly, wi))
    end
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    set_max_ratio_kelly_return_constraints!(model, obj)
    @expression(model, kret, k .+ X)
    @constraint(model, ekelly_ret[i = 1:T],
                [sc * t_ekelly[i], sc * k, sc * kret[i]] in MOI.ExponentialCone())
    set_return_bounds!(model, lb)
    return nothing
end
function add_to_objective_penalty!(model::JuMP.Model, expr)
    op = if !haskey(model, :op)
        @expression(model, op, zero(AffExpr))
    else
        model[:op]
    end
    add_to_expression!(op, expr)
    return nothing
end
function add_penalty_to_objective!(model::JuMP.Model, factor::Integer, expr)
    if !haskey(model, :op)
        return nothing
    end
    op = model[:op]
    add_to_expression!(expr, factor, op)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MinimumRisk,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult)
    so = model[:so]
    risk = model[:risk]
    @expression(model, obj_expr, risk)
    add_penalty_to_objective!(model, 1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr)
    @objective(model, Min, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumUtility,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult)
    so = model[:so]
    ret = model[:ret]
    risk = model[:risk]
    l = obj.l
    @expression(model, obj_expr, ret - l * risk)
    add_penalty_to_objective!(model, -1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr)
    @objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::KellyReturn,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult)
    so = model[:so]
    ret = model[:ret]
    k = model[:k]
    rf = obj.rf
    @expression(model, obj_expr, ret - rf * k)
    add_penalty_to_objective!(model, -1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr)
    @objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult)
    so = model[:so]
    if haskey(model, :sr_risk)
        ret = model[:ret]
        k = model[:k]
        rf = obj.rf
        @expression(model, obj_expr, ret - rf * k)
        add_penalty_to_objective!(model, -1, obj_expr)
        add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr)
        @objective(model, Max, so * obj_expr)
    else
        risk = model[:risk]
        @expression(model, obj_expr, risk)
        add_penalty_to_objective!(model, 1, obj_expr)
        add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr)
        @objective(model, Min, so * obj_expr)
    end
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumReturn,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult)
    so = model[:so]
    ret = model[:ret]
    @expression(model, obj_expr, ret)
    add_penalty_to_objective!(model, -1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr)
    @objective(model, Max, so * obj_expr)
    return nothing
end

export ArithmeticReturn, KellyReturn, MinimumRisk, MaximumUtility, MaximumRatio,
       MaximumReturn, bounds_returns_estimator
