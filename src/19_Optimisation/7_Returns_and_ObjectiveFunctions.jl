struct ArithmeticReturn{T1 <: Union{Nothing, <:AbstractUncertaintySetResult,
                                    <:AbstractUncertaintySetEstimator},
                        T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:Frontier}} <:
       JuMPReturnsEstimator
    ucs::T1
    lb::T2
end
function ArithmeticReturn(;
                          ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                     <:AbstractUncertaintySetEstimator} = nothing,
                          lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:Frontier} = nothing)
    if isa(ucs, EllipseUncertaintySetResult)
        @smart_assert(isa(ucs,
                          EllipseUncertaintySetResult{<:Any, <:Any,
                                                      <:MuEllipseUncertaintySetResult}))
    end
    if isa(lb, Real)
        @smart_assert(isfinite(lb))
    elseif isa(lb, AbstractVector)
        @smart_assert(!isempty(lb) && all(isfinite, lb))
    end
    return ArithmeticReturn{typeof(ucs), typeof(lb)}(ucs, lb)
end
function jump_returns_view(r::ArithmeticReturn, i::AbstractVector, args...)
    uset = ucs_view(r.ucs, i)
    return ArithmeticReturn(; ucs = uset, lb = r.lb)
end
function no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool = true)
    return flag ? ArithmeticReturn(; ucs = r.ucs) : ArithmeticReturn()
end
struct KellyReturn{T1 <: Union{Nothing, <:AbstractWeights},
                   T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:Frontier}} <:
       JuMPReturnsEstimator
    w::T1
    lb::T2
end
function KellyReturn(; w::Union{Nothing, <:AbstractWeights} = nothing,
                     lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:Frontier} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    if isa(lb, Real)
        @smart_assert(isfinite(lb))
    elseif isa(lb, AbstractVector)
        @smart_assert(!isempty(lb) && all(isfinite, lb))
    end
    return KellyReturn{typeof(w), typeof(lb)}(w, lb)
end
function no_bounds_returns_estimator(r::KellyReturn, args...)
    return r
end
for r ∈ traverse_subtypes(JuMPReturnsEstimator)
    eval(quote
             function bounds_returns_estimator(r::$(r), lb::Real)
                 pnames = setdiff(propertynames(r), (:lb,))
                 return if isempty(pnames)
                     $(r)(lb)
                 else
                     $(r)(getproperty.(r, pnames)..., lb)
                 end
             end
         end)
end
function jump_returns_factory(r::KellyReturn, pr::AbstractPriorResult; kwargs...)
    return KellyReturn(; w = nothing_scalar_array_factory(r.w, pr.w), lb = r.lb)
end
struct MinimumRisk <: ObjectiveFunction end
struct MaximumUtility{T1 <: Real} <: ObjectiveFunction
    l::T1
end
function MaximumUtility(; l::Real = 2)
    @smart_assert(l >= zero(l))
    return MaximumUtility{typeof(l)}(l)
end
struct MaximumRatio{T1 <: Real, T2 <: Union{Nothing, <:Real}} <: ObjectiveFunction
    rf::T1
    ohf::T2
end
function MaximumRatio(; rf::Real = 0.0, ohf::Union{Nothing, <:Real} = nothing)
    if !isnothing(ohf)
        @smart_assert(ohf > zero(ohf))
    end
    return MaximumRatio{typeof(rf), typeof(ohf)}(rf, ohf)
end
struct MaximumReturn <: ObjectiveFunction end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, mu::AbstractVector,
                                             obj::MaximumRatio)
    ohf = if isnothing(obj.ohf)
        min(1e3, max(1e-3, mean(abs.(mu))))
    else
        @smart_assert(obj.ohf > zero(obj.ohf))
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
function set_return_bounds!(model::JuMP.Model, lb::Real)
    sc = model[:sc]
    k = model[:k]
    ret = model[:ret]
    @constraint(model, ret_lb, sc * (ret - lb * k) >= 0)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, lb::Union{<:AbstractVector, <:Frontier})
    @expression(model, ret_frontier, lb)
    return nothing
end
function set_max_ratio_return_constraints!(args...)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio,
                                           mu::AbstractVector)
    sc = model[:sc]
    k = model[:k]
    ohf = model[:ohf]
    ret = model[:ret]
    rf = obj.rf
    if haskey(model, :bucs_w) || haskey(model, :t_eucs_gw) || all(mu .<= rf)
        risk = model[:risk]
        add_to_expression!(ret, -rf, k)
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
function set_return_constraints!(model::JuMP.Model, pret::ArithmeticReturn{Nothing, <:Any},
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    w = model[:w]
    lb = pret.lb
    mu = pr.mu
    @expression(model, ret, dot(mu, w))
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    set_max_ratio_return_constraints!(model, obj, mu)
    set_return_bounds!(model, lb)
    return nothing
end
function set_ucs_return_constraints!(model::JuMP.Model, ucs::BoxUncertaintySetResult,
                                     mu::AbstractVector)
    sc = model[:sc]
    w = model[:w]
    N = length(w)
    d_mu = (ucs.ub - ucs.lb) * 0.5
    @variable(model, bucs_w[1:N])
    @constraint(model, bucs_ret[i = 1:N], [sc * bucs_w[i]; sc * w[i]] ∈ MOI.NormOneCone(2))
    @expression(model, ret, dot(mu, w) - dot(d_mu, bucs_w))
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
function set_ucs_return_constraints!(model::JuMP.Model, ucs::EllipseUncertaintySetResult,
                                     mu::AbstractVector)
    sc = model[:sc]
    w = model[:w]
    G = cholesky(ucs.sigma).U
    k = ucs.k
    @expression(model, x_eucs_w, G * w)
    @variable(model, t_eucs_gw)
    @constraint(model, eucs_ret, [sc * t_eucs_gw; sc * x_eucs_w] ∈ SecondOrderCone())
    @expression(model, ret, dot(mu, w) - k * t_eucs_gw)
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
function set_return_constraints!(model::JuMP.Model,
                                 pret::ArithmeticReturn{<:Union{<:AbstractUncertaintySetResult,
                                                                <:AbstractUncertaintySetEstimator},
                                                        <:Any}, obj::ObjectiveFunction,
                                 pr::AbstractPriorResult; kwargs...)
    lb = pret.lb
    ucs = pret.ucs
    X = pr.X
    mu = pr.mu
    set_ucs_return_constraints!(model, mu_ucs(ucs, X; kwargs...), mu)
    set_max_ratio_return_constraints!(model, obj, mu)
    set_return_bounds!(model, lb)
    return nothing
end
function set_max_ratio_kelly_return_constraints!(args...)
    return nothing
end
function set_max_ratio_kelly_return_constraints!(model::JuMP.Model, obj::MaximumRatio, k,
                                                 sc, ret)
    ohf = model[:ohf]
    risk = model[:risk]
    rf = obj.rf
    add_to_expression!(ret, -rf, k)
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
    set_max_ratio_kelly_return_constraints!(model, obj, k, sc, ret)
    @expression(model, kret, k .+ X)
    @constraint(model, ekelly_ret[i = 1:T],
                [sc * t_ekelly[i], sc * k, sc * kret[i]] ∈ MOI.ExponentialCone())
    set_return_bounds!(model, lb)
    return nothing
end
function add_penalty_to_objective!(model::JuMP.Model, sign::Integer, expr)
    if !haskey(model, :op)
        return nothing
    end
    op = model[:op]
    add_to_expression!(expr, sign, op)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MinimumRisk,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Union{Nothing, <:CustomObjective},
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
                                           cobj::Union{Nothing, <:CustomObjective},
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
                                           cobj::Union{Nothing, <:CustomObjective},
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
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Union{Nothing, <:CustomObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult)
    so = model[:so]
    if haskey(model, :sr_risk)
        ret = model[:ret]
        @expression(model, obj_expr, ret)
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
                                           cobj::Union{Nothing, <:CustomObjective},
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
       MaximumReturn
