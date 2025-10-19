abstract type BudgetConstraintEstimator <: JuMPConstraintEstimator end
abstract type BudgetEstimator <: BudgetConstraintEstimator end
abstract type BudgetCostEstimator <: BudgetConstraintEstimator end
function set_budget_costs!(args...)
    return nothing
end
struct BudgetRange{T1, T2} <: BudgetEstimator
    lb::T1
    ub::T2
    function BudgetRange(lb::Union{Nothing, <:Real}, ub::Union{Nothing, <:Real})
        lb_flag = isnothing(lb)
        ub_flag = isnothing(ub)
        @argcheck(lb_flag ⊼ ub_flag)
        if !lb_flag
            @argcheck(isfinite(lb))
        end
        if !ub_flag
            @argcheck(isfinite(ub))
        end
        if !lb_flag && !ub_flag
            @argcheck(lb <= ub)
        end
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function BudgetRange(; lb::Union{Nothing, <:Real} = 1.0, ub::Union{Nothing, <:Real} = 1.0)
    return BudgetRange(lb, ub)
end
function budget_view(bgt::Union{<:Real, <:BudgetRange}, ::Any)
    return bgt
end
function set_budget_constraints!(args...)
    return nothing
end
struct BudgetCosts{T1, T2, T3, T4, T5, T6} <: BudgetCostEstimator
    bgt::T1
    w::T2
    vp::T3
    vn::T4
    up::T5
    un::T6
    function BudgetCosts(bgt::Union{<:Real, <:BudgetRange}, w::AbstractVector{<:Real},
                         vp::Union{<:Real, <:AbstractVector{<:Real}},
                         vn::Union{<:Real, <:AbstractVector{<:Real}},
                         up::Union{<:Real, <:AbstractVector{<:Real}},
                         un::Union{<:Real, <:AbstractVector{<:Real}})
        @argcheck(!isempty(w))
        if isa(vp, AbstractVector)
            @argcheck(!isempty(vp) && all(x -> x >= zero(x), vp))
        else
            @argcheck(vp >= zero(vp))
        end
        if isa(vn, AbstractVector)
            @argcheck(!isempty(vn) && all(x -> x >= zero(x), vn))
        else
            @argcheck(vn >= zero(vn))
        end
        if isa(up, AbstractVector)
            @argcheck(!isempty(up) && all(x -> x >= zero(x), up))
        else
            @argcheck(up >= zero(up))
        end
        if isa(un, AbstractVector)
            @argcheck(!isempty(un) && all(x -> x >= zero(x), un))
        else
            @argcheck(un >= zero(un))
        end
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un)}(bgt,
                                                                                           w,
                                                                                           vp,
                                                                                           vn,
                                                                                           up,
                                                                                           un)
    end
end
function BudgetCosts(; bgt::Union{<:Real, <:BudgetRange} = 1.0, w::AbstractVector{<:Real},
                     vp::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                     vn::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                     up::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                     un::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    return BudgetCosts(bgt, w, vp, vn, up, un)
end
function budget_view(bgt::BudgetCosts, i::AbstractVector)
    w = view(bgt.w, i)
    vp = nothing_scalar_array_view(bgt.vp, i)
    vn = nothing_scalar_array_view(bgt.vn, i)
    up = nothing_scalar_array_view(bgt.up, i)
    un = nothing_scalar_array_view(bgt.un, i)
    return BudgetCosts(; bgt = bgt.bgt, w = w, vp = vp, vn = vn, up = up, un = un)
end
struct BudgetMarketImpact{T1, T2, T3, T4, T5, T6, T7} <: BudgetCostEstimator
    bgt::T1
    w::T2
    vp::T3
    vn::T4
    up::T5
    un::T6
    beta::T7
    function BudgetMarketImpact(bgt::Union{<:Real, <:BudgetRange},
                                w::AbstractVector{<:Real},
                                vp::Union{<:Real, <:AbstractVector{<:Real}},
                                vn::Union{<:Real, <:AbstractVector{<:Real}},
                                up::Union{<:Real, <:AbstractVector{<:Real}},
                                un::Union{<:Real, <:AbstractVector{<:Real}}, beta::Real)
        @argcheck(!isempty(w))
        if isa(vp, AbstractVector)
            @argcheck(!isempty(vp) && all(x -> x >= zero(x), vp))
        else
            @argcheck(vp >= zero(vp))
        end
        if isa(vn, AbstractVector)
            @argcheck(!isempty(vn) && all(x -> x >= zero(x), vn))
        else
            @argcheck(vn >= zero(vn))
        end
        if isa(up, AbstractVector)
            @argcheck(!isempty(up) && all(x -> x >= zero(x), up))
        else
            @argcheck(up >= zero(up))
        end
        if isa(un, AbstractVector)
            @argcheck(!isempty(un) && all(x -> x >= zero(x), un))
        else
            @argcheck(un >= zero(un))
        end
        @argcheck(zero(beta) <= beta <= one(beta),
                  DomainError(beta,
                              range_msg("`beta`", zero(beta), one(beta), nothing, true,
                                        true) * "."))
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un),
                   typeof(beta)}(bgt, w, vp, vn, up, un, beta)
    end
end
function BudgetMarketImpact(; bgt::Union{<:Real, <:BudgetRange} = 1.0,
                            w::AbstractVector{<:Real},
                            vp::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                            vn::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                            up::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                            un::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                            beta::Real = 2 / 3)
    return BudgetMarketImpact(bgt, w, vp, vn, up, un, beta)
end
function budget_view(bgt::BudgetMarketImpact, i::AbstractVector)
    w = view(bgt.w, i)
    vp = nothing_scalar_array_view(bgt.vp, i)
    vn = nothing_scalar_array_view(bgt.vn, i)
    up = nothing_scalar_array_view(bgt.up, i)
    un = nothing_scalar_array_view(bgt.un, i)
    return BudgetMarketImpact(; bgt = bgt.bgt, w = w, vp = vp, vn = vn, up = up, un = un,
                              beta = bgt.beta)
end
function set_budget_constraints!(model::JuMP.Model, val::Real, w::AbstractVector)
    k = model[:k]
    sc = model[:sc]
    @constraint(model, bgt, sc * (sum(w) - k * val) == 0)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::AbstractVector)
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    ub = bgt.ub
    if !isnothing(lb)
        @constraint(model, bgt_lb, sc * (sum(w) - k * lb) >= 0)
    end
    if !isnothing(ub)
        @constraint(model, bgt_ub, sc * (sum(w) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(args...)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lbgt, sc * (sum(lw) - k * bgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::Real)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, sbgt, sc * (sum(sw) - k * sbgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::Real)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    @constraints(model, begin
                     lbgt, sc * (sum(lw) - k * (bgt + sbgt)) == 0
                     sbgt, sc * (sum(sw) - k * sbgt) == 0
                 end)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        @constraint(model, lbgt_lb, sc * (sum(lw) - k * lb) >= 0)
    end
    ub = bgt.ub
    if !isnothing(ub)
        @constraint(model, lbgt_ub, sc * (sum(lw) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::BudgetRange)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        @constraint(model, sbgt_lb, sc * (sum(sw) - k * lb) >= 0)
    end
    ub = sbgt.ub
    if !isnothing(ub)
        @constraint(model, sbgt_ub, sc * (sum(sw) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, sbgt::Real)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        @constraint(model, lbgt_lb, sc * (sum(lw) - k * (lb + sbgt)) >= 0)
    end
    ub = bgt.ub
    if !isnothing(ub)
        @constraint(model, lbgt_ub, sc * (sum(lw) - k * (ub + sbgt)) <= 0)
    end
    @constraint(model, sbgt, sc * (sum(sw) - k * sbgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::BudgetRange)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        @constraints(model, begin
                         lbgt_lb, sc * (sum(lw) - k * (lb + bgt)) >= 0
                         sbgt_lb, sc * (sum(sw) - k * lb) >= 0
                     end)
    end
    ub = sbgt.ub
    if !isnothing(ub)
        @constraints(model, begin
                         lbgt_ub, sc * (sum(lw) - k * (ub + bgt)) <= 0
                         sbgt_ub, sc * (sum(sw) - k * ub) <= 0
                     end)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::BudgetRange)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    slb = sbgt.lb
    lb_flag = isnothing(lb)
    slb_flag = isnothing(slb)
    if !lb_flag && !slb_flag
        @constraints(model, begin
                         lbgt_lb, sc * (sum(lw) - k * (lb + slb)) >= 0
                         sbgt_lb, sc * (sum(sw) - k * slb) >= 0
                     end)
    elseif !lb_flag && slb_flag
        @constraint(model, lbgt_lb, sc * (sum(lw) - k * lb) >= 0)
    elseif lb_flag && !slb_flag
        @constraint(model, sbgt_lb, sc * (sum(sw) - k * slb) >= 0)
    end
    ub = bgt.ub
    sub = sbgt.ub
    ub_flag = isnothing(ub)
    sub_flag = isnothing(sub)
    if !ub_flag && !sub_flag
        @constraints(model, begin
                         lbgt_ub, sc * (sum(lw) - k * (ub + sub)) <= 0
                         sbgt_ub, sc * (sum(sw) - k * sub) <= 0
                     end)
    elseif !ub_flag && sub_flag
        @constraint(model, lbgt_ub, sc * (sum(lw) - k * ub) <= 0)
    elseif ub_flag && !sub_flag
        @constraint(model, sbgt_ub, sc * (sum(sw) - k * sub) <= 0)
    end
    return nothing
end
function set_cost_budget_constraints!(model::JuMP.Model,
                                      vp::Union{<:Real, <:AbstractVector{<:Real}},
                                      vn::Union{<:Real, <:AbstractVector{<:Real}},
                                      val::Real, w::AbstractVector)
    k = model[:k]
    sc = model[:sc]
    wp = model[:wp]
    wn = model[:wn]
    @expression(model, cost_bgt_expr, dot_scalar(vp, wp) + dot_scalar(vn, wn))
    @constraint(model, cost_bgt, sc * (sum(w) + cost_bgt_expr - k * val) == 0)
    return nothing
end
function set_cost_budget_constraints!(model::JuMP.Model,
                                      vp::Union{<:Real, <:AbstractVector{<:Real}},
                                      vn::Union{<:Real, <:AbstractVector{<:Real}},
                                      bgt::BudgetRange, w::AbstractVector)
    k = model[:k]
    sc = model[:sc]
    wp = model[:wp]
    wn = model[:wn]
    lb = bgt.lb
    ub = bgt.ub
    @expression(model, cost_bgt_expr, dot_scalar(vp, wp) + dot_scalar(vn, wn))
    if !isnothing(lb)
        @constraint(model, cost_bgt_lb, sc * (sum(w) + cost_bgt_expr - k * lb) >= 0)
    end
    if !isnothing(ub)
        @constraint(model, cost_bgt_ub, sc * (sum(w) + cost_bgt_expr - k * ub) <= 0)
    end
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetCosts, w::AbstractVector)
    wb = bgt.w
    up = bgt.up
    un = bgt.un
    sc = model[:sc]
    N = length(w)
    @variables(model, begin
                   wp[1:N], (lower_bound = 0)
                   wn[1:N], (lower_bound = 0)
               end)
    @constraints(model, begin
                     sc * (wp ⊖ up) <= 0
                     sc * (wn ⊖ un) <= 0
                     sc * (w - wb - wp + wn) == 0
                 end)
    set_cost_budget_constraints!(model, bgt.vp, bgt.vn, bgt.bgt, w)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetMarketImpact,
                                 w::AbstractVector)
    wb = bgt.w
    up = bgt.up
    un = bgt.un
    beta = bgt.beta
    sc = model[:sc]
    N = length(w)
    @variables(model, begin
                   wp[1:N], (lower_bound = 0)
                   wn[1:N], (lower_bound = 0)
                   wip[1:N]
                   win[1:N]
               end)
    @constraints(model, begin
                     sc * (wp ⊖ up) <= 0
                     sc * (wn ⊖ un) <= 0
                     sc * (w - wb - wp + wn) == 0
                     [i = 1:N], [sc * wip[i], sc, sc * wp[i]] in MOI.PowerCone(beta)
                     [i = 1:N], [sc * win[i], sc, sc * wn[i]] in MOI.PowerCone(beta)
                 end)
    set_cost_budget_constraints!(model, wip, win, bgt.bgt, w)
    return nothing
end

export BudgetRange, BudgetCosts, BudgetMarketImpact
