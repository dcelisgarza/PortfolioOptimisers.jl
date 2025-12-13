abstract type BudgetConstraintEstimator <: JuMPConstraintEstimator end
const Num_BgtCE = Union{<:Number, <:BudgetConstraintEstimator}
abstract type BudgetEstimator <: BudgetConstraintEstimator end
abstract type BudgetCostEstimator <: BudgetConstraintEstimator end
function set_budget_costs!(args...)
    return nothing
end
struct BudgetRange{T1, T2} <: BudgetEstimator
    lb::T1
    ub::T2
    function BudgetRange(lb::Option{<:Number}, ub::Option{<:Number})
        lb_flag = isnothing(lb)
        ub_flag = isnothing(ub)
        @argcheck(!(lb_flag && ub_flag))
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
function BudgetRange(; lb::Option{<:Number} = 1.0, ub::Option{<:Number} = 1.0)
    return BudgetRange(lb, ub)
end
const Num_BgtRg = Union{<:Number, <:BudgetRange}
function budget_view(bgt::Num_BgtRg, ::Any)
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
    function BudgetCosts(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                         up::Num_VecNum, un::Num_VecNum)
        @argcheck(!isempty(w))
        if isa(vp, VecNum)
            @argcheck(!isempty(vp))
            @argcheck(all(x -> zero(x) <= x, vp))
        else
            @argcheck(vp >= zero(vp))
        end
        if isa(vn, VecNum)
            @argcheck(!isempty(vn))
            @argcheck(all(x -> zero(x) <= x, vn))
        else
            @argcheck(vn >= zero(vn))
        end
        if isa(up, VecNum)
            @argcheck(!isempty(up))
            @argcheck(all(x -> zero(x) <= x, up))
        else
            @argcheck(up >= zero(up))
        end
        if isa(un, VecNum)
            @argcheck(!isempty(un))
            @argcheck(all(x -> zero(x) <= x, un))
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
function BudgetCosts(; bgt::Num_BgtRg = 1.0, w::VecNum, vp::Num_VecNum = 1.0,
                     vn::Num_VecNum = 1.0, up::Num_VecNum = 1.0, un::Num_VecNum = 1.0)
    return BudgetCosts(bgt, w, vp, vn, up, un)
end
function budget_view(bgt::BudgetCosts, i)
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
    function BudgetMarketImpact(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                                up::Num_VecNum, un::Num_VecNum, beta::Number)
        @argcheck(!isempty(w))
        if isa(vp, VecNum)
            @argcheck(!isempty(vp))
            @argcheck(all(x -> zero(x) <= x, vp))
        else
            @argcheck(vp >= zero(vp))
        end
        if isa(vn, VecNum)
            @argcheck(!isempty(vn))
            @argcheck(all(x -> zero(x) <= x, vn))
        else
            @argcheck(vn >= zero(vn))
        end
        if isa(up, VecNum)
            @argcheck(!isempty(up))
            @argcheck(all(x -> zero(x) <= x, up))
        else
            @argcheck(up >= zero(up))
        end
        if isa(un, VecNum)
            @argcheck(!isempty(un))
            @argcheck(all(x -> zero(x) <= x, un))
        else
            @argcheck(un >= zero(un))
        end
        @argcheck(zero(beta) <= beta <= one(beta), DomainError)
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un),
                   typeof(beta)}(bgt, w, vp, vn, up, un, beta)
    end
end
function BudgetMarketImpact(; bgt::Num_BgtRg = 1.0, w::VecNum, vp::Num_VecNum = 1.0,
                            vn::Num_VecNum = 1.0, up::Num_VecNum = 1.0,
                            un::Num_VecNum = 1.0, beta::Number = 2 / 3)
    return BudgetMarketImpact(bgt, w, vp, vn, up, un, beta)
end
function budget_view(bgt::BudgetMarketImpact, i)
    w = view(bgt.w, i)
    vp = nothing_scalar_array_view(bgt.vp, i)
    vn = nothing_scalar_array_view(bgt.vn, i)
    up = nothing_scalar_array_view(bgt.up, i)
    un = nothing_scalar_array_view(bgt.un, i)
    return BudgetMarketImpact(; bgt = bgt.bgt, w = w, vp = vp, vn = vn, up = up, un = un,
                              beta = bgt.beta)
end
function set_budget_constraints!(model::JuMP.Model, val::Number, w::VecNum)
    k = model[:k]
    sc = model[:sc]
    @constraint(model, bgt, sc * (sum(w) - k * val) == 0)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::VecNum)
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
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lbgt, sc * (sum(lw) - k * bgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::Number)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    @constraint(model, sbgt, sc * (sum(sw) - k * sbgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, sbgt::Number)
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
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::Number)
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
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number,
                                            sbgt::BudgetRange)
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
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      val::Number, w::VecNum)
    k = model[:k]
    sc = model[:sc]
    wp = model[:wp]
    wn = model[:wn]
    @expression(model, cost_bgt_expr, dot_scalar(vp, wp) + dot_scalar(vn, wn))
    @constraint(model, cost_bgt, sc * (sum(w) + cost_bgt_expr - k * val) == 0)
    return nothing
end
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      bgt::BudgetRange, w::VecNum)
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
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetCosts, w::VecNum)
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
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetMarketImpact, w::VecNum)
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
