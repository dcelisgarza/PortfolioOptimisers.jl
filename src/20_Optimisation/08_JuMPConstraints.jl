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
function w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function w_neg_flag(wb::AbstractVector)
    return any(x -> x < zero(x), wb)
end
function w_finite_flag(wb::Real)
    return isfinite(wb)
end
function w_finite_flag(wb::AbstractVector)
    return any(isfinite, wb)
end
function set_weight_constraints!(args...)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 bgt::Union{Nothing, <:Real, <:BudgetRange},
                                 sbgt::Union{Nothing, <:Real, <:BudgetRange},
                                 long::Bool = false)
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    @argcheck(long ⊼ flag, "Long-only strategy cannot have negative weight limits")
    w = model[:w]
    N = length(w)
    k = model[:k]
    sc = model[:sc]
    if !isnothing(lb) && w_finite_flag(lb)
        @constraint(model, w_lb, sc * (w - k * lb) >= 0)
    end
    if !isnothing(ub) && w_finite_flag(ub)
        @constraint(model, w_ub, sc * (w - k * ub) <= 0)
    end
    set_budget_constraints!(model, bgt, w)
    if flag
        @variables(model, begin
                       lw[1:N] >= 0
                       sw[1:N] >= 0
                   end)
        @constraints(model, begin
                         w_lw, sc * (w - lw) <= 0
                         w_sw, sc * (w + sw) >= 0
                     end)
        set_long_short_budget_constraints!(model, bgt, sbgt)
    else
        @expression(model, lw, w)
    end
    return nothing
end
function non_zero_real_or_vec(::Nothing)
    return false
end
function non_zero_real_or_vec(x::Real)
    return !iszero(x)
end
function non_zero_real_or_vec(x::AbstractVector{<:Real})
    return any(!iszero, x)
end
function set_linear_weight_constraints!(args...)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model,
                                        lcms::Union{<:LinearConstraint,
                                                    <:AbstractVector{<:LinearConstraint}},
                                        key_ineq::Symbol, key_eq::Symbol)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    for (i, lcm) in enumerate(lcms)
        if !isnothing(lcm.ineq)
            A = lcm.ineq.A
            B = lcm.ineq.B
            model[Symbol(key_ineq, i)] = @constraint(model, sc * (A * w - k * B) <= 0)
        end
        if !isnothing(lcm.eq)
            A = lcm.eq.A
            B = lcm.eq.B
            model[Symbol(key_eq, i)] = @constraint(model, sc * (A * w - k * B) == 0)
        end
    end
    return nothing
end
function mip_wb(::Any, ::Nothing, args...)
    return nothing
end
function add_to_fees!(model::JuMP.Model, expr::AbstractJuMPScalar)
    if !haskey(model, :fees)
        @expression(model, fees, expr)
    else
        fees = model[:fees]
        add_to_expression!(fees, expr)
    end
    return nothing
end
function mip_wb(model::JuMP.Model, wb::WeightBounds, il::AbstractVector, is::AbstractVector)
    sc = model[:sc]
    w = model[:w]
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        @constraint(model, w_mip_lb, sc * (w - is ⊙ lb) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        @constraint(model, w_mip_ub, sc * (w - il ⊙ ub) <= 0)
    end
    return nothing
end
function short_mip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                         lt::Union{Nothing, <:BuyInThreshold},
                                         st::Union{Nothing, <:BuyInThreshold},
                                         ffl::Union{Nothing, <:Real, <:AbstractVector},
                                         ffs::Union{Nothing, <:Real, <:AbstractVector},
                                         ss::Union{Nothing, <:Real}, lt_flag::Bool,
                                         st_flag::Bool, ffl_flag::Bool, ffs_flag::Bool)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    if isnothing(ss)
        ss = 1_000
    end
    N = length(w)
    @variables(model, begin
                   ilb[1:N], (binary = true)
                   isb[1:N], (binary = true)
               end)
    @expression(model, i_mip, ilb + isb)
    if isa(k, Real)
        @expressions(model, begin
                         il, ilb
                         is, isb
                     end)
    else
        @variables(model, begin
                       ilf[1:N] >= 0
                       isf[1:N] >= 0
                   end)
        @constraints(model, begin
                         ilf_ub, ilf .- k <= 0
                         isf_ub, isf .- k <= 0
                         ilfd_ub, ilf - ss * ilb <= 0
                         isfd_ub, isf - ss * isb <= 0
                         ilfd_lb, (ilf + ss * (1 .- ilb)) .- k >= 0
                         isfd_lb, (isf + ss * (1 .- isb)) .- k >= 0
                     end)
        @expressions(model, begin
                         il, ilf
                         is, isf
                     end)
    end
    @constraint(model, i_mip_ub, i_mip .- 1 <= 0)
    mip_wb(model, wb, il, is)
    if lt_flag
        @constraint(model, w_mip_lt, sc * (w - il ⊙ lt.val + ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        @constraint(model, w_mip_st, sc * (w + is ⊙ st.val - ss * (1 .- isb)) <= 0)
    end
    if ffl_flag
        @expression(model, ffl, dot_scalar(ffl, ilb))
        add_to_fees!(model, ffl)
    end
    if ffs_flag
        @expression(model, ffs, dot_scalar(ffs, isb))
        add_to_fees!(model, ffs)
    end
    return i_mip
end
function mip_constraints(model::JuMP.Model, wb::WeightBounds,
                         ffl::Union{Nothing, <:Real, <:AbstractVector},
                         lt::Union{Nothing, <:BuyInThreshold}, ss::Union{Nothing, <:Real},
                         lt_flag::Bool, ffl_flag::Bool)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    @variable(model, ib[1:N], binary = true)
    if isa(k, Real)
        @expression(model, i_mip, ib)
    else
        if isnothing(ss)
            ss = 100_000.0
        end
        @variable(model, ibf[1:N] >= 0)
        @constraints(model, begin
                         ibf_ub, ibf .- k <= 0
                         ibfd_ub, ibf - ss * ib <= 0
                         ibfd_lb, (ibf + ss * (1 .- ib)) .- k >= 0
                     end)
        @expression(model, i_mip, ibf)
    end
    mip_wb(model, wb, i_mip, i_mip)
    if lt_flag
        @constraint(model, w_mip_lt, sc * (w - i_mip ⊙ lt.val) >= 0)
    end
    if ffl_flag
        @expression(model, ffl, dot_scalar(ffl, ib))
        add_to_fees!(model, ffl)
    end
    return ib
end
function set_iplg_constraints!(model::JuMP.Model,
                               plgs::Union{<:AbstractPhylogenyConstraintResult,
                                           <:AbstractVector{<:AbstractPhylogenyConstraintResult}})
    ib = model[:ib]
    sc = model[:sc]
    for (i, plg) in enumerate(plgs)
        if !isa(plg, IntegerPhylogeny)
            continue
        end
        A = plg.A
        B = plg.B
        model[Symbol(:card_plg_, i)] = @constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    return nothing
end
function set_mip_constraints!(model::JuMP.Model, wb::WeightBounds,
                              card::Union{Nothing, <:Integer},
                              gcard::Union{Nothing, <:LinearConstraint},
                              plg::Union{Nothing, <:AbstractPhylogenyConstraintResult,
                                         <:AbstractVector{<:AbstractPhylogenyConstraintResult}},
                              lt::Union{Nothing, <:BuyInThreshold},
                              st::Union{Nothing, <:BuyInThreshold},
                              fees::Union{Nothing, <:Fees}, ss::Union{Nothing, <:Real})
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    iplg_flag = isa(plg, IntegerPhylogeny) ||
                isa(plg, AbstractVector) && any(x -> isa(x, IntegerPhylogeny), plg)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    ffl_flag, ffs_flag, ffl, ffs = if !isnothing(fees)
        non_zero_real_or_vec(fees.fl), non_zero_real_or_vec(fees.fs), fees.fl, fees.fs
    else
        false, false, nothing, nothing
    end
    if !(card_flag || gcard_flag || iplg_flag || lt_flag || st_flag || ffl_flag || ffs_flag)
        return nothing
    end
    ib = if (st_flag || ffl_flag || ffs_flag) && haskey(model, :sw)
        short_mip_threshold_constraints(model, wb, lt, st, ffl, ffs, ss, lt_flag, st_flag,
                                        ffl_flag, ffs_flag)
    else
        mip_constraints(model, wb, ffl, lt, ss, lt_flag, ffl_flag)
    end
    sc = model[:sc]
    if card_flag
        @constraint(model, card, sc * (sum(ib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            @constraint(model, gcard_ineq, sc * (A * ib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            @constraint(model, gcard_eq, sc * (A * ib ⊖ B) == 0)
        end
    end
    if iplg_flag
        set_iplg_constraints!(model, plg)
    end
    return nothing
end
function smip_wb(::Any, ::Nothing, args...)
    return nothing
end
function smip_wb(model::JuMP.Model, wb::WeightBounds, smtx::AbstractMatrix,
                 smtx_expr::AbstractVector{<:AbstractJuMPScalar}, il::AbstractVector,
                 is::AbstractVector, key::Symbol = :set_w_mip_, i::Integer = 1)
    sc = model[:sc]
    lb = wb.lb
    if !isnothing(lb) && w_finite_flag(lb)
        lb = [sum(lb[view(smtx, j, :)]) for j in axes(smtx, 1)]
        model[Symbol(key, :lb_, i)] = @constraint(model, sc * (smtx_expr - lb ⊙ is) >= 0)
    end
    ub = wb.ub
    if !isnothing(ub) && w_finite_flag(ub)
        ub = [sum(ub[view(smtx, j, :)]) for j in axes(smtx, 1)]
        model[Symbol(key, :ub_, i)] = @constraint(model, sc * (smtx_expr - ub ⊙ il) <= 0)
    end
    return nothing
end
function short_smip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                          smtx::Union{Nothing, <:AbstractMatrix},
                                          lt::Union{Nothing, <:BuyInThreshold},
                                          st::Union{Nothing, <:BuyInThreshold},
                                          ss::Union{Nothing, <:Real}, lt_flag::Bool,
                                          st_flag::Bool, key1::Symbol = :si,
                                          key7::Symbol = :smtx_expr_,
                                          key8::Symbol = :set_w_mip_, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    if isnothing(ss)
        ss = 1_000
    end
    N = size(smtx, 1)
    ilb, isb = model[Symbol(key1, :lb_, i)], model[Symbol(key1, :ub_, i)] = @variables(model,
                                                                                       begin
                                                                                           [1:N],
                                                                                           (binary = true)
                                                                                           [1:N],
                                                                                           (binary = true)
                                                                                       end)
    key2 = Symbol(key1, :_mip_)
    i_mip = model[Symbol(key2, i)] = @expression(model, ilb + isb)
    il, is = if isa(k, Real)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = @expressions(model,
                                                                                begin
                                                                                    ilb
                                                                                    isb
                                                                                end)
    else
        key3 = Symbol(key1, :lf_)
        key4 = Symbol(key1, :sf_)
        ilf, isf = model[Symbol(key3, i)], model[Symbol(key4, i)] = @variables(model,
                                                                               begin
                                                                                   [1:N],
                                                                                   (lower_bound = 0)
                                                                                   [1:N],
                                                                                   (lower_bound = 0)
                                                                               end)
        key5 = Symbol(key1, :lfd_)
        key6 = Symbol(key1, :sfd_)
        model[Symbol(key3, :ub_, i)], model[Symbol(key4, :ub_, i)], model[Symbol(key5, :ub_, i)], model[Symbol(key6, :ub_, i)], model[Symbol(key5, :lb_, i)], model[Symbol(key6, :lb_, i)] = @constraints(model,
                                                                                                                                                                                                          begin
                                                                                                                                                                                                              ilf .-
                                                                                                                                                                                                              k <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              isf .-
                                                                                                                                                                                                              k <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              ilf -
                                                                                                                                                                                                              ss *
                                                                                                                                                                                                              ilb <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              isf -
                                                                                                                                                                                                              ss *
                                                                                                                                                                                                              isb <=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              (ilf +
                                                                                                                                                                                                               ss *
                                                                                                                                                                                                               (1 .-
                                                                                                                                                                                                                ilb)) .-
                                                                                                                                                                                                              k >=
                                                                                                                                                                                                              0
                                                                                                                                                                                                              (isf +
                                                                                                                                                                                                               ss *
                                                                                                                                                                                                               (1 .-
                                                                                                                                                                                                                isb)) .-
                                                                                                                                                                                                              k >=
                                                                                                                                                                                                              0
                                                                                                                                                                                                          end)
        model[Symbol(key1, :l_, i)], model[Symbol(key1, :s_, i)] = @expressions(model,
                                                                                begin
                                                                                    ilf
                                                                                    isf
                                                                                end)
    end
    model[Symbol(key2, :ub, i)] = @constraint(model, i_mip .- 1 <= 0)
    smtx_expr = model[Symbol(key7, i)] = @expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, il, is, key8, i)
    if lt_flag
        model[Symbol(key8, :lt, i)] = @constraint(model,
                                                  sc * (smtx_expr - il ⊙ lt.val +
                                                        ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        model[Symbol(key8, :st, i)] = @constraint(model,
                                                  sc * (smtx_expr + is ⊙ st.val -
                                                        ss * (1 .- isb)) <= 0)
    end
    return i_mip
end
function smip_constraints(model::JuMP.Model, wb::WeightBounds,
                          smtx::Union{Nothing, <:AbstractMatrix},
                          lt::Union{Nothing, <:BuyInThreshold}, ss::Union{Nothing, <:Real},
                          lt_flag::Bool, key1::Symbol = :sib_, key2::Symbol = :i_smip_,
                          key3::Symbol = :isbf_, key4::Symbol = :smtx_expr_,
                          key5::Symbol = :set_w_mip_, key6::Symbol = :w_smip_lt_,
                          i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = size(smtx, 1)
    sib = model[Symbol(key1, i)] = @variable(model, [1:N], binary = true)
    i_smip = if isa(k, Real)
        model[Symbol(key2, i)] = @expression(model, sib)
    else
        if isnothing(ss)
            ss = 100_000.0
        end
        isbf = model[Symbol(key3, i)] = @variable(model, [1:N], lower_bound = 0)
        model[Symbol(key3, :_ub_, i)], model[Symbol(key3, :d_ub_, i)], model[Symbol(key3, :d_lb_, i)] = @constraints(model,
                                                                                                                     begin
                                                                                                                         isbf .-
                                                                                                                         k <=
                                                                                                                         0
                                                                                                                         isbf -
                                                                                                                         ss *
                                                                                                                         sib <=
                                                                                                                         0
                                                                                                                         (isbf +
                                                                                                                          ss *
                                                                                                                          (1 .-
                                                                                                                           sib)) .-
                                                                                                                         k >=
                                                                                                                         0
                                                                                                                     end)
        model[Symbol(key2, i)] = @expression(model, isbf)
    end
    smtx_expr = model[Symbol(key4, i)] = @expression(model, smtx * w)
    smip_wb(model, wb, smtx, smtx_expr, i_smip, i_smip, key5, i)
    if lt_flag
        model[Symbol(key6, i)] = @constraint(model, sc * (smtx_expr - i_smip ⊙ lt.val) >= 0)
    end
    return sib
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Union{Nothing, <:Integer},
                                   gcard::Union{Nothing, <:LinearConstraint},
                                   smtx::Union{Nothing, <:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold},
                                   st::Union{Nothing, <:BuyInThreshold},
                                   ss::Union{Nothing, <:Real}, i::Integer = 1)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = model[:sc]
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag, :si,
                                         :smtx_expr_, :set_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sib_, :i_smip_, :isbf_,
                         :smtx_expr_, :set_w_mip_, :w_smip_lt_, i)
    end
    if card_flag
        model[Symbol(:scard_, i)] = @constraint(model, sc * (sum(sib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            model[Symbol(:sgcard_ineq_, i)] = @constraint(model,
                                                          sc *
                                                          (A * transpose(smtx) * sib ⊖ B) <=
                                                          0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            model[Symbol(:sgcard_eq_, i)] = @constraint(model,
                                                        sc *
                                                        (A * transpose(smtx) * sib ⊖ B) ==
                                                        0)
        end
    end
    return nothing
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::AbstractVector{<:Integer},
                                   gcard::AbstractVector{<:LinearConstraint},
                                   smtx::AbstractVector{<:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   st::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   ss::Union{Nothing, <:Real})
    for (i, (c, g, s)) in enumerate(zip(card, gcard, smtx))
        lti = isa(lt, Union{Nothing, <:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Union{Nothing, <:BuyInThreshold}) ? st : st[i]
        set_all_smip_constraints!(model, wb, c, g, s, lti, sti, ss, i)
    end
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Union{Nothing, <:Integer},
                                   smtx::Union{Nothing, <:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold},
                                   st::Union{Nothing, <:BuyInThreshold},
                                   ss::Union{Nothing, <:Real}, i::Integer = 1)
    card_flag = !isnothing(card)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || lt_flag || st_flag)
        return nothing
    end
    sc = model[:sc]
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag, :si,
                                         :smtx_expr_, :set_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sib_, :i_smip_, :isbf_,
                         :smtx_expr_, :set_w_mip_, :w_smip_lt_, i)
    end
    model[Symbol(:scard_, i)] = @constraint(model, sc * (sum(sib) - card) <= 0)
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::AbstractVector{<:Integer},
                                   smtx::AbstractVector{<:AbstractMatrix},
                                   lt::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   st::Union{Nothing, <:BuyInThreshold,
                                             <:AbstractVector{<:BuyInThreshold},
                                             <:AbstractVector{<:Union{Nothing,
                                                                      <:BuyInThreshold}}},
                                   ss::Union{Nothing, <:Real})
    for (i, (c, s)) in enumerate(zip(card, smtx))
        lti = isa(lt, Union{Nothing, <:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Union{Nothing, <:BuyInThreshold}) ? st : st[i]
        set_scardmip_constraints!(model, wb, c, s, lti, sti, ss, i)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Union{Nothing, <:LinearConstraint},
                                    smtx::Union{Nothing, <:AbstractMatrix},
                                    lt::Union{Nothing, <:BuyInThreshold},
                                    st::Union{Nothing, <:BuyInThreshold},
                                    ss::Union{Nothing, <:Real}, i::Integer = 1)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = model[:sc]
    sib = if st_flag && haskey(model, :sw)
        short_smip_threshold_constraints(model, wb, smtx, lt, st, ss, lt_flag, st_flag,
                                         :sgi, :sgmtx_expr_, :setg_w_mip_, i)
    else
        smip_constraints(model, wb, smtx, lt, ss, lt_flag, :sgib_, :i_sgmip_, :isgbf_,
                         :sgmtx_expr_, :setg_w_mip_, :w_sgmip_lt_, i)
    end
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        model[Symbol(:sgcard_ineq_, i)] = @constraint(model,
                                                      sc *
                                                      (A * transpose(smtx) * sib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[Symbol(:sgcard_eq_, i)] = @constraint(model,
                                                    sc * (A * transpose(smtx) * sib ⊖ B) ==
                                                    0)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::AbstractVector{<:LinearConstraint},
                                    smtx::AbstractVector{<:AbstractMatrix},
                                    lt::Union{Nothing, <:BuyInThreshold,
                                              <:AbstractVector{<:BuyInThreshold},
                                              <:AbstractVector{<:Union{Nothing,
                                                                       <:BuyInThreshold}}},
                                    st::Union{Nothing, <:BuyInThreshold,
                                              <:AbstractVector{<:BuyInThreshold},
                                              <:AbstractVector{<:Union{Nothing,
                                                                       <:BuyInThreshold}}},
                                    ss::Union{Nothing, <:Real})
    for (i, (gc, s)) in enumerate(zip(gcard, smtx))
        lti = isa(lt, Union{Nothing, <:BuyInThreshold}) ? lt : lt[i]
        sti = isa(st, Union{Nothing, <:BuyInThreshold}) ? st : st[i]
        set_sgcardmip_constraints!(model, wb, gc, s, lti, sti, ss, i)
    end
    return nothing
end
function set_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                               card::Union{Nothing, <:Integer, <:AbstractVector{<:Integer}},
                               gcard::Union{Nothing, <:LinearConstraint,
                                            <:AbstractVector{<:LinearConstraint}},
                               smtx::Union{Nothing, <:AbstractMatrix,
                                           <:AbstractVector{<:AbstractMatrix}},
                               sgmtx::Union{Nothing, <:AbstractMatrix,
                                            <:AbstractVector{<:AbstractMatrix}},
                               lt::Union{Nothing, <:BuyInThreshold},
                               st::Union{Nothing, <:BuyInThreshold},
                               glt::Union{Nothing, <:BuyInThreshold},
                               gst::Union{Nothing, <:BuyInThreshold},
                               ss::Union{Nothing, <:Real})
    if smtx === sgmtx
        set_all_smip_constraints!(model, wb, card, gcard, smtx, lt, st, ss)
    else
        set_scardmip_constraints!(model, wb, card, smtx, lt, st, ss)
        set_sgcardmip_constraints!(model, wb, gcard, sgmtx, glt, gst, ss)
    end
    return nothing
end
function set_turnover_fees!(args...)
    return nothing
end
function set_turnover_fees!(model::JuMP.Model, tn::Turnover)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wt = tn.w
    val = tn.val
    @variable(model, t_ftn[1:N])
    @expressions(model, begin
                     x_ftn, w - wt * k
                     ftn, dot_scalar(val, t_ftn)
                 end)
    @constraint(model, cftn[i = 1:N], [sc * t_ftn[i]; sc * x_ftn[i]] in MOI.NormOneCone(2))
    add_to_fees!(model, ftn)
    return nothing
end
function set_non_fixed_fees!(args...)
    return nothing
end
function set_long_non_fixed_fees!(args...)
    return nothing
end
function set_short_non_fixed_fees!(args...)
    return nothing
end
function set_long_non_fixed_fees!(model::JuMP.Model, fl::Union{<:Real, <:AbstractVector})
    lw = model[:lw]
    @expression(model, fl, dot_scalar(fl, lw))
    add_to_fees!(model, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::Union{<:Real, <:AbstractVector})
    if !haskey(model, :sw)
        return nothing
    end
    sw = model[:sw]
    @expression(model, fs, dot_scalar(fs, sw))
    add_to_fees!(model, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_long_non_fixed_fees!(model, fees.l)
    set_short_non_fixed_fees!(model, fees.s)
    set_turnover_fees!(model, fees.tn)
    return nothing
end
function set_tracking_error_constraints!(args...; kwargs...)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::TrackingError{<:Any, <:Any, <:NOCTracking},
                                         args...; kwargs...)
    X = pr.X
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(te.tracking, X)
    err = te.err
    T = size(X, 1)
    f = err * T
    t_te = model[Symbol(:t_te_, i)] = @variable(model)
    te = model[Symbol(:te_, i)] = @expression(model, net_X - wb * k)
    model[Symbol(:cte_noc_, i)], model[Symbol(:cte_, i)] = @constraints(model,
                                                                        begin
                                                                            [sc * t_te;
                                                                             sc * te] in
                                                                            MOI.NormOneCone(1 +
                                                                                            T)
                                                                            sc *
                                                                            (t_te - f * k) <=
                                                                            0
                                                                        end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::TrackingError{<:Any, <:Any, <:SOCTracking},
                                         args...; kwargs...)
    X = pr.X
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(te.tracking, X)
    err = te.err
    f = err * sqrt(size(X, 1) - te.alg.ddof)
    t_te = model[Symbol(:t_te_, i)] = @variable(model)
    te = model[Symbol(:te_, i)] = @expression(model, net_X - wb * k)
    model[Symbol(:cte_soc_, i)], model[Symbol(:cte_, i)] = @constraints(model,
                                                                        begin
                                                                            [sc * t_te;
                                                                             sc * te] in
                                                                            SecondOrderCone()
                                                                            sc *
                                                                            (t_te - f * k) <=
                                                                            0
                                                                        end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:IndependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         plg::Union{Nothing,
                                                    <:AbstractPhylogenyConstraintResult,
                                                    <:AbstractVector{<:AbstractPhylogenyConstraintResult}},
                                         fees::Union{Nothing, <:Fees}, args...; kwargs...)
    r = te.r
    wb = te.tracking.w
    err = te.err
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    te_dw = Symbol(:te_dw_, i)
    model[:oldw] = model[:w]
    unregister(model, :w)
    model[:w] = @expression(model, w - wb * k)
    risk_expr = set_triv_risk_constraints!(model, te_dw, r, opt, pr, plg, fees, args...;
                                           kwargs...)
    model[Symbol(:triv_, i, :_w)] = model[:w]
    model[:w] = model[:oldw]
    unregister(model, :oldw)
    model[Symbol(:cter_, i)] = @constraint(model, sc * (risk_expr - err * k) <= 0)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         te::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:DependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         plg::Union{Nothing,
                                                    <:AbstractPhylogenyConstraintResult,
                                                    <:AbstractVector{<:AbstractPhylogenyConstraintResult}},
                                         fees::Union{Nothing, <:Fees}, args...; kwargs...)
    ri = te.r
    wb = te.tracking.w
    err = te.err
    rb = expected_risk(factory(ri, pr, opt.opt.slv), wb, pr.X, fees)
    k = model[:k]
    sc = model[:sc]
    key = Symbol(:t_dr_, i)
    t_dr = model[key] = @variable(model)
    risk_expr = set_trdv_risk_constraints!(model, key, ri, opt, pr, plg, fees, args...;
                                           kwargs...)
    dr = model[Symbol(:dr_, i)] = @expression(model, risk_expr - rb * k)
    model[Symbol(:cter_noc_, i)], model[Symbol(:cter_, i)] = @constraints(model,
                                                                          begin
                                                                              [sc * t_dr;
                                                                               sc * dr] in
                                                                              MOI.NormOneCone(2)
                                                                              sc *
                                                                              (t_dr -
                                                                               err * k) <=
                                                                              0
                                                                          end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, pr::AbstractPriorResult,
                                         tres::Union{<:AbstractTracking,
                                                     <:AbstractVector{<:AbstractTracking}},
                                         args...; kwargs...)
    for (i, te) in enumerate(tres)
        set_tracking_error_constraints!(model, i, pr, te, args...; kwargs...)
    end
    return nothing
end
function set_turnover_constraints!(args...)
    return nothing
end
function _set_turnover_constraints!(model::JuMP.Model, tn::Turnover, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wb = tn.w
    val = tn.val
    t_tn = model[Symbol(:t_tn_, i)] = @variable(model, [1:N])
    tn = model[Symbol(:tn_, i)] = @expression(model, w - wb * k)
    model[Symbol(:ctn_noc_, i)], model[Symbol(:ctn_, i)] = @constraints(model,
                                                                        begin
                                                                            [i = 1:N],
                                                                            [sc * t_tn[i];
                                                                             sc * tn[i]] in
                                                                            MOI.NormOneCone(2)
                                                                            sc *
                                                                            (t_tn ⊖ val * k) <=
                                                                            0
                                                                        end)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model,
                                   tns::Union{<:Turnover, <:AbstractVector{<:Turnover}})
    for (i, tn) in enumerate(tns)
        _set_turnover_constraints!(model, tn, i)
    end
    return nothing
end
function set_number_effective_assets!(args...)
    return nothing
end
function set_number_effective_assets!(model::JuMP.Model, val::Real)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    @variable(model, nea)
    @constraints(model, begin
                     cnea_soc, [sc * nea; sc * w] in SecondOrderCone()
                     cnea, sc * (nea * sqrt(val) - k) <= 0
                 end)
    return nothing
end
function set_l1_regularisation!(args...)
    return nothing
end
function set_l2_regularisation!(args...)
    return nothing
end
function set_l1_regularisation!(model::JuMP.Model, l1::Real)
    w = model[:w]
    sc = model[:sc]
    @variable(model, t_l1)
    @constraint(model, cl1_noc, [sc * t_l1; sc * w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1, l1 * t_l1)
    add_to_objective_penalty!(model, l1)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2::Real)
    w = model[:w]
    sc = model[:sc]
    @variable(model, t_l2)
    @constraint(model, cl2_soc, [sc * t_l2; sc * w] in SecondOrderCone())
    @expression(model, l2, l2 * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
function set_sdp_constraints!(model::JuMP.Model)
    if haskey(model, :W)
        return model[:W]
    end
    w = model[:w]
    k = ifelse(haskey(model, :crkb), 1, model[:k])
    sc = model[:sc]
    N = length(w)
    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, M_PSD, sc * M in PSDCone())
    return W
end
function set_sdp_frc_constraints!(model::JuMP.Model)
    if haskey(model, :frc_W)
        return model[:frc_W]
    end
    w1 = model[:w1]
    sc = model[:sc]
    k = model[:k]
    Nf = length(w1)
    @variable(model, frc_W[1:Nf, 1:Nf], Symmetric)
    @expression(model, frc_M, hcat(vcat(frc_W, transpose(w1)), vcat(w1, k)))
    @constraint(model, frc_M_PSD, sc * frc_M in PSDCone())
    return frc_W
end
function set_sdp_phylogeny_constraints!(model::JuMP.Model,
                                        plgs::Union{Nothing,
                                                    <:AbstractPhylogenyConstraintResult,
                                                    <:AbstractVector{<:AbstractPhylogenyConstraintResult}})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    for (i, plg) in enumerate(plgs)
        if !isa(plg, SemiDefinitePhylogeny)
            continue
        end
        key = Symbol(:sdp_plg_, i)
        A = plg.A
        model[key] = @constraint(model, sc * A ⊙ W == 0)
        if !haskey(model, :variance_flag)
            key = Symbol(key, :_p)
            p = plg.p
            plp = model[key] = @expression(model, p * tr(W))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
function set_sdp_frc_phylogeny_constraints!(model::JuMP.Model,
                                            plgs::Union{Nothing,
                                                        <:AbstractPhylogenyConstraintResult,
                                                        <:AbstractVector{<:AbstractPhylogenyConstraintResult}})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = model[:sc]
    W = set_sdp_frc_constraints!(model)
    for (i, plg) in enumerate(plgs)
        if !isa(plg, SemiDefinitePhylogeny)
            continue
        end
        key = Symbol(:frc_sdp_plg_, i)
        A = plg.A
        model[key] = @constraint(model, sc * A ⊙ W == 0)
        if !haskey(model, :variance_flag)
            key = Symbol(key, :_p)
            p = plg.p
            plp = model[key] = @expression(model, p * tr(W))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end

export BudgetRange, BudgetCosts, BudgetMarketImpact
