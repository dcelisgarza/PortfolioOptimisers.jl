struct BudgetRange{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       JuMPConstraintEstimator
    lb::T1
    ub::T2
end
function BudgetRange(; lb::Union{Nothing, <:Real} = 1.0, ub::Union{Nothing, <:Real} = 1.0)
    lb_flag = isnothing(lb)
    ub_flag = isnothing(ub)
    @smart_assert(lb_flag ⊼ ub_flag)
    if !lb_flag
        @smart_assert(isfinite(lb) && lb >= 0)
    end
    if !ub_flag
        @smart_assert(isfinite(ub) && ub >= 0)
    end
    if !lb_flag && !ub_flag
        @smart_assert(lb <= ub)
    end
    return BudgetRange{typeof(lb), typeof(ub)}(lb, ub)
end
function set_budget_constraints!(args...; kwargs...)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, val::Real, w::AbstractVector;
                                 key::Symbol = :bgt, kwargs...)
    k = model[:k]
    sc = model[:sc]
    model[key] = @constraint(model, sc * (sum(w) - k * val) == 0)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::AbstractVector;
                                 key_lb::Symbol = :bgt_lb, key_ub::Symbol = :bgt_ub,
                                 kwargs...)
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    ub = bgt.ub
    if !isnothing(lb)
        model[key_lb] = @constraint(model, sc * (sum(w) - k * lb) >= 0)
    end
    if !isnothing(ub)
        model[key_ub] = @constraint(model, sc * (sum(w) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(::JuMP.Model, ::Nothing, ::Nothing)
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
    k = model[:k]
    sc = model[:sc]
    @constraint(model, lsbgt, sc * (sum(lw) - k * (bgt + sbgt)) == 0)
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
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Real, sbgt::BudgetRange)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        @constraint(model, lbgt_lb, sc * (sum(lw) - k * (lb + bgt)) >= 0)
    end
    ub = sbgt.ub
    if !isnothing(ub)
        @constraint(model, lbgt_ub, sc * (sum(lw) - k * (ub + bgt)) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::BudgetRange)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    slb = sbgt.lb
    lb_flag = isnothing(lb)
    slb_flag = isnothing(slb)
    if !lb_flag && !slb_flag
        @constraint(model, lbgt_lb, sc * (sum(lw) - k * (lb + slb)) >= 0)
    elseif !lb_flag && slb_flag
        @constraint(model, lbgt_lb, sc * (sum(lw) - k * lb) >= 0)
    end
    ub = bgt.ub
    sub = sbgt.ub
    ub_flag = isnothing(ub)
    sub_flag = isnothing(sub)
    if !ub_flag && !sub_flag
        @constraint(model, lbgt_ub, sc * (sum(lw) - k * (ub + sub)) <= 0)
    elseif !ub_flag && sub_flag
        @constraint(model, lbgt_ub, sc * (sum(lw) - k * ub) <= 0)
    end
    return nothing
end
function w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function w_neg_flag(wb::AbstractVector)
    return any(wb .< zero(eltype(wb)))
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
function set_weight_constraints!(model::JuMP.Model, wb::WeightBoundsResult,
                                 bgt::Union{Nothing, <:Real, <:BudgetRange},
                                 sbgt::Union{Nothing, <:Real, <:BudgetRange},
                                 long_only::Bool = false)
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    @smart_assert(long_only ⊼ flag, "Long-only strategy cannot have negative weight limits")
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
    set_budget_constraints!(model, bgt, w; key = :bgt, key_lb = :bgt_lb, key_ub = :bgt_ub)
    if flag
        @variables(model, begin
                       lw[1:N] >= 0
                       sw[1:N] >= 0
                   end)
        @constraints(model, begin
                         w_lw, sc * (w - lw) <= 0
                         w_sw, sc * (w + sw) >= 0
                     end)
        set_budget_constraints!(model, sbgt, sw; key = :sbgt, key_lb = :sbgt_lb,
                                key_ub = :sbgt_ub)
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
function set_linear_weight_constraints!(::JuMP.Model, ::Nothing, ::Symbol, ::Symbol)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcm::LinearConstraintResult,
                                        key_ineq::Symbol, key_eq::Symbol)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    if !isnothing(lcm.ineq)
        A = lcm.ineq.A
        B = lcm.ineq.B
        model[key_ineq] = @constraint(model, sc * (A * w - k * B) <= 0)
    end
    if !isnothing(lcm.eq)
        A = lcm.eq.A
        B = lcm.eq.B
        model[key_eq] = @constraint(model, sc * (A * w - k * B) == 0)
    end
    return nothing
end
function mip_wb(args...)
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
function mip_wb(model::JuMP.Model, wb::WeightBoundsResult, il::AbstractVector,
                is::AbstractVector)
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
function short_mip_threshold_constraints(model::JuMP.Model, wb::WeightBoundsResult,
                                         lt::Union{Nothing, <:Real, <:AbstractVector},
                                         st::Union{Nothing, <:Real, <:AbstractVector},
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
        @constraint(model, w_mip_lt, sc * (w - il ⊙ lt + ss * (1 .- ilb)) >= 0)
    end
    if st_flag
        @constraint(model, w_mip_st, sc * (w + is ⊙ st - ss * (1 .- isb)) <= 0)
    end
    if ffl_flag || ffs_flag
        if ffl_flag
            @expression(model, ffl, sum(ffl ⊙ ilb))
            add_to_fees!(model, ffl)
        end
        if ffs_flag
            @expression(model, ffs, sum(ffs ⊙ isb))
            add_to_fees!(model, ffs)
        end
    end
    return i_mip
end
function mip_constraints(model::JuMP.Model, wb::WeightBoundsResult,
                         ffl::Union{Nothing, <:Real, <:AbstractVector},
                         lt::Union{Nothing, <:Real, <:AbstractVector},
                         ss::Union{Nothing, <:Real}, lt_flag::Bool, ffl_flag::Bool)
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
        @constraint(model, w_mip_lt, sc * (w - i_mip ⊙ lt) >= 0)
    end
    if ffl_flag
        @expression(model, ffl, sum(ffl ⊙ ib))
        add_to_fees!(model, ffl)
    end
    return ib
end
function set_mip_constraints!(model::JuMP.Model, wb::WeightBoundsResult,
                              card::Union{Nothing, <:Integer},
                              gcard::Union{Nothing, <:LinearConstraintResult},
                              nplg::Union{Nothing, <:PhilogenyConstraintResult},
                              cplg::Union{Nothing, <:PhilogenyConstraintResult},
                              lt::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                              st::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                              fees::Union{Nothing, <:Fees}, ss::Union{Nothing, <:Real})
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    n_flag = isa(nplg, IntegerPhilogenyResult)
    c_flag = isa(cplg, IntegerPhilogenyResult)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    ffl_flag, ffs_flag, ffl, ffs = if !isnothing(fees)
        non_zero_real_or_vec(fees.fixed_long), non_zero_real_or_vec(fees.fixed_short),
        fees.fixed_long, fees.fixed_short
    else
        false, false, nothing, nothing
    end
    if !(card_flag ||
         gcard_flag ||
         n_flag ||
         c_flag ||
         lt_flag ||
         st_flag ||
         ffl_flag ||
         ffs_flag)
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
            @constraint(model, gcard_ineq, sc * (A * ib - B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            @constraint(model, gcard_eq, sc * (A * ib - B) == 0)
        end
    end
    if n_flag
        A = nplg.A
        B = nplg.B
        @constraint(model, card_nplg, sc * (A * ib - B) <= 0)
    end
    if c_flag
        A = cplg.A
        B = cplg.B
        @constraint(model, card_cplg, sc * (A * ib - B) <= 0)
    end
    return nothing
end
function set_turnover_fees!(args...)
    return nothing
end
function set_turnover_fees!(model::JuMP.Model, turnover::Turnover)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wt = turnover.w
    val = turnover.val
    @variable(model, t_ftn[1:N])
    @expressions(model, begin
                     x_ftn, w - wt * k
                     ftn, sum(val ⊙ t_ftn)
                 end)
    @constraint(model, cftn[i = 1:N], [sc * t_ftn[i]; sc * x_ftn[i]] ∈ MOI.NormOneCone(2))
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
    @expression(model, fl, sum(fl ⊙ lw))
    add_to_fees!(model, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::Union{<:Real, <:AbstractVector})
    if !haskey(model, :sw)
        return nothing
    end
    sw = model[:sw]
    @expression(model, fs, sum(fs ⊙ sw))
    add_to_fees!(model, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_long_non_fixed_fees!(model, fees.long)
    set_short_non_fixed_fees!(model, fees.short)
    set_turnover_fees!(model, fees.turnover)
    return nothing
end
function set_tracking_error_constraints!(args...)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, X::AbstractMatrix,
                                         tre::TrackingError)
    k = model[:k]
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tre.tracking, X)
    err = tre.err
    f = err * sqrt(size(X, 1) - 1)
    @variable(model, t_tr)
    @expression(model, te, net_X - wb * k)
    @constraints(model, begin
                     ctr_soc, [sc * t_tr; sc * te] ∈ SecondOrderCone()
                     ctr, sc * (t_tr - f * k) <= 0
                 end)
    return nothing
end
function set_turnover_constraints!(args...)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model, tn::Turnover)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wi = tn.w
    val = tn.val
    @variable(model, t_tn[1:N])
    @expression(model, tn, w - wi * k)
    @constraints(model,
                 begin
                     ctr_noc[i = 1:N], [sc * t_tn[i]; sc * tn[i]] ∈ MOI.NormOneCone(2)
                     ctr, sc * (t_tn ⊖ val * k) <= 0
                 end)
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
                     c_nea_soc, [sc * nea; sc * w] ∈ SecondOrderCone()
                     c_nea, sc * (nea * sqrt(val) - k) <= 0
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
    @constraint(model, l1_noc, [sc * t_l1; sc * w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1, l1 * t_l1)
    add_to_objective_penalty!(model, l1)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2::Real)
    w = model[:w]
    sc = model[:sc]
    @variable(model, t_l2)
    @constraint(model, l2_soc, [sc * t_l2; sc * w] in SecondOrderCone())
    @expression(model, l2, l2 * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
function set_sdp_constraints!(model::JuMP.Model)
    if haskey(model, :W)
        return model[:W]
    end
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    @variable(model, W[1:N, 1:N], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w)), vcat(w, k)))
    @constraint(model, M_PSD, sc * M ∈ PSDCone())
    return W
end
function set_sdp_frc_constraints!(model::JuMP.Model)
    if haskey(model, :W)
        return model[:W]
    end
    w1 = model[:w1]
    sc = model[:sc]
    k = model[:k]
    Nf = length(w1)
    @variable(model, W[1:Nf, 1:Nf], Symmetric)
    @expression(model, M, hcat(vcat(W, transpose(w1)), vcat(w1, k)))
    @constraint(model, M_PSD, sc * M ∈ PSDCone())
    return W
end
function set_sdp_philogeny_constraints!(args...)
    return nothing
end
function set_sdp_philogeny_constraints!(model::JuMP.Model, adj::SemiDefinitePhilogenyResult,
                                        key::Symbol)
    sc = model[:sc]
    W = set_sdp_constraints!(model)
    A = adj.A
    model[key] = @constraint(model, sc * A ⊙ W == 0)
    if !haskey(model, :variance_flag)
        key = Symbol(key, :_p)
        p = adj.p
        plp = model[key] = @expression(model, p * tr(W))
        add_to_objective_penalty!(model, plp)
    end
    return nothing
end
function set_sdp_frc_philogeny_constraints!(args...)
    return nothing
end
function set_sdp_frc_philogeny_constraints!(model::JuMP.Model,
                                            adj::SemiDefinitePhilogenyResult, key::Symbol)
    sc = model[:sc]
    set_sdp_frc_constraints!(model)
    W = model[:W]
    A = adj.A
    model[key] = @constraint(model, sc * A ⊙ W == 0)
    if !haskey(model, :variance_flag)
        key = Symbol(key, :_p)
        p = adj.p
        plp = model[key] = @expression(model, p * tr(W))
        add_to_objective_penalty!(model, plp)
    end
    return nothing
end

export BudgetRange
