struct BuyInThreshold{T1 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                      T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}}
    sbi::T1
    lbi::T2
end
function BuyInThreshold(; sbi::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                        lbi::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0)
    sbi_flag = !isnothing(sbi)
    lbi_flag = !isnothing(lbi)
    @smart_assert(!sbi_flag ⊼ !lbi_flag)
    if sbi_flag && lbi_flag
        sbi_flag = isa(sbi, Real)
        lbi_flag = isa(lbi, Real)
        if sbi_flag
            @smart_assert(isfinite(sbi) && sbi >= zero(sbi))
        else
            @smart_assert(all(isfinite, sbi) && all(sbi .>= zero(sbi)))
        end
        if lbi_flag
            @smart_assert(isfinite(lbi) && lbi >= zero(lbi))
        else
            @smart_assert(all(isfinite, lbi) && all(lbi .>= zero(lbi)))
        end
        if !sbi_flag && !lbi_flag
            @smart_assert(length(sbi) == length(lbi))
        end
    elseif lbi_flag && !sbi_flag
        if isa(sbi, Real)
            @smart_assert(isfinite(sbi) && sbi >= zero(sbi))
        else
            @smart_assert(all(isfinite, sbi) && all(sbi .>= zero(sbi)))
        end
    elseif !lbi_flag && sbi_flag
        if isa(lbi, Real)
            @smart_assert(isfinite(lbi) && lbi >= zero(lbi))
        else
            @smart_assert(all(isfinite, lbi) && all(lbi .>= zero(lbi)))
        end
    end
    return BuyInThreshold{typeof(sbi), typeof(lbi)}(sbi, lbi)
end
function mip_wb(::JuMP.Model, ::Nothing, args...)
    return nothing
end
function mip_wb(model::JuMP.Model, wb::WeightBounds, il, is)
    sc = model[:sc]
    w = model[:w]
    lb = wb.lb
    if !isnothing(lb) && isfinite(lb)
        @constraint(model, w_mip_lb, sc * w .>= sc * is .* lb)
    end
    ub = wb.ub
    if !isnothing(ub) && isfinite(ub)
        @constraint(model, w_mip_ub, sc * w .<= sc * il .* ub)
    end
    return nothing
end
function _short_mip_threshold_constraints(model::JuMP.Model, wb::WeightBounds,
                                          ffl::Union{<:Real, <:AbstractVector},
                                          ffs::Union{<:Real, <:AbstractVector},
                                          lbi::Union{<:Real, <:AbstractVector},
                                          sbi::Union{<:Real, <:AbstractVector},
                                          lbi_flag::Bool, sbi_flag::Bool, ffl_flag::Bool,
                                          ffs_flag::Bool)
    w, k, sc = get_w_k_sc(model)
    ss = model[:ss]
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
                       isf[1:N] >= 0
                       ilf[1:N] >= 0
                   end)
        @constraints(model, begin
                         ilf_ub, ilf <= k
                         isf_ub, isf <= k
                         ilfd_ub, ilf <= ss * ilb
                         isfd_ub, isf <= ss * isb
                         ilfd_lb, ilf >= k .- ss * (1 .- ilb)
                         isfd_lb, isf >= k .- ss * (1 .- isb)
                     end)
        @expressions(model, begin
                         il, ilf
                         is, isf
                     end)
    end
    @constraint(model, i_mip_ub, i_mip <= 1)
    mip_wb(model, wb, il, is)
    if lbi_flag
        @constraint(model, w_mip_lbi, sc * w >= sc * (il .* lbi .- ss * (1 .- ilb)))
    end
    if sbi_flag
        @constraint(model, w_mip_sbi, sc * w <= sc * (is .* sbi .+ ss * (1 .- isb)))
    end
    if ffl_flag || ffs_flag
        fees = model[:fees]
        if ffl_flag
            @expression(model, ffl, sum(ffl .* il))
            add_to_expression!(fees, ffl)
        end
        if ffs_flag
            @expression(model, ffs, sum(ffs .* is))
            add_to_expression!(fees, ffs)
        end
    end
    return ib
end
function _mip_constraints(model::JuMP.Model, wb::WeightBounds,
                          ffl::Union{<:Real, <:AbstractVector},
                          lbi::Union{<:Real, <:AbstractVector}, lbi_flag::Bool,
                          ffl_flag::Bool)
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variable(model, ib[1:N], binary = true)
    if isa(k, Real)
        @expression(model, i_mip, ib)
    else
        ss = model[:ss]
        @variable(model, ibf[1:N] >= 0)
        @constraints(model, begin
                         ibf_ub, ibf <= k
                         ibfd_ub, ibf <= ss * ib
                         ibfd_lb, ibf >= k .- ss * (1 .- ib)
                     end)
        @expression(model, i_mip, ibf)
    end
    ub = wb.ub
    if !isnothing(ub) && isfinite(ub)
        @constraint(model, w_mip_ub, sc * w .<= sc * i_mip .* ub)
    end
    if lbi_flag
        @constraint(model, w_mip_lbi, sc * w .>= sc * i_mip .* lbi)
    end
    if ffl_flag
        fees = model[:fees]
        @expression(model, ffl, sum(ffl .* i_mip))
        add_to_expression!(fees, ffl)
    end
    if haskey(model, :sw)
        @constraint(model, w_mip_lb, sc * w >= sc * i_mip .* wb.lb)
    end
    return ib
end
function set_mip_constraints!(model::JuMP.Model, bit::Union{Nothing, <:BuyInThreshold},
                              card::Union{Nothing, <:Integer},
                              gcard::Union{Nothing, <:LinearConstraintResult},
                              fees::Union{Nothing, <:Fees},
                              nplg::Union{Nothing, <:PhilogenyConstraintResult},
                              cplg::Union{Nothing, <:PhilogenyConstraintResult},
                              wb::WeightBounds)
    lbi_flag, sbi_flag = if !isnothing(bit)
        non_zero_real_or_vec(bit.lbi), non_zero_real_or_vec(bit.sbi)
    else
        false, false
    end
    card_flag = !isnothing(card)
    gcard_flag = !isa(gcard,
                      Union{Nothing,
                            <:LinearConstraintResult{<:PartialLinearConstraintResult{Nothing,
                                                                                     Nothing},
                                                     <:PartialLinearConstraintResult{Nothing,
                                                                                     Nothing}}})
    ffl_flag, ffs_flag = if !isnothing(fees)
        non_zero_real_or_vec(fees.fixed_long)
        non_zero_real_or_vec(fees.fixed_short)
    else
        false, false
    end
    n_flag = isa(nplg, IntegerPhilogenyResult)
    c_flag = isa(cplg, IntegerPhilogenyResult)
    if !(lbi_flag ||
         sbi_flag ||
         card_flag ||
         gcard_flag ||
         ffl_flag ||
         ffs_flag ||
         n_flag ||
         c_flag)
        return nothing
    end
    ib = if (sbi_flag || ffl_flag || ffs_flag) && haskey(model, :sw)
        _short_mip_threshold_constraints(model, wb, fees.fixed_long, fees.fixed_short,
                                         bit.lbi, bit.sbi, lbi_flag, sbi_flag, ffl_flag,
                                         ffs_flag)
    else
        _mip_constraints(model, wb, fees.fixed_long, bit.lbi, lbi_flag, ffl_flag)
    end

    sc = model[:sc]
    if card_flag
        @constraint(model, card, sc * sum(ib) <= sc * card)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            @constraint(model, gcard_ineq, sc * A * ib <= sc * B)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            @constraint(model, gcard_eq, sc * A * ib == sc * B)
        end
    end
    if n_flag
        A = nplg.A
        B = nplg.B
        @constraint(model, card_nplg, sc * A * ib <= sc * B)
    end
    if c_flag
        A = cplg.A
        B = cplg.B
        @constraint(model, card_cplg, sc * A * ib <= sc * B)
    end
    return nothing
end
