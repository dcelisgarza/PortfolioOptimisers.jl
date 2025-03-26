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
            @smart_assert(all(isfinite.(sbi)) && all(sbi .>= zero(sbi)))
        end
        if lbi_flag
            @smart_assert(isfinite(lbi) && lbi >= zero(lbi))
        else
            @smart_assert(all(isfinite.(lbi)) && all(lbi .>= zero(lbi)))
        end
        if !sbi_flag && !lbi_flag
            @smart_assert(length(sbi) == length(lbi))
        end
    elseif lbi_flag && !sbi_flag
        if isa(sbi, Real)
            @smart_assert(isfinite(sbi) && sbi >= zero(sbi))
        else
            @smart_assert(all(isfinite.(sbi)) && all(sbi .>= zero(sbi)))
        end
    elseif !lbi_flag && sbi_flag
        if isa(lbi, Real)
            @smart_assert(isfinite(lbi) && lbi >= zero(lbi))
        else
            @smart_assert(all(isfinite.(lbi)) && all(lbi .>= zero(lbi)))
        end
    end
    return BuyInThreshold{typeof(sbi), typeof(lbi)}(sbi, lbi)
end
function _short_mip_threshold_constraints(model::JuMP.Model, ss::Real, wb::WeightBounds,
                                          ffl::Union{<:Real, <:AbstractVector},
                                          ffs::Union{<:Real, <:AbstractVector},
                                          lbi::Union{<:Real, <:AbstractVector},
                                          sbi::Union{<:Real, <:AbstractVector},
                                          lbi_flag::Bool, sbi_flag::Bool, ffl_flag::Bool,
                                          ffs_flag::Bool)
    w, k, sc = get_w_k_sc(model)
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
    @constraints(model, begin
                     i_mip_ub, i_mip <= 1
                     w_mip_ub, sc * w <= sc * il .* wb.ub
                     w_mip_lb, sc * w >= sc * is .* wb.lb
                 end)

    if lbi_flag
        @constraint(model, w_mip_lbi, sc * w >= sc * (il .* lbi .- ss * (1 .- ilb)))
    end
    if sbi_flag
        @constraint(model, w_mip_sbi, sc * w <= sc * (is .* sbi .+ ss * (1 .- isb)))
    end
    if ffl_flag
        @expression(model, ffl, sum(ffl .* il))
    end
    if ffs_flag
        @expression(model, ffs, sum(ffs .* is))
    end
    return ib
end
function _mip_constraints(model::JuMP.Model, ss::Real, wb::WeightBounds,
                          ffl::Union{<:Real, <:AbstractVector},
                          lbi::Union{<:Real, <:AbstractVector}, lbi_flag::Bool,
                          ffl_flag::Bool)
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variable(model, ib[1:N], binary = true)
    if isa(k, Real)
        @expression(model, i_mip, ib)
    else
        @variable(model, ibf[1:N] >= 0)
        @constraints(model, begin
                         ibf_ub, ibf <= k
                         ibfd_ub, ibf <= ss * ib
                         ibfd_lb, ibf >= k .- ss * (1 .- ib)
                     end)
        @expression(model, i_mip, ibf)
    end
    @constraint(model, w_mip_ub, sc * w <= sc * i_mip .* wb.ub)
    if lbi_flag
        @constraint(model, w_mip_lbi, sc * w >= sc * i_mip .* lbi)
    end
    if ffl_flag
        @expression(model, ffl, sum(ffl .* i_mip))
    end
    if haskey(model, :sw)
        @constraint(model, w_mip_lb, sc * w >= sc * i_mip .* wb.lb)
    end
    return ib
end
function set_mip_constraints!(model::JuMP.Model, card::Integer,
                              gcard::Union{Nothing, <:LinearConstraintModel},
                              bit::Union{Nothing, <:BuyInThreshold},
                              fees::Union{Nothing, <:Fees},
                              cadj::Union{Nothing, <:AdjacencyConstraint},
                              nadj::Union{Nothing, <:AdjacencyConstraint}, wb::WeightBounds,
                              ss::Real = 100_000.0)
    card_flag = card > zero(card)
    gcard_flag = !isa(gcard,
                      Union{Nothing,
                            <:LinearConstraintModel{<:PartialLinearConstraintModel{Nothing,
                                                                                   Nothing},
                                                    <:PartialLinearConstraintModel{Nothing,
                                                                                   Nothing}}})
    lbi_flag, sbi_flag = if !isnothing(bit)
        non_zero_real_or_vec(bit.lbi), non_zero_real_or_vec(bit.sbi)
    else
        false, false
    end
    ffl_flag, ffs_flag = if !isnothing(fees)
        non_zero_real_or_vec(fees.fixed_long)
        non_zero_real_or_vec(fees.fixed_short)
    else
        false, false
    end
    c_flag = isa(cadj, IntegerAdjacency)
    n_flag = isa(nadj, IntegerAdjacency)
    if !(card_flag ||
         gcard_flag ||
         lbi_flag ||
         sbi_flag ||
         ffl_flag ||
         ffs_flag ||
         n_flag ||
         c_flag)
        return nothing
    end
    ib = if (sbi_flag || ffl_flag || ffs_flag) && haskey(model, :sw)
        _short_mip_threshold_constraints(model, ss, wb, fees.fixed_long, fees.fixed_short,
                                         bit.lbi, bit.sbi, lbi_flag, sbi_flag, ffl_flag,
                                         ffs_flag)
    else
        _mip_constraints(model, ss, wb, fees.fixed_long, bit.lbi, lbi_flag, ffl_flag)
    end
    if card_flag
        @constraint(model, card, sum(ib) <= card)
    end
    if gcard_flag
        w, k, sc = get_w_k_sc(model)
        if !isnothing(gcard.A_ineq)
            @constraint(model, gcard_ineq, sc * gcard.A_ineq * w <= sc * k * gcard.B_ineq)
        end
        if !isnothing(gcard.A_eq)
            @constraint(model, gcard_eq, sc * gcard.A_eq * w == sc * k * gcard.B_eq)
        end
        @constraint(model, gcard, gcard.A * ib <= gcard.B)
    end
    if c_flag
        @constraint(model, ccard, cadj.A * ib <= cadj.B)
    end
    if n_flag
        @constraint(model, ncard, nadj.A * ib <= nadj.B)
    end
    return nothing
end
