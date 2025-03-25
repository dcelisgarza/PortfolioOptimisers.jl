struct BuyInThreshold{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                      T2 <: Union{<:Real, <:AbstractVector{<:Real}}}
    sbi::T1
    lbi::T2
end
function BuyInThreshold(; sbi::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                        lbi::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)
    if isa(sbi, Real)
        @smart_assert(isfinite(sbi) && isfinite(lbi))
        @smart_assert(sbi >= zero(sbi) && lbi >= zero(lbi))
    else
        @smart_assert(!isempty(sbi) && !isempty(lbi) && length(sbi) == length(lbi))
        @smart_assert(all(sbi .>= zero(sbi)) && all(lbi .>= zero(lbi)))
    end
    return BuyInThreshold{typeof(sbi), typeof(lbi)}(sbi, lbi)
end
function _short_mip_threshold_constraints(model::JuMP.Model, ss::Real, wl::WeightLimits,
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
                     w_mip_ub, sc * w <= sc * il .* wl.ub
                     w_mip_lb, sc * w >= sc * is .* wl.lb
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
function _mip_constraints(model::JuMP.Model, ss::Real, wl::WeightLimits,
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
    @constraint(model, w_mip_ub, sc * w <= sc * i_mip .* wl.ub)
    if lbi_flag
        @constraint(model, w_mip_lbi, sc * w >= sc * i_mip .* lbi)
    end
    if ffl_flag
        @expression(model, ffl, sum(ffl .* i_mip))
    end
    if haskey(model, :sw)
        @constraint(model, w_mip_lb, sc * w >= sc * i_mip .* wl.lb)
    end
    return ib
end
function set_mip_constraints!(model::JuMP.Model, card::Integer,
                              gcard::PartialLinearConstraintModel, bit::BuyInThreshold,
                              fees::Fees, cadj::AdjacencyConstraint,
                              nadj::AdjacencyConstraint, wl::WeightLimits,
                              ss::Real = 100_000.0)
    card_flag = card > zero(card)
    gcard_flag = !isa(gcard, <:PartialLinearConstraintModel{Nothing, Nothing})
    lbi_flag = non_zero_real_or_vec(bit.lbi)
    sbi_flag = non_zero_real_or_vec(bit.sbi)
    ffl_flag = non_zero_real_or_vec(fees.fixed_long)
    ffs_flag = non_zero_real_or_vec(fees.fixed_short)
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
        _short_mip_threshold_constraints(model, ss, wl, fees.fixed_long, fees.fixed_short,
                                         bit.lbi, bit.sbi, lbi_flag, sbi_flag, ffl_flag,
                                         ffs_flag)
    else
        _mip_constraints(model, ss, wl, fees.fixed_long, bit.lbi, lbi_flag, ffl_flag)
    end
    if card_flag
        @constraint(model, card, sum(ib) <= card)
    end
    if gcard_flag
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
