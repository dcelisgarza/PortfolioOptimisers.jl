function set_turnover_fees!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_turnover_fees!(model::JuMP.Model, turnover::Turnover)
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variable(model, t_ftr[1:N])
    @expressions(model, begin
                     x_ftr, w - turnover.w * k
                     ftr, sum(turnover.val .* t_ftr)
                 end)
    @constraint(model, cftr[i = 1:N], [sc * t_ftr[i]; sc * x_ftr[i]] ∈ MOI.NormOneCone(2))
    fees = model[:fees]
    add_to_expression!(fees, ftr)
    return nothing
end
function set_non_fixed_fees!(::JuMP.Model, args...)
    return nothing
end
function set_long_non_fixed_fees!(::JuMP.Model, ::LongShortSum, ::Nothing)
    return nothing
end
function set_short_non_fixed_fees!(::JuMP.Model, ::LongShortSum, ::Nothing)
    return nothing
end
function set_long_non_fixed_fees!(model::JuMP.Model, lss::LongShortSum,
                                  fl::Union{<:Real, <:AbstractVector})
    fees = model[:fees]
    ls = lss.ls
    if !isnothing(ls)
        @expression(model, fl, sum(fl .* w))
    elseif haskey(model, :lw)
        @expression(model, fl, sum(fl .* lw))
    else
        @warn("Long fees require setting long bounds, default to 1.0.")
        w, k, sc = get_w_k_sc(model)
        N = length(w)
        @variable(model, lw[1:N] >= 0)
        @constraints(model, begin
                         w_lw, sc * w <= sc * lw
                         lw_ub, sc * sum(lw) <= sc * k
                     end)
        @expression(model, fl, sum(fl .* lw))
    end
    add_to_expression!(fees, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, lss::LongShortSum,
                                   fs::Union{<:Real, <:AbstractVector})
    fees = model[:fees]
    ss = lss.ss
    if !isnothing(ss)
        @expression(model, fs, sum(fs .* w))
    elseif haskey(model, :sw)
        @expression(model, fs, sum(fs .* sw))
    else
        @warn("Short fees require setting short bounds, default to 1.0.")
        w, k, sc = get_w_k_sc(model)
        N = length(w)
        @variable(model, sw[1:N] >= 0)
        @constraints(model, begin
                         w_sw, sc * w >= -sc * sw
                         sw_lb, sc * sum(sw) <= sc * k
                     end)
        @expression(model, fs, sum(fs .* sw))
    end
    add_to_expression!(fees, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, lss::LongShortSum, fees::Fees)
    set_long_non_fixed_fees!(model, lss, fees.long)
    set_short_non_fixed_fees!(model, lss, fees.short)
    set_turnover_fees!(model, fees.turnover)
    return nothing
end
