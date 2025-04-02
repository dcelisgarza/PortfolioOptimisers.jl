function set_turnover_fees!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_turnover_fees!(model::JuMP.Model, turnover::Turnover)
    if !non_zero_real_or_vec(turnover.val)
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    N = length(w)
    @variable(model, t_trfs[1:N])
    @expressions(model, begin
                     x_trfs, w - turnover.w * k
                     trfs, sum(turnover.val .* t_trfs)
                 end)
    @constraint(model, ctrfs[i = 1:N], sc * [t_trfs[i]; x_trfs[i]] ∈ MOI.NormOneCone(2))
    return nothing
end
function set_non_fixed_fees!(::JuMP.Model, args...)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, wb::WeightBounds, fees::Fees)
    if non_zero_real_or_vec(fees.long)
        if !isnothing(wb.lb) && all(wb.lb .>= zero(wb.lb))
            @expression(model, fl, sum(fees.long .* w))
        elseif haskey(model, :lw)
            @expression(model, fl, sum(fees.long .* lw))
        else
            ub = !isnothing(wb.ub) ? maximum(wb.ub) : 1.0
            @warn("Long fees require setting long bounds, default to $ub.")
            w, k, sc = get_w_k_sc(model)
            N = length(w)
            @variable(model, lw[1:N] >= 0)
            @constraints(model, begin
                             w_lw, sc * w <= sc * lw
                             lw_ub, sc * sum(lw) <= sc * k * ub
                         end)
            @expression(model, fl, sum(fees.long .* lw))
        end
    end
    if non_zero_real_or_vec(fees.short)
        if !isnothing(wb.ub) && all(wb.ub .< zero(wb.ub))
            @expression(model, fs, -sum(fees.short .* w))
        elseif haskey(model, :sw)
            sw = model[:sw]
            @expression(model, fs, sum(fees.short .* sw))
        else
            lb = !isnothing(wb.lb) ? minimum(wb.lb) : -1.0
            @warn("Short fees require setting short bounds, default to $lb.")
            w, k, sc = get_w_k_sc(model)
            N = length(w)
            @variable(model, sw[1:N] >= 0)
            @constraints(model, begin
                             w_sw, sc * w >= -sc * sw
                             sw_lb, sc * sum(sw) <= sc * k * lss.ss
                         end)
            @expression(model, fs, sum(fees.short .* sw))
        end
    end
    set_turnover_fees!(model, fees.turnover)
    return nothing
end
