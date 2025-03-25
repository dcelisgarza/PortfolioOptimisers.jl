function set_turnover_fees!(::JuMP.Model, ::NoTurnover)
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
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    if haskey(model, :lw) && non_zero_real_or_vec(fees.long)
        @expression(model, fl, sum(fees.long .* lw))
    elseif !haskey(model, :lw) && non_zero_real_or_vec(fees.long)
        @warn("Long fees require")
        w, k, sc = get_w_k_sc(model)
        N = length(w)
        @variable(model, lw[1:N] >= 0)
        @constraints(model, begin
                         w_lw, sc * w <= sc * lw
                         lw_ub, sc * sum(lw) <= sc * k * 1
                     end)
        @expression(model, fl, sum(fees.long .* lw))
    end
    if haskey(model, :sw) && non_zero_real_or_vec(fees.short)
        sw = model[:sw]
        @expression(model, fs, sum(fees.short .* sw))
    end
    set_turnover_fees!(model, fees.turnover)
    return nothing
end
