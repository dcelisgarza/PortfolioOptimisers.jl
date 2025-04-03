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
function set_long_non_fixed_fees!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_short_non_fixed_fees!(::JuMP.Model, ::Nothing)
    return nothing
end
function set_long_non_fixed_fees!(model::JuMP.Model, fl::Union{<:Real, <:AbstractVector})
    lw = model[:lw]
    fees = model[:fees]
    @expression(model, fl, sum(fl .* lw))
    add_to_expression!(fees, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::Union{<:Real, <:AbstractVector})
    if !haskey(model, :sw)
        return nothing
    end
    sw = model[:sw]
    fees = model[:fees]
    @expression(model, fs, sum(fs .* sw))
    add_to_expression!(fees, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_long_non_fixed_fees!(model, fees.long)
    set_short_non_fixed_fees!(model, fees.short)
    set_turnover_fees!(model, fees.turnover)
    return nothing
end
