function add_to_fees!(model::JuMP.Model, expr::AbstractJuMPScalar)
    if !haskey(model, :fees)
        @expression(model, fees, expr)
    else
        fees = model[:fees]
        add_to_expression!(fees, expr)
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
function set_long_non_fixed_fees!(model::JuMP.Model, fl::UNumVec)
    lw = model[:lw]
    @expression(model, fl, dot_scalar(fl, lw))
    add_to_fees!(model, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::UNumVec)
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
