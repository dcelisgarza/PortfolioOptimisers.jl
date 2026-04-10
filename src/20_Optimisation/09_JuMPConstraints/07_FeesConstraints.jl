"""
    add_to_fees!(model, expr)

Accumulate `expr` into the model's `:fees` expression, creating it if absent.
"""
function add_to_fees!(model::JuMP.Model, expr::JuMP.AbstractJuMPScalar)
    if !haskey(model, :fees)
        JuMP.@expression(model, fees, expr)
    else
        fees = model[:fees]
        JuMP.add_to_expression!(fees, expr)
    end
    return nothing
end
"""
    set_turnover_fees!(args...)
    set_turnover_fees!(model, tn)

Add turnover-based transaction fee expression to the JuMP model.

The fall-through method does nothing. The concrete method computes
``\\mathrm{val}^\\intercal |w - w_t|`` and accumulates it into `:fees`.
"""
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
    JuMP.@variable(model, t_ftn[1:N])
    JuMP.@expressions(model, begin
                          x_ftn, w - wt * k
                          ftn, dot_scalar(val, t_ftn)
                      end)
    JuMP.@constraint(model, cftn[i = 1:N],
                     [sc * t_ftn[i]; sc * x_ftn[i]] in JuMP.MOI.NormOneCone(2))
    add_to_fees!(model, ftn)
    return nothing
end
"""
    set_non_fixed_fees!(args...)
    set_non_fixed_fees!(model, fees)

Add all non-fixed (proportional and turnover) fee expressions to the JuMP model.

The fall-through method does nothing. The concrete method delegates to
`set_long_non_fixed_fees!`, `set_short_non_fixed_fees!`, and `set_turnover_fees!`.
"""
function set_non_fixed_fees!(args...)
    return nothing
end
"""
    set_long_non_fixed_fees!(args...)
    set_long_non_fixed_fees!(model, fl)

Add proportional long-side fee expression `fl' * lw` to the JuMP model.
"""
function set_long_non_fixed_fees!(args...)
    return nothing
end
"""
    set_short_non_fixed_fees!(args...)
    set_short_non_fixed_fees!(model, fs)

Add proportional short-side fee expression `fs' * sw` to the JuMP model.
Does nothing when no short-weight variable `:sw` exists.
"""
function set_short_non_fixed_fees!(args...)
    return nothing
end
function set_long_non_fixed_fees!(model::JuMP.Model, fl::Num_VecNum)
    lw = model[:lw]
    JuMP.@expression(model, fl, dot_scalar(fl, lw))
    add_to_fees!(model, fl)
    return nothing
end
function set_short_non_fixed_fees!(model::JuMP.Model, fs::Num_VecNum)
    if !haskey(model, :sw)
        return nothing
    end
    sw = model[:sw]
    JuMP.@expression(model, fs, dot_scalar(fs, sw))
    add_to_fees!(model, fs)
    return nothing
end
function set_non_fixed_fees!(model::JuMP.Model, fees::Fees)
    set_long_non_fixed_fees!(model, fees.l)
    set_short_non_fixed_fees!(model, fees.s)
    set_turnover_fees!(model, fees.tn)
    return nothing
end
