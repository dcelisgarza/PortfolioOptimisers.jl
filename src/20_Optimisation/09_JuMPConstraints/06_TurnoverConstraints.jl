"""
    set_turnover_constraints!(args...)
    set_turnover_constraints!(model, tns)

Add turnover constraints to the JuMP model for each [`Turnover`](@ref) in `tns`.

The fall-through method does nothing. Each concrete constraint limits
``\\|w - w_b\\|_1 \\le val``.
"""
function set_turnover_constraints!(args...)
    return nothing
end
"""
    _set_turnover_constraints!(model, tn, i=1)

Add a single turnover constraint for [`Turnover`](@ref) `tn` to the JuMP model,
using index `i` to generate unique variable/constraint names.
"""
function _set_turnover_constraints!(model::JuMP.Model, tn::Turnover, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wb = tn.w
    val = tn.val
    t_tn = model[Symbol(:t_tn_, i)] = JuMP.@variable(model, [1:N])
    tn = model[Symbol(:tn_, i)] = JuMP.@expression(model, w - wb * k)
    model[Symbol(:ctn_noc_, i)], model[Symbol(:ctn_, i)] = JuMP.@constraints(model,
                                                                             begin
                                                                                 [i = 1:N],
                                                                                 [sc *
                                                                                  t_tn[i]
                                                                                  sc *
                                                                                  tn[i]] in
                                                                                 JuMP.MOI.NormOneCone(2)
                                                                                 sc *
                                                                                 (t_tn ⊖
                                                                                  val * k) <=
                                                                                 0
                                                                             end)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model, tns::Tn_VecTn)
    for (i, tn) in enumerate(tns)
        _set_turnover_constraints!(model, tn, i)
    end
    return nothing
end
