function set_turnover_constraints!(args...)
    return nothing
end
function _set_turnover_constraints!(model::JuMP.Model, tn::Turnover, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wb = tn.w
    val = tn.val
    t_tn = model[Symbol(:t_tn_, i)] = @variable(model, [1:N])
    tn = model[Symbol(:tn_, i)] = @expression(model, w - wb * k)
    model[Symbol(:ctn_noc_, i)], model[Symbol(:ctn_, i)] = @constraints(model,
                                                                        begin
                                                                            [i = 1:N],
                                                                            [sc * t_tn[i];
                                                                             sc * tn[i]] in
                                                                            MOI.NormOneCone(2)
                                                                            sc *
                                                                            (t_tn ⊖ val * k) <=
                                                                            0
                                                                        end)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model, tns::TnVecTn)
    for (i, tn) in enumerate(tns)
        _set_turnover_constraints!(model, tn, i)
    end
    return nothing
end
