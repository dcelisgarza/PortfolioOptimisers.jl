function set_l1_regularisation!(args...)
    return nothing
end
function set_l2_regularisation!(args...)
    return nothing
end
function set_l1_regularisation!(model::JuMP.Model, l1::Real)
    w = model[:w]
    sc = model[:sc]
    @variable(model, t_l1)
    @constraint(model, cl1_noc, [sc * t_l1; sc * w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1, l1 * t_l1)
    add_to_objective_penalty!(model, l1)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2::Real)
    w = model[:w]
    sc = model[:sc]
    @variable(model, t_l2)
    @constraint(model, cl2_soc, [sc * t_l2; sc * w] in SecondOrderCone())
    @expression(model, l2, l2 * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
