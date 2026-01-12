function set_l1_regularisation!(args...)
    return nothing
end
function set_l2_regularisation!(args...)
    return nothing
end
function set_l1_regularisation!(model::JuMP.Model, l1_val::Number)
    w = model[:w]
    sc = model[:sc]
    JuMP.@variable(model, t_l1)
    JuMP.@constraint(model, cl1_noc,
                     [sc * t_l1; sc * w] in JuMP.MOI.NormOneCone(1 + length(w)))
    JuMP.@expression(model, l1, l1_val * t_l1)
    add_to_objective_penalty!(model, l1)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2_val::Number)
    w = model[:w]
    sc = model[:sc]
    JuMP.@variable(model, t_l2)
    JuMP.@constraint(model, cl2_soc, [sc * t_l2; sc * w] in JuMP.SecondOrderCone())
    JuMP.@expression(model, l2, l2_val * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
