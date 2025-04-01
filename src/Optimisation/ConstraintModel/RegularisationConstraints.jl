function set_l1_regularisation!(model::JuMP.Model, l1::Real)
    sc = model[:sc]
    w = model[:w]
    @variable(model, t_l1)
    @constraint(model, l1_noc, sc * [t_l1; w] in MOI.NormOneCone(1 + length(w)))
    @expression(model, l1, l1 * t_l1)
    add_to_expression!(model[:op], l1)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2::Real)
    sc = model[:sc]
    w = model[:w]
    @variable(model, t_l2)
    @constraint(model, l2_soc, sc * [t_l2; w] in SecondOrderCone())
    @expression(model, l2, l2 * t_l2)
    add_to_expression!(model[:op], l2)
    return nothing
end
