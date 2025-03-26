function set_linear_weight_constraints!(::JuMP.Model,
                                        ::Union{Nothing,
                                                <:LinearConstraintModel{<:PartialLinearConstraintModel{Nothing,
                                                                                                       Nothing},
                                                                        <:PartialLinearConstraintModel{Nothing,
                                                                                                       Nothing}}})
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcm::LinearConstraintModel)
    w, k, sc = get_w_k_sc(model)
    if !isnothing(lcm.A_ineq)
        @constraint(model, c_ineq, sc * lcm.A_ineq * w <= sc * k * lcm.B_ineq)
    end
    if !isnothing(lcm.A_eq)
        @constraint(model, c_eq, sc * lcm.A_eq * w == sc * k * lcm.B_eq)
    end
    return nothing
end
