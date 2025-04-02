function set_linear_weight_constraints!(::JuMP.Model,
                                        ::Union{Nothing,
                                                <:LinearConstraintModel{<:PartialLinearConstraintModel{Nothing,
                                                                                                       Nothing},
                                                                        <:PartialLinearConstraintModel{Nothing,
                                                                                                       Nothing}}},
                                        args...)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcm::LinearConstraintModel,
                                        key_ineq::Symbol, key_eq::Symbol)
    w, k, sc = get_w_k_sc(model)
    if !isnothing(lcm.A_ineq)
        model[key_ineq] = @constraint(model, sc * lcm.A_ineq * w <= sc * k * lcm.B_ineq)
    end
    if !isnothing(lcm.A_eq)
        model[key_eq] = @constraint(model, sc * lcm.A_eq * w == sc * k * lcm.B_eq)
    end
    return nothing
end
