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
        A = lcm.A_ineq
        B = lcm.B_ineq
        model[key_ineq] = @constraint(model, sc * A * w <= sc * k * B)
    end
    if !isnothing(lcm.A_eq)
        A = lcm.A_eq
        B = lcm.B_eq
        model[key_eq] = @constraint(model, sc * A * w == sc * k * B)
    end
    return nothing
end
