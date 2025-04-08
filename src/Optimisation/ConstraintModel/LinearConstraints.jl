function set_linear_weight_constraints!(::JuMP.Model,
                                        ::Union{Nothing,
                                                <:LinearConstraintResult{<:PartialLinearConstraintResult{Nothing,
                                                                                                         Nothing},
                                                                         <:PartialLinearConstraintResult{Nothing,
                                                                                                         Nothing}}},
                                        ::Symbol, ::Symbol)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcm::LinearConstraintResult,
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
