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
    if !isnothing(lcm.ineq)
        A = lcm.ineq.A
        B = lcm.ineq.B
        model[key_ineq] = @constraint(model, sc * A * w <= sc * k * B)
    end
    if !isnothing(lcm.eq)
        A = lcm.eq.A
        B = lcm.eq.B
        model[key_eq] = @constraint(model, sc * A * w == sc * k * B)
    end
    return nothing
end
