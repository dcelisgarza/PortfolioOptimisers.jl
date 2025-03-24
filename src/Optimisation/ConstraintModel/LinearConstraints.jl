function set_linear_weight_constraints!(model::JuMP.Model, lcm::LinearConstraintModel)
    ineq_flag = isnothing(lcm.A_ineq)
    eq_flag = isnothing(lcm.A_eq)
    if ineq_flag && eq_flag
        return nothing
    end
    w, k, sc = get_w_k_sc(model)
    if ineq_flag
        @constraint(model, c_ineq, sc * lcm.A_ineq * w <= sc * k * lcm.B_ineq)
    end
    if eq_flag
        @constraint(model, c_eq, sc * lcm.A_eq * w == sc * k * lcm.B_eq)
    end
    return nothing
end
