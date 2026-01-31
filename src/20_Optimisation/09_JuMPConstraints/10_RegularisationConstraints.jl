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
struct LnRegularisation{T1, T2}
    n::T1
    val::T2
    function LnRegularisation(n::Num_VecNum, val::Num_VecNum)
        n_flag = isa(n, VecNum)
        val_flag = isa(val, VecNum)
        if n_flag || val_flag
            @argcheck(n_flag && val_flag,
                      "If either `n` or `val` is a vector, both must be vectors of the same length.")
            @argcheck(length(n) == length(val))
        end
        return new{typeof(n), typeof(val)}(n, val)
    end
end
function LnRegularisation(; n::Num_VecNum = 3, val::Num_VecNum = 1e-3)
    return LnRegularisation(n, val)
end
function set_ln_regularisation(model::JuMP.Model, ln::LnRegularisation)
    w = model[:w]
    sc = model[:sc]
    for (i, (n, val)) in enumerate(zip(ln.n, ln.val))
        t_ln, ln = model[Symbol(:t_ln_, i)], model[Symbol(:ln_, i)] = JuMP.@variable(model,
                                                                                     (), ())
        model[(Symbol(:cln_, i))] = JuMP.@constraint(model,
                                                     [sc * t_ln, 1, t_ln] in
                                                     JuMP.MOI.PowerCone(inv(n)))
    end
end
