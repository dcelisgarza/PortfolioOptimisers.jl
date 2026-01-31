function set_l1_regularisation!(args...)
    return nothing
end
function set_l2_regularisation!(args...)
    return nothing
end
function set_ln_regularisation!(args...)
    return nothing
end
function set_linf_regularisation!(args...)
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
struct LnRegularisation{T1, T2} <: AbstractEstimator
    n::T1
    val::T2
    function LnRegularisation(n::Number, val::Number)
        assert_nonempty_gt0_finite_val(n, :n)
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(n), typeof(val)}(n, val)
    end
end
function LnRegularisation(; n::Number = 3, val::Number = 1e-3)
    return LnRegularisation(n, val)
end
const VecLnReg = AbstractVector{<:LnRegularisation}
const LnReg_VecLnReg = Union{<:LnRegularisation, <:VecLnReg}
function set_ln_regularisation!(model::JuMP.Model, lns::LnReg_VecLnReg)
    w = model[:w]
    sc = model[:sc]
    N = length(w)
    for (i, ln) in enumerate(lns)
        val = ln.val
        n_inv = inv(ln.n)
        t_ln, r_ln = model[Symbol(:t_ln_, i)], model[Symbol(:r_ln_, i)] = JuMP.@variables(model,
                                                                                          begin
                                                                                              ()
                                                                                              [1:N]
                                                                                          end)
        model[(Symbol(:cln_, i))], model[Symbol(:csln_, i)] = JuMP.@constraints(model,
                                                                                begin
                                                                                    [i = 1:N],
                                                                                    [sc *
                                                                                     r_ln[i],
                                                                                     1,
                                                                                     sc *
                                                                                     w[i]] in
                                                                                    JuMP.MOI.PowerCone(n_inv)
                                                                                    sc *
                                                                                    (sum(r_ln) -
                                                                                     t_ln) ==
                                                                                    0
                                                                                end)
        ln_expr = model[Symbol(:ln_, i)] = JuMP.@expression(model, val * t_ln)
        add_to_objective_penalty!(model, ln_expr)
    end
end
function set_linf_regularisation!(model::JuMP.Model, linf_val::Number)
    w = model[:w]
    sc = model[:sc]
    JuMP.@variable(model, t_linf)
    JuMP.@constraint(model, clinf_noc,
                     [sc * t_linf; sc * w] in JuMP.MOI.NormInfinityCone(1 + length(w)))
    JuMP.@expression(model, linf, linf_val * t_linf)
    add_to_objective_penalty!(model, linf)
    return nothing
end

export LnRegularisation
