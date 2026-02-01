function set_l1_regularisation!(args...)
    return nothing
end
function set_l2_regularisation!(args...)
    return nothing
end
function set_lp_regularisation!(args...)
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
struct LpRegularisation{T1, T2} <: AbstractEstimator
    p::T1
    val::T2
    function LpRegularisation(p::Number, val::Number)
        assert_nonempty_gt0_finite_val(p, :p)
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(p), typeof(val)}(p, val)
    end
end
function LpRegularisation(; p::Number = 3, val::Number = 1e-3)
    return LpRegularisation(p, val)
end
const VecLpReg = AbstractVector{<:LpRegularisation}
const LpReg_VecLpReg = Union{<:LpRegularisation, <:VecLpReg}
function set_lp_regularisation!(model::JuMP.Model, lps::LpReg_VecLpReg)
    w = model[:w]
    sc = model[:sc]
    N = length(w)
    for (i, lp) in enumerate(lps)
        val = lp.val
        p_inv = inv(lp.p)
        t_lp, r_lp = model[Symbol(:t_lp_, i)], model[Symbol(:r_lp_, i)] = JuMP.@variables(model,
                                                                                          begin
                                                                                              ()
                                                                                              [1:N]
                                                                                          end)
        model[(Symbol(:clp_, i))], model[Symbol(:cslp_, i)] = JuMP.@constraints(model,
                                                                                begin
                                                                                    [i = 1:N],
                                                                                    [sc *
                                                                                     r_lp[i],
                                                                                     1,
                                                                                     sc *
                                                                                     w[i]] in
                                                                                    JuMP.MOI.PowerCone(p_inv)
                                                                                    sc *
                                                                                    (sum(r_lp) -
                                                                                     t_lp) ==
                                                                                    0
                                                                                end)
        lp_expr = model[Symbol(:lp_, i)] = JuMP.@expression(model, val * t_lp)
        add_to_objective_penalty!(model, lp_expr)
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

export LpRegularisation
