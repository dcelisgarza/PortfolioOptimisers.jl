"""
    set_l1_regularisation!(model::JuMP.Model, l1_val::Number)
    set_l1_regularisation!(args...)

Add an L1 regularisation penalty to the objective function.

The penalty is applied as given, so `l1_val` must be commensurate with the scale of the objective it is added to.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_1 \\lVert \\boldsymbol{w} \\rVert_1\\,.
\\end{align}
```

Where:

  - ``\\lambda_1``: L1 regularisation coefficient.
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `l1_val::Number`: L1 regularisation coefficient.

# Returns

  - `nothing`.

# Details

  - `l1_val::Number`: Introduces the auxiliary variable `t_l1`, bounds it with `[t_l1; w] in MOI.NormOneCone`, and adds `l1_val * t_l1` to the objective penalty via [`add_to_objective_penalty!`](@ref).
  - `args...`: No-op, used when no L1 regularisation is configured.

# Related

  - [`set_l2_regularisation!`](@ref)
  - [`set_lp_regularisation!`](@ref)
  - [`set_linf_regularisation!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_l1_regularisation!(args...)
    return nothing
end
"""
    set_l2_regularisation!(model::JuMP.Model, l2s::L2Reg_VecL2Reg)
    set_l2_regularisation!(args...)

Add one or more L2 regularisation penalties to the objective function.

Each [`L2Regularisation`](@ref) carries its own coefficient and cone formulation, so several may be combined in a single model. The penalties are applied as given, so each coefficient must be commensurate with the scale of the objective it is added to, and with the formulation used to express it.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\sum_{i} \\lambda_{2,\\, i} \\lVert \\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``\\lambda_{2,\\, i}``: L2 regularisation coefficient of the ``i``-th term.
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `l2s::L2Reg_VecL2Reg`: One or more L2 regularisation terms.

# Returns

  - `nothing`.

# Details

  - `l2s::L2Reg_VecL2Reg`: Dispatches each term to [`_set_l2_regularisation!`](@ref), which selects the cone formulation from the term's `alg` field. Variables and constraints are suffixed by the term's index, so terms do not collide.
  - `args...`: No-op, used when no L2 regularisation is configured.

# Related

  - [`L2Regularisation`](@ref)
  - [`L2Reg_VecL2Reg`](@ref)
  - [`_set_l2_regularisation!`](@ref)
  - [`set_l1_regularisation!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_l2_regularisation!(args...)
    return nothing
end
"""
    set_lp_regularisation!(model::JuMP.Model, lps::LpReg_VecLpReg)
    set_lp_regularisation!(args...)

Add one or more Lp regularisation penalties to the objective function.

Each [`LpRegularisation`](@ref) carries its own norm order and coefficient, so several may be combined in a single model. The penalties are applied as given, so each coefficient must be commensurate with the scale of the objective it is added to.

The same estimator also specifies a p-norm *constraint* when it is placed in the `lpc` field of [`JuMPOptimiser`](@ref) instead of the `lp` field, in which case its `val` field is an upper bound on the p-norm rather than a coefficient. See [`set_weight_norm_p_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\sum_{i} \\lambda_{p,\\, i} \\lVert \\boldsymbol{w} \\rVert_{p_i}\\,.
\\end{align}
```

Where:

  - ``\\lambda_{p,\\, i}``: Regularisation coefficient of the ``i``-th term, its `val` field.
  - ``p_i``: Norm order of the ``i``-th term, its `p` field.
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `lps::LpReg_VecLpReg`: One or more Lp regularisation terms.

# Returns

  - `nothing`.

# Details

  - `lps::LpReg_VecLpReg`: For each term, introduces the auxiliary variables `t_lp_i` and `r_lp_i`, bounds ``\\lVert \\boldsymbol{w} \\rVert_{p_i}`` above by `t_lp_i` with a set of `MOI.PowerCone` constraints, and adds `val * t_lp_i` to the objective penalty via [`add_to_objective_penalty!`](@ref). Variables and constraints are suffixed by the term's index, so terms do not collide.
  - `args...`: No-op, used when no Lp regularisation is configured.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`set_l1_regularisation!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_lp_regularisation!(args...)
    return nothing
end
"""
    set_linf_regularisation!(model::JuMP.Model, linf::Number)
    set_linf_regularisation!(args...)

Add an L∞ regularisation penalty to the objective function.

The penalty is applied as given, so `linf` must be commensurate with the scale of the objective it is added to and with the range of values the weights can take.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_\\infty \\lVert \\boldsymbol{w} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - ``\\lambda_\\infty``: L∞ regularisation coefficient.
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `linf::Number`: L∞ regularisation coefficient.

# Returns

  - `nothing`.

# Details

  - `linf::Number`: Introduces the auxiliary variable `t_linf`, bounds it with `[t_linf; w] in MOI.NormInfinityCone`, and adds `linf * t_linf` to the objective penalty via [`add_to_objective_penalty!`](@ref).
  - `args...`: No-op, used when no L∞ regularisation is configured.

# Related

  - [`set_l1_regularisation!`](@ref)
  - [`set_l2_regularisation!`](@ref)
  - [`set_lp_regularisation!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_linf_regularisation!(args...)
    return nothing
end
function set_l1_regularisation!(model::JuMP.Model, l1_val::Number)
    w = get_w(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, t_l1)
    JuMP.@constraint(model, cl1_noc,
                     [sc * t_l1; sc * w] in JuMP.MOI.NormOneCone(1 + length(w)))
    JuMP.@expression(model, l1, l1_val * t_l1)
    add_to_objective_penalty!(model, l1)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all portfolio weight regularisation estimators.

# Related

  - [`LpRegularisation`](@ref)
"""
abstract type AbstractRegularisationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

L2-norm regularisation term added to the optimisation objective.

Penalises concentrated portfolios by shrinking the weights towards zero. The `alg` field selects the formulation used to express the penalty, which determines both the cones the solver sees and the power of the norm being penalised, so `val` must be interpreted in the light of the chosen formulation — see [`_set_l2_regularisation!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2\\,, \\\\
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2^2\\,.
\\end{align}
```

Where:

  - ``\\lambda_2``: L2 regularisation coefficient, the `val` field.
  - $(math_dict[:w_port])

The first form is used by [`SOCRiskExpr`](@ref); the second by [`SquaredSOCRiskExpr`](@ref), [`QuadRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L2Regularisation(;
        val::Number = 1e-4,
        alg::SecondMomentFormulation = SOCRiskExpr()
    ) -> L2Regularisation

Keywords correspond to the struct's fields.

## Validation

  - `val > 0` and finite.

# Related

  - [`AbstractRegularisationEstimator`](@ref)
  - [`L2Reg_VecL2Reg`](@ref)
  - [`VecL2Reg`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`set_l2_regularisation!`](@ref)
  - [`_set_l2_regularisation!`](@ref)
  - [`LpRegularisation`](@ref)
"""
@concrete struct L2Regularisation <: AbstractRegularisationEstimator
    """
    $(field_dict[:l2reg_val])
    """
    val
    """
    $(field_dict[:l2reg_alg])
    """
    alg
    function L2Regularisation(val::Number, alg::SecondMomentFormulation)
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(val), typeof(alg)}(val, alg)
    end
end
function L2Regularisation(; val::Number = 1e-4,
                          alg::SecondMomentFormulation = SOCRiskExpr())
    return L2Regularisation(val, alg)
end
"""
    const VecL2Reg = AbstractVector{<:L2Regularisation}

Alias for a vector of [`L2Regularisation`](@ref) objects.

Represents a collection of L2-norm regularisation terms to be added to the optimisation objective.

# Related

  - [`L2Regularisation`](@ref)
  - [`L2Reg_VecL2Reg`](@ref)
  - [`set_l2_regularisation!`](@ref)
"""
const VecL2Reg = AbstractVector{<:L2Regularisation}
"""
    const L2Reg_VecL2Reg = Union{<:L2Regularisation, <:VecL2Reg}

Alias for a single or vector of L2 regularisation terms.

Matches either a single [`L2Regularisation`](@ref) or a vector of them ([`VecL2Reg`](@ref)).

# Related

  - [`L2Regularisation`](@ref)
  - [`VecL2Reg`](@ref)
  - [`set_l2_regularisation!`](@ref)
"""
const L2Reg_VecL2Reg = Union{<:L2Regularisation, <:VecL2Reg}
"""
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:SOCRiskExpr}, sc::Number)
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:SquaredSOCRiskExpr}, sc::Number)
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:QuadRiskExpr}, args...)
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:RSOCRiskExpr}, sc::Number)

Add the `i`-th L2 regularisation penalty to the objective, in the formulation given by `l2.alg`.

The coefficient `l2.val` is applied as given, so it must be consistent with the formulation it multiplies: [`SOCRiskExpr`](@ref) penalises ``\\lVert \\boldsymbol{w} \\rVert_2``, whereas [`SquaredSOCRiskExpr`](@ref), [`QuadRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref) penalise ``\\lVert \\boldsymbol{w} \\rVert_2^2``. The same `val` therefore does not produce the same penalty across formulations.

[`QuadRiskExpr`](@ref) yields a quadratic objective term rather than a conic one; [`add_to_objective_penalty!`](@ref) promotes an affine penalty expression to a `JuMP.QuadExpr` to accommodate it.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2\\,, \\\\
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2^2\\,.
\\end{align}
```

Where:

  - ``\\lambda_2``: L2 regularisation coefficient, the `val` field of `l2`.
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `i::Integer`: Index of the term, used to suffix the variables and constraints it creates so that multiple terms do not collide.
  - `w::VecNum`: Portfolio weights variable.
  - `l2::L2Regularisation`: The L2 regularisation term.
  - `sc::Number`: Constraint scale factor.

# Returns

  - `nothing`.

# Details

  - `l2::L2Regularisation{<:Any, <:SOCRiskExpr}`: Introduces `t_l2_i`, constrains `[t_l2_i; w] in SecondOrderCone` so that `t_l2_i >= norm(w, 2)`, and penalises `val * t_l2_i`.
  - `l2::L2Regularisation{<:Any, <:SquaredSOCRiskExpr}`: Introduces `t_l2_i`, constrains `[t_l2_i; w] in SecondOrderCone` so that `t_l2_i >= norm(w, 2)`, and penalises `val * t_l2_i^2`.
  - `l2::L2Regularisation{<:Any, <:QuadRiskExpr}`: Introduces no auxiliary variable, and penalises `val * dot(w, w)` directly.
  - `l2::L2Regularisation{<:Any, <:RSOCRiskExpr}`: Introduces `t_l2_i`, constrains `[t_l2_i; 0.5; w] in RotatedSecondOrderCone` so that `t_l2_i >= norm(w, 2)^2`, and penalises `val * t_l2_i`.

# Related

  - [`L2Regularisation`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`set_l2_regularisation!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:SOCRiskExpr}, sc::Number)
    val = l2.val
    t_l2 = model[Symbol(:t_l2_, i)] = JuMP.@variable(model)
    model[Symbol(:cl2_soc_, i)] = JuMP.@constraint(model,
                                                   [sc * t_l2; sc * w] in
                                                   JuMP.SecondOrderCone())
    l2 = model[Symbol(:l2_, i)] = JuMP.@expression(model, val * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:SquaredSOCRiskExpr},
                                 sc::Number)
    val = l2.val
    t_l2 = model[Symbol(:t_l2_, i)] = JuMP.@variable(model)
    model[Symbol(:cl2_soc_, i)] = JuMP.@constraint(model,
                                                   [sc * t_l2; sc * w] in
                                                   JuMP.SecondOrderCone())
    l2 = model[Symbol(:l2_, i)] = JuMP.@expression(model, val * t_l2^2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:QuadRiskExpr}, args...)
    val = l2.val
    l2 = model[Symbol(:l2_, i)] = JuMP.@expression(model, val * LinearAlgebra.dot(w, w))
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:RSOCRiskExpr}, sc::Number)
    val = l2.val
    t_l2 = model[Symbol(:t_l2_, i)] = JuMP.@variable(model)
    model[Symbol(:cl2_rsoc_, i)] = JuMP.@constraint(model,
                                                    [sc * t_l2; 0.5; sc * w] in
                                                    JuMP.RotatedSecondOrderCone())
    l2 = model[Symbol(:l2_, i)] = JuMP.@expression(model, val * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2s::L2Reg_VecL2Reg)
    w = get_w(model)
    sc = get_constraint_scale(model)
    for (i, l2) in enumerate(l2s)
        _set_l2_regularisation!(model, i, w, l2, sc)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Lp-norm regularisation term added to the optimisation objective.

Penalises concentrated portfolios by encouraging weight smoothness for ``p > 1``.

The same estimator serves double duty. In the `lp` field of [`JuMPOptimiser`](@ref) it is a *penalty*, and `val` is the coefficient multiplying the norm. In the `lpc` field it is a *constraint*, and `val` is instead an upper bound on the p-norm of the weights. The norm order `p` means the same thing in both. See [`set_lp_regularisation!`](@ref) and [`set_weight_norm_p_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\mathrm{val} \\cdot \\lVert \\boldsymbol{w} \\rVert_p\\,.
\\end{align}
```

Where:

  - ``\\mathrm{val}``: Regularisation coefficient.
  - ``p > 1``: Norm order.
  - $(math_dict[:w_port])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LpRegularisation(; p::Number = 3, val::Number = 1e-3) -> LpRegularisation

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)`.
  - `p > 1`.
  - `val > 0` and finite.

# Related

  - [`AbstractRegularisationEstimator`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`VecLpReg`](@ref)
  - [`set_lp_regularisation!`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`L2Regularisation`](@ref)
"""
@concrete struct LpRegularisation <: AbstractRegularisationEstimator
    """
    $(field_dict[:lpreg_p])
    """
    p
    """
    $(field_dict[:lpreg_val])
    """
    val
    function LpRegularisation(p::Number, val::Number)
        @argcheck(isfinite(p), IsNonFiniteError)
        @argcheck(p > one(p), DomainError)
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(p), typeof(val)}(p, val)
    end
end
function LpRegularisation(; p::Number = 3, val::Number = 1e-3)
    return LpRegularisation(p, val)
end
"""
    const VecLpReg = AbstractVector{<:LpRegularisation}

Alias for a vector of [`LpRegularisation`](@ref) objects.

Represents a collection of Lp-norm regularisation terms to be added to the optimisation objective.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
"""
const VecLpReg = AbstractVector{<:LpRegularisation}
"""
    const LpReg_VecLpReg = Union{<:LpRegularisation, <:VecLpReg}

Alias for a single or vector of Lp regularisation terms.

Matches either a single [`LpRegularisation`](@ref) or a vector of them ([`VecLpReg`](@ref)).

# Related

  - [`LpRegularisation`](@ref)
  - [`VecLpReg`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
const LpReg_VecLpReg = Union{<:LpRegularisation, <:VecLpReg}
function set_lp_regularisation!(model::JuMP.Model, lps::LpReg_VecLpReg)
    w = get_w(model)
    sc = get_constraint_scale(model)
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
                                                                                     sc *
                                                                                     t_lp,
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sets the L∞ regularisation term in the objective function. The penalty value is not adjusted so it must be consistent with the range of values the weights can take.
"""
function set_linf_regularisation!(model::JuMP.Model, linf::Number)
    w = get_w(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, t_linf)
    JuMP.@constraint(model, clinf_nic,
                     [sc * t_linf;
                      sc * w] in JuMP.MOI.NormInfinityCone(1 + length(w)))
    JuMP.@expression(model, linf, linf * t_linf)
    add_to_objective_penalty!(model, linf)
    return nothing
end

export L2Regularisation, LpRegularisation
