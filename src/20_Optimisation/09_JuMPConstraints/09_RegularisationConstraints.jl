"""
    set_l1_regularisation!(args...)

No-op fallback for L1 regularisation setup.

Called when no L1 regularisation is configured. Returns `nothing`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_1 \\|\\boldsymbol{w}\\|_1\\,.
\\end{align}
```

Where:

  - ``\\lambda_1``: L1 regularisation coefficient.
  - $(math_dict[:w_port])

# Related

  - [`LpRegularisation`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
function set_l1_regularisation!(args...)
    return nothing
end
"""
    set_l2_regularisation!(args...)

No-op fallback for L2 regularisation setup.

Called when no L2 regularisation is configured. Returns `nothing`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\|\\boldsymbol{w}\\|_2\\,.
\\end{align}
```

Where:

  - ``\\lambda_2``: L2 regularisation coefficient.
  - $(math_dict[:w_port])

# Related

  - [`LpRegularisation`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
function set_l2_regularisation!(args...)
    return nothing
end
"""
    set_lp_regularisation!(args...)

No-op fallback for Lp regularisation setup.

Called when no Lp regularisation is configured. Returns `nothing`.

# Related

  - [`LpRegularisation`](@ref)
"""
function set_lp_regularisation!(args...)
    return nothing
end
"""
    set_linf_regularisation!(args...)

No-op fallback for L∞ regularisation setup.

Called when no L∞ regularisation is configured. Returns `nothing`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_\\infty \\|\\boldsymbol{w}\\|_\\infty\\,.
\\end{align}
```

Where:

  - ``\\lambda_\\infty``: L∞ regularisation coefficient.
  - $(math_dict[:w_port])

# Related

  - [`LpRegularisation`](@ref)
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

Defines the L2-norm regularisation term added to the optimisation objective.

# Fields

$(DocStringExtensions.FIELDS)

## Validation

  - `val > 0` and finite.

# Related

  - [`LpRegularisation`](@ref)
"""
@concrete struct L2Regularisation <: AbstractRegularisationEstimator
    """
    $(field_dict[:val])
    """
    val
    """
    $(field_dict[:alg])
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

# Related

  - [`L2Regularisation`](@ref)
  - [`L2Reg_VecL2Reg`](@ref)
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
                            l2::L2Regularisation, sc::Number)

Sets the L2 regularisation term for a single [`L2Regularisation`](@ref) object in the optimisation model using the formulation specified by `l2.alg`. The penalty value is not adjusted so it must be consistent with the cone being used.
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

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\mathrm{val} \\cdot \\left( \\sum_{i=1}^N |w_i|^p \\right)^{1/p}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{val}``: Regularisation coefficient.
  - ``w_i``: Portfolio weight for asset ``i``.
  - ``p > 1``: Norm order.
  - $(math_dict[:N])

Penalises concentrated portfolios by encouraging weight smoothness for ``p > 1``.

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
"""
@concrete struct LpRegularisation <: AbstractRegularisationEstimator
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:val])
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
