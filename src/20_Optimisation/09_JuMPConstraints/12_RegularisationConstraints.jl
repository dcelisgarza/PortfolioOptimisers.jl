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

A regularisation term penalises the norm of the weight vector, which is the Lagrangian form
of a constraint on that norm.

# Related

  - [`L2Regularisation`](@ref)
  - [`LpRegularisation`](@ref)
  - [`set_l2_regularisation!`](@ref)
  - [`set_lp_regularisation!`](@ref)

# References

  - $(ref_dict[:demiguel2009])
"""
abstract type AbstractRegularisationEstimator <: AbstractEstimator end
"""
    squared_norm_radius_msg(alg) -> String

Build the refusal message of [`assert_ambiguity_radius_formulation`](@ref), naming the formulation that was paired with a rule.

The message is written once and every refusing method reads it, so the two formulations that refuse a radius today, and any that a later type adds, all say the same thing. The formulation is named with `nameof`, not printed, for the reason [`assert_calibrated_slots`](@ref) states: a printed type carries a module prefix wherever the name is not visible from `Main`.

# Arguments

  - `alg`: The formulation that penalises the squared norm.

# Returns

  - `msg::String`: The refusal message.

# Related

  - [`assert_ambiguity_radius_formulation`](@ref)
  - [`L2Regularisation`](@ref)
"""
function squared_norm_radius_msg(alg)
    return "`L2Regularisation.val` holds an ambiguity radius, an `$(nameof(AmbiguityRadiusCalibration))`, beside a `$(nameof(typeof(alg)))` formulation. That formulation penalises `norm(w, 2)^2`, and an ambiguity radius is the coefficient of `norm(w, 2)`: the Blanchet-Chen-Zhou identity that makes a radius and an L2 coefficient the same number holds for the un-squared penalty alone. A radius beside a squared penalty therefore has no reading. Use `SOCRiskExpr()`, which is the default, or state a plain number."
end
"""
    assert_ambiguity_radius_formulation(val, alg) -> Nothing

Refuse an ambiguity radius that was placed beside a formulation which penalises the squared norm.

[`L2Regularisation`](@ref) emits `val * norm(w, 2)` under [`SOCRiskExpr`](@ref), and `val * norm(w, 2)^2` under the three others. Only the first is the Wasserstein form, so only the first reads `val` as a radius. A plain number stays legal with every formulation, because a number is whatever coefficient the caller meant it to be; a rule states that the number *is* a radius, and that statement is false beside a squared penalty.

The check is a set of methods rather than a list of types in one body, so a formulation that a later change adds refuses a radius by adding one method here. The permissive fallback is the first method, and each refusing method names one formulation of the family.

# Algorithm

 1. The pair is anything but a radius rule beside a squared formulation: return `nothing`.
 2. A radius rule stands beside a [`VarianceFormulation`](@ref), which covers [`QuadRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref): refuse.
 3. A radius rule stands beside an [`RSOCRiskExpr`](@ref): refuse.

# Arguments

  - `val`: The coefficient slot's occupant: a number, or an [`AmbiguityRadiusCalibration`](@ref).
  - `alg`: The formulation the coefficient multiplies.

# Returns

  - `nothing`.

# Related

  - [`L2Regularisation`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`squared_norm_radius_msg`](@ref)
  - [`SecondMomentFormulation`](@ref)
"""
function assert_ambiguity_radius_formulation(::Any, ::Any)
    return nothing
end
function assert_ambiguity_radius_formulation(::AmbiguityRadiusCalibration,
                                             alg::VarianceFormulation)
    return throw(ArgumentError(squared_norm_radius_msg(alg)))
end
function assert_ambiguity_radius_formulation(::AmbiguityRadiusCalibration,
                                             alg::RSOCRiskExpr)
    return throw(ArgumentError(squared_norm_radius_msg(alg)))
end
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
        val::Num_AmbRadCal = 1e-4,
        alg::SecondMomentFormulation = SOCRiskExpr()
    ) -> L2Regularisation

Keywords correspond to the struct's fields.

## Validation

  - If `val` is a number: `val > 0` and finite.
  - If `val` holds an [`AmbiguityRadiusCalibration`](@ref): `alg` must be [`SOCRiskExpr`](@ref).

# Related

  - [`AbstractRegularisationEstimator`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`assert_ambiguity_radius_formulation`](@ref)
  - [`L2Reg_VecL2Reg`](@ref)
  - [`VecL2Reg`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`set_l2_regularisation!`](@ref)
  - [`_set_l2_regularisation!`](@ref)
  - [`LpRegularisation`](@ref)

# References

  - $(ref_dict[:demiguel2009])
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
    function L2Regularisation(val::Num_AmbRadCal, alg::SecondMomentFormulation)
        assert_nonempty_gt0_finite_val(val, :val)
        assert_ambiguity_radius_formulation(val, alg)
        return new{typeof(val), typeof(alg)}(val, alg)
    end
end
function L2Regularisation(; val::Num_AmbRadCal = 1e-4,
                          alg::SecondMomentFormulation = SOCRiskExpr())
    return L2Regularisation(val, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the ambiguity radius in `val` against prior result `pr`, and return an [`L2Regularisation`](@ref) holding the number.

This is the channel [`JuMPOptimiser`](@ref) uses: the estimator is not `@propagatable`, so the resolution takes the ordinary `factory` verb rather than the generated one, and [`assemble_jump_model!`](@ref) calls it with the optimisation's own prior result and solver. The rebuild goes through the keyword constructor, so the positivity check and [`assert_ambiguity_radius_formulation`](@ref) both re-run on the calibrated number.

The weights-only `factory(l2, w)` and [`port_opt_view`](@ref) carry the estimator through untouched, which is right: neither holds a prior result, so neither can resolve a rule, and a rule that survives a cluster slice is resolved against that cluster's own prior when the model is assembled.

The slot is named `val` and its key is `:l2reg_val`, because [`LpRegularisation`](@ref) names its own coefficient `val` too and the two carry two different ground metrics. A key that read `:val` could not tell the type-2 metric of this penalty from the type-``q`` metric of that one, so [`DualNormRadius`](@ref) would have no reading of either. The two keys are the two names [`field_dict`](@ref) already uses for the two slots.

# Arguments

  - `x`: The regularisation term.
  - `pr`: Prior result the rule reads.
  - `slv`: Effective solver, or `nothing`.

# Returns

  - `L2Regularisation`: The term, with `val` holding a number.

# Related

  - [`L2Regularisation`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assert_ambiguity_radius_formulation`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function factory(x::L2Regularisation, pr::AbstractPriorResult, slv = nothing)
    val = resolve_calibration_slot(x.val, :l2reg_val, pr, pr.w, slv)
    if val === x.val
        return x
    end
    return L2Regularisation(; val = val, alg = x.alg)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::L2Regularisation) = (; val = x.val)
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
    t_l2 = state_set!(model, Symbol(""), :t_l2_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :cl2_soc_, i,
               JuMP.@constraint(model, [sc * t_l2; sc * w] in JuMP.SecondOrderCone()))
    l2 = state_set!(model, Symbol(""), :l2_, i, JuMP.@expression(model, val * t_l2))
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:SquaredSOCRiskExpr},
                                 sc::Number)
    val = l2.val
    t_l2 = state_set!(model, Symbol(""), :t_l2_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :cl2_soc_, i,
               JuMP.@constraint(model, [sc * t_l2; sc * w] in JuMP.SecondOrderCone()))
    l2 = state_set!(model, Symbol(""), :l2_, i, JuMP.@expression(model, val * t_l2^2))
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:QuadRiskExpr}, args...)
    val = l2.val
    l2 = state_set!(model, Symbol(""), :l2_, i,
                    JuMP.@expression(model, val * LinearAlgebra.dot(w, w)))
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:RSOCRiskExpr}, sc::Number)
    val = l2.val
    t_l2 = state_set!(model, Symbol(""), :t_l2_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :cl2_rsoc_, i,
               JuMP.@constraint(model,
                                [sc * t_l2; sc * 0.5; sc * w] in
                                JuMP.RotatedSecondOrderCone()))
    l2 = state_set!(model, Symbol(""), :l2_, i, JuMP.@expression(model, val * t_l2))
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

    LpRegularisation(;
        p::Number = 3,
        val::Num_AmbRadNormCeilCal = 1e-3
    ) -> LpRegularisation

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)`.
  - `p > 1`.
  - If `val` is a number: `val > 0` and finite.
  - The role in `val` is checked by the field that holds the term, not here. `val` is the one dual-use slot in the library, so this constructor cannot know which reading applies: [`JuMPOptimiser`](@ref)'s constructor refuses a norm-ceiling role in `lp` and a radius role in `lpc`, and the two `factory` routes refuse the same pairings again.

# Related

  - [`AbstractRegularisationEstimator`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`VecLpReg`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`assert_penalty_coefficient_role`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
  - [`set_lp_regularisation!`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`L2Regularisation`](@ref)

# References

  - $(ref_dict[:demiguel2009])
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
    function LpRegularisation(p::Number, val::Num_AmbRadNormCeilCal)
        @argcheck(isfinite(p), IsNonFiniteError)
        @argcheck(p > one(p), DomainError)
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(p), typeof(val)}(p, val)
    end
end
function LpRegularisation(; p::Number = 3, val::Num_AmbRadNormCeilCal = 1e-3)
    return LpRegularisation(p, val)
end
"""
    assert_penalty_coefficient_role(x) -> Nothing

Refuse a norm-ceiling rule that was placed in a slot which reads its number as a penalty coefficient.

The `val` field of [`LpRegularisation`](@ref) is the one dual-use slot in the library. [`JuMPOptimiser`](@ref)'s `lp` field adds `val * norm(w, p)` to the objective, where `val` is an ambiguity radius, and its `lpc` field bounds `norm(w, p) <= val * k`, where `val` is a norm ceiling. One field cannot carry two bounds, so [`Num_AmbRadNormCeilCal`](@ref) admits both roles and the slot's *owner* settles which reading is legal.

This is the penalty half. A [`NormCeilingCalibration`](@ref) states that the number **is** a ceiling, and a ceiling has no reading as a penalty coefficient: its reciprocal is a floor on the effective number of assets, which says nothing about how strongly the objective should shrink the weights. A plain number stays legal on both routes, because a number is whatever quantity the caller meant it to be.

The check is a set of methods rather than a list of types in one body, on the same terms as [`assert_ambiguity_radius_formulation`](@ref). It runs in [`JuMPOptimiser`](@ref)'s constructor, where the caller wrote the field, and again in [`factory`](@ref) for a term that reached the objective by another route.

# Algorithm

 1. The occupant is anything but a norm-ceiling role: return `nothing`.
 2. The occupant is an [`LpRegularisation`](@ref): check its `val`.
 3. The occupant is a vector of them: check each.
 4. The occupant is a [`NormCeilingCalibration`](@ref): refuse.

# Arguments

  - `x`: A regularisation term, a vector of them, or the occupant of a `val` slot.

# Returns

  - `nothing`.

# Related

  - [`LpRegularisation`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
function assert_penalty_coefficient_role(::Any)
    return nothing
end
function assert_penalty_coefficient_role(x::LpRegularisation)
    return assert_penalty_coefficient_role(x.val)
end
function assert_penalty_coefficient_role(xs::AbstractVector{<:LpRegularisation})
    return foreach(assert_penalty_coefficient_role, xs)
end
function assert_penalty_coefficient_role(::NormCeilingCalibration)
    return throw(ArgumentError("`LpRegularisation.val` holds a norm ceiling, a `$(nameof(NormCeilingCalibration))`, in a slot that reads it as a penalty coefficient. `JuMPOptimiser.lp` adds `val * norm(w, p)` to the objective, and a ceiling is an upper bound on that norm instead: the two are different quantities. Move the term to `JuMPOptimiser.lpc`, which reads `val` as a ceiling, or state an `$(nameof(AmbiguityRadiusCalibration))` or a plain number."))
end
"""
    assert_norm_ceiling_role(x) -> Nothing

Refuse an ambiguity-radius rule that was placed in a slot which reads its number as a norm ceiling.

This is the norm-constraint half of the pair [`assert_penalty_coefficient_role`](@ref) opens, and it carries that method's reading unchanged. An [`AmbiguityRadiusCalibration`](@ref) states that the number **is** a radius, the coefficient of a norm penalty in the objective. `JuMPOptimiser.lpc` bounds the norm instead, so the statement is false there.

# Algorithm

 1. The occupant is anything but a radius role: return `nothing`.
 2. The occupant is an [`LpRegularisation`](@ref): check its `val`.
 3. The occupant is a vector of them: check each.
 4. The occupant is an [`AmbiguityRadiusCalibration`](@ref): refuse.

# Arguments

  - `x`: A norm-constraint term, a vector of them, or the occupant of a `val` slot.

# Returns

  - `nothing`.

# Related

  - [`LpRegularisation`](@ref)
  - [`assert_penalty_coefficient_role`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`norm_ceiling_factory`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
"""
function assert_norm_ceiling_role(::Any)
    return nothing
end
function assert_norm_ceiling_role(x::LpRegularisation)
    return assert_norm_ceiling_role(x.val)
end
function assert_norm_ceiling_role(xs::AbstractVector{<:LpRegularisation})
    return foreach(assert_norm_ceiling_role, xs)
end
function assert_norm_ceiling_role(::AmbiguityRadiusCalibration)
    return throw(ArgumentError("`LpRegularisation.val` holds an ambiguity radius, an `$(nameof(AmbiguityRadiusCalibration))`, in a slot that reads it as a norm ceiling. `JuMPOptimiser.lpc` bounds `norm(w, p) <= val * k`, and a radius is the coefficient of that norm in the objective instead: the two are different quantities. Move the term to `JuMPOptimiser.lp`, which reads `val` as a radius, or state a `$(nameof(NormCeilingCalibration))` or a plain number."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the ambiguity radius in `val` against prior result `pr`, and return an [`LpRegularisation`](@ref) holding the number.

It carries the reading of [`factory`](@ref) on [`L2Regularisation`](@ref) unchanged. The estimator has one norm order and no formulation slot, so no pairing can be wrong and no guard runs: `val` multiplies `norm(w, p)` and never its square.

The same estimator also serves as a norm *constraint* through the `lpc` field of [`JuMPOptimiser`](@ref), where `val` is an upper bound and not a coefficient. Both routes share one field and one bound, so the route settles the reading: this method refuses a norm-ceiling role through [`assert_penalty_coefficient_role`](@ref), and [`norm_ceiling_factory`](@ref) refuses a radius role on the other side.

The slot is named `val` and its key is `:lpreg_val`, because [`L2Regularisation`](@ref) names its own coefficient `val` too and the two carry two different ground metrics. The two keys are the two names [`field_dict`](@ref) already uses for the two slots.

The key still names no norm order, because `p` lives on this estimator and one rule may stand in several terms. So this route hands the term's own order to the rule with [`bind_norm_order`](@ref) before it resolves the slot, on the same terms as [`norm_ceiling_factory`](@ref). [`DualNormRadius`](@ref) is the rule that reads it.

# Arguments

  - `x`: The regularisation term.
  - `pr`: Prior result the rule reads.
  - `slv`: Effective solver, or `nothing`.

# Returns

  - `LpRegularisation`: The term, with `val` holding a number.

# Related

  - [`LpRegularisation`](@ref)
  - [`L2Regularisation`](@ref)
  - [`assert_penalty_coefficient_role`](@ref)
  - [`norm_ceiling_factory`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function factory(x::LpRegularisation, pr::AbstractPriorResult, slv = nothing)
    assert_penalty_coefficient_role(x.val)
    val = resolve_calibration_slot(bind_norm_order(x.val, x.p), :lpreg_val, pr, pr.w, slv)
    if val === x.val
        return x
    end
    return LpRegularisation(; p = x.p, val = val)
end
"""
    norm_ceiling_factory(x, pr::AbstractPriorResult, slv = nothing)

Resolve the norm ceiling in `val` against prior result `pr`, and return an [`LpRegularisation`](@ref) holding the number.

It is the norm-constraint counterpart of [`factory`](@ref) on the same type, and it is a second verb because the two routes read one field as two quantities. A `factory` call on the `lpc` field would resolve a radius rule that has no reading there, and would bind no norm order.

This route does two things the penalty route does not. It refuses a radius role through [`assert_norm_ceiling_role`](@ref). It also hands the term's own norm order to the rule with [`bind_norm_order`](@ref) before it resolves the slot, because one rule placed in `lpc` serves every term and each term carries its own `p`.

The fallback returns its argument unchanged, so `nothing` and a stated number both cross untouched.

# Arguments

  - `x`: The norm-constraint term, or a vector of them.
  - `pr`: Prior result the rule reads.
  - `slv`: Effective solver, or `nothing`.

# Returns

  - The term, or the vector of terms, with each `val` holding a number.

# Related

  - [`LpRegularisation`](@ref)
  - [`factory`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
  - [`bind_norm_order`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function norm_ceiling_factory(x, ::AbstractPriorResult, ::Any = nothing)
    return x
end
function norm_ceiling_factory(x::LpRegularisation, pr::AbstractPriorResult, slv = nothing)
    assert_norm_ceiling_role(x.val)
    val = resolve_calibration_slot(bind_norm_order(x.val, x.p), :lpc, pr, pr.w, slv)
    if val === x.val
        return x
    end
    return LpRegularisation(; p = x.p, val = val)
end
function norm_ceiling_factory(xs::AbstractVector{<:LpRegularisation},
                              pr::AbstractPriorResult, slv = nothing)
    return [norm_ceiling_factory(x, pr, slv) for x in xs]
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::LpRegularisation) = (; val = x.val)
"""
    const VecLpReg = AbstractVector{<:LpRegularisation}

Alias for a vector of [`LpRegularisation`](@ref) objects.

Represents a collection of Lp-norm regularisation terms to be added to the optimisation objective.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`set_lp_regularisation!`](@ref)
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
        t_lp, r_lp = JuMP.@variables(model, begin
                                         ()
                                         [1:N]
                                     end)
        state_set!(model, Symbol(""), :t_lp_, i, t_lp)
        state_set!(model, Symbol(""), :r_lp_, i, r_lp)
        clp, cslp = JuMP.@constraints(model,
                                      begin
                                          [i = 1:N],
                                          [sc * r_lp[i], sc * t_lp, sc * w[i]] in
                                          JuMP.MOI.PowerCone(p_inv)
                                          sc * (sum(r_lp) - t_lp) == 0
                                      end)
        state_set!(model, Symbol(""), :clp_, i, clp)
        state_set!(model, Symbol(""), :cslp_, i, cslp)
        lp_expr = state_set!(model, Symbol(""), :lp_, i,
                             JuMP.@expression(model, val * t_lp))
        add_to_objective_penalty!(model, lp_expr)
    end
end
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
