"""
    set_l1_regularisation!(model::JuMP.Model, l1_val::Number)
    set_l1_regularisation!(args...)

Add an L1 penalty on the portfolio weights to the objective penalty.

The objective takes the penalty in its own units, so `l1_val` is a price in those units. With no short positions and a budget of one, ``\\lVert \\boldsymbol{w} \\rVert_1 = 1`` at every feasible point, so the penalty is a constant and moves no weight. It acts on the gross exposure of a portfolio that can hold short positions. Under [`MaximumRatio`](@ref) the model's `w` is the homogenised vector ``k \\boldsymbol{w}``, so the penalty acts on ``k \\boldsymbol{w}``. The method on `args...` does nothing, and it runs when no L1 coefficient is set.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_1 \\lVert \\boldsymbol{w} \\rVert_1\\,.
\\end{align}
```

Where:

  - ``\\lambda_1``: L1 penalty coefficient, `l1_val`.
  - $(math_dict[:w_port])

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `t_l1`: created, the epigraph variable of the 1-norm.

## Expressions

  - `l1`: ``\\lambda_1 t_{1}``. [`add_to_objective_penalty!`](@ref) adds it to the objective penalty `op`.

## Constraints

  - `cl1_noc`: ``\\left(s_c t_{1},\\, s_c \\boldsymbol{w}\\right)`` lies in the 1-norm cone, which is ``t_{1} \\geq \\lVert \\boldsymbol{w} \\rVert_1``. The objective penalty always makes the objective worse, and ``\\lambda_1 > 0``, so at the optimum ``t_{1}`` equals the norm.

Where:

  - ``t_{1}``: The epigraph variable `t_l1`.
  - ``\\lambda_1``: L1 penalty coefficient, `l1_val`.
  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `l1_val::Number`: L1 penalty coefficient.

# Validation

  - `l1_val > 0` and finite.

# Returns

  - `nothing`.

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

Add the penalty of each L2 term to the objective penalty.

Each [`L2Regularisation`](@ref) carries its own coefficient and formulation, and the penalties of several terms add. The objective takes each penalty in its own units, and the formulation sets the power of the norm, so one `val` gives a different penalty under each formulation. The method on `args...` does nothing, and it runs when no L2 term is set.

# Mathematical definition

Each term adds

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2^{q}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_2_reg])
  - $(math_dict[:q_l2_power])
  - $(math_dict[:w_port])

# Algorithm

 1. Read the weights `w` and the constraint scale `sc` from the model.
 2. For the term at position `i`, call [`_set_l2_regularisation!`](@ref), which registers the rows and the penalty of that term under names that end in `i`.

# Arguments

  - $(arg_dict[:model])
  - `l2s::L2Reg_VecL2Reg`: One L2 term, or a vector of them.

# Returns

  - `nothing`.

# Related

  - [`L2Regularisation`](@ref)
  - [`L2Reg_VecL2Reg`](@ref)
  - [`_set_l2_regularisation!`](@ref): Registers the model entries of one term.
  - [`set_l1_regularisation!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_l2_regularisation!(args...)
    return nothing
end
"""
    set_lp_regularisation!(model::JuMP.Model, lps::LpReg_VecLpReg)
    set_lp_regularisation!(args...)

Add the penalty of each Lp term to the objective penalty.

Each [`LpRegularisation`](@ref) carries its own norm order and coefficient, and the penalties of several terms add. The objective takes each penalty in its own units. Under [`MaximumRatio`](@ref) the model's `w` is the homogenised vector ``k \\boldsymbol{w}``, so the penalty acts on ``k \\boldsymbol{w}``. The method on `args...` does nothing, and it runs when no Lp term is set.

The same estimator in the `lpc` field of [`JuMPOptimiser`](@ref) is a norm constraint, and [`set_weight_norm_p_constraints!`](@ref) reads its `val` as a ceiling on the norm.

# Mathematical definition

Each term adds

```math
\\begin{align}
\\text{penalty} &= \\lambda_p \\lVert \\boldsymbol{w} \\rVert_p\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_p_reg])
  - $(math_dict[:p_norm_order])
  - $(math_dict[:w_port])

# JuMP formulation

The entries below belong to the term at position ``i``. Each name ends in ``i``, so the entries of two terms do not collide.

## Variables

  - `w`: read from the model.
  - `t_lp_i`: created, the epigraph variable of the p-norm.
  - `r_lp_i`: created, one entry for each asset.

## Expressions

  - `lp_i`: ``\\lambda_p t_p``. [`add_to_objective_penalty!`](@ref) adds it to the objective penalty `op`.

## Constraints

  - `clp_i`: one row for each asset ``j``, ``\\left(s_c r_j,\\, s_c t_p,\\, s_c w_j\\right)`` lies in the power cone of exponent ``1/p``, which is ``r_j^{1/p} t_p^{1 - 1/p} \\geq \\lvert w_j \\rvert``.
  - `cslp_i`: ``s_c \\left(\\sum_{j=1}^{N} r_j - t_p\\right) = 0``.

The two rows together give ``t_p \\geq \\lVert \\boldsymbol{w} \\rVert_p``. The objective penalty always makes the objective worse, and ``\\lambda_p > 0``, so at the optimum ``t_p`` equals the norm.

Where:

  - ``t_p``: The epigraph variable `t_lp_i`.
  - ``r_j``: Entry ``j`` of `r_lp_i`.
  - $(math_dict[:lambda_p_reg])
  - $(math_dict[:p_norm_order])
  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port])
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:model])
  - `lps::LpReg_VecLpReg`: One Lp term, or a vector of them.

# Returns

  - `nothing`.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref): Reads the same estimator as a norm ceiling.
  - [`set_l1_regularisation!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_lp_regularisation!(args...)
    return nothing
end
"""
    set_linf_regularisation!(model::JuMP.Model, linf::Number)
    set_linf_regularisation!(args...)

Add an L∞ penalty on the portfolio weights to the objective penalty.

The penalty prices the largest absolute weight, so it moves weight away from the largest position. The objective takes the penalty in its own units, so `linf` is a price in those units. Under [`MaximumRatio`](@ref) the model's `w` is the homogenised vector ``k \\boldsymbol{w}``, so the penalty acts on ``k \\boldsymbol{w}``. The method on `args...` does nothing, and it runs when no L∞ coefficient is set.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_\\infty \\lVert \\boldsymbol{w} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - ``\\lambda_\\infty``: L∞ penalty coefficient, `linf`.
  - $(math_dict[:w_port])

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `t_linf`: created, the epigraph variable of the ∞-norm.

## Expressions

  - `linf`: ``\\lambda_\\infty t_\\infty``. [`add_to_objective_penalty!`](@ref) adds it to the objective penalty `op`.

## Constraints

  - `clinf_nic`: ``\\left(s_c t_\\infty,\\, s_c \\boldsymbol{w}\\right)`` lies in the ∞-norm cone, which is ``t_\\infty \\geq \\lVert \\boldsymbol{w} \\rVert_\\infty``. The objective penalty always makes the objective worse, and ``\\lambda_\\infty > 0``, so at the optimum ``t_\\infty`` equals the norm.

Where:

  - ``t_\\infty``: The epigraph variable `t_linf`.
  - ``\\lambda_\\infty``: L∞ penalty coefficient, `linf`.
  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `linf::Number`: L∞ penalty coefficient.

# Validation

  - `linf > 0` and finite.

# Returns

  - `nothing`.

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
    assert_nonempty_gt0_finite_val(l1_val, :l1)
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

Every refusing method reads this one message, so every formulation that refuses a radius gives the same text. The message names the formulation and the rule family with `nameof`, because a printed type carries a module prefix wherever the name is not visible from `Main`. [`assert_calibrated_slots`](@ref) names its types the same way.

# Arguments

  - `alg`: The formulation that penalises the squared norm.

# Returns

  - `msg::String`: The refusal message.

# Related

  - [`assert_ambiguity_radius_formulation`](@ref)
  - [`L2Regularisation`](@ref)
"""
function squared_norm_radius_msg(alg)
    return "`L2Regularisation.val` holds an ambiguity radius, an `$(nameof(AbstractAmbiguityRadiusCalibrationAlgorithm))`, beside a `$(nameof(typeof(alg)))` formulation. That formulation penalises `norm(w, 2)^2`, and an ambiguity radius is the coefficient of `norm(w, 2)`: the Blanchet-Chen-Zhou identity that makes a radius and an L2 coefficient the same number holds for the un-squared penalty alone. A radius beside a squared penalty therefore has no reading. Use `SOCRiskExpr()`, which is the default, or state a plain number."
end
"""
    assert_ambiguity_radius_formulation(val, alg) -> Nothing

Refuse an ambiguity radius that was placed beside a formulation which penalises the squared norm.

[`L2Regularisation`](@ref) emits `val * norm(w, 2)` under [`SOCRiskExpr`](@ref), and `val * norm(w, 2)^2` under the three others. Only the first is the Wasserstein form, so only the first reads `val` as a radius. A plain number stays legal with every formulation, because a number is whatever coefficient the caller meant it to be. A rule of the radius family states that the number *is* a radius, and that statement is false beside a squared penalty.

The check is a set of methods rather than a list of types in one body, so a formulation that a later change adds refuses a radius by adding one method here. The permissive fallback is the first method, and each refusing method names one formulation of the family.

# Algorithm

 1. The pair is anything but a radius rule beside a squared formulation: return `nothing`.
 2. A radius rule stands beside a [`VarianceFormulation`](@ref), which covers [`QuadRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref): refuse.
 3. A radius rule stands beside an [`RSOCRiskExpr`](@ref): refuse.

# Arguments

  - `val`: The coefficient slot's occupant: a number, or an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref).
  - `alg`: The formulation the coefficient multiplies.

# Returns

  - `nothing`.

# Related

  - [`L2Regularisation`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`squared_norm_radius_msg`](@ref)
  - [`SecondMomentFormulation`](@ref)
"""
function assert_ambiguity_radius_formulation(::Any, ::Any)
    return nothing
end
function assert_ambiguity_radius_formulation(::AbstractAmbiguityRadiusCalibrationAlgorithm,
                                             alg::VarianceFormulation)
    return throw(ArgumentError(squared_norm_radius_msg(alg)))
end
function assert_ambiguity_radius_formulation(::AbstractAmbiguityRadiusCalibrationAlgorithm,
                                             alg::RSOCRiskExpr)
    return throw(ArgumentError(squared_norm_radius_msg(alg)))
end
"""
$(DocStringExtensions.TYPEDEF)

L2-norm penalty term that a [`JuMPOptimiser`](@ref) adds to the objective.

The penalty shrinks the weights towards zero, and under a budget it spreads them over the assets. The `alg` field selects the formulation, which sets the cones the solver sees and the power of the norm, so one `val` gives a different penalty under each formulation. Under [`MaximumRatio`](@ref) the model's weights are the homogenised vector ``k \\boldsymbol{w}``, and ``k`` scales with the normalisation `ohf` of the ratio. Each formulation gives a penalty of degree one in ``k``: the squared formulations divide ``\\lVert k \\boldsymbol{w} \\rVert_2^2`` by ``k``, through [`squared_norm_over_k!`](@ref). When the ratio bounds the risk and maximises the return, the weights do not depend on `ohf`. When the ratio fixes the return and minimises the risk, they do not depend on `ohf` if the risk term also has degree one, as a standard deviation or a conditional value at risk has. A [`Variance`](@ref) in the SOC or quadratic formulation has degree two, so beside it the strength of every penalty changes with `ohf`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2^{q}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_2_reg])
  - $(math_dict[:q_l2_power])
  - $(math_dict[:w_port])

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
  - If `val` holds an ambiguity-radius rule: `alg` must be [`SOCRiskExpr`](@ref).

# Related

  - [`AbstractRegularisationEstimator`](@ref)
  - [`Num_AmbRadCal`](@ref)
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

[`assemble_jump_model!`](@ref) calls it with the prior result and the solver of the optimisation. The estimator is not `@propagatable`, so this method is the ordinary `factory` verb and not the generated one. The rebuild goes through the keyword constructor, so the positivity check and [`assert_ambiguity_radius_formulation`](@ref) run again on the resolved number.

The weights-only `factory(l2, w)` and [`port_opt_view`](@ref) return the estimator unchanged. Neither holds a prior result, so neither can resolve a rule. A rule that stays after a cluster slice is resolved against the prior of that cluster when the model is assembled.

The slot resolves under the key `:l2reg_val` and not `:val`, because [`LpRegularisation`](@ref) also names its coefficient `val`, and the two coefficients belong to two different ground metrics. [`DualNormRadius`](@ref) reads the key to tell the type-2 metric of this penalty from the type-``q`` metric of the Lp penalty.

# Algorithm

 1. Resolve the slot `x.val` under the key `:l2reg_val` against `pr`, its observation weights `pr.w` and `slv`, giving `val`.
 2. Pair the declaration of [`calibration_slots`](@ref) with this resolver through [`assert_declared_calibration_resolver`](@ref).
 3. `val` is the object that `x` already holds: return `x`. Otherwise rebuild the term through the keyword constructor with `val` and `x.alg`.

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
  - [`assert_declared_calibration_resolver`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function factory(x::L2Regularisation, pr::AbstractPriorResult, slv = nothing)
    val = resolve_calibration_slot(x.val, :l2reg_val, pr, pr.w, slv)
    assert_declared_calibration_resolver(x, (; val = val))
    if val === x.val
        return x
    end
    return L2Regularisation(; val = val, alg = x.alg)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::L2Regularisation) = (; val = x.val)
# The derived calibration recursion does not own this slot — see `resolve_calibration_slots`.
# The declaration names the field, `val`, and the resolution above names the quantity,
# `:l2reg_val`, which ADR 0097 parts from the other three. A derivation reads the field name
# and would hand a rule the wrong key, so this term opts out and `factory` stays the one
# route.
resolve_calibration_slots(::L2Regularisation, ::AbstractPriorResult, ::Any = nothing) = (;)
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
    squared_norm_over_k!(model::JuMP.Model, i::Integer, x, k::Number, sc::Number)
    squared_norm_over_k!(model::JuMP.Model, i::Integer, x, k, sc::Number)

Return a term equal to the squared 2-norm of `x` divided by the homogenisation variable `k`, for the squared L2 penalty at position `i`.

Under [`MaximumRatio`](@ref) the model's weights are the homogenised vector ``\\boldsymbol{y} = k \\boldsymbol{w}``, so for ``\\boldsymbol{x} = \\boldsymbol{y}`` the term is ``k \\lVert \\boldsymbol{w} \\rVert_2^2``. It has degree one in ``k``, as the other penalties have, so it does not change the balance of the ratio when the normalisation `ohf` changes. When `k` is a number, the term is the quadratic ``\\boldsymbol{x}^\\intercal \\boldsymbol{x} / k``, and every objective except [`MaximumRatio`](@ref) sets ``k = 1``. When `k` is a variable, the quotient is not quadratic, and the function states it as a rotated second-order cone.

# Mathematical definition

```math
\\begin{align}
\\text{term} &= \\frac{\\lVert \\boldsymbol{x} \\rVert_2^2}{k}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: The vector argument `x`. A scalar `x` is the vector of one entry.
  - $(math_dict[:k_budget])

# JuMP formulation

The number method registers no entry. The variable method registers these entries, and each name ends in `i`.

## Variables

  - `t_l2_sq_i`: the epigraph variable ``s`` of the term.

## Constraints

  - `cl2_sq_rsoc_i`: ``\\left(s_c s,\\, s_c k / 2,\\, s_c \\boldsymbol{x}\\right)`` lies in the rotated second-order cone, which is ``s k \\geq \\lVert \\boldsymbol{x} \\rVert_2^2``. The penalty makes the objective worse as ``s`` rises, so at the optimum ``s`` equals its bound ``\\lVert \\boldsymbol{x} \\rVert_2^2 / k``.

Where:

  - $(math_dict[:sc_scale])

# Arguments

  - $(arg_dict[:model])
  - `i::Integer`: Position of the term, which ends the name of each entry it registers.
  - `x`: The vector, or the scalar, whose squared norm the term measures.
  - `k`: The homogenisation variable from [`effective_k`](@ref), or the number that stands for it.
  - `sc::Number`: Constraint scale. The number method registers no row and ignores it.

# Returns

  - The quadratic expression ``\\boldsymbol{x}^\\intercal \\boldsymbol{x} / k`` when `k` is a number, else the epigraph variable `t_l2_sq_i`.

# Related

  - [`_set_l2_regularisation!`](@ref)
  - [`effective_k`](@ref)
  - [`MaximumRatio`](@ref)
"""
function squared_norm_over_k!(::JuMP.Model, ::Integer, x, k::Number, ::Number)
    return LinearAlgebra.dot(x, x) / k
end
function squared_norm_over_k!(model::JuMP.Model, i::Integer, x, k, sc::Number)
    t_sq = state_set!(model, Symbol(""), :t_l2_sq_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :cl2_sq_rsoc_, i,
               JuMP.@constraint(model,
                                [sc * t_sq; sc * k / 2; sc * x] in
                                JuMP.RotatedSecondOrderCone()))
    return t_sq
end
"""
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:SOCRiskExpr}, sc::Number)
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:SquaredSOCRiskExpr}, sc::Number)
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:QuadRiskExpr}, sc::Number)
    _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                            l2::L2Regularisation{<:Any, <:RSOCRiskExpr}, sc::Number)

Add the penalty of the L2 term at position `i` to the objective penalty, in the formulation that `l2.alg` selects.

[`SOCRiskExpr`](@ref) penalises ``\\lVert \\boldsymbol{w} \\rVert_2``, and [`SquaredSOCRiskExpr`](@ref), [`QuadRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref) penalise ``\\lVert \\boldsymbol{w} \\rVert_2^2``, so one `val` gives a different penalty under each. Under [`MaximumRatio`](@ref) the model's `w` is the homogenised vector ``k \\boldsymbol{w}``, so the penalty acts on ``k \\boldsymbol{w}``. The three squared formulations divide the square by ``k``, so every penalty has degree one in ``k``. Under every other objective ``k = 1``: [`QuadRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref) then give a quadratic penalty, and [`add_to_objective_penalty!`](@ref) promotes an affine objective penalty to a `JuMP.QuadExpr` to hold it. Under [`MaximumRatio`](@ref) the quotient is not quadratic, so [`squared_norm_over_k!`](@ref) states it as a rotated second-order cone.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\lVert \\boldsymbol{w} \\rVert_2^{q}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_2_reg])
  - $(math_dict[:q_l2_power])
  - $(math_dict[:w_port])

# JuMP formulation

Each name ends in `i`, so the entries of two terms do not collide.

## Variables

  - `w`: read from the model, passed in as the argument `w`.
  - `t_l2_i`: created by every formulation except [`QuadRiskExpr`](@ref), the epigraph variable of the norm or of its square.
  - `t_l2_sq_i`: created by [`squared_norm_over_k!`](@ref) under [`SquaredSOCRiskExpr`](@ref) and [`QuadRiskExpr`](@ref) when ``k`` is a variable.

## Expressions

  - `l2_i`: the penalty. [`add_to_objective_penalty!`](@ref) adds it to the objective penalty `op`.
      + [`SOCRiskExpr`](@ref): ``\\lambda_2 t_2``.
      + [`SquaredSOCRiskExpr`](@ref): ``\\lambda_2 t_2^2 / k``.
      + [`QuadRiskExpr`](@ref): ``\\lambda_2 \\boldsymbol{w}^\\intercal \\boldsymbol{w} / k``.
      + [`RSOCRiskExpr`](@ref): ``\\lambda_2 t_2``.

## Constraints

  - `cl2_soc_i`, under [`SOCRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref): ``\\left(s_c t_2,\\, s_c \\boldsymbol{w}\\right)`` lies in the second-order cone, which is ``t_2 \\geq \\lVert \\boldsymbol{w} \\rVert_2``.
  - `cl2_rsoc_i`, under [`RSOCRiskExpr`](@ref): ``\\left(s_c t_2,\\, s_c k / 2,\\, s_c \\boldsymbol{w}\\right)`` lies in the rotated second-order cone, which is ``t_2 \\geq \\lVert \\boldsymbol{w} \\rVert_2^2 / k``.
  - `cl2_sq_rsoc_i`, under [`SquaredSOCRiskExpr`](@ref) and [`QuadRiskExpr`](@ref) when ``k`` is a variable: the row of [`squared_norm_over_k!`](@ref).

[`QuadRiskExpr`](@ref) registers no row when ``k`` is a number. The objective penalty always makes the objective worse, ``\\lambda_2 > 0``, and ``t_2^2`` rises with ``t_2 \\geq 0``, so at the optimum ``t_2`` equals its bound.

Where:

  - ``t_2``: The epigraph variable `t_l2_i`.
  - $(math_dict[:k_budget])
  - $(math_dict[:lambda_2_reg])
  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `i::Integer`: Position of the term, which ends the name of each entry it registers.
  - `w::VecNum`: The weights variable of the model.
  - `l2::L2Regularisation`: The L2 term.
  - `sc::Number`: Constraint scale. The [`QuadRiskExpr`](@ref) method uses it only when ``k`` is a variable.

# Returns

  - `nothing`.

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
    t_l2_sq = squared_norm_over_k!(model, i, t_l2, effective_k(model), sc)
    l2 = state_set!(model, Symbol(""), :l2_, i, JuMP.@expression(model, val * t_l2_sq))
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:QuadRiskExpr}, sc::Number)
    val = l2.val
    w_sq = squared_norm_over_k!(model, i, w, effective_k(model), sc)
    l2 = state_set!(model, Symbol(""), :l2_, i, JuMP.@expression(model, val * w_sq))
    add_to_objective_penalty!(model, l2)
    return nothing
end
function _set_l2_regularisation!(model::JuMP.Model, i::Integer, w::VecNum,
                                 l2::L2Regularisation{<:Any, <:RSOCRiskExpr}, sc::Number)
    val = l2.val
    k = effective_k(model)
    t_l2 = state_set!(model, Symbol(""), :t_l2_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :cl2_rsoc_, i,
               JuMP.@constraint(model,
                                [sc * t_l2; sc * k / 2; sc * w] in
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

Lp-norm penalty term that a [`JuMPOptimiser`](@ref) adds to the objective, or a ceiling on the Lp norm of the weights.

For ``p > 1`` and a budget of one, Hölder's inequality gives ``\\lVert \\boldsymbol{w} \\rVert_p \\geq N^{1/p - 1}``, with equality at equal weights alone. So the penalty moves weight away from concentrated positions. A larger ``p`` puts more of the price on the largest weights.

In the `lp` field of [`JuMPOptimiser`](@ref) the term is a penalty, and `val` is the coefficient of the norm. In the `lpc` field it is a constraint, and `val` is a ceiling on the norm. The norm order `p` has one meaning in both fields. [`set_lp_regularisation!`](@ref) builds the penalty, and [`set_weight_norm_p_constraints!`](@ref) builds the constraint.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_p \\lVert \\boldsymbol{w} \\rVert_p\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_p_reg])
  - $(math_dict[:p_norm_order])
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
        @argcheck(p > one(p),
                  DomainError(p,
                              "`LpRegularisation.p` is $p, and the penalty is the `p`-norm of the weights, which the model states with a power cone of exponent `1 / p`, so `1 < p` must hold. State a finite value greater than `1`."))
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

The `val` field of [`LpRegularisation`](@ref) is the one dual-use slot in the library. [`JuMPOptimiser`](@ref)'s `lp` field adds `val * norm(w, p)` to the objective, where `val` is an ambiguity radius, and its `lpc` field bounds `norm(w, p) <= val * k`, where `val` is a norm ceiling. The type bound of the field is [`Num_AmbRadNormCeilCal`](@ref), which admits both rule families, and the field of [`JuMPOptimiser`](@ref) that holds the term decides which reading is legal.

This method checks the penalty side. A rule of the norm-ceiling family states that the number is a ceiling, and a ceiling has no reading as a penalty coefficient. Its reciprocal is a floor on the effective number of assets, which says nothing about how strongly the objective shrinks the weights. A plain number is legal on both sides, because the caller states what it means.

The check is a set of methods rather than a list of types in one body, on the same terms as [`assert_ambiguity_radius_formulation`](@ref). It runs in [`JuMPOptimiser`](@ref)'s constructor, where the caller wrote the field, and again in [`factory`](@ref) for a term that reached the objective by another route.

# Algorithm

 1. The occupant is anything but a norm-ceiling role: return `nothing`.
 2. The occupant is an [`LpRegularisation`](@ref): check its `val`.
 3. The occupant is a vector of them: check each.
 4. The occupant is an [`AbstractNormCeilingCalibrationAlgorithm`](@ref): refuse.

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
function assert_penalty_coefficient_role(::AbstractNormCeilingCalibrationAlgorithm)
    return throw(ArgumentError("`LpRegularisation.val` holds a norm ceiling, an `$(nameof(AbstractNormCeilingCalibrationAlgorithm))`, in a slot that reads it as a penalty coefficient. `JuMPOptimiser.lp` adds `val * norm(w, p)` to the objective, and a ceiling is an upper bound on that norm instead: the two are different quantities. Move the term to `JuMPOptimiser.lpc`, which reads `val` as a ceiling, or state an `$(nameof(AbstractAmbiguityRadiusCalibrationAlgorithm))` or a plain number."))
end
"""
    assert_norm_ceiling_role(x) -> Nothing

Refuse an ambiguity-radius rule that was placed in a slot which reads its number as a norm ceiling.

This method checks the norm-constraint side, and [`assert_penalty_coefficient_role`](@ref) checks the penalty side. A rule of the ambiguity-radius family states that the number is a radius, the coefficient of a norm penalty in the objective. `JuMPOptimiser.lpc` bounds the norm instead, so the statement is false there.

# Algorithm

 1. The occupant is anything but a radius role: return `nothing`.
 2. The occupant is an [`LpRegularisation`](@ref): check its `val`.
 3. The occupant is a vector of them: check each.
 4. The occupant is an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref): refuse.

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
function assert_norm_ceiling_role(::AbstractAmbiguityRadiusCalibrationAlgorithm)
    return throw(ArgumentError("`LpRegularisation.val` holds an ambiguity radius, an `$(nameof(AbstractAmbiguityRadiusCalibrationAlgorithm))`, in a slot that reads it as a norm ceiling. `JuMPOptimiser.lpc` bounds `norm(w, p) <= val * k`, and a radius is the coefficient of that norm in the objective instead: the two are different quantities. Move the term to `JuMPOptimiser.lp`, which reads `val` as a radius, or state an `$(nameof(AbstractNormCeilingCalibrationAlgorithm))` or a plain number."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the ambiguity radius in `val` against prior result `pr`, and return an [`LpRegularisation`](@ref) holding the number.

It reads the slot as [`factory`](@ref) on [`L2Regularisation`](@ref) does. The estimator has no formulation slot, and `val` multiplies `norm(w, p)` and never its square, so [`assert_ambiguity_radius_formulation`](@ref) has nothing to refuse here.

In the `lpc` field of [`JuMPOptimiser`](@ref) the same estimator is a norm constraint, and `val` is a ceiling. The field decides the reading. This method refuses a norm-ceiling rule through [`assert_penalty_coefficient_role`](@ref), and [`norm_ceiling_factory`](@ref) refuses a radius rule on the constraint side.

The slot resolves under the key `:lpreg_val`, which [`L2Regularisation`](@ref) does not use, because the two coefficients belong to two different ground metrics. The key names no norm order: `p` is a field of the term, and one rule can stand in several terms. So the method states the order of the term in the [`CalibrationContext`](@ref), and [`DualNormRadius`](@ref) reads it there.

# Algorithm

 1. Refuse a norm-ceiling rule in `x.val` through [`assert_penalty_coefficient_role`](@ref).
 2. Resolve the slot `x.val` under the key `:lpreg_val` against `pr`, its observation weights `pr.w`, `slv` and a [`CalibrationContext`](@ref) that holds `x.p`, giving `val`.
 3. Pair the declaration of [`calibration_slots`](@ref) with this resolver through [`assert_declared_calibration_resolver`](@ref).
 4. `val` is the object that `x` already holds: return `x`. Otherwise rebuild the term through the keyword constructor with `x.p` and `val`.

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
  - [`assert_declared_calibration_resolver`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function factory(x::LpRegularisation, pr::AbstractPriorResult, slv = nothing)
    assert_penalty_coefficient_role(x.val)
    val = resolve_calibration_slot(x.val, :lpreg_val, pr, pr.w, slv,
                                   CalibrationContext(; p = x.p))
    assert_declared_calibration_resolver(x, (; val = val))
    if val === x.val
        return x
    end
    return LpRegularisation(; p = x.p, val = val)
end
"""
    norm_ceiling_factory(x, pr::AbstractPriorResult, slv = nothing)

Resolve the norm ceiling in `val` against prior result `pr`, and return an [`LpRegularisation`](@ref) holding the number.

It is the norm-constraint counterpart of [`factory`](@ref) on the same type, and it is a second verb because the two routes read one field as two quantities. A `factory` call on the `lpc` field would refuse the ceiling rule that belongs there, and would resolve the rule under `:lpreg_val`, the key of the penalty slot.

The two verbs differ in the guard and in the key, and in nothing else. Each refuses the family that has no reading on its own route, this one through [`assert_norm_ceiling_role`](@ref). Each states the term's own norm order in the [`CalibrationContext`](@ref) it resolves the slot against, because one rule placed in `lp` or in `lpc` serves every term and each term carries its own `p`.

The fallback returns its argument unchanged, so an empty `lpc` field, `nothing`, passes through it. A term whose `val` is already a number comes back as the same object, because the resolution returns the number it holds.

# Algorithm

 1. The argument is neither a term nor a vector of them: return it unchanged.
 2. The argument is a vector of terms: resolve each of them, and return the vector of the results.
 3. The argument is one term: refuse a radius rule with [`assert_norm_ceiling_role`](@ref).
 4. State the term's own `p` in a [`CalibrationContext`](@ref), then resolve the slot under the key `:lpc` against it, giving `val`.
 5. Pair the declaration with this resolver through [`assert_declared_calibration_resolver`](@ref), which refuses the declared slot when the resolution above did not reach it.
 6. `val` is the number the term already holds: return the term itself. Otherwise rebuild the term through the keyword constructor.

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
  - [`CalibrationContext`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assert_declared_calibration_resolver`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function norm_ceiling_factory(x, ::AbstractPriorResult, ::Any = nothing)
    return x
end
function norm_ceiling_factory(x::LpRegularisation, pr::AbstractPriorResult, slv = nothing)
    assert_norm_ceiling_role(x.val)
    val = resolve_calibration_slot(x.val, :lpc, pr, pr.w, slv,
                                   CalibrationContext(; p = x.p))
    assert_declared_calibration_resolver(x, (; val = val))
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
# The derived calibration recursion does not own this slot — see `resolve_calibration_slots`.
# This slot carries two readings and two keys, `:lpreg_val` and `:lpc`, and the caller's own
# norm order reaches the rule in a `CalibrationContext`. A derivation states neither the
# key nor the context, so this term opts out and the two factories stay the two routes.
resolve_calibration_slots(::LpRegularisation, ::AbstractPriorResult, ::Any = nothing) = (;)
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
    assert_nonempty_gt0_finite_val(linf, :linf)
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
