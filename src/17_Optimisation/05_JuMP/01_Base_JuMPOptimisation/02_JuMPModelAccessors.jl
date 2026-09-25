"""
    set_model_scales!(model::JuMP.Model, sc::Number, so::Number)

Register the constraint scale `sc` and the objective scale `so` in the JuMP model.

The constraint scale comes first, in the order in which every head passes `opt.sc, opt.so` from its [`JuMPOptimiser`](@ref). JuMP stores a number given to `@expressions` as the number itself, so `model[:sc]` and `model[:so]` hold `sc` and `so` with their own types.

# JuMP formulation

## Expressions

  - `so`: ``s_o``, the factor that multiplies the objective.
  - `sc`: ``s_c``, the factor that multiplies both sides of each scaled row.

Where:

  - $(math_dict[:so_scale])
  - $(math_dict[:sc_scale])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `sc::Number`: Constraint scale, read back by [`get_constraint_scale`](@ref).
  - `so::Number`: Objective scale, read back by [`get_objective_scale`](@ref).

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`get_constraint_scale`](@ref)
  - [`get_objective_scale`](@ref)
"""
function set_model_scales!(model::JuMP.Model, sc::Number, so::Number)
    JuMP.@expressions(model, begin
                          so, so
                          sc, sc
                      end)
    return nothing
end
"""
    set_model_observations!(model::JuMP.Model, T::Integer)

Register the observation count of the fit as the model entry `model[:T]`.

Every head calls it beside [`set_model_scales!`](@ref), before a builder runs. One fit has one returns matrix, and its row count is the holding period that every builder measures against, so the count is one entry for the whole model. A builder can therefore read it with [`get_T`](@ref) whatever else the model carries.

The row count of a fold, of a benchmark or of a stacked meta-optimisation panel is a different number. A builder that needs one of those reads the row count of its own matrix.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `T::Integer`: Observation count of the fit, read back by [`get_T`](@ref).

# Returns

  - `nothing`.

# Related

  - [`set_model_scales!`](@ref)
  - [`get_T`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_model_observations!(model::JuMP.Model, T::Integer)
    shared_set!(model, :T, T)
    return nothing
end
# ---------------------------------------------------------------------------
# Model-state read interface
#
# The JuMP model is the shared blackboard between every constraint and risk
# builder. These accessors give its always-present singleton entries a named,
# checked interface: each asserts the entry has been populated and returns it,
# turning an absent-key bug into a clear error at the read site instead of an
# opaque `KeyError` (or, worse, silent misbehaviour) deep in a builder.
#
# Prefer these over raw `model[:sym]` for the entries they cover. See ADR 0004.
# ---------------------------------------------------------------------------
"""
    get_constraint_scale(model::JuMP.Model)

Return the constraint scale `model[:sc]`.

# Validation

  - `model[:sc]` is registered, by [`set_model_scales!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_model_scales!`](@ref)
  - [`get_objective_scale`](@ref)
"""
function get_constraint_scale(model::JuMP.Model)
    @argcheck(haskey(model, :sc),
              ArgumentError("model[:sc] (constraint scale) has not been registered; call set_model_scales! first"))
    return model[:sc]
end
"""
    get_objective_scale(model::JuMP.Model)

Return the objective scale `model[:so]`.

# Validation

  - `model[:so]` is registered, by [`set_model_scales!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_model_scales!`](@ref)
  - [`get_constraint_scale`](@ref)
"""
function get_objective_scale(model::JuMP.Model)
    @argcheck(haskey(model, :so),
              ArgumentError("model[:so] (objective scale) has not been registered; call set_model_scales! first"))
    return model[:so]
end
"""
    get_T(model::JuMP.Model)

Return the observation count of the fit, `model[:T]`.

It is the row count of the returns matrix of the fit. It is not the row count of a fold, of a tracking benchmark or of a stacked meta-optimisation panel. A builder that needs one of those reads its own matrix.

# Validation

  - `model[:T]` is registered, by [`set_model_observations!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_model_observations!`](@ref)
  - [`get_constraint_scale`](@ref)
  - [`get_objective_scale`](@ref)
"""
function get_T(model::JuMP.Model)
    @argcheck(haskey(model, :T),
              ArgumentError("model[:T] (observation count of the fit) has not been registered; call set_model_observations! first"))
    return model[:T]
end
"""
    get_w(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the portfolio weight variables `model[Symbol(prefix, :w)]`.

A nested build passes its `prefix` and reads the shifted weights that it registered under that prefix. The empty prefix reads the weights of the head.

# Validation

  - The weights are registered, by [`set_w!`](@ref) for the empty prefix. Otherwise an `ArgumentError` names the missing key.

# Related

  - [`set_w!`](@ref)
"""
function get_w(model::JuMP.Model, prefix::Symbol = Symbol(""))
    @argcheck(haskey(model, Symbol(prefix, :w)),
              ArgumentError("model[$(Symbol(prefix, :w))] (portfolio weights) have not been registered; call set_w! first"))
    return model[Symbol(prefix, :w)]
end
"""
    get_k(model::JuMP.Model)

Return the homogenisation variable `model[:k]`.

A ratio objective such as [`MaximumRatio`](@ref) solves for the scaled weights ``k \\boldsymbol{w}``, and [`process_model`](@ref) divides the solved weights by `k`. Two builders register `k`, and a head reaches one of them. The error message names both.

  - [`set_maximum_ratio_factor_variables!`](@ref) registers a variable ``k \\geq 0`` under [`MaximumRatio`](@ref), and the number `1` under any other objective. Every head that builds an objective calls it.
  - [`_set_risk_budgeting_constraints!`](@ref) declares a free variable `k`, because its log barrier fixes the scale. The same head calls [`set_unit_budget!`](@ref), so a builder that reads [`effective_k`](@ref) gets `1` and not the free variable.

# Validation

  - `model[:k]` is registered. Otherwise an `ArgumentError` names the two builders and the heads that call each.

# Related

  - [`process_model`](@ref)
  - [`get_w`](@ref)
  - [`effective_k`](@ref)
  - [`set_maximum_ratio_factor_variables!`](@ref)
"""
function get_k(model::JuMP.Model)
    @argcheck(haskey(model, :k),
              ArgumentError("model[:k] (homogenisation variable) has not been registered. A head registers it either through set_maximum_ratio_factor_variables! (objective-shaped heads: MeanRisk, FactorRiskContribution, NearOptimalCentering, RelaxedRiskBudgeting) or through _set_risk_budgeting_constraints!, whose log barrier declares a free k (RiskBudgeting). Call the one that matches the head before any builder that reads k."))
    return model[:k]
end
"""
    set_unit_budget!(model::JuMP.Model)

Record that the head fixed the budget scale of the model to one.

A head calls it when its own constraints make the formulation invariant to scale, so a builder can use the number `1` in place of the homogenisation variable `k`. [`RiskBudgeting`](@ref) is the only such head. Its log barrier fixes the scale, and it normalises the weights after the solve. `k` stays a free variable under this record. The budget scale is one, and `k` is not a constant.

# Returns

  - `nothing`.

# Related

  - [`is_unit_budget`](@ref)
  - [`effective_k`](@ref)
"""
function set_unit_budget!(model::JuMP.Model)
    model[:unit_budget] = true
    return nothing
end
"""
    is_unit_budget(model::JuMP.Model)

Return `true` when the head fixed the budget scale to one through [`set_unit_budget!`](@ref).

# Related

  - [`set_unit_budget!`](@ref)
  - [`effective_k`](@ref)
"""
function is_unit_budget(model::JuMP.Model)
    return haskey(model, :unit_budget)
end
"""
    effective_k(model::JuMP.Model)

Return the budget scale that a builder multiplies a bound by: `1` under a unit budget, and `model[:k]` otherwise.

A builder that multiplies a bound by the budget reads this and not [`get_k`](@ref). A head that is invariant to scale then needs no check in each builder.

# Validation

  - `model[:k]` is registered, also under a unit budget, because the function reads it on both branches. Otherwise [`get_k`](@ref) raises.

# Related

  - [`is_unit_budget`](@ref)
  - [`get_k`](@ref)
"""
function effective_k(model::JuMP.Model)
    return ifelse(is_unit_budget(model), 1, get_k(model))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for how the head relates the weights to their long and short parts.

The weights are `model[:w]`, the long part is `model[:lw]` and the short part is `model[:sw]`. A head builds that relation in one of two ways, and a builder that pins the split into parts needs different constraints for each. The head records its way with [`set_decomposition_contract!`](@ref). A builder reads it back with [`decomposition_contract`](@ref) and dispatches on it.

# Related

  - [`WeightsFromParts`](@ref)
  - [`PartsBoundWeights`](@ref)
  - [`set_decomposition_contract!`](@ref)
  - [`decomposition_contract`](@ref)
"""
abstract type AbstractDecompositionContract end
"""
$(DocStringExtensions.TYPEDEF)

Selects the relation in which the head defines the weights from their long and short parts.

The relation is ``\\boldsymbol{w} = \\boldsymbol{w}^{l} - \\boldsymbol{w}^{s}``, with `lw` and `sw` the model variables. [`set_rb_mip_w!`](@ref) records it.

The identity always holds, so a long-or-short sign pattern pins the parts. With `sw = 0` the identity gives `lw == w`, and `lw >= 0` makes that ``\\max(w_i, 0)``. No slack is left to close.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`PartsBoundWeights`](@ref)
"""
struct WeightsFromParts <: AbstractDecompositionContract end
"""
$(DocStringExtensions.TYPEDEF)

Selects the relation in which the long and short parts bound the weights.

The relation is ``\\boldsymbol{w}^{l} \\geq \\boldsymbol{w}``, ``\\boldsymbol{w}^{s} \\geq -\\boldsymbol{w}`` and ``\\boldsymbol{w}^{l}, \\boldsymbol{w}^{s} \\geq 0``, with `w` the model variable. [`set_weight_constraints!`](@ref) records it.

The parts are upper bounds on the true long and short exposures. A budget built on them, `bgt`, `sbgt` or `gbgt`, therefore bounds the exposure that the weights realise and does not fix it. A sign pattern leaves that slack in place, so a builder that pins the parts under this relation adds two more rows to close it.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`WeightsFromParts`](@ref)
"""
struct PartsBoundWeights <: AbstractDecompositionContract end
"""
    set_decomposition_contract!(model::JuMP.Model, dc::AbstractDecompositionContract)

Record how the head relates `model[:w]` to `model[:lw]` and `model[:sw]`, unless a head already recorded it.

A head can run both builders. The mixed-integer [`RiskBudgeting`](@ref) head calls [`set_rb_mip_w!`](@ref), which records [`WeightsFromParts`](@ref), and then passes the same `lw` and `sw` to [`set_weight_constraints!`](@ref), which adds them as bounds and tries to record [`PartsBoundWeights`](@ref). The identity is the stronger statement and still holds, so the first record stays.

# Algorithm

 1. When the model has no entry `:decomposition_contract`, register `dc` under that key.
 2. Otherwise leave the entry unchanged.

# Returns

  - `nothing`.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`decomposition_contract`](@ref)
"""
function set_decomposition_contract!(model::JuMP.Model, dc::AbstractDecompositionContract)
    if !haskey(model, :decomposition_contract)
        model[:decomposition_contract] = dc
    end
    return nothing
end
"""
    decomposition_contract(model::JuMP.Model)

Return the relation that the head recorded, or `nothing` when the head recorded none.

`nothing` means that the model has no short side. The weights are their own long part, `lw` is a second name for `w`, and the model has no `sw`, so no split needs to be pinned.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`set_decomposition_contract!`](@ref)
"""
function decomposition_contract(model::JuMP.Model)
    return haskey(model, :decomposition_contract) ? model[:decomposition_contract] : nothing
end
"""
    get_ret(model::JuMP.Model)

Return the expected portfolio return expression `model[:ret]`.

# Validation

  - `model[:ret]` is registered, by [`scalarise_return_expression!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`scalarise_return_expression!`](@ref)
  - [`get_risk`](@ref)
"""
function get_ret(model::JuMP.Model)
    @argcheck(haskey(model, :ret),
              ArgumentError("model[:ret] (portfolio expected-return expression) has not been registered; call scalarise_return_expression! first"))
    return model[:ret]
end
"""
    get_risk(model::JuMP.Model)

Return the scalar portfolio risk expression `model[:risk]`.

# Validation

  - `model[:risk]` is registered, by [`scalarise_risk_expression!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`get_ret`](@ref)
"""
function get_risk(model::JuMP.Model)
    @argcheck(haskey(model, :risk),
              ArgumentError("model[:risk] (scalarised risk expression) has not been registered; call scalarise_risk_expression! first"))
    return model[:risk]
end
"""
    has_X(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return `true` when [`set_portfolio_returns!`](@ref) registered the portfolio returns `model[Symbol(prefix, :X)]`.

# Related

  - [`get_X`](@ref)
"""
function has_X(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :X))
end
"""
    get_X(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the portfolio returns expression `model[Symbol(prefix, :X)]`.

# Validation

  - The entry is registered, by [`set_portfolio_returns!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_portfolio_returns!`](@ref)
  - [`has_X`](@ref)
"""
function get_X(model::JuMP.Model, prefix::Symbol = Symbol(""))
    @argcheck(has_X(model, prefix),
              ArgumentError("model[$(Symbol(prefix, :X))] (portfolio returns) have not been registered; call set_portfolio_returns! first"))
    return model[Symbol(prefix, :X)]
end
"""
    has_net_X(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return `true` when [`set_net_portfolio_returns!`](@ref) registered the net portfolio returns `model[Symbol(prefix, :net_X)]`.

# Related

  - [`get_net_X`](@ref)
"""
function has_net_X(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :net_X))
end
"""
    get_net_X(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the net portfolio returns expression `model[Symbol(prefix, :net_X)]`.

# Validation

  - The entry is registered, by [`set_net_portfolio_returns!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_net_portfolio_returns!`](@ref)
  - [`has_net_X`](@ref)
"""
function get_net_X(model::JuMP.Model, prefix::Symbol = Symbol(""))
    @argcheck(has_net_X(model, prefix),
              ArgumentError("model[$(Symbol(prefix, :net_X))] (net portfolio returns) have not been registered; call set_net_portfolio_returns! first"))
    return model[Symbol(prefix, :net_X)]
end
"""
    has_Xap1(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return `true` when [`set_asset_returns_plus_one!`](@ref) registered the gross asset returns `model[Symbol(prefix, :Xap1)]`.

# Related

  - [`get_Xap1`](@ref)
"""
function has_Xap1(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :Xap1))
end
"""
    get_Xap1(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the gross asset returns `model[Symbol(prefix, :Xap1)]`, the matrix ``\\mathbf{X} + 1``.

# Validation

  - The entry is registered, by [`set_asset_returns_plus_one!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_asset_returns_plus_one!`](@ref)
  - [`has_Xap1`](@ref)
"""
function get_Xap1(model::JuMP.Model, prefix::Symbol = Symbol(""))
    @argcheck(has_Xap1(model, prefix),
              ArgumentError("model[$(Symbol(prefix, :Xap1))] (gross asset returns X.+1) have not been registered; call set_asset_returns_plus_one! first"))
    return model[Symbol(prefix, :Xap1)]
end
"""
    has_ddap1(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return `true` when [`set_portfolio_drawdowns_plus_one!`](@ref) registered the asset drawdowns plus one `model[Symbol(prefix, :ddap1)]`.

# Related

  - [`get_ddap1`](@ref)
"""
function has_ddap1(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :ddap1))
end
"""
    get_ddap1(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the asset drawdowns plus one, `model[Symbol(prefix, :ddap1)]`.

# Validation

  - The entry is registered, by [`set_portfolio_drawdowns_plus_one!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_portfolio_drawdowns_plus_one!`](@ref)
  - [`has_ddap1`](@ref)
"""
function get_ddap1(model::JuMP.Model, prefix::Symbol = Symbol(""))
    @argcheck(has_ddap1(model, prefix),
              ArgumentError("model[$(Symbol(prefix, :ddap1))] (drawdowns-plus-one) have not been registered; call set_portfolio_drawdowns_plus_one! first"))
    return model[Symbol(prefix, :ddap1)]
end
"""
    has_dd(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return `true` when [`set_drawdown_constraints!`](@ref) registered the drawdown variables `model[Symbol(prefix, :dd)]`.

# Related

  - [`get_dd`](@ref)
"""
function has_dd(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :dd))
end
"""
    get_dd(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the drawdown variables `model[Symbol(prefix, :dd)]`, one more than the number of observations.

# Validation

  - The entry is registered, by [`set_drawdown_constraints!`](@ref). Otherwise an `ArgumentError` names that builder.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`has_dd`](@ref)
"""
function get_dd(model::JuMP.Model, prefix::Symbol = Symbol(""))
    @argcheck(has_dd(model, prefix),
              ArgumentError("model[$(Symbol(prefix, :dd))] (cumulative-drawdown variables) have not been registered; call set_drawdown_constraints! first"))
    return model[Symbol(prefix, :dd)]
end

public get_w, get_k, get_constraint_scale
