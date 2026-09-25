"""
    set_model_scales!(model::JuMP.Model, sc::Number, so::Number)

Register constraint scale `sc` and objective scale `so` as named expressions in the JuMP model.

The positional order is `sc` first, matching every head, which passes `opt.sc, opt.so`
straight out of its [`JuMPOptimiser`](@ref).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `sc::Number`: Constraint scale factor, read back by [`get_constraint_scale`](@ref).
  - `so::Number`: Objective scale factor, read back by [`get_objective_scale`](@ref).

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

Register the observation count of the fit as the named entry `model[:T]`.

The sibling of [`set_model_scales!`](@ref), and every head calls it in the same place, before
any builder runs. The count is a model-wide singleton: one fit produces one returns matrix, and
its row count is the holding period every builder measures against. Registering it here rather
than inside a builder means a reader may rely on it whatever the model carries, and
[`get_T`](@ref) reads it back.

The row count of a **fold**, of a **benchmark**, or of a stacked meta-optimisation panel is a
different number. A site that means one of those keeps its own `size(..., 1)`.

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

Return the constraint scale expression `model[:sc]`.

Asserts the scale has been registered (via [`set_model_scales!`](@ref)); errors otherwise.

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

Return the objective scale expression `model[:so]`.

Asserts the scale has been registered (via [`set_model_scales!`](@ref)); errors otherwise.

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

Asserts the count has been registered (via [`set_model_observations!`](@ref)); errors otherwise.

This is the row count of the **fit's own** returns matrix. It is not the row count of a fold, of
a tracking benchmark, or of a stacked meta-optimisation panel; a site that means one of those
reads its own matrix instead.

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
    get_w(model::JuMP.Model)

Return the portfolio weight variables `model[:w]`.

Asserts the weights have been registered (via [`set_w!`](@ref)); errors otherwise.

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

`k` is the auxiliary scaling variable used to homogenise fractional/ratio objectives (e.g.
maximum ratio); recovered weights are `w / k`. Asserts `:k` has been registered; errors
otherwise.

Two producers register it, and the error message names both because a head reaches only
one of them:

  - [`set_maximum_ratio_factor_variables!`](@ref) is the head-level producer. Every head
    that shapes `w` from an objective calls it: `k >= 0` under [`MaximumRatio`](@ref), and
    the literal `1` otherwise.
  - [`_set_risk_budgeting_constraints!`](@ref) declares a *free* `k` instead, because the
    log barrier it builds is what pins the scale. That head also declares
    [`set_unit_budget!`](@ref), so downstream builders read [`effective_k`](@ref) and get
    `1` rather than the free variable.

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

Record that the head normalised the model's budget scale to unit.

A head declares this when its own constraints make the formulation scale-invariant, so
downstream builders may substitute the literal `1` for the homogenisation variable `k`.
[`RiskBudgeting`](@ref) is the only such head: its log-barrier normalisation pins the scale,
and the weights are renormalised after the solve. Note that `k` remains a *free variable*
under this declaration — it is the budget *scale* that is unit, not `k` that is constant.

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

Return whether the head normalised the budget scale to unit (see [`set_unit_budget!`](@ref)).

# Related

  - [`set_unit_budget!`](@ref)
  - [`effective_k`](@ref)
"""
function is_unit_budget(model::JuMP.Model)
    return haskey(model, :unit_budget)
end
"""
    effective_k(model::JuMP.Model)

Return the budget scale a builder should use: `1` under a unit budget, else `model[:k]`.

Builders that multiply a bound by the budget want this rather than [`get_k`](@ref), so a
scale-invariant head is honoured without each builder re-deriving that fact for itself.

# Related

  - [`is_unit_budget`](@ref)
  - [`get_k`](@ref)
"""
function effective_k(model::JuMP.Model)
    return ifelse(is_unit_budget(model), 1, get_k(model))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the head's *decomposition contract*: how `model[:w]` relates to the
long/short parts `model[:lw]` and `model[:sw]`.

Heads build that relationship in one of two incompatible ways, and a builder that pins the
decomposition needs to know which, because the two need different constraints to become exact.
The head declares its own with [`set_decomposition_contract!`](@ref); builders read it back
with [`decomposition_contract`](@ref) and dispatch.

# Related

  - [`WeightsFromParts`](@ref)
  - [`PartsBoundWeights`](@ref)
  - [`set_decomposition_contract!`](@ref)
  - [`decomposition_contract`](@ref)
"""
abstract type AbstractDecompositionContract end
"""
$(DocStringExtensions.TYPEDEF)

The head defines the weights *from* the parts: `w = lw - sw` is an identity, `lw` and `sw`
being the primitive variables. Declared by [`set_rb_mip_w!`](@ref).

Because the identity always holds, forcing the long-xor-short sign pattern is enough to pin the
decomposition: with `sw = 0` the identity leaves `lw == w`, and `lw >= 0` makes that
`max(w, 0)`. No slack remains to close.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`PartsBoundWeights`](@ref)
"""
struct WeightsFromParts <: AbstractDecompositionContract end
"""
$(DocStringExtensions.TYPEDEF)

The head defines the parts as *bounds* on the weights: `lw >= w`, `sw >= -w`, `lw, sw >= 0`,
`w` being the primitive variable. Declared by [`set_weight_constraints!`](@ref).

The parts are only upper bounds on the true long/short exposures, so every budget built on
them (`bgt`, `sbgt`, `gbgt`) bounds the realised exposure rather than pinning it. Forcing the
sign pattern does not change that — the slack survives it — so pinning the decomposition under
this contract needs two further constraints to close it.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`WeightsFromParts`](@ref)
"""
struct PartsBoundWeights <: AbstractDecompositionContract end
"""
    set_decomposition_contract!(model::JuMP.Model, dc::AbstractDecompositionContract)

Record how the head related `model[:w]` to `model[:lw]`/`model[:sw]`.

The first declaration wins: a head may run both builders (the mixed-integer
[`RiskBudgeting`](@ref) head calls [`set_rb_mip_w!`](@ref), then hands the same `lw`/`sw` to
[`set_weight_constraints!`](@ref), which re-states them as bounds). The identity is the
stronger statement and still holds, so the bounds must not overwrite it.

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

Return the head's decomposition contract, or `nothing` when no head declared one.

`nothing` means the model has no short side — the weights are their own long part, `lw` is an
alias for `w` and there is no `sw`, so there is no decomposition to pin.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`set_decomposition_contract!`](@ref)
"""
function decomposition_contract(model::JuMP.Model)
    return haskey(model, :decomposition_contract) ? model[:decomposition_contract] : nothing
end
"""
    get_ret(model::JuMP.Model)

Return the portfolio expected-return expression `model[:ret]`.

Asserts the return expression has been registered; errors otherwise.

# Related

  - [`get_risk`](@ref)
"""
function get_ret(model::JuMP.Model)
    @argcheck(haskey(model, :ret),
              ArgumentError("model[:ret] (portfolio expected-return expression) has not been registered; call set_return_constraints! first"))
    return model[:ret]
end
"""
    get_risk(model::JuMP.Model)

Return the scalarised portfolio risk expression `model[:risk]`.

Asserts the risk expression has been registered (via [`scalarise_risk_expression!`](@ref));
errors otherwise.

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

Return `true` if the portfolio returns `model[Symbol(prefix, :X)]` have been
registered (via [`set_portfolio_returns!`](@ref)).

# Related

  - [`get_X`](@ref)
"""
function has_X(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :X))
end
"""
    get_X(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the portfolio returns expression `model[Symbol(prefix, :X)]`.

Asserts it has been registered (via [`set_portfolio_returns!`](@ref)); errors otherwise.

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

Return `true` if the net portfolio returns `model[Symbol(prefix, :net_X)]` have been
registered (via [`set_net_portfolio_returns!`](@ref)).

# Related

  - [`get_net_X`](@ref)
"""
function has_net_X(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :net_X))
end
"""
    get_net_X(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the net portfolio returns expression `model[Symbol(prefix, :net_X)]`.

Asserts it has been registered (via [`set_net_portfolio_returns!`](@ref)); errors otherwise.

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

Return `true` if the gross portfolio returns `model[Symbol(prefix, :Xap1)]` have been
registered (via [`set_asset_returns_plus_one!`](@ref)).

# Related

  - [`get_Xap1`](@ref)
"""
function has_Xap1(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :Xap1))
end
"""
    get_Xap1(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the gross portfolio returns expression `model[Symbol(prefix, :Xap1)]` (`X .+ 1`).

Asserts it has been registered (via [`set_asset_returns_plus_one!`](@ref)); errors otherwise.

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

Return `true` if the drawdowns-plus-one `model[Symbol(prefix, :ddap1)]` have been
registered (via [`set_portfolio_drawdowns_plus_one!`](@ref)).

# Related

  - [`get_ddap1`](@ref)
"""
function has_ddap1(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :ddap1))
end
"""
    get_ddap1(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the drawdowns-plus-one expression `model[Symbol(prefix, :ddap1)]`.

Asserts it has been registered (via [`set_portfolio_drawdowns_plus_one!`](@ref)); errors otherwise.

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

Return `true` if the cumulative-drawdown variables `model[Symbol(prefix, :dd)]` have
been registered (via [`set_drawdown_constraints!`](@ref)).

# Related

  - [`get_dd`](@ref)
"""
function has_dd(model::JuMP.Model, prefix::Symbol = Symbol(""))
    return haskey(model, Symbol(prefix, :dd))
end
"""
    get_dd(model::JuMP.Model, prefix::Symbol = Symbol(""))

Return the cumulative-drawdown variables `model[Symbol(prefix, :dd)]`.

Asserts they have been registered (via [`set_drawdown_constraints!`](@ref)); errors otherwise.

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
