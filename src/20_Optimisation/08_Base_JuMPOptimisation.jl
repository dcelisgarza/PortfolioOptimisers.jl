"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base JuMP-based portfolio optimisation estimators.

These are configuration-level types (e.g., `JuMPOptimiser`) that define the optimisation problem setup for JuMP-based optimisers.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based portfolio optimisation estimators.

JuMP optimisers formulate and solve portfolio optimisation problems using mathematical programming via the JuMP.jl framework.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
"""
abstract type JuMPOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the estimator's own problem-definition fields, the inner JuMP optimiser, or the fallback carry time-dependent constraints.
"""
function is_time_dependent(opt::JuMPOptimisationEstimator)
    return (!isempty(time_dependent_fields(opt)) ||
            is_time_dependent(opt.opt) ||
            is_time_dependent(opt.fb))
end
function assert_time_dependent_fold_count(opt::JuMPOptimisationEstimator, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    assert_time_dependent_fold_count(opt.opt, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve time-dependent constraints for the fold described by `ctx`: the estimator's own scheduled fields (risk measure, objective, warm start, fallback, …) are swapped for their per-fold values, then the inner JuMP optimiser and the (possibly just-swapped-in) fallback are recursed into with the same context.
"""
function update_time_dependent_estimator(opt::JuMPOptimisationEstimator,
                                         ctx::TimeDependentContext, all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              opt = update_time_dependent_estimator(opt.opt, ctx,
                                                                    all_binds),
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace time-dependent constraints with their static defaults, both on the estimator's own fields and by recursing into the inner JuMP optimiser and fallback.
"""
function reset_time_dependent_estimator(opt::JuMPOptimisationEstimator)
    if !is_time_dependent(opt)
        return opt
    end
    opt = reset_time_dependent_fields(opt)
    return rebuild_estimator(opt,
                             (; opt = reset_time_dependent_estimator(opt.opt),
                              fb = reset_time_dependent_estimator(opt.fb)))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk-based JuMP portfolio optimisation estimators.

Subtype `RiskJuMPOptimisationEstimator` to implement optimisers that minimise or constrain risk measures as the primary objective.

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
"""
abstract type RiskJuMPOptimisationEstimator <: JuMPOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the embedded JuMP optimisation result core.

Mirrors [`BaseJuMPOptimisationEstimator`](@ref): the factored-out struct holding the fields common to every JuMP-based optimisation result lives on this branch and is *not* part of the optimisation result hierarchy. The concrete core is [`JuMPOptimisationResult`](@ref).

# Related

  - [`JuMPOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
"""
abstract type BaseJuMPOptimisationResult <: AbstractResult end
# The concrete core `JuMPOptimisationResult` is defined in 10_JuMPOptimiser.jl, after
# `ProcessedJuMPOptimiserAttributes` and `JuMPOptimiser` — it is the result-side analogue
# of `JuMPOptimiser` and its typed constructor binds `pa::ProcessedJuMPOptimiserAttributes`,
# which must be in scope.
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based continuous optimisation results.

The JuMP half of the result split; mirrors [`RiskJuMPOptimisationEstimator`](@ref). Concrete subtypes embed a [`JuMPOptimisationResult`](@ref) as their first field (`jr`) and add only their unique fields plus the trailing `fb`. The default `getproperty` resolves unique fields directly and delegates everything else (including `:w` and the `pa` fall-through) to `jr`; types with composed sub-result fields override it to forward into those first.

# Related

  - [`NonJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type RiskJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Default property access for [`RiskJuMPOptimisationResult`](@ref): unique fields resolve directly; everything else delegates to the embedded [`JuMPOptimisationResult`](@ref) `jr`.
"""
function Base.getproperty(r::RiskJuMPOptimisationResult, sym::Symbol)
    return if sym in fieldnames(typeof(r))
        getfield(r, sym)
    else
        getproperty(getfield(r, :jr), sym)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Default property enumeration for [`RiskJuMPOptimisationResult`](@ref): mirrors the default `getproperty` by unioning the receiver's own field names with everything forwarded from the embedded [`JuMPOptimisationResult`](@ref) `jr` (which itself forwards `pa`). Concrete subtypes that override `getproperty` (e.g. via [`@forward_properties`](@ref)) emit their own, more-specific `propertynames`.
"""
function Base.propertynames(r::RiskJuMPOptimisationResult)
    return Tuple(unique((fieldnames(typeof(r))..., propertynames(getfield(r, :jr))...)))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio objective functions.

Subtype `ObjectiveFunction` to implement portfolio optimisation objectives such as minimum risk, maximum return, or maximum Sharpe ratio.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumReturn`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumUtility`](@ref)
"""
abstract type ObjectiveFunction <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based returns estimators used in optimisation models.

`JuMPReturnsEstimator` types define how expected returns are incorporated into JuMP models.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
"""
abstract type JuMPReturnsEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP constraint estimators.

The extension point for user-defined constraints and objectives. Rather than subtyping this directly, subtype one of the two purpose-built children and implement its one contract method:

  - [`CustomJuMPConstraint`](@ref) ⇒ implement [`add_custom_constraint!`](@ref), supply via `JuMPOptimiser`'s `ccnt`.
  - [`CustomJuMPObjective`](@ref) ⇒ implement [`add_custom_objective_term!`](@ref), supply via `JuMPOptimiser`'s `cobj`.

(The objective child subtypes [`AbstractEstimator`](@ref) directly — it is grouped here as the sibling extension point, not by type hierarchy.)

# Related

  - [`CustomJuMPConstraint`](@ref) / [`add_custom_constraint!`](@ref)
  - [`CustomJuMPObjective`](@ref) / [`add_custom_objective_term!`](@ref)
"""
abstract type JuMPConstraintEstimator <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom JuMP constraint implementations.

Subtype this and implement [`add_custom_constraint!`](@ref) — the single method the type exists to make you define — to add custom constraints to the JuMP model. Pass the resulting estimator (or a vector of them) as the `ccnt` field of [`JuMPOptimiser`](@ref).

# Related

  - [`add_custom_constraint!`](@ref) — the method to implement
  - [`CustomJuMPObjective`](@ref) / [`add_custom_objective_term!`](@ref) — the objective-side analogue
  - [`JuMPOptimiser`](@ref) — its `ccnt` field is where a custom constraint is supplied
"""
abstract type CustomJuMPConstraint <: JuMPConstraintEstimator end
"""
    const VecJuMPConstr = AbstractVector{<:CustomJuMPConstraint}

Alias for a vector of JuMP constraint estimators.

# Related

  - [`CustomJuMPConstraint`](@ref)
"""
const VecJuMPConstr = AbstractVector{<:CustomJuMPConstraint}
"""
    const JuMPConstr_VecJuMPConstr = Union{<:CustomJuMPConstraint, <:VecJuMPConstr}

Alias for a single JuMP constraint estimator or a vector of them.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`VecJuMPConstr`](@ref)
"""
const JuMPConstr_VecJuMPConstr = Union{<:CustomJuMPConstraint, <:VecJuMPConstr}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom JuMP objective implementations.

Subtype this and implement [`add_custom_objective_term!`](@ref) — the single method the type exists to make you define — to add custom penalty or reward terms to the JuMP model objective. Pass the resulting estimator (or a vector of them) as the `cobj` field of [`JuMPOptimiser`](@ref).

# Related

  - [`add_custom_objective_term!`](@ref) — the method to implement
  - [`CustomJuMPConstraint`](@ref) / [`add_custom_constraint!`](@ref) — the constraint-side analogue
  - [`JuMPOptimiser`](@ref) — its `cobj` field is where a custom objective is supplied
"""
abstract type CustomJuMPObjective <: AbstractEstimator end
"""
    const VecJuMPObj = AbstractVector{<:CustomJuMPObjective}

Alias for a vector of JuMP objective estimators.

# Related

  - [`CustomJuMPObjective`](@ref)
"""
const VecJuMPObj = AbstractVector{<:CustomJuMPObjective}
"""
    const JuMPObj_VecJuMPObj = Union{<:CustomJuMPObjective, <:VecJuMPObj}

Alias for a single JuMP objective estimator or a vector of them.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`VecJuMPObj`](@ref)
"""
const JuMPObj_VecJuMPObj = Union{<:CustomJuMPObjective, <:VecJuMPObj}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: custom JuMP constraints never require previous portfolio weights.
"""
function needs_previous_weights(::CustomJuMPConstraint)
    return false
end
function needs_previous_weights(c::VecJuMPConstr)
    return any(needs_previous_weights, c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: custom JuMP objectives never require previous portfolio weights.
"""
function needs_previous_weights(::CustomJuMPObjective)
    return false
end
function needs_previous_weights(c::VecJuMPObj)
    return any(needs_previous_weights, c)
end
function port_opt_view(::CustomJuMPConstraint, ::Any, args...; kwargs...)
    return nothing
end
function port_opt_view(cs::VecJuMPConstr, i::Any, args...; kwargs...)
    return [port_opt_view(c, i, args...; kwargs...) for c in cs]
end
function port_opt_view(::CustomJuMPObjective, ::Any, args...; kwargs...)
    return nothing
end
function port_opt_view(cs::VecJuMPObj, i::Any, args...; kwargs...)
    return [port_opt_view(c, i, args...; kwargs...) for c in cs]
end
"""
    add_custom_objective_term!(args...; kwargs...)

Add a custom objective term to the JuMP model.

No-op fallback. Override this method for subtypes of [`CustomJuMPObjective`](@ref) to add custom penalty or reward terms to the JuMP model objective.

# Arguments

  - `args...`: JuMP model and custom objective type (ignored in fallback).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`add_custom_constraint!`](@ref)
"""
function add_custom_objective_term!(args...; kwargs...)
    return nothing
end
"""
    add_custom_constraint!(args...; kwargs...)

Add a custom constraint to the JuMP model.

No-op fallback. Override this method for subtypes of [`CustomJuMPConstraint`](@ref) to add custom constraints to the JuMP model.

# Arguments

  - `args...`: JuMP model and custom constraint type (ignored in fallback).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
function add_custom_constraint!(args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the solution (portfolio weights) from a JuMP optimisation model.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimisationSolution(; w::ArrNum) -> JuMPOptimisationSolution

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.

# Related

  - [`OptimisationModelResult`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
@concrete struct JuMPOptimisationSolution <: OptimisationModelResult
    """
    $(field_dict[:pw])
    """
    w
    function JuMPOptimisationSolution(w::ArrNum)
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        return new{typeof(w)}(w)
    end
end
function JuMPOptimisationSolution(; w::ArrNum)::JuMPOptimisationSolution
    return JuMPOptimisationSolution(w)
end
"""
    const VecJuMPOptSol = AbstractVector{<:JuMPOptimisationSolution}

Alias for a vector of JuMP optimisation solutions.

# Related

  - [`JuMPOptimisationSolution`](@ref)
"""
const VecJuMPOptSol = AbstractVector{<:JuMPOptimisationSolution}
"""
    const JuMPOptSol_VecJuMPOptSol = Union{<:JuMPOptimisationSolution, <:VecJuMPOptSol}

Alias for a single JuMP optimisation solution or a vector of them.

# Related

  - [`JuMPOptimisationSolution`](@ref)
  - [`VecJuMPOptSol`](@ref)
"""
const JuMPOptSol_VecJuMPOptSol = Union{<:JuMPOptimisationSolution, <:VecJuMPOptSol}
"""
    set_model_scales!(model::JuMP.Model, so::Number, sc::Number)

Register objective scale `so` and constraint scale `sc` as named expressions in the JuMP model.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `so::Number`: Objective scale factor.
  - `sc::Number`: Constraint scale factor.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
"""
function set_model_scales!(model::JuMP.Model, so::Number, sc::Number)
    JuMP.@expressions(model, begin
                          so, so
                          sc, sc
                      end)
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

`k >= 0` is the auxiliary scaling variable used to homogenise fractional/ratio
objectives (e.g. maximum ratio); recovered weights are `w / k`. Asserts `:k` has
been registered; errors otherwise.

# Related

  - [`process_model`](@ref)
  - [`get_w`](@ref)
"""
function get_k(model::JuMP.Model)
    @argcheck(haskey(model, :k),
              ArgumentError("model[:k] (homogenisation variable) has not been registered; call set_maximum_ratio_factor_variables! first"))
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
"""
    preg!(model::JuMP.Model, prefix::Symbol, name::Symbol, val)

Register `val` in the model under the prefixed key `Symbol(prefix, name)` and return it.

The single place the model-state namespacing convention lives: a nested risk build
(e.g. risk tracking) passes a non-empty `prefix` so the shared infrastructure keys it
creates (`:X`, `:net_X`, `:W`, `:dd`, …) do not collide with the outer model's; the
default empty prefix reproduces the bare key. Pairs with the prefixed read accessors.
See ADR 0004.

# Related

  - [`get_w`](@ref)
"""
function preg!(model::JuMP.Model, prefix::Symbol, name::Symbol, val)
    model[Symbol(prefix, name)] = val
    return val
end
"""
    set_initial_w!(args...)
    set_initial_w!(w::VecNum, wi::VecNum)

Set initial (warm-start) values for portfolio weight variables in the JuMP model.

The no-op fallback does nothing when `wi` is not provided. The two-argument method sets JuMP start values for each weight variable.

# Arguments

  - `w::VecNum`: Vector of JuMP weight variables.
  - `wi::VecNum`: Vector of initial weight values.

# Returns

  - `nothing`.

# Related

  - [`set_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_initial_w!(args...)
    return nothing
end
function set_initial_w!(w::VecNum, wi::VecNum)
    @argcheck(length(wi) == length(w),
              DimensionMismatch("wi ($(length(wi))) must match w ($(length(w)))"))
    JuMP.set_start_value.(w, wi)
    return nothing
end
"""
    set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})

Create portfolio weight variables in the JuMP model and optionally set initial values.

Registers a vector of weight variables `w` of length `size(X, 2)` in the model. If `wi` is provided, sets the initial values via [`set_initial_w!`](@ref).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (shape: observations × assets).
  - `wi`: Optional initial weight values.

# Returns

  - `nothing`.

# Related

  - [`set_initial_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})
    JuMP.@variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
"""
    process_model(model, retcode)

Extract the solution from an optimised JuMP model based on the return code.

On success, extracts the optimised weights from the model. On failure, returns an empty solution.

# Arguments

  - `model`: Optimised JuMP model.
  - `retcode`: Optimisation return code ([`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref)).

# Returns

  - Solution object.

# Related

  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function process_model(model::JuMP.Model, ::OptimisationSuccess)
    k = JuMP.value(model[:k])
    ik = !iszero(k) ? inv(k) : 1
    w = JuMP.value.(model[:w]) * ik
    return JuMPOptimisationSolution(; w = w)
end
function process_model(model::JuMP.Model, ::OptimisationFailure)
    return JuMPOptimisationSolution(; w = fill(NaN, length(model[:w])))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Attempt to solve the JuMP model using each solver in `opt.opt.slv` in order.

Tries each solver sequentially, checking feasibility and finite non-zero weights. Returns a `(retcode, solution)` tuple where `retcode` is [`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref) and `solution` is a [`JuMPOptimisationSolution`](@ref).

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`JuMPOptimisationSolution`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function optimise_JuMP_model!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                              datatype::DataType = Float64)
    trials = Dict()
    success = false
    for solver in opt.opt.slv
        try
            JuMP.set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
        catch err
            trials[solver.name] = Dict(:set_optimizer => err)
            continue
        end
        set_solver_attributes(model, solver.settings)
        try
            JuMP.optimize!(model)
        catch err
            trials[solver.name] = Dict(:optimize! => err)
            continue
        end
        trial = Dict{Symbol, Any}(:settings => solver.settings)
        try
            JuMP.assert_is_solved_and_feasible(model; solver.check_sol...)
            all_finite_weights = all(isfinite, JuMP.value.(model[:w]))
            all_non_zero_weights = !all(x -> isapprox(x, zero(datatype)),
                                        abs.(JuMP.value.(model[:w])))
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            trial[:assert_is_solved_and_feasible] = err
        end
        trial[:err] = JuMP.solution_summary(model)
        trials[solver.name] = trial
    end
    retcode = if success
        OptimisationSuccess(; res = trials)
    else
        @warn("Failed to solve optimisation problem. Check `retcode.res` for details.")
        OptimisationFailure(; res = trials)
    end
    return retcode, process_model(model, retcode)
end
"""
    set_portfolio_returns!(model::JuMP.Model, X::MatNum)

Compute and register portfolio returns expression `X * w` in the JuMP model.

If the expression already exists in the model, returns it directly (idempotent).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.

# Returns

  - The portfolio returns expression.

# Related

  - [`set_net_portfolio_returns!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_portfolio_returns!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :X))
        return model[Symbol(prefix, :X)]
    end
    w = get_w(model, prefix)
    return preg!(model, prefix, :X, JuMP.@expression(model, X * w))
end
"""
    set_net_portfolio_returns!(model::JuMP.Model, X::MatNum)

Compute and register net portfolio returns (after fees) in the JuMP model.

Calls [`set_portfolio_returns!`](@ref) and subtracts fees if present.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.

# Returns

  - The net portfolio returns expression.

# Related

  - [`set_portfolio_returns!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_net_portfolio_returns!(model::JuMP.Model, X::MatNum;
                                    prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :net_X))
        return model[Symbol(prefix, :net_X)]
    end
    X = set_portfolio_returns!(model, X; prefix = prefix)
    # `:fees` is shared and not recreated by a nested build, so it is read bare.
    if haskey(model, :fees)
        fees = model[:fees]
        return preg!(model, prefix, :net_X, JuMP.@expression(model, X .- fees))
    else
        return preg!(model, prefix, :net_X, JuMP.@expression(model, X))
    end
end
"""
    set_asset_returns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register portfolio asset gross returns `X .+ 1` in the JuMP model.

Used in drawdown and logarithmic return computations.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns expression.

# Returns

  - The gross asset returns expression `X .+ 1`.

# Related

  - [`set_portfolio_drawdowns_plus_one!`](@ref)
  - [`set_portfolio_returns!`](@ref)
"""
function set_asset_returns_plus_one!(model::JuMP.Model, X::MatNum;
                                     prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :Xap1))
        return model[Symbol(prefix, :Xap1)]
    end
    return preg!(model, prefix, :Xap1, JuMP.@expression(model, X .+ one(eltype(X))))
end
"""
    set_asset_neg_returns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register negative asset gross returns `-X .+ 1` in the JuMP model.

Used in drawdown and logarithmic return computations.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns expression.

# Returns

  - The negative gross asset returns expression `-X .+ 1`.

# Related

  - [`set_portfolio_drawdowns_plus_one!`](@ref)
  - [`set_portfolio_returns!`](@ref)
"""
function set_asset_neg_returns_plus_one!(model::JuMP.Model, X::MatNum;
                                         prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :nXap1))
        return model[Symbol(prefix, :nXap1)]
    end
    return preg!(model, prefix, :nXap1, JuMP.@expression(model, -X .+ one(eltype(X))))
end
"""
    set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register absolute drawdowns plus one in the JuMP model.

Computes `absolute_drawdown_arr(X) .+ 1` and registers it in the model.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Portfolio returns expression.

# Returns

  - The drawdowns-plus-one expression.

# Related

  - [`set_asset_returns_plus_one!`](@ref)
"""
function set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum;
                                           prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :ddap1))
        return model[Symbol(prefix, :ddap1)]
    end
    _ddap1 = absolute_drawdown_arr(X) .+ one(eltype(X))
    return preg!(model, prefix, :ddap1, JuMP.@expression(model, _ddap1))
end
"""
    scalarise_risk_expression!(model, r, X, T, ...) -> nothing

Scalarise a risk expression and add it to the JuMP model objective.

Generic function stub; concrete methods are defined in constraint and risk measure files. Each method adds the appropriate risk objective term for a given risk measure type `r`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function scalarise_risk_expression! end
"""
    set_risk_constraints!(model, r, X, T, ...) -> nothing

Set risk constraints in the JuMP model for a given risk measure.

Generic function stub; concrete methods are defined in constraint and risk measure files. Each method configures the appropriate risk constraint expressions for a given risk measure type `r`.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function set_risk_constraints! end

export JuMPOptimisationSolution, JuMPOptimisationResult
