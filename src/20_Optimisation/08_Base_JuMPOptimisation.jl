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
    add_custom_objective_term!(model::JuMP.Model, obj, cobj::Nothing, optimiser, attrs)
    add_custom_objective_term!(model::JuMP.Model, obj, cobj::CustomJuMPObjective, optimiser, attrs)

Add a custom objective term to the JuMP model.

Implement this for a subtype of [`CustomJuMPObjective`](@ref) to price a preference the library does not already name. Contribute the term with [`add_to_objective_penalty!`](@ref) rather than touching the objective expression: the accumulated penalty is folded in by [`add_penalty_to_objective!`](@ref) with the sign factor matching the objective's optimisation sense, so a contribution always worsens the objective and **a reward is a negative contribution**. This is what makes a term correct under every objective, [`MaximumRatio`](@ref) included, without the implementer consulting the sense (ADR 0036).

`add_to_objective_penalty!` promotes an affine accumulator to a quadratic one as needed, so a quadratic term is safe on every configuration.

Terms that are not homogeneous of degree one in `w` must still multiply any constant by [`get_k`](@ref): under a ratio objective the weights are solved in a rescaled space.

There is no no-op fallback for a [`CustomJuMPObjective`](@ref) — a subtype with no method of its own raises, so a mis-shaped or stale signature fails loudly instead of silently contributing nothing. `nothing` (no custom term configured) is the only no-op.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model, mid-assembly.
  - `obj`: The [`ObjectiveFunction`](@ref) *being built*. During a [`Frontier`](@ref) sweep this differs from the objective the user declared — the endpoint sub-problems are built as [`MinimumRisk`](@ref) and [`MaximumReturn`](@ref).
  - `cobj`: The custom objective estimator; the argument to dispatch on.
  - `optimiser`: The outer optimisation estimator (e.g. the [`MeanRisk`](@ref) itself); its `opt` field is the [`JuMPOptimiser`](@ref).
  - `attrs::ProcessedJuMPOptimiserAttributes`: Processed problem data — `attrs.pr` (prior), `attrs.ret` (returns estimator), `attrs.wb` (bounds), and the rest.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`add_to_objective_penalty!`](@ref) — how to contribute a term
  - [`add_custom_constraint!`](@ref) — the constraint-side analogue
"""
function add_custom_objective_term!(::JuMP.Model, ::Any, ::Nothing, ::Any, ::Any)
    return nothing
end
function add_custom_objective_term!(::JuMP.Model, ::Any, cobj::CustomJuMPObjective, ::Any,
                                    ::Any)
    return throw(ArgumentError("""
                               `$(nameof(typeof(cobj)))` subtypes `CustomJuMPObjective` but defines no `add_custom_objective_term!` method, so its term would contribute nothing.

                               Define one with the signature:

                                   PortfolioOptimisers.add_custom_objective_term!(model::JuMP.Model, obj, cobj::$(nameof(typeof(cobj))), optimiser, attrs)

                               and contribute the term with `add_to_objective_penalty!(model, expr)`. A reward is a negative contribution; the optimisation sense is applied for you. See ADR 0036."""))
end
"""
    add_custom_objective_term!(model::JuMP.Model, obj, cobjs::VecJuMPObj, optimiser, attrs)

Apply each custom objective term in a vector, in order. Dispatches to the per-type [`add_custom_objective_term!`](@ref) for every element, so a `cobj` vector composes several custom terms into one objective — they accumulate additively in the shared objective penalty.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`VecJuMPObj`](@ref)
"""
function add_custom_objective_term!(model::JuMP.Model, obj, cobjs::VecJuMPObj, optimiser,
                                    attrs)
    for cobj in cobjs
        add_custom_objective_term!(model, obj, cobj, optimiser, attrs)
    end
    return nothing
end
"""
    add_custom_constraint!(model::JuMP.Model, ccnt::Nothing, optimiser, attrs)
    add_custom_constraint!(model::JuMP.Model, ccnt::CustomJuMPConstraint, optimiser, attrs)

Add a custom constraint to the JuMP model.

Implement this for a subtype of [`CustomJuMPConstraint`](@ref) to mandate a preference the library does not already name. Two idioms keep a hand-written constraint correct (ADR 0008): scale it by [`get_constraint_scale`](@ref), and multiply any constant bound by [`get_k`](@ref), the homogenisation variable, so the bound is compared against unrescaled weights under a ratio objective.

There is no no-op fallback for a [`CustomJuMPConstraint`](@ref) — a subtype with no method of its own raises, so a mis-shaped or stale signature fails loudly instead of silently adding no constraint. `nothing` (no custom constraint configured) is the only no-op.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model, mid-assembly.
  - `ccnt`: The custom constraint estimator; the argument to dispatch on.
  - `optimiser`: The outer optimisation estimator (e.g. the [`MeanRisk`](@ref) itself).
  - `attrs::ProcessedJuMPOptimiserAttributes`: Processed problem data.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`add_custom_objective_term!`](@ref) — the objective-side analogue
"""
function add_custom_constraint!(::JuMP.Model, ::Nothing, ::Any, ::Any)
    return nothing
end
function add_custom_constraint!(::JuMP.Model, ccnt::CustomJuMPConstraint, ::Any, ::Any)
    return throw(ArgumentError("""
                               `$(nameof(typeof(ccnt)))` subtypes `CustomJuMPConstraint` but defines no `add_custom_constraint!` method, so it would add no constraint.

                               Define one with the signature:

                                   PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model, ccnt::$(nameof(typeof(ccnt))), optimiser, attrs)

                               Scale the constraint by `get_constraint_scale(model)` and multiply any constant bound by `get_k(model)`. See ADR 0008."""))
end
"""
    add_custom_constraint!(model::JuMP.Model, ccnts::VecJuMPConstr, optimiser, attrs)

Apply each custom constraint in a vector, in order. Dispatches to the per-type [`add_custom_constraint!`](@ref) for every element, so a `ccnt` vector adds several custom constraints to the same model.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`VecJuMPConstr`](@ref)
"""
function add_custom_constraint!(model::JuMP.Model, ccnts::VecJuMPConstr, optimiser, attrs)
    for ccnt in ccnts
        add_custom_constraint!(model, ccnt, optimiser, attrs)
    end
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
    SHARED_STATE

The Model State entries deliberately shared **bare** across a nested risk build.

The complement of Per-Build Risk State: an entry belongs here iff it is *not* a function of
the weights being optimised and *not* a build-scoped presence flag, so the inner and outer
builds want the same object and prefixing it would break sharing rather than protect it
(ADR 0005). [`shared_get`](@ref) and friends validate against this set, so the classification
is enforced at run time rather than only by the seam-lock test (ADR 0037).

Each grouping records *why* those entries are shared. Adding a name here is a claim that a
nested build may safely see the enclosing build's copy — check that claim before adding.
"""
const SHARED_STATE = Set{Symbol}([# Pure functions of the prior `pr`: identical in the inner
                                  # and outer build (Cholesky/eigendecompositions, factor
                                  # risk contribution lift).
                                  :G, :GV, :Gkt, :vals_Akt, :vecs_Akt, :frc_W, :frc_M,
                                  :frc_M_PSD,
                                  # Model-wide singletons established once, before the risk
                                  # spine runs.
                                  :sc, :so, :k, :w, :ret, :risk, :fees, :unit_budget,
                                  :decomposition_contract, :mip_indicators, :ss,
                                  # Weight shaping: outer level only. A nested build shifts
                                  # the weights through its own prefixed `:w`; it does not
                                  # reshape the long/short parts.
                                  :lw, :sw, :wp, :wn, :w1, :w_obj, :wip,
                                  # Returns and objective plumbing, outer level only.
                                  :ohf, :op, :cost_bgt_expr, :bucs_w, :t_eucs_gw, :sr_risk,
                                  # Risk accumulation and frontier bookkeeping: collected by
                                  # the terminal scalarise seam (ADR 0024), which runs once
                                  # at the outer level. A nested build returns its
                                  # expression to its caller instead of pushing here.
                                  :risk_vec, :risk_frontier, :ret_frontier,
                                  # Per-optimiser scratch on the outer model.
                                  :noc_rk, :noc_rt, :psi,
                                  # The one deliberate write-prefixed / read-bare entry
                                  # (ADR 0005). The inner write is prefixed so a nested
                                  # variance cannot leak its presence outward; the only
                                  # readers are the outer-level phylogeny builders, which
                                  # add a `p·tr(W)` penalty when no variance is present and
                                  # so must see the OUTER flag. Reading this prefixed would
                                  # silently re-add the penalty under tracking.
                                  :variance_flag])
"""
    assert_shared_state(name::Symbol)

Assert `name` is a sanctioned bare Model State entry.

Guards the [`shared_get`](@ref) family so that reaching for a per-build entry without a
prefix fails loudly at the call site, rather than silently aliasing the enclosing build's
copy — the regression class that broke `IndependentVariableTracking`.
"""
function assert_shared_state(name::Symbol)
    @argcheck(name in SHARED_STATE,
              ArgumentError("model[:$name] is not a sanctioned bare Model State entry. If it is per-build risk state (a function of the weights, or a build-scoped presence flag) reach it with a prefix via state_get/state_build!; if it is genuinely shared across nested builds, add it to SHARED_STATE with the reason."))
    return nothing
end
"""
    shared_set!(model::JuMP.Model, name::Symbol, val)

Register `val` as the sanctioned bare Model State entry `name` and return it.

The unprefixed counterpart of [`state_set!`](@ref), for entries on [`SHARED_STATE`](@ref).

# Related

  - [`shared_get`](@ref)
"""
function shared_set!(model::JuMP.Model, name::Symbol, val)
    assert_shared_state(name)
    model[name] = val
    return val
end
"""
    shared_has(model::JuMP.Model, name::Symbol)

Return `true` if the sanctioned bare Model State entry `name` is registered.
"""
function shared_has(model::JuMP.Model, name::Symbol)
    assert_shared_state(name)
    return haskey(model, name)
end
"""
    shared_get(model::JuMP.Model, name::Symbol)

Return the sanctioned bare Model State entry `name`, asserting it has been registered.

The unprefixed counterpart of [`state_get`](@ref). Prefer a named accessor
([`get_w`](@ref), [`get_k`](@ref), [`get_ret`](@ref), …) where one exists.

# Related

  - [`shared_has`](@ref)
  - [`SHARED_STATE`](@ref)
"""
function shared_get(model::JuMP.Model, name::Symbol)
    assert_shared_state(name)
    @argcheck(haskey(model, name),
              ArgumentError("model[:$name] has not been registered; it is being read before the builder that produces it has run"))
    return model[name]
end
"""
    state_key(prefix::Symbol, name::Symbol)

Resolve the Model State key for entry `name` under `prefix`.

Internal to the Model State interface: the single place the prefix-namespacing convention
is spelled. Keeping it here is what lets the seam-lock test assert that no emitter builds a
key by hand — emitters reach Model State through [`state_get`](@ref), [`state_has`](@ref),
[`state_set!`](@ref) and [`state_build!`](@ref). See ADR 0037.

# Related

  - [`state_build!`](@ref)
"""
function state_key(prefix::Symbol, name::Symbol)
    return Symbol(prefix, name)
end
"""
    state_set!(model::JuMP.Model, prefix::Symbol, name::Symbol, val)

Register `val` in the model under the prefixed Model State key and return it.

A nested risk build (e.g. risk tracking) passes a non-empty `prefix` so the shared
infrastructure entries it creates (`:X`, `:net_X`, `:W`, `:dd`, …) do not collide with the
outer model's; the default empty prefix reproduces the bare key. See ADR 0005.

# Related

  - [`state_get`](@ref)
  - [`state_build!`](@ref)
"""
function state_set!(model::JuMP.Model, prefix::Symbol, name::Symbol, val)
    model[state_key(prefix, name)] = val
    return val
end
"""
    state_has(model::JuMP.Model, prefix::Symbol, name::Symbol)

Return `true` if Model State entry `name` is registered under `prefix`.

# Related

  - [`state_get`](@ref)
"""
function state_has(model::JuMP.Model, prefix::Symbol, name::Symbol)
    return haskey(model, state_key(prefix, name))
end
"""
    state_get(model::JuMP.Model, prefix::Symbol, name::Symbol)

Return Model State entry `name` under `prefix`, asserting it has been registered.

Prefer a named accessor ([`get_X`](@ref), [`get_net_X`](@ref), [`get_dd`](@ref), …) where
one exists: those name the builder that produces the entry, so an out-of-order read reports
which builder to call instead of a generic missing-entry error.

# Related

  - [`state_has`](@ref)
  - [`state_build!`](@ref)
"""
function state_get(model::JuMP.Model, prefix::Symbol, name::Symbol)
    key = state_key(prefix, name)
    @argcheck(haskey(model, key),
              ArgumentError("model[$key] has not been registered; it is being read before the builder that produces it has run"))
    return model[key]
end
"""
    state_build!(f, model::JuMP.Model, prefix::Symbol, name::Symbol)

Return Model State entry `name` under `prefix`, building it with `f()` exactly once.

The memoise-on-prefixed-key idiom shared by every risk and constraint emitter: if the entry
is already registered — an earlier measure in the same build produced it, or an outer build
already did — it is returned untouched; otherwise `f()` runs and its value is registered
under the prefixed key. Companion entries created inside `f` register with
[`state_set!`](@ref).

Because the key is resolved here rather than at the call site, a Model State entry added in
future participates in the prefix discipline with no further work. That is what closes the
residual hole ADR 0004 §2 accepted; see ADR 0037.

# Related

  - [`state_set!`](@ref)
  - [`state_get`](@ref)
"""
function state_build!(f, model::JuMP.Model, prefix::Symbol, name::Symbol)
    key = state_key(prefix, name)
    if haskey(model, key)
        return model[key]
    end
    val = f()
    model[key] = val
    return val
end
"""
    mark_state!(model::JuMP.Model, prefix::Symbol, name::Symbol)

Record that this build has `name` present, idempotently.

A build-scoped presence flag: `name` carries no value beyond its own existence, and readers
test it with [`state_has`](@ref) rather than reading it. Marking under `prefix` is what keeps
a nested build's flags out of the enclosing build — the second half of Per-Build Risk State
(ADR 0005), the half that is not weight-dependent.

# Related

  - [`state_has`](@ref)
  - [`state_build!`](@ref)
"""
function mark_state!(model::JuMP.Model, prefix::Symbol, name::Symbol)
    state_build!(() -> true, model, prefix, name)
    return nothing
end
"""
    nested_prefix(prefix::Symbol, tag::Symbol)
    nested_prefix(prefix::Symbol, tag::Symbol, i)

Compose the Model State namespace a nested build threads down its own spine.

Distinct from a Model State *key*: this produces a `prefix`, not an entry name, so a nested
build's entries cannot alias the enclosing build's. `tag` names the nesting kind (`:tr_iv_`,
`:tr_dv_`, `:te_ir_`, `:te_dr_`, `:gain_`) and the optional `i` disambiguates the measure
index, which is what makes tracking-nested-in-tracking collision-free. See ADR 0005.

# Related

  - [`state_build!`](@ref)
"""
function nested_prefix(prefix::Symbol, tag::Symbol)
    return Symbol(prefix, tag)
end
function nested_prefix(prefix::Symbol, tag::Symbol, i)
    return Symbol(prefix, tag, i, :_)
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
        @warn("Failed to solve optimisation problem.\nCheck `result.retcode.res` on the returned result for per-solver diagnostics.")
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
    return state_set!(model, prefix, :X, JuMP.@expression(model, X * w))
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
        return state_set!(model, prefix, :net_X, JuMP.@expression(model, X .- fees))
    else
        return state_set!(model, prefix, :net_X, JuMP.@expression(model, X))
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
    return state_set!(model, prefix, :Xap1, JuMP.@expression(model, X .+ one(eltype(X))))
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
    return state_set!(model, prefix, :nXap1, JuMP.@expression(model, -X .+ one(eltype(X))))
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
    return state_set!(model, prefix, :ddap1, JuMP.@expression(model, _ddap1))
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

# The custom-term extension point (ADR 0036). Public API, but deliberately not exported:
# `get_w`/`get_k` are too generic to put in a user's namespace, and the two builder methods
# must be qualified anyway to be extended. Reach them as `PortfolioOptimisers.name`, or
# import the ones you need. (`ProcessedJuMPOptimiserAttributes`, the `attrs` bundle a hook
# receives, is already exported by `10_JuMPOptimiser.jl`.)
public CustomJuMPObjective, CustomJuMPConstraint, VecJuMPObj, VecJuMPConstr,
       add_custom_objective_term!, add_custom_constraint!, add_to_objective_penalty!, get_w,
       get_k, get_constraint_scale
