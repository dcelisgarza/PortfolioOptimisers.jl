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
# The concrete core `JuMPOptimisationResult` is defined in 03_JuMPOptimiser_a.jl, after
# `ProcessedJuMPOptimiserAttributes` and `JuMPOptimiser` — it is the result-side analogue
# of `JuMPOptimiser` and its typed constructor binds `pa::ProcessedJuMPOptimiserAttributes`,
# which must be in scope.
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based continuous optimisation results that **carry a risk measure**.

One of the two JuMP result halves; mirrors [`RiskJuMPOptimisationEstimator`](@ref). The sibling half is [`NonRiskJuMPOptimisationResult`](@ref), for the JuMP results that carry no risk measure at all. Concrete subtypes embed a [`JuMPOptimisationResult`](@ref) as their first field (`jr`) and add only their unique fields plus the trailing `fb`. Every subtype carries a resolved `r`; a JuMP result with no `r` belongs on the sibling branch. The default `getproperty` resolves unique fields directly and delegates everything else (including `:w` and the `pa` fall-through) to `jr`; types with composed sub-result fields override it to forward into those first.

# Related

  - [`NonRiskJuMPOptimisationResult`](@ref)
  - [`RJR_NRJR`](@ref)
  - [`NonJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type RiskJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based continuous optimisation results that **carry no risk measure**.

The sibling half of [`RiskJuMPOptimisationResult`](@ref). A relaxed risk budgeting run builds its constraints straight from `pr.sigma` and never resolves a measure, so its result has no `r` to carry. Splitting the branch keeps `r` mandatory on the risk half instead of optional on a shared type. Concrete subtypes follow the same shape: an embedded [`JuMPOptimisationResult`](@ref) `jr` first, their unique fields, then the trailing `fb`.

# Related

  - [`RiskJuMPOptimisationResult`](@ref)
  - [`RJR_NRJR`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
"""
abstract type NonRiskJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
    const RJR_NRJR = Union{<:RiskJuMPOptimisationResult, <:NonRiskJuMPOptimisationResult}

Union of both JuMP result halves.

The default `getproperty` and `propertynames` are bound here, not on either half alone. [`MeanRiskResult`](@ref) and [`NearOptimalCenteringResult`](@ref) declare no [`@forward_properties`](@ref) rule and depend on that default for `res.w`, so a half without it would silently cost the next measure-less leaf its property forwarding.

# Related

  - [`RiskJuMPOptimisationResult`](@ref)
  - [`NonRiskJuMPOptimisationResult`](@ref)
"""
const RJR_NRJR = Union{<:RiskJuMPOptimisationResult, <:NonRiskJuMPOptimisationResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Default property access for [`RJR_NRJR`](@ref): unique fields resolve directly; everything else delegates to the embedded [`JuMPOptimisationResult`](@ref) `jr`.
"""
function Base.getproperty(r::RJR_NRJR, sym::Symbol)
    return if sym in fieldnames(typeof(r))
        getfield(r, sym)
    else
        getproperty(getfield(r, :jr), sym)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Default property enumeration for [`RJR_NRJR`](@ref): mirrors the default `getproperty` by unioning the receiver's own field names with everything forwarded from the embedded [`JuMPOptimisationResult`](@ref) `jr` (which itself forwards `pa`). Concrete subtypes that override `getproperty` (e.g. via [`@forward_properties`](@ref)) emit their own, more-specific `propertynames`.
"""
function Base.propertynames(r::RJR_NRJR)
    return Tuple(unique((fieldnames(typeof(r))..., propertynames(getfield(r, :jr))...)))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio objective functions.

Subtype `ObjectiveFunction` to implement portfolio optimisation objectives such as minimum risk, maximum return, or maximum Sharpe ratio.

The four concrete children are the source's four classic objective functions, one per subsection.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumReturn`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumUtility`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.
"""
abstract type ObjectiveFunction <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based returns estimators used in optimisation models.

`JuMPReturnsEstimator` types define how expected returns are incorporated into JuMP models.

The two children are the source's two return definitions: the arithmetic return of Section 8.1.1 and the geometric return of Section 8.1.2.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.1.
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

# Interfaces

In order to implement a new constraint that works seamlessly with the library, subtype `CustomJuMPConstraint` with the constraint's parameters as fields, and implement the following method. The verb is `public` and not exported, so the method is defined on the qualified name, `PortfolioOptimisers.add_custom_constraint!`.

## `add_custom_constraint!`

  - `add_custom_constraint!(model::JuMP.Model, ccnt::MyConstraint, optimiser, attrs::ProcessedJuMPOptimiserAttributes) -> Nothing`: Add the constraint to `model`. There is no fallback: a subtype with no method of its own raises. Scale the constraint by [`get_constraint_scale`](@ref), and multiply any constant bound by [`get_k`](@ref), the homogenisation variable, so the bound is compared against unrescaled weights under a ratio objective.

### Arguments

  - `model`: The JuMP model, mid-assembly; read the weights with [`get_w`](@ref).
  - `ccnt`: The concrete subtype instance.
  - `optimiser`: The outer optimisation estimator, e.g. the [`MeanRisk`](@ref) itself.
  - `attrs`: The processed problem data: `attrs.pr` is the prior, `attrs.wb` the bounds.

### Returns

  - `nothing`.

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

# Interfaces

In order to implement a new objective term that works seamlessly with the library, subtype `CustomJuMPObjective` with the term's parameters as fields, and implement the following method. The verb is `public` and not exported, so the method is defined on the qualified name, `PortfolioOptimisers.add_custom_objective_term!`.

## `add_custom_objective_term!`

  - `add_custom_objective_term!(model::JuMP.Model, obj::ObjectiveFunction, cobj::MyObjective, optimiser, attrs::ProcessedJuMPOptimiserAttributes) -> Nothing`: Contribute the term to the model's objective. There is no fallback: a subtype with no method of its own raises. Contribute through [`add_to_objective_penalty!`](@ref) rather than by touching the objective expression: the accumulated penalty is folded in with the sign the objective's sense needs, so a contribution always worsens the objective and a reward is a negative contribution. A term that is not homogeneous of degree one in the weights multiplies its constants by [`get_k`](@ref).

### Arguments

  - `model`: The JuMP model, mid-assembly; read the weights with [`get_w`](@ref).
  - `obj`: The objective being built, which differs from the declared one inside a [`Frontier`](@ref) sweep.
  - `cobj`: The concrete subtype instance.
  - `optimiser`: The outer optimisation estimator, e.g. the [`MeanRisk`](@ref) itself.
  - `attrs`: The processed problem data: `attrs.pr` is the prior, `attrs.wb` the bounds.

### Returns

  - `nothing`.

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
function port_opt_view(::CustomJuMPObjective, ::Any, args...; kwargs...)
    return nothing
end
"""
    add_custom_objective_term!(model::JuMP.Model, obj, cobj::Nothing, optimiser, attrs)
    add_custom_objective_term!(model::JuMP.Model, obj, cobj::CustomJuMPObjective, optimiser, attrs)

Add a custom objective term to the JuMP model.

Implement this for a subtype of [`CustomJuMPObjective`](@ref) to price a preference the library does not already name. Contribute the term with [`add_to_objective_penalty!`](@ref) rather than touching the objective expression: the accumulated penalty is folded in by [`add_penalty_to_objective!`](@ref) with the sign factor matching the objective's optimisation sense, so a contribution always worsens the objective and **a reward is a negative contribution**. This is what makes a term correct under every objective, [`MaximumRatio`](@ref) included, without the implementer consulting the sense.

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

                               and contribute the term with `add_to_objective_penalty!(model, expr)`. A reward is a negative contribution; the optimisation sense is applied for you."""))
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

Implement this for a subtype of [`CustomJuMPConstraint`](@ref) to mandate a preference the library does not already name. Two idioms keep a hand-written constraint correct: scale it by [`get_constraint_scale`](@ref), and multiply any constant bound by [`get_k`](@ref), the homogenisation variable, so the bound is compared against unrescaled weights under a ratio objective.

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

                               Scale the constraint by `get_constraint_scale(model)` and multiply any constant bound by `get_k(model)`, the homogenisation variable, so that the bound holds against the unrescaled weights under a ratio objective."""))
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

export JuMPOptimisationSolution, JuMPOptimisationResult
public CustomJuMPObjective, CustomJuMPConstraint, VecJuMPObj, VecJuMPConstr,
       add_custom_objective_term!, add_custom_constraint!, add_to_objective_penalty!
