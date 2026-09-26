"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the configuration of a JuMP-based portfolio optimisation.

A subtype, such as [`JuMPOptimiser`](@ref), holds the problem definition that every JuMP-based optimiser shares: the prior, the solvers, the weight bounds and the constraints.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisers that build and solve a JuMP model.

A subtype holds a [`JuMPOptimiser`](@ref) in its `opt` field and a fallback in its `fb` field, and adds the fields of its own formulation.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
"""
abstract type JuMPOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the optimiser, its inner [`JuMPOptimiser`](@ref) or its fallback carries a time-dependent field.

# Algorithm

 1. Return `true` when [`time_dependent_fields`](@ref) of `opt` is not empty.
 2. Otherwise return `true` when the inner optimiser `opt.opt` is time dependent.
 3. Otherwise return whether the fallback `opt.fb` is time dependent.
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

Resolve the time-dependent fields of the optimiser for the fold that `ctx` describes.

The optimiser's own scheduled fields take their value for the fold first. The fallback can be one of those fields, so the recursion into the fallback reads the value that the first step put in place.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false`.
 2. Replace each scheduled field of `opt` with its value for the fold, through [`update_time_dependent_fields`](@ref), giving the new `opt`.
 3. Resolve the inner optimiser `opt.opt` and the fallback `opt.fb` of the new `opt` with the same `ctx`.
 4. Rebuild `opt` with the two resolved values.
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

Replace the time-dependent fields of the optimiser, its inner [`JuMPOptimiser`](@ref) and its fallback with their static defaults.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false`.
 2. Replace each scheduled field of `opt` with its static default, through [`reset_time_dependent_fields`](@ref), giving the new `opt`.
 3. Reset the inner optimiser `opt.opt` and the fallback `opt.fb` of the new `opt`.
 4. Rebuild `opt` with the two reset values.
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

Abstract supertype for JuMP optimisers whose formulation reads a risk measure.

A subtype minimises a risk measure, or bounds one, as the core of its model.

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
"""
abstract type RiskJuMPOptimisationEstimator <: JuMPOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the core that every JuMP optimisation result embeds.

It is the result-side twin of [`BaseJuMPOptimisationEstimator`](@ref). It is not a member of the optimisation result hierarchy. A JuMP result holds its one concrete subtype, [`JuMPOptimisationResult`](@ref), in the field `jr`.

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

Abstract supertype for the JuMP optimisation results that carry a resolved risk measure.

It is the result-side twin of [`RiskJuMPOptimisationEstimator`](@ref), and [`NonRiskJuMPOptimisationResult`](@ref) is its sibling for the results that carry no risk measure. A concrete subtype holds an embedded [`JuMPOptimisationResult`](@ref) in its first field `jr`, then its own fields, then the fallback `fb`. Every subtype carries a resolved `r`.

The default `getproperty` of [`RJR_NRJR`](@ref) reads a field of the subtype directly and forwards every other name, `w` and the fields of `pa` among them, to `jr`. A subtype that holds a composed result in a field overrides it and forwards into that field first.

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

Abstract supertype for the JuMP optimisation results that carry no risk measure.

It is the sibling of [`RiskJuMPOptimisationResult`](@ref). A relaxed risk budgeting run builds its constraints from the covariance matrix `pr.sigma` and resolves no risk measure, so its result has no `r`. Two branches keep `r` mandatory on the risk branch, where one shared branch would make it optional. A concrete subtype has the same layout as its sibling: the embedded [`JuMPOptimisationResult`](@ref) `jr` first, then its own fields, then the fallback `fb`.

# Related

  - [`RiskJuMPOptimisationResult`](@ref)
  - [`RJR_NRJR`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
"""
abstract type NonRiskJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
    const RJR_NRJR = Union{<:RiskJuMPOptimisationResult, <:NonRiskJuMPOptimisationResult}

Union of the two branches of JuMP optimisation results.

The default `getproperty` and `propertynames` dispatch on this union, not on one branch. [`MeanRiskResult`](@ref) and [`NearOptimalCenteringResult`](@ref) declare no [`@forward_properties`](@ref) rule, and read `res.w` through the default. A default on the risk branch alone would leave a new result on the other branch with no forwarding.

# Related

  - [`RiskJuMPOptimisationResult`](@ref)
  - [`NonRiskJuMPOptimisationResult`](@ref)
"""
const RJR_NRJR = Union{<:RiskJuMPOptimisationResult, <:NonRiskJuMPOptimisationResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the property `sym` of a JuMP optimisation result.

# Algorithm

 1. When `sym` is a field of the result, return that field.
 2. Otherwise return `getproperty` of the embedded [`JuMPOptimisationResult`](@ref) `jr`, which in turn reads the fields of `pa`.
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

Return the names that the default `getproperty` of [`RJR_NRJR`](@ref) reads.

The tuple holds the field names of the result, then the property names of the embedded [`JuMPOptimisationResult`](@ref) `jr` that are not field names of the result. The names of `jr` include the fields of `pa`. A subtype that overrides `getproperty` through [`@forward_properties`](@ref) gets its own `propertynames` from that macro.
"""
function Base.propertynames(r::RJR_NRJR)
    return Tuple(unique((fieldnames(typeof(r))..., propertynames(getfield(r, :jr))...)))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the objective functions of a JuMP portfolio optimisation.

Four concrete subtypes are the four classic objectives of the source, one for each subsection of its Section 8.2: [`MinimumRisk`](@ref), [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and [`MaximumRatio`](@ref). The fifth, [`MaximumElementReturn`](@ref), is internal. It maximises one return term, for the corner solves of a return frontier.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumReturn`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumElementReturn`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.
"""
abstract type ObjectiveFunction <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that write the expected portfolio return into a JuMP model.

Two concrete subtypes are the two return definitions of the source: [`ArithmeticReturn`](@ref), the arithmetic return of its Section 8.1.1, and [`LogarithmicReturn`](@ref), the geometric return of its Section 8.1.2. The third, [`NoReturn`](@ref), writes a return term that is identically zero.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`NoReturn`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.1.
"""
abstract type JuMPReturnsEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the constraints that a user adds to a JuMP model.

Do not subtype it directly. Subtype [`CustomJuMPConstraint`](@ref), implement [`add_custom_constraint!`](@ref) for the subtype, and give an instance to the `ccnt` field of [`JuMPOptimiser`](@ref).

The objective-side extension point, [`CustomJuMPObjective`](@ref), subtypes [`AbstractEstimator`](@ref) and not this type. It is named here because it is the sibling extension point.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`add_custom_constraint!`](@ref)
  - [`CustomJuMPObjective`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
abstract type JuMPConstraintEstimator <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for a constraint that the user writes into the JuMP model.

Subtype it and implement [`add_custom_constraint!`](@ref) for the subtype. Give an instance, or a vector of instances, to the `ccnt` field of [`JuMPOptimiser`](@ref).

# Interfaces

To add a constraint, subtype `CustomJuMPConstraint` with the parameters of the constraint as fields, and implement the method below. The verb is `public` and not exported, so define the method on the qualified name, `PortfolioOptimisers.add_custom_constraint!`.

## `add_custom_constraint!`

  - `add_custom_constraint!(model::JuMP.Model, ccnt::MyConstraint, optimiser, attrs::ProcessedJuMPOptimiserAttributes) -> Nothing`: Add the constraint to `model`. The verb has no fallback for a subtype, so a subtype with no method of its own raises. Multiply the constraint by [`get_constraint_scale`](@ref). Multiply a constant bound by [`get_k`](@ref), the homogenisation variable, so that the bound holds on the reported weights under a ratio objective.

### Arguments

  - `model`: The JuMP model while it is built. Read the weights with [`get_w`](@ref).
  - `ccnt`: The instance of the subtype.
  - `optimiser`: The outer optimiser, for example the [`MeanRisk`](@ref).
  - `attrs`: The processed problem data. `attrs.pr` is the prior and `attrs.wb` the weight bounds.

### Returns

  - `nothing`.

# Related

  - [`add_custom_constraint!`](@ref): the method to implement.
  - [`CustomJuMPObjective`](@ref): the extension point for an objective term.
  - [`add_custom_objective_term!`](@ref)
  - [`JuMPOptimiser`](@ref): its `ccnt` field holds the custom constraint.
"""
abstract type CustomJuMPConstraint <: JuMPConstraintEstimator end
"""
    const VecJuMPConstr = AbstractVector{<:CustomJuMPConstraint}

Alias for a vector of custom JuMP constraints.

The `ccnt` field of [`JuMPOptimiser`](@ref) accepts a vector, and [`add_custom_constraint!`](@ref) dispatches on this alias to add each entry in order.

# Related

  - [`CustomJuMPConstraint`](@ref)
"""
const VecJuMPConstr = AbstractVector{<:CustomJuMPConstraint}
"""
    const JuMPConstr_VecJuMPConstr = Union{<:CustomJuMPConstraint, <:VecJuMPConstr}

Alias for one custom JuMP constraint or a vector of them.

The `ccnt` field of [`JuMPOptimiser`](@ref) accepts a value of this alias, `nothing`, or a time-dependent schedule of either.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`VecJuMPConstr`](@ref)
"""
const JuMPConstr_VecJuMPConstr = Union{<:CustomJuMPConstraint, <:VecJuMPConstr}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for a term that the user adds to the objective of the JuMP model.

Subtype it and implement [`add_custom_objective_term!`](@ref) for the subtype. Give an instance, or a vector of instances, to the `cobj` field of [`JuMPOptimiser`](@ref).

# Interfaces

To add a penalty or a reward, subtype `CustomJuMPObjective` with the parameters of the term as fields, and implement the method below. The verb is `public` and not exported, so define the method on the qualified name, `PortfolioOptimisers.add_custom_objective_term!`.

## `add_custom_objective_term!`

  - `add_custom_objective_term!(model::JuMP.Model, obj::ObjectiveFunction, cobj::MyObjective, optimiser, attrs::ProcessedJuMPOptimiserAttributes) -> Nothing`: Add the term to the objective of `model`. The verb has no fallback for a subtype, so a subtype with no method of its own raises. Add the term through [`add_to_objective_penalty!`](@ref), and do not write the objective expression. The model adds the accumulated penalty with the sign that the sense of the objective needs, so a positive term always makes the objective worse, and a reward is a negative term. A term that is not homogeneous of degree one in the weights multiplies its constants by [`get_k`](@ref).

### Arguments

  - `model`: The JuMP model while it is built. Read the weights with [`get_w`](@ref).
  - `obj`: The objective that the model builds. Inside a [`Frontier`](@ref) sweep it is not the objective that the user declared.
  - `cobj`: The instance of the subtype.
  - `optimiser`: The outer optimiser, for example the [`MeanRisk`](@ref).
  - `attrs`: The processed problem data. `attrs.pr` is the prior and `attrs.wb` the weight bounds.

### Returns

  - `nothing`.

# Related

  - [`add_custom_objective_term!`](@ref): the method to implement.
  - [`CustomJuMPConstraint`](@ref): the extension point for a constraint.
  - [`add_custom_constraint!`](@ref)
  - [`JuMPOptimiser`](@ref): its `cobj` field holds the custom term.
"""
abstract type CustomJuMPObjective <: AbstractEstimator end
"""
    const VecJuMPObj = AbstractVector{<:CustomJuMPObjective}

Alias for a vector of custom JuMP objective terms.

The `cobj` field of [`JuMPOptimiser`](@ref) accepts a vector, and [`add_custom_objective_term!`](@ref) dispatches on this alias to add each entry in order.

# Related

  - [`CustomJuMPObjective`](@ref)
"""
const VecJuMPObj = AbstractVector{<:CustomJuMPObjective}
"""
    const JuMPObj_VecJuMPObj = Union{<:CustomJuMPObjective, <:VecJuMPObj}

Alias for one custom JuMP objective term or a vector of them.

The `cobj` field of [`JuMPOptimiser`](@ref) accepts a value of this alias, `nothing`, or a time-dependent schedule of either.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`VecJuMPObj`](@ref)
"""
const JuMPObj_VecJuMPObj = Union{<:CustomJuMPObjective, <:VecJuMPObj}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`, because a custom JuMP constraint never reads the previous portfolio weights.

The method for a vector returns `true` when one of its entries does, so a vector reports `true` when one of its entries defines its own `true` method.
"""
function needs_previous_weights(::CustomJuMPConstraint)
    return false
end
function needs_previous_weights(c::VecJuMPConstr)
    return any(needs_previous_weights, c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`, because a custom JuMP objective term never reads the previous portfolio weights.

The method for a vector returns `true` when one of its entries does, so a vector reports `true` when one of its entries defines its own `true` method.
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

Add a custom term to the objective of the JuMP model.

Implement this verb for a subtype of [`CustomJuMPObjective`](@ref) to price a preference that the library does not name. Add the term with [`add_to_objective_penalty!`](@ref), and do not write the objective expression. [`add_penalty_to_objective!`](@ref) adds the accumulated penalty with the sign that the sense of the objective needs, so a positive term always makes the objective worse, and a reward is a negative term. The implementer therefore does not read the sense, and one method is correct under every objective, [`MaximumRatio`](@ref) included.

`add_to_objective_penalty!` changes an affine accumulator into a quadratic one when a term needs it, so a quadratic term is valid on every configuration.

A term that is not homogeneous of degree one in `w` multiplies each constant by [`get_k`](@ref), because a ratio objective solves for the weights in a rescaled space.

`nothing`, the value when no custom term is configured, is the only no-op.

# Arguments

  - `model::JuMP.Model`: The JuMP model while it is built.
  - `obj`: The [`ObjectiveFunction`](@ref) that the model builds. Inside a [`Frontier`](@ref) sweep it is not the objective that the user declared, because the sweep builds the sub-problems of its end points as [`MinimumRisk`](@ref) and [`MaximumReturn`](@ref).
  - `cobj`: The custom term, the argument to dispatch on.
  - `optimiser`: The outer optimiser, for example the [`MeanRisk`](@ref). Its `opt` field is the [`JuMPOptimiser`](@ref).
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed problem data: the prior `attrs.pr`, the returns estimator `attrs.ret`, the weight bounds `attrs.wb` and the rest.

# Validation

  - A subtype of [`CustomJuMPObjective`](@ref) with no method of its own raises an `ArgumentError` that names the subtype and the signature to define. A term with a wrong or stale signature therefore raises, and never adds an empty term in silence.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`add_to_objective_penalty!`](@ref): how to add a term.
  - [`add_custom_constraint!`](@ref): the verb for a constraint.
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

Add each custom term of a vector to the objective, in order.

The terms add to one shared objective penalty, so a vector in the `cobj` field puts the sum of its terms into the objective.

# Algorithm

 1. For each entry `cobj` of `cobjs`, in order, call [`add_custom_objective_term!`](@ref) with `cobj`.

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

Implement this verb for a subtype of [`CustomJuMPConstraint`](@ref) to impose a preference that the library does not name. Multiply the constraint by [`get_constraint_scale`](@ref). Multiply a constant bound by [`get_k`](@ref), the homogenisation variable, so that the bound holds on the reported weights under a ratio objective.

`nothing`, the value when no custom constraint is configured, is the only no-op.

# Arguments

  - `model::JuMP.Model`: The JuMP model while it is built.
  - `ccnt`: The custom constraint, the argument to dispatch on.
  - `optimiser`: The outer optimiser, for example the [`MeanRisk`](@ref).
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed problem data.

# Validation

  - A subtype of [`CustomJuMPConstraint`](@ref) with no method of its own raises an `ArgumentError` that names the subtype and the signature to define. A constraint with a wrong or stale signature therefore raises, and never adds an empty constraint in silence.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`add_custom_objective_term!`](@ref): the verb for an objective term.
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

Add each custom constraint of a vector to the model, in order.

# Algorithm

 1. For each entry `ccnt` of `ccnts`, in order, call [`add_custom_constraint!`](@ref) with `ccnt`.

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

Holds the portfolio weights that one solve of a JuMP model returns.

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
  - [`process_model`](@ref)
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

A [`Frontier`](@ref) sweep solves one model per point, and its result holds one solution for each point in a vector of this type.

# Related

  - [`JuMPOptimisationSolution`](@ref)
"""
const VecJuMPOptSol = AbstractVector{<:JuMPOptimisationSolution}
"""
    const JuMPOptSol_VecJuMPOptSol = Union{<:JuMPOptimisationSolution, <:VecJuMPOptSol}

Alias for one JuMP optimisation solution or a vector of them.

It is the type of the `sol` argument of [`JuMPOptimisationResult`](@ref), which holds one solution for a single solve and a vector for a [`Frontier`](@ref) sweep.

# Related

  - [`JuMPOptimisationSolution`](@ref)
  - [`VecJuMPOptSol`](@ref)
"""
const JuMPOptSol_VecJuMPOptSol = Union{<:JuMPOptimisationSolution, <:VecJuMPOptSol}

export JuMPOptimisationSolution, JuMPOptimisationResult
public CustomJuMPObjective, CustomJuMPConstraint, VecJuMPObj, VecJuMPConstr,
       add_custom_objective_term!, add_custom_constraint!, add_to_objective_penalty!
