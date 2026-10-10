"""
    supports_partial_fit(est) -> Bool

Answers whether [`partial_fit!`](@ref) folds this estimator.

An outer estimator that carries the observations asks each of its members. This is the rule for an estimator with mixed members. The outer estimator folds each member that folds, and it runs the batch verb over its own rows for each member that does not. So a caller writes the same estimator as in batch, with no wrapper at the call site, and no member carries a second copy of the sample.

The default is the buffering route. An estimator folds when it carries a [`SampleBufferState`](@ref), which [`Online`](@ref) seeds. A family with an exact fold of its own adds a method that returns `true`, beside that fold, under the same type bound without the `cache` parameter. An estimator of that family folds exactly when its bound matches, and by buffering when a wrapper seeded a buffer. A configuration that a family refuses adds no method, and it takes the default. The `SemiMoment` arms are such a configuration, because their clip moves when the mean moves. For them, the default is `true` only when a wrapper gave the estimator a buffer.

The predicate reads a type and a field, never a method table. A read of the dispatch would count a refusal as a fold, because a refusal is a method too, and the members that refuse are the members that must buffer.

# Arguments

  - `est`: The estimator to ask about.

# Returns

  - `folds::Bool`: `true` when [`partial_fit!`](@ref) folds this estimator.

# Related

  - [`partial_fit!`](@ref)
  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
"""
function supports_partial_fit(est::Union{<:AbstractEstimator,
                                         <:StatsBase.CovarianceEstimator})
    return hasfield(typeof(est), :cache) && isa(getfield(est, :cache), SampleBufferState)
end
function supports_partial_fit(::Nothing)
    # A member that the outer estimator does not hold folds vacuously: there is nothing to
    # fold and nothing to refit, so the call with no data skips it either way.
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the empty state an [`Online`](@ref) seeds into the estimator it wraps.

One sample buffer works for each estimator that refits, so the seed needs no knowledge of the type. A prior whose batch verb reads a factor matrix beside the returns records the factor rows in the same buffer, and [`needs_factor_returns`](@ref) decides whether its fold receives them. An estimator whose state is not a sample buffer has a method of its own that returns the state that its call with no data reads. A prior-less optimiser head is such an estimator, because its state is a fold context. Nothing else about the wrapper changes.

The function takes the estimator and not its type, so a family that sizes its seed from a field can read that field.

# Arguments

  - `est`: The estimator the wrapper wraps, read for its type.
  - `max_history`: The wrapper's cap, carried into the state.

# Returns

  - `state::AbstractPartialFitState`: The empty state to seed, carrying the cap and no observations.

# Related

  - [`Online`](@ref)
  - [`SampleBufferState`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_state_seed(::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
                           max_history::Option{<:Integer})
    return SampleBufferState(; max_history = max_history)
end
function update_online_estimator(o::Online)
    est = rebuild_estimator(o.est, (; cache = online_state_seed(o.est, o.max_history)))
    return update_online_estimator(est)
end
function update_online_estimator(::Online{<:CrossValidationEstimator})
    return throw(ArgumentError("`Online` on a scheme is an Online Scheme, and it reached the warm-up in an estimator slot: a scheme seeds no sample buffer, because it is the `cv` argument of `cross_val_predict`, not a field of the estimator that runs through it."))
end

"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Allocation Set of an online portfolio selection head, the set that contains each allocation of the recursion.

The head is [`OnlinePortfolioSelection`](@ref), and it holds the set in its `set` field. The budget is one on each set, so `⟨w, x⟩` is the wealth factor that the formula of each rule assumes. Cash is an asset with price relative one, and the rule allocates it like any other asset.

# Interfaces

In order to implement a new set kind, subtype `AbstractAllocationSet` with its constraint objects as part of the struct, and implement:

  - `resolve_allocation_set(set::AbstractAllocationSet, N::Integer, strict::Bool, datatype::DataType) -> AbstractAllocationSet`: The set with every estimator resolved to a value over the `N` assets, which the projection then reads.
  - `project(proj::AbstractProjectionGeometry, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector) -> AbstractVector`: The projection onto the resolved set, for every geometry the set admits.
  - `rows_needed(set::AbstractAllocationSet) -> Union{Nothing, Integer}`: The number of rows the set's constraints read at a step, `0` for a set that reads none.

## Arguments

  - `set`: The set.
  - `N`: The number of assets of the universe the set is resolved over.
  - `strict`: Whether an unknown name in a set is an error.
  - `datatype`: The element type a scalar bound is expanded in.

## Returns

  - `set::AbstractAllocationSet`: The resolved set.

# Related

  - [`BoundedAllocationSet`](@ref)
  - [`project`](@ref)
  - [`AbstractProjectionGeometry`](@ref)
"""
abstract type AbstractAllocationSet <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for an Allocation Set whose projection is a programme, a bare JuMP model that holds the constraints of the set.

The programme lets the set own a risk constraint that the shared risk-measure builders write, as a [`RiskConstraintOwner`](@ref) beside the JuMP optimisers.

# Interfaces

In order to implement a new programme set, subtype `AbstractProgrammeAllocationSet`, implement everything [`AbstractAllocationSet`](@ref) asks for, and implement:

  - `risk_constraint_solver(set::AbstractProgrammeAllocationSet)`: The solver a Deferred Quantity of the set's risk measure is resolved against.

# Related

  - [`AbstractAllocationSet`](@ref)
  - [`RiskConstraintOwner`](@ref)
  - [`risk_constraint_solver`](@ref)
"""
abstract type AbstractProgrammeAllocationSet <: AbstractAllocationSet end

public AbstractProgrammeAllocationSet, risk_constraint_solver, supports_partial_fit
