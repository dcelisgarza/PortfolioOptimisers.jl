"""
    online_wrapper_path(est)
    online_wrapper_path(::Online)
    online_wrapper_path(::TimeDependent)

Names the first field of an estimator tree that holds an [`Online`](@ref) declaration, or answers `nothing`.

[`assert_batch_entry`](@ref) uses this walk. Only the warm-up of the online arm of the fold loop resolves a wrapper, with [`update_online_estimator`](@ref). Without the check, a batch fit that reaches a wrapper gets a `MethodError` at `prior(pe, X)`, not a refusal. The walk reads the fields of [`estimator_fields`](@ref), and the answer is the dotted path from the root. It is `"pe"` for a wrapped prior on a naive optimiser, and `"opt.pe"` for a wrapped prior under a JuMP head. The walk does not enter a field that holds a wrapper, because the warm-up resolves the wrappers below it. A wrapper at the root answers `""`, the empty path. A [`TimeDependent`](@ref) schedule answers `nothing`, because its constructor refuses a wrapper among its entries. A value that is not an estimator also answers `nothing`.

# Algorithm

 1. Answer `""` when `est` is an [`Online`](@ref).
 2. Walk the fields of [`estimator_fields`](@ref) in order, and answer the name of the first field that holds a wrapper.
 3. For a field that holds no wrapper, find the path in the field with this function, and answer it with the field name and a dot before it.
 4. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, or any value a field holds.

# Returns

  - `path::Option{<:String}`: The dotted path of the first wrapper found, `""` when `est` is itself one, or `nothing`.

# Related

  - [`assert_batch_entry`](@ref)
  - [`online_entry_state`](@ref)
  - [`estimator_fields`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_wrapper_path(est::Union{<:AbstractEstimator,
                                        <:StatsBase.CovarianceEstimator})
    for f in estimator_fields(est)
        v = getfield(est, f)
        if isa(v, Online)
            return string(f)
        end
        path = online_wrapper_path(v)
        if !isnothing(path)
            return string(f, ".", path)
        end
    end
    return nothing
end
function online_wrapper_path(::Online)
    return ""
end
function online_wrapper_path(::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the window rule of a windowed moment estimator, the rule that says which observations its window keeps on the online step.

A windowed estimator, such as [`WindowedCovariance`](@ref), fits its inner estimator on a window of observations. In a batch fit every rule keeps the same window, the last `window` observations. The rules differ when the estimator receives its observations one block at a time.

# Interfaces

A rule is a marker for dispatch, and it holds no data. The windowed estimators hold it in their `rule` field, and the methods that fold them dispatch on its type. A new rule needs its own fold methods for the five windowed estimators.

# Related

  - [`RollingWindow`](@ref)
  - [`SeedWindow`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
abstract type AbstractWindowRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the last observations of the window at every fit, so the window rolls with each new observation. This is the default rule.

The estimator does not fold. On the online step, a prior that carries the observations, such as the carry of [`EmpiricalPrior`](@ref), refits it over the carried rows at each read, and the window of each refit ends at the last row. So the call with no data after any stream of blocks equals the batch fit over every row received.

# Constructors

    RollingWindow() -> RollingWindow

# Examples

```jldoctest
julia> RollingWindow()
RollingWindow()
```

# Related

  - [`AbstractWindowRule`](@ref)
  - [`SeedWindow`](@ref)
"""
struct RollingWindow <: AbstractWindowRule end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the last observations of the window at the first fit alone. After the first fit, the inner estimator folds every new observation, and no observation leaves the estimate.

In a batch fit the rule keeps the same window as [`RollingWindow`](@ref). On the online step the estimator folds. Its first fold cuts the first block to the window, folds the rows that remain into the inner estimator, and returns the estimator with `window = nothing`, so each later fold passes every row. The call with no data after a stream of blocks then gives the fit over the window of the first block, continued by the fold of every later row, and it equals no batch fit.

The rule needs an inner estimator that folds exactly, such as [`ExpWeightedCovariance`](@ref), and a prior that folds, such as the carry of [`EmpiricalPrior`](@ref). [`Online`](@ref) refuses it, because the wrapper refits its estimator at each step, and a refit has no first fit to remember.

# Constructors

    SeedWindow() -> SeedWindow

# Examples

```jldoctest
julia> SeedWindow()
SeedWindow()
```

# Related

  - [`AbstractWindowRule`](@ref)
  - [`RollingWindow`](@ref)
  - [`seed_window_path`](@ref)
  - [`partial_fit!`](@ref)
"""
struct SeedWindow <: AbstractWindowRule end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Choice Rule of an estimator, the rule that says how a choice that a fit makes over its whole sample behaves on the online step.

Some fits make a choice that reads every observation, such as the dropped member of a Factor Family in [`CrossSectionalFactorPrior`](@ref), the factor set of each asset in [`StepwiseRegression`](@ref), or the components of [`DimensionReductionRegression`](@ref). In a batch fit every rule makes the same choice. The rules differ when the estimator receives its observations one block at a time.

# Interfaces

A rule is a marker for dispatch, and it holds no data. An estimator that makes such a choice holds the rule in its `choice` field. The step of its online fold dispatches on the type of the rule. A new rule needs its own method of that step for each estimator that holds it.

# Related

  - [`BatchChoice`](@ref)
  - [`PinnedChoice`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
abstract type AbstractChoiceRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Makes the choice again over every observation at each fit. This is the default rule.

On the online step a refit reads every row of its buffer, so the choice can move from one step to the next, and the call with no data after any stream of blocks equals the batch fit over the same rows. On the carry fold of a [`CrossSectionalFactorPrior`](@ref), a step that moves the dropped member of a Factor Family solves no observation of full rank again. The raw factor returns of such an observation do not depend on the dropped member, so the step selects the columns that the new member keeps, and folds the factor prior again over them.

# Constructors

    BatchChoice() -> BatchChoice

# Examples

```jldoctest
julia> BatchChoice()
BatchChoice()
```

# Related

  - [`AbstractChoiceRule`](@ref)
  - [`PinnedChoice`](@ref)
"""
struct BatchChoice <: AbstractChoiceRule end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the choice of the first fit. After the first fit, every later fit reads the same choice.

In a batch fit the rule makes the same choice as [`BatchChoice`](@ref). An [`EntropyPoolingPrior`](@ref) and a [`MeucciEntropyPoolingPrior`](@ref) build their views on the fit under the prior probabilities, and refit under the posterior weights through [`scenario_factory`](@ref), which holds the regression of that fit, so each refit reads the choice of the fit that the views read under either rule. On the online step the rule holds no state. After the first step whose rows the fit accepts, the step writes the choice into the configuration of the estimator that it returns, for example the name of the dropped member of each Factor Family into the `families` of a [`CrossSectionalFactorPrior`](@ref), the factor set of each asset into `included` of a [`StepwiseRegression`](@ref), or the components into `proj` of a [`DimensionReductionRegression`](@ref). Each later step then reads that choice. The call with no data after a stream of blocks equals the batch fit over the same rows under the choice of the first fit, which differs from a batch fit when the choice of the whole sample moves.

# Constructors

    PinnedChoice() -> PinnedChoice

# Examples

```jldoctest
julia> PinnedChoice()
PinnedChoice()
```

# Related

  - [`AbstractChoiceRule`](@ref)
  - [`BatchChoice`](@ref)
"""
struct PinnedChoice <: AbstractChoiceRule end
"""
    seed_window_path(est)
    seed_window_path(v::AbstractVector)
    seed_window_path(p::Pair)
    seed_window_path(::Any)

Names the first field of an estimator tree that holds a windowed estimator under [`SeedWindow`](@ref), or answers `nothing`.

The constructor of [`Online`](@ref) uses this walk to refuse a seed window, because the wrapper refits its estimator at each step. The answer is the path from the root to the `rule` field, such as `"ce.rule"` for the covariance member of a prior. A windowed estimator under a seed window answers `"rule"` through its own method. A value that is not an estimator answers `nothing`.

The walk reads every field, not only the fields of [`estimator_fields`](@ref). A refit reaches the estimators that a vector holds, such as the factor list of a [`CrossSectionalFactorPrior`](@ref), the Descriptors of a [`DescriptorScores`](@ref) and the priors of an [`OpinionPoolingPrior`](@ref), so the refusal must reach them too (#1638). A vector answers with the index of its element in brackets, and a Pair answers through its value, so a Descriptor in a factor list gives a path such as `"factors[2].descriptors[1].ce.rule"`. [`element_path`](@ref) walks the vector.

# Algorithm

 1. Walk every field of `est` in order, and find the path in each field with this function.
 2. Answer the first path found, with the field name before it, as [`tree_path`](@ref) joins them.
 3. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, or any value a field holds.
  - `v`: A vector that a field holds.
  - `p`: A Pair that a vector holds, such as a member of a factor list.

# Returns

  - `path::Option{<:String}`: The path of the first seed window found, or `nothing`.

# Related

  - [`SeedWindow`](@ref)
  - [`Online`](@ref)
  - [`element_path`](@ref)
  - [`tree_path`](@ref)
  - [`estimator_fields`](@ref)
"""
function seed_window_path(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    for f in fieldnames(typeof(est))
        path = seed_window_path(getfield(est, f))
        if !isnothing(path)
            return tree_path(f, path)
        end
    end
    return nothing
end
function seed_window_path(::Any)
    return nothing
end
"""
    assert_batch_entry(est, entry::AbstractString)

Refuses an estimator that holds an [`Online`](@ref) at the entry of a batch fit, with an error that names the path of the wrapper.

A wrapper is a declaration that the online arm of the fold loop resolves at its warm-up. A batch fit runs no warm-up. A batch fit is a plain [`optimise`](@ref), or a fold of a scheme that is not an Online Scheme. So the wrapper would reach `prior(pe, X)` unresolved and get a `MethodError` that names the whole type. The refusal names the dotted path that [`online_wrapper_path`](@ref) finds, and the two exits. One exit is an Online Scheme, such as [`OnlineIndexWalkForward`](@ref). The other exit is the estimator without the wrapper. Each caller passes an optimiser, which is never a wrapper, so the path at this entry is never the empty path.

# Algorithm

 1. Find the path of the first wrapper in `est` with [`online_wrapper_path`](@ref).
 2. Refuse `est` when the path is not `nothing`, and name the path and the entry function.

# Arguments

  - `est`: The estimator handed to the entry function.
  - `entry`: The name of the entry function, as the message reads it.

# Validation

  - No field in the tree of `est` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`online_wrapper_path`](@ref)
  - [`assert_online_entry`](@ref)
  - [`optimise`](@ref)
  - [`fold_loop`](@ref)
"""
function assert_batch_entry(est, entry::AbstractString)
    path = online_wrapper_path(est)
    @argcheck(isnothing(path),
              ArgumentError("`$(typeof(est).name.name)` enters $(entry) holding an `Online` at `$(path)`, and a batch fit cannot resolve it: `Online` declares the sample buffer the fold loop's online arm seeds at its warm-up and folds the wrapped estimator's rows into, and nothing else seeds one, so the wrapper would reach the batch verb unresolved. Run the estimator through an Online Scheme — `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit` — or set `$(path)` to the estimator it wraps and let the batch fit refit it from its rows."))
    return nothing
end
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
$(DocStringExtensions.TYPEDSIGNATURES)

Joins the name of a field, or the bracketed index of an element, to the path that a walk found below it.

A path that starts with a bracket follows the name with no dot, so a walk names a member of a vector field as `"factors[2]"` and not as `"factors.[2]"`. Every other path follows the name after a dot.

# Arguments

  - `head`: The field name, or the bracketed index of an element, such as `"[2]"`.
  - `path`: The path below `head`.

# Returns

  - `path::String`: The joined path.

# Related

  - [`seed_window_path`](@ref)
  - [`online_entry_state`](@ref)
  - [`element_path`](@ref)
"""
function tree_path(head::Union{<:Symbol, <:AbstractString}, path::AbstractString)
    return startswith(path, '[') ? string(head, path) : string(head, ".", path)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Walks the elements of a vector in order with `walk`, and answers the path of the first element that gives one, or `nothing`.

The walks of an estimator tree that must see the estimators a vector holds call this function: [`seed_window_path`](@ref) and [`online_entry_state`](@ref). The answer starts with the index of the element in brackets, as [`tree_path`](@ref) joins it.

# Arguments

  - `walk`: The walk, a function that answers a path or `nothing` for one element.
  - `v`: The vector.

# Returns

  - `path::Option{<:String}`: The path of the first element that gives one, or `nothing`.

# Related

  - [`seed_window_path`](@ref)
  - [`online_entry_state`](@ref)
  - [`tree_path`](@ref)
"""
function element_path(walk, v::AbstractVector)
    i = findfirst(x -> !isnothing(walk(x)), v)
    return isnothing(i) ? nothing : tree_path(string("[", i, "]"), walk(v[i]))
end
function seed_window_path(v::AbstractVector)
    return element_path(seed_window_path, v)
end
function seed_window_path(p::Pair)
    return seed_window_path(last(p))
end
function online_entry_state(v::AbstractVector)
    return element_path(online_entry_state, v)
end
function online_entry_state(p::Pair)
    return online_entry_state(last(p))
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

export RollingWindow, SeedWindow, BatchChoice, PinnedChoice
public AbstractWindowRule, AbstractChoiceRule
public AbstractProgrammeAllocationSet, risk_constraint_solver, supports_partial_fit
