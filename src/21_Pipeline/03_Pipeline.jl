"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that a [`TrainTestSplit`](@ref) is the first step of a [`Pipeline`](@ref), and that no nested step contains one.

The split keeps the test window away from every fitted step. A stateful step before the split reads the rows of the test window. A [`MissingDataFilter`](@ref) selects the universe from them, and a [`PriceGapFill`](@ref) computes fill values from them, so the fitted state carries test data into the training workflow. At the first position, no step runs before the split. The check refuses a split inside a nested pipeline at any position, because [`holdout_window`](@ref) reads the steps of the outer pipeline only.

# Arguments

  - `ests`: The step estimators.

# Validation

  - A `TrainTestSplit` step is at index 1. Raises an `ArgumentError`.
  - No nested `Pipeline` and no [`PipelineStep`](@ref) contains a `TrainTestSplit`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`Pipeline`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`has_split`](@ref)
"""
function assert_split_position(ests)::Nothing
    for (i, e) in enumerate(ests)
        if isa(e, TrainTestSplit)
            @argcheck(i == 1,
                      ArgumentError("a TrainTestSplit step must be the first step of a Pipeline, but one appears at step $i; a stateful step fitted before the split would have seen the held-out test rows, leaking them into the fitted workflow"))
        elseif has_split(e)
            throw(ArgumentError("a TrainTestSplit step is nested inside a $(Base.typename(typeof(e)).wrapper) step of a Pipeline; the holdout must be the first step of the outermost Pipeline, where no step has yet touched the data"))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that an optimisation step of a [`Pipeline`](@ref) is the last step.

The optimiser writes the `:opt` slot, which is the output of the workflow. No step reads `:opt`. A step after the optimiser changes the context after the weights were computed, so the weights no longer describe the context. With the optimiser last, [`PIPELINE_INVALIDATES`](@ref) can omit `:opt` from the slots that a write makes stale. A pipeline with no optimiser, such as a prior-only pipeline, passes.

# Arguments

  - `ests`: The step estimators.

# Validation

  - Only the last step writes `:opt`. Raises an `ArgumentError`. A nested [`Pipeline`](@ref) writes the slot of its own last step, and its own constructor checks the steps inside it.

# Returns

  - `nothing`.

# Related

  - [`Pipeline`](@ref)
  - [`PIPELINE_INVALIDATES`](@ref)
  - [`pipe_writes`](@ref)
"""
function assert_opt_last(ests)::Nothing
    n = length(ests)
    for (i, e) in enumerate(ests)
        if i < n && pipe_writes(e) === :opt
            throw(ArgumentError("an optimisation step writes the terminal :opt slot, so it must be the last step of a Pipeline, but one appears at step $i of $n; move it to the end, or drop the steps that follow it"))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that each constraint step of a [`Pipeline`](@ref) resolves to one [routing target](@ref PIPELINE_ROUTING_TARGETS).

The check calls [`resolve_constraint_target`](@ref) on each constraint step. [`run_constraint_step`](@ref) makes the same call when the step runs, so three failures move from the fold loop to the constructor. The first is a family that computes nothing for the `constraints` slot. The second is a family that names several targets when the step names none. The third is a declared target that another family owns.

# Arguments

  - `ests`: The step estimators.

# Validation

  - The family of each constraint step declares at least one target, see [`pipe_constraint_targets`](@ref).
  - When the family declares several targets, the `target` of the [`PipelineStep`](@ref) names one of them.

# Returns

  - `nothing`.

# Related

  - [`resolve_constraint_target`](@ref)
  - [`pipe_constraint_targets`](@ref)
  - [`assert_routable`](@ref)
"""
function assert_constraint_targets(ests)::Nothing
    for e in ests
        est = isa(e, PipelineStep) ? e.est : e
        if !isa(est, AbstractConstraintEstimator) || !(pipe_writes(e) === :constraints)
            continue
        end
        resolve_constraint_target(est, isa(e, PipelineStep) ? e.target : nothing)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that each step of a [`Pipeline`](@ref) declares the slot and the target that the step writes when it runs.

[`run_step`](@ref) selects the method for a [`PipelineStep`](@ref) that wraps an estimator from the family of that estimator. So the family decides the slot that the step writes, and the `writes` field does not. An uncertainty-set step computes the halves that its `target` names, and it stops the fit when it has no such target. The constructor calls this check, so a wrong declaration stops the construction and not the first fold.

# Arguments

  - `ests`: The step estimators.

# Validation

  - An uncertainty-set estimator is wrapped in a `PipelineStep` whose `target` is `:mu`, `:sigma` or `:both`. Raises an `ArgumentError`.
  - A `PipelineStep` that wraps a preprocessing, prior, phylogeny, uncertainty-set, constraint or optimisation estimator declares the slot of that family in `writes`, see [`pipe_writes`](@ref). Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`PipelineStep`](@ref)
  - [`pipe_writes`](@ref)
  - [`run_uncertainty_step`](@ref)
  - [`assert_constraint_targets`](@ref)
"""
function assert_step_declarations(ests)::Nothing
    n = length(ests)
    families = Union{<:AbstractPricesPreprocessingEstimator,
                     <:AbstractReturnsPreprocessingEstimator, <:AbstractPriorEstimator,
                     <:AbstractPhylogenyEstimator, <:AbstractUncertaintySetEstimator,
                     <:AbstractConstraintEstimator, <:OptimisationEstimator}
    for (i, e) in enumerate(ests)
        est = isa(e, PipelineStep) ? e.est : e
        if isa(est, AbstractUncertaintySetEstimator)
            target = isa(e, PipelineStep) ? e.target : nothing
            @argcheck(target in (:mu, :sigma, :both),
                      ArgumentError("step $i of $n is a $(Base.typename(typeof(est)).wrapper) uncertainty-set step with target $(repr(target)); wrap it in a PipelineStep with target = :mu, :sigma or :both, which names the parameter the set bounds"))
        end
        if isa(e, PipelineStep) && isa(est, families)
            slot = pipe_writes(est)
            @argcheck(e.writes === slot,
                      ArgumentError("step $i of $n wraps a $(Base.typename(typeof(est)).wrapper) in a PipelineStep that declares writes = :$(e.writes), but the step writes the :$slot slot of its family when it runs; declare writes = :$slot"))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that a nested [`Pipeline`](@ref) step changes no Data Slot that the outer pipeline does not receive.

A nested pipeline gives one slot to the outer Pipeline Context, the slot of its last step. Assume that a step inside it rewrites `:prices` or `:returns`, and that its last step writes a different slot. Then the outer pipeline receives a result for assets that its own Data Slot does not hold. The weights and the prior then describe two different universes, and a prediction fails at [`assert_universe_aligned`](@ref).

# Arguments

  - `ests`: The step estimators.

# Validation

  - A nested `Pipeline` whose last step writes a slot outside [`PIPELINE_DATA_SLOTS`](@ref) contains no step that writes a Data Slot. Raises an `ArgumentError`. Put such a step in the outer pipeline.

# Returns

  - `nothing`.

# Related

  - [`Pipeline`](@ref)
  - [`PIPELINE_DATA_SLOTS`](@ref)
  - [`pipe_writes`](@ref)
"""
function assert_nested_data_slots(ests)::Nothing
    n = length(ests)
    for (i, e) in enumerate(ests)
        p = isa(e, PipelineStep) ? e.est : e
        if !isa(p, Pipeline) || pipe_writes(p) in PIPELINE_DATA_SLOTS
            continue
        end
        for s in p.steps
            slot = pipe_writes(s)
            @argcheck(!(slot in PIPELINE_DATA_SLOTS),
                      ArgumentError("step $i of $n is a nested Pipeline that writes the :$(pipe_writes(p)) slot, and one of its steps rewrites the :$slot slot, which the outer pipeline never receives; the :$(pipe_writes(p)) result would describe other assets than the outer :$slot slot. Move the $(Base.typename(typeof(s)).wrapper) step into the outer pipeline"))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [routing targets](@ref PIPELINE_ROUTING_TARGETS) of a step that are known before any step runs.

Two kinds of step have such targets. An uncertainty-set step declares the parameters that it bounds in the `target` field of its [`PipelineStep`](@ref). A constraint step takes its target from its family through [`pipe_constraint_targets`](@ref), or from the `target` field of its `PipelineStep` when the family names several. [`run_constraint_step`](@ref) resolves the target from the same declaration, so this function returns the target that the step writes.

Every other step returns an empty tuple. A callable step declares no family, and a precomputed result names its target only by its type.

# Arguments

  - `est`: A step estimator.

# Returns

  - `(:mu_ucs,)`, `(:sigma_ucs,)` or `(:mu_ucs, :sigma_ucs)` for a step that writes `:uncertainty` with the target `:mu`, `:sigma` or `:both`.
  - A tuple of one routing target for a constraint step.
  - `()` for every other step. This includes a callable step that writes `:uncertainty` with no such target.

# Related

  - [`assert_routable`](@ref)
  - [`pipe_constraint_targets`](@ref)
  - [`PIPELINE_ROUTING_TARGETS`](@ref)
"""
function pipe_required_targets(ps::PipelineStep)
    if pipe_writes(ps) === :uncertainty
        return if ps.target === :mu
            (:mu_ucs,)
        elseif ps.target === :sigma
            (:sigma_ucs,)
        elseif ps.target === :both
            (:mu_ucs, :sigma_ucs)
        else
            ()
        end
    end
    if isa(ps.est, AbstractConstraintEstimator) && pipe_writes(ps) === :constraints
        return (resolve_constraint_target(ps.est, ps.target),)
    end
    return ()
end
function pipe_required_targets(ce::AbstractConstraintEstimator)
    return (resolve_constraint_target(ce, nothing),)
end
pipe_required_targets(::Any) = ()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse at construction a pipeline whose last optimiser cannot receive a target that an earlier step writes.

Without this check, [`inject_context`](@ref) finds a target that it cannot route when the optimisation step runs. Under [`cross_val_predict`](@ref), that is after the fold loop fitted every earlier step of the first fold. The check asks the optimiser through [`pipe_accepts`](@ref), so it follows the fields that the optimiser has.

The check is structural. It finds whether the optimiser family can receive the target, and not whether this configuration accepts the value. A [`JuMPOptimiser`](@ref) always accepts `:mu_ucs`, but one with a return estimator other than [`ArithmeticReturn`](@ref) fails at injection. [`pipe_route`](@ref) owns that condition.

# Algorithm

 1. Stop when the pipeline has one step, or when its last step writes a slot other than `:opt`.
 2. Read the optimiser `opt` from the last step, without its [`PipelineStep`](@ref) wrapper. Stop when `opt` is not an [`OptimisationEstimator`](@ref). A [`TimeDependent`](@ref) schedule and a precomputed result are such steps, and the fold loop resolves them later.
 3. Read the targets of each earlier step with [`pipe_required_targets`](@ref).
 4. Check each target with [`pipe_accepts`](@ref).

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Validation

  - `pipe_accepts(opt, Val(target))` holds for each target of each earlier step. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`pipe_required_targets`](@ref)
  - [`pipe_accepts`](@ref)
  - [`Pipeline`](@ref)
"""
function assert_routable(ests)::Nothing
    n = length(ests)
    if !(n > 1)
        return nothing
    end
    terminal = ests[n]
    if !(pipe_writes(terminal) === :opt)
        return nothing
    end
    opt = isa(terminal, PipelineStep) ? terminal.est : terminal
    if !(isa(opt, OptimisationEstimator))
        return nothing
    end
    for i in 1:(n - 1)
        for target in pipe_required_targets(ests[i])
            @argcheck(pipe_accepts(opt, Val(target)),
                      ArgumentError("step $i of $n writes the :$target target, which the terminal $(Base.typename(typeof(opt)).wrapper) cannot receive; a computed value that reaches no optimiser field would be silently dropped, so change the step's target, drop the step, or use an optimiser that accepts it"))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the first element of `xs` that repeats an earlier element.

The error for a repeated step name shows this element, and not the whole collection. Only the failure path of the [`Pipeline`](@ref) constructor calls this function.

# Arguments

  - `xs`: A collection of names.

# Returns

  - The first element that repeats an earlier one, or `nothing` when every element is unique.

# Related

  - [`Pipeline`](@ref)
"""
function first_duplicate(xs)
    if isempty(xs)
        return nothing
    end
    #! `typeof(first(xs))`, not `eltype(xs)`: over an abstract `Tuple{Vararg{String}}` the
    #! latter is inferred as `Union{}`, and JET reports the set's lookup.
    seen = Set{typeof(first(xs))}()
    for x in xs
        if x in seen
            return x
        end
        push!(seen, x)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Runs an ordered list of steps as one end-to-end portfolio workflow.

The steps run from left to right over a [`PipelineContext`](@ref). A step is an ordinary estimator. It is a preprocessing, prior, phylogeny, uncertainty-set, constraint or optimisation estimator, a nested `Pipeline`, or a [`PipelineStep`](@ref) wrapper. [`pipe_writes`](@ref) and [`pipe_reads`](@ref) give the slots that the family of a step writes and reads. [`fit`](@ref) runs the steps in order, and the computed slots replace the configuration of the last optimiser, see [`inject_context`](@ref). When the pipeline has no step for a stage, the optimiser computes that value itself, so each stage is optional.

A last optimiser is optional too. A prior-only pipeline is valid, but a prediction needs weights.

# Algorithm

The keyword constructor runs these steps.

 1. Split each element of `steps` into its explicit name, or `nothing`, and its estimator, giving `explicit` and `ests`.
 2. Run on `ests` each check of `## Validation` that has a function of its own.
 3. Walk `ests` in order. Check each slot that a step reads against `avail`, which holds the Data Slots and the slots of the earlier steps. Check each slot that the write of the step makes stale against `written`, which holds the slots of the earlier steps.
 4. Count the steps that write each slot, giving `counts`. The count includes the steps with an explicit name.
 5. Name each step, giving `names`. A step keeps its explicit name. A step whose slot no other step writes takes the slot name, such as `"prior"`. Every other step takes the slot name and its position among the steps of that slot, such as `"prices_1"` and `"prices_2"`.
 6. Call the positional constructor, which checks that the names are unique.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Pipeline(; steps::Union{<:Tuple, <:AbstractVector},
               cache::Option{<:AbstractPartialFitState} = nothing) -> Pipeline

Give the steps in the order of execution. Each element is a step estimator or a `"name" => estimator` pair.

## Validation

  - `!isempty(steps)`. Raises an `IsEmptyError`.
  - [`pipe_writes`](@ref) is defined for each step. Raises an `ArgumentError`.
  - A [`TrainTestSplit`](@ref) is the first step only, see [`assert_split_position`](@ref).
  - An optimisation step is the last step only, see [`assert_opt_last`](@ref). It writes the `:opt` slot, and no step runs after it.
  - Each step declares the slot and the target that it writes, see [`assert_step_declarations`](@ref).
  - A nested pipeline changes no Data Slot that the outer pipeline does not receive, see [`assert_nested_data_slots`](@ref).
  - Each constraint step resolves to one routing target, see [`assert_constraint_targets`](@ref).
  - The last optimiser can receive each target that an earlier step writes, see [`assert_routable`](@ref).
  - An earlier step writes each slot that a step reads, or the pipeline input fills it with `prices` or `returns`. Raises an `ArgumentError`.
  - No step writes a slot that makes stale a slot that an earlier step wrote, see [`PIPELINE_INVALIDATES`](@ref). A step that rewrites `:returns` after a prior, phylogeny, uncertainty or constraint step leaves that result on an old asset universe. Raises an `ArgumentError`.
  - The step names are unique. Raises an `ArgumentError`.

# Online form

A Pipeline takes the online step. [`partial_fit!`](@ref) runs the steps in order, and folds each block of observations through them into the **row owner**. The row owner is the prior step, or the optimiser step when the pipeline has no prior step. `fit(pipe)` with no data returns the fitted [`PipelineResult`](@ref). Each step before the row owner is one of three kinds.

  - A row-local step folds its rows and emits the transformed rows. [`PricesToReturns`](@ref), [`PriceGapFill`](@ref) with a [`CarriedPrice`](@ref), and [`MissingDataFilter`](@ref) at `row_thr = 1` are row-local.
  - A universe-only step folds nothing. An [`AbstractAssetSelector`](@ref) and the column filter of a `MissingDataFilter` are universe-only. `fit(pipe)` with no data fits such a step again over the rows of the row owner, and applies its universe as a view.
  - A window-valued step, and every other step that writes a Data Slot, stops the warm-up with an error that names it. `Online(pipe)` is the declared refit that accepts such a step.

A cap on the row owner alone, `Online(pe; max_history = w)`, is a window that counts the rows of the row owner. A row-local step before it folds a state across the front edge of that window, so the warm-up refuses that pair with an error that names it. The rolling window through a Pipeline is `Online(pipe; max_history = w)`, which fits every step again over the window. `cache` holds the Fold Context of the Pipeline when a prior step owns the rows, or the buffer of the input data that `Online(pipe)` seeds. It is `nothing` until a step writes one. See [`partial_fit!(pipe::Pipeline{<:Any, <:Any, <:Option{<:Union{<:PipelineBufferState, <:ReturnsBufferState}}}, data::Prices_RR)`](@ref) and [`fit(pipe::Pipeline)`](@ref).

# Examples

```jldoctest
julia> pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()));

julia> pipe.names
("returns", "prior", "opt")
```

# Related

  - [`AbstractPipelineEstimator`](@ref)
  - [`PipelineResult`](@ref)
  - [`PipelineStep`](@ref)
  - [`fit`](@ref)
"""
@concrete struct Pipeline <: AbstractPipelineEstimator
    """
    The name of each step, in the order of `steps`.
    """
    names
    """
    The step estimators, in the order of execution.
    """
    steps
    """
    $(field_dict[:pfcache])
    """
    cache
    function Pipeline(names::Tuple{Vararg{String}}, steps::Tuple,
                      cache::Option{<:AbstractPartialFitState} = nothing)
        @argcheck(!isempty(steps), IsEmptyError("steps cannot be empty"))
        @argcheck(length(names) == length(steps), DimensionMismatch)
        @argcheck(allunique(names),
                  ArgumentError("pipeline step names must be unique; the name $(repr(first_duplicate(names))) is repeated among the $(length(names)) steps"))
        return new{typeof(names), typeof(steps), typeof(cache)}(names, steps, cache)
    end
end
function Pipeline(; steps::Union{<:Tuple, <:AbstractVector},
                  cache::Option{<:AbstractPartialFitState} = nothing)::Pipeline
    @argcheck(!isempty(steps), IsEmptyError("steps cannot be empty"))
    ests = Vector{Any}(undef, length(steps))
    explicit = Vector{Union{Nothing, String}}(undef, length(steps))
    for (i, s) in enumerate(steps)
        if isa(s, Pair)
            explicit[i] = String(s.first)
            ests[i] = s.second
        else
            explicit[i] = nothing
            ests[i] = s
        end
    end
    assert_split_position(ests)
    assert_opt_last(ests)
    assert_step_declarations(ests)
    assert_nested_data_slots(ests)
    assert_constraint_targets(ests)
    assert_routable(ests)
    slots = Symbol[pipe_writes(e) for e in ests]
    avail = Set{Symbol}(PIPELINE_DATA_SLOTS)
    written = Dict{Symbol, Any}()
    for (e, slot) in zip(ests, slots)
        for r in pipe_reads(e)
            @argcheck(r in avail,
                      ArgumentError("a $(Base.typename(typeof(e)).wrapper) step reads the :$r slot, which no earlier step writes and the pipeline input cannot fill"))
        end
        for inv in get(PIPELINE_INVALIDATES, slot, ())
            @argcheck(!haskey(written, inv),
                      ArgumentError("a $(Base.typename(typeof(e)).wrapper) step writes the :$slot slot, invalidating the :$inv slot written by an earlier $(Base.typename(typeof(written[inv])).wrapper) step; the stale :$inv result would no longer match the assets of the new :$slot data. Move the $(Base.typename(typeof(e)).wrapper) step before the $(Base.typename(typeof(written[inv])).wrapper) step, or drop one of them."))
        end
        written[slot] = e
        push!(avail, slot)
    end
    counts = Dict{Symbol, Int}()
    for s in slots
        counts[s] = get(counts, s, 0) + 1
    end
    seen = Dict{Symbol, Int}()
    names = Vector{String}(undef, length(ests))
    for i in eachindex(ests)
        s = slots[i]
        seen[s] = get(seen, s, 0) + 1
        names[i] = if !isnothing(explicit[i])
            explicit[i]
        elseif counts[s] == 1
            string(s)
        else
            string(s, '_', seen[s])
        end
    end
    return Pipeline(Tuple(names), Tuple(ests), cache)
end
pipe_writes(p::Pipeline) = pipe_writes(p.steps[end])
pipe_reads(p::Pipeline) = pipe_reads(p.steps[1])
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a step is a [`TrainTestSplit`](@ref), or contains one.

The search enters each nested [`Pipeline`](@ref) and each [`PipelineStep`](@ref). A split inside a nested pipeline sees data that an outer step already changed, and the first position of the split exists to prevent that. The same search tells whether a whole pipeline has a Holdout Split, which the cross-validation functions check before they run.

# Returns

  - `true` for a `TrainTestSplit`, and for a `Pipeline` or a `PipelineStep` that contains one. `false` for every other step.

# Related

  - [`assert_split_position`](@ref)
  - [`assert_no_holdout`](@ref)
  - [`TrainTestSplit`](@ref)
"""
has_split(::Any)::Bool = false
has_split(::TrainTestSplit)::Bool = true
has_split(p::Pipeline) = any(has_split, p.steps)
has_split(ps::PipelineStep) = has_split(ps.est)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a [`Pipeline`](@ref) that contains a [`TrainTestSplit`](@ref) in the cross-validation functions.

A Holdout Split and a cross-validation scheme are two methods of evaluation. The scheme sets the training and the test window of each fold. A split in the pipeline takes a second holdout from the training window of each fold, and keeps a test window that no function reads. So each fold loses training data without a message. One call uses one method of evaluation.

# Arguments

  - `pipe`: The pipeline.

# Validation

  - `!has_split(pipe)`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`TrainTestSplit`](@ref)
  - [`search_cross_validation`](@ref)
  - [`has_split`](@ref)
"""
function assert_no_holdout(pipe::Pipeline)::Nothing
    @argcheck(!has_split(pipe),
              ArgumentError("this Pipeline contains a TrainTestSplit step, so it cannot also be cross-validated: cross-validation already defines the train and test window of every fold, and the split would shave a second holdout off each fold's training data. Remove the TrainTestSplit step, or evaluate the pipeline with fit_predict instead of cross-validating it."))
    return nothing
end
"""
    port_opt_view(pipe::Pipeline, i, args...; kwargs...)

Refuse an asset view of a [`Pipeline`](@ref).

A meta-optimiser such as [`NestedClustered`](@ref), [`Stacking`](@ref) or [`SubsetResampling`](@ref) takes a `port_opt_view` of its inner estimator to build a portfolio over a subset of the assets. The asset universe of a pipeline is fitted state, because a missing-data filter selects it from the training window. So an asset view before the fit has no defined meaning. A meta-optimiser cannot wrap a `Pipeline`, but it can be the optimisation step of one.

# Validation

  - Each call raises an `ArgumentError`.

# Related

  - [`Pipeline`](@ref)
  - [`optimise(::Pipeline)`](@ref)
"""
function port_opt_view(::Pipeline, args...; kwargs...)
    return throw(ArgumentError("a Pipeline cannot be sub-selected with port_opt_view: its asset universe is fitted state, so wrapping a Pipeline inside a meta-optimiser is unsupported. A meta-optimiser may be used as the optimisation step of a Pipeline instead."))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the fitted result of each step of a pipeline, and the last Pipeline Context.

The context is a [`PipelineContext`](@ref). Its slots hold the computed data, prior, phylogeny, uncertainty sets, constraints and optimisation result of the [`Pipeline`](@ref).

Read the result of a step by its name with `getindex`, as in `res["prior"]`, or by its position through the `results` field, as in `res.results[2]`. A name that no step has raises an `ArgumentError` that suggests the nearest name. An integer index keeps the length-1 container behaviour of the package, so `res[1] === res`. The `w` property returns the weights of the optimisation result, `res.ctx.opt.w`. It raises a [`PropertyPathError`](@ref) when the pipeline has no optimisation result.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Pipeline`](@ref)
  - [`AbstractPipelineResult`](@ref)
  - [`fit`](@ref)
"""
@concrete struct PipelineResult <: AbstractPipelineResult
    """
    The name of each step, in the order of `results`.
    """
    names
    """
    The fitted result of each step, in step order.
    """
    results
    """
    The last [`PipelineContext`](@ref).
    """
    ctx
end
@forward_properties PipelineResult begin
    compute(w, ctx.opt.w; broadcast)
end
function Base.getindex(pr::PipelineResult, name::AbstractString)
    names = getfield(pr, :names)
    i = findfirst(==(name), names)
    @argcheck(!isnothing(i),
              ArgumentError("no pipeline step named $(repr(name)) among the $(length(names)) named steps" *
                            did_you_mean(name, names)))
    return getfield(pr, :results)[i]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the elements of the `constraints` slot as one iterable.

# Arguments

  - `x`: `nothing`, one [`AbstractConstraintResult`](@ref), or a vector of them.

# Returns

  - An iterable of constraint results. It is empty for `nothing`, and holds one element for one result.

# Related

  - [`constraint_targets`](@ref)
"""
constraint_results(::Nothing) = ()
constraint_results(c::AbstractConstraintResult) = (c,)
constraint_results(c::AbstractVector{<:AbstractConstraintResult}) = c
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [routing target](@ref PIPELINE_ROUTING_TARGETS) that a constraint result names by its type alone, or `nothing`.

Four result types name one optimiser field each. A [`WeightBounds`](@ref) goes to `:wb`, a [`LinearConstraint`](@ref) to `:lcse`, a phylogeny constraint result to `:ple`, and a [`RiskBudget`](@ref) to `:rkb`. Every other type returns `nothing`, and needs a target beside it, see [`TargetedConstraint`](@ref).

This function is the only part of the fan-out that the type decides. It also decides whether the value of a step needs a wrapper. [`add_constraint_result`](@ref) wraps a value only when the value cannot name its own target, so the `constraints` slot holds a bare result where it can.

# Arguments

  - `c`: A constraint result.

# Returns

  - `target::Union{Nothing, Symbol}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref), or `nothing`.

# Related

  - [`constraint_target_of`](@ref)
  - [`TargetedConstraint`](@ref)
"""
implicit_constraint_target(::WeightBounds)::Symbol = :wb
implicit_constraint_target(::LinearConstraint)::Symbol = :lcse
implicit_constraint_target(::AbstractPhylogenyConstraintResult)::Symbol = :ple
implicit_constraint_target(::RiskBudget)::Symbol = :rkb
implicit_constraint_target(::Any) = nothing
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [routing target](@ref PIPELINE_ROUTING_TARGETS) that one element of the `constraints` slot goes to.

[`run_constraint_step`](@ref) pairs a value that its type cannot place with its target, and this function reads that target. [`implicit_constraint_target`](@ref) places every other element.

# Arguments

  - `c`: One element of the `constraints` slot.

# Validation

  - A [`Threshold`](@ref) without a target raises an `ArgumentError`. It names six optimiser fields, so its type cannot place it. The error names the declaration that places it.
  - A result of any other type that names no target raises an `ArgumentError`. The function refuses it here, before it reaches an optimiser.

# Returns

  - `target::Symbol`: One of [`PIPELINE_ROUTING_TARGETS`](@ref).

# Related

  - [`constraint_targets`](@ref)
  - [`implicit_constraint_target`](@ref)
  - [`constraint_value_of`](@ref)
  - [`TargetedConstraint`](@ref)
"""
function constraint_target_of(c)::Symbol
    target = implicit_constraint_target(c)
    if !isnothing(target)
        return target
    end
    return throw(ArgumentError("cannot route a $(Base.typename(typeof(c)).wrapper) constraint result into any optimiser; supported: WeightBounds, LinearConstraint, RiskBudget, phylogeny constraint results, and any value a constraint step paired with a routing target"))
end
function constraint_target_of(c::TargetedConstraint)::Symbol
    return c.target
end
function constraint_target_of(::Threshold)::Symbol
    return throw(ArgumentError("cannot route a Threshold constraint result on its own: it names $(length(PIPELINE_THRESHOLD_TARGETS)) optimiser fields, $PIPELINE_THRESHOLD_TARGETS, and the result does not say which is meant. Wrap the step in a PipelineStep with target = one of them, or pass the Threshold to the optimiser field directly."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the value that one element of the `constraints` slot delivers, without its routing wrapper.

# Arguments

  - `c`: One element of the `constraints` slot.

# Returns

  - The value to route.

# Related

  - [`constraint_target_of`](@ref)
  - [`TargetedConstraint`](@ref)
"""
constraint_value_of(c) = c
constraint_value_of(c::TargetedConstraint) = c.res
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Combine the values that reached one [accumulating](@ref PIPELINE_ACCUMULATING_TARGETS) routing target.

The default puts the values in a vector, in write order. Each field that holds one result per estimator reads this shape.

`:cte` is different. Its field takes a vector of [`CentralityConstraint`](@ref) estimators, and [`centrality_constraints`](@ref) appends every row of every estimator into one [`LinearConstraint`](@ref). So the values of separate steps merge through [`merge_linear_constraints`](@ref). Then *n* centrality steps in a [`Pipeline`](@ref) give the optimiser the value that one `cte` field with *n* estimators gives.

[`constraint_targets`](@ref) calls this function with more than one value only, because it unwraps a single value first.

# Arguments

  - `::Val{target}`: The routing target that the values reached.
  - `vals`: The values, in write order.

# Validation

  - For `:cte`, each value is a `LinearConstraint`. Raises an `ArgumentError`.

# Returns

  - The combined value.

# Related

  - [`constraint_targets`](@ref)
  - [`PIPELINE_ACCUMULATING_TARGETS`](@ref)
  - [`merge_linear_constraints`](@ref)
"""
function accumulate_constraint_values(::Val, vals)
    return identity.(vals)
end
function accumulate_constraint_values(::Val{:cte}, vals)
    @argcheck(all(v -> isa(v, LinearConstraint), vals),
              ArgumentError("every value routed to :cte must be a LinearConstraint so that separate steps can be merged into the single constraint the field holds, got $(unique(Base.typename(typeof(v)).wrapper for v in vals))"))
    return merge_linear_constraints(identity.(vals))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Split the `constraints` slot into [routing targets](@ref PIPELINE_ROUTING_TARGETS).

# Algorithm

 1. For each element `c` of the `constraints` slot, in write order, read its target with [`constraint_target_of`](@ref) and its value with [`constraint_value_of`](@ref).
 2. Add the value to the group of its target in `out`. A target seen for the first time starts a new group.
 3. Return the value itself for a group of one value. This is the scalar shape that the fields accept outside a pipeline. For a larger group, combine the values with [`accumulate_constraint_values`](@ref). It puts them in a vector in write order, or merges them into one constraint for `:cte`.

# Arguments

  - `cs`: The `constraints` slot.

# Validation

  - A second value for a target outside [`PIPELINE_ACCUMULATING_TARGETS`](@ref) raises an `ArgumentError`. Such a field holds one value, and the second value replaces the first without a message.

# Returns

  - A vector of `target => value` pairs, in the order of the first write to each target.

# Related

  - [`inject_context`](@ref)
  - [`constraint_results`](@ref)
  - [`constraint_target_of`](@ref)
  - [`accumulate_constraint_values`](@ref)
  - [`PIPELINE_ACCUMULATING_TARGETS`](@ref)
"""
function constraint_targets(cs)
    out = Pair{Symbol, Any}[]
    for c in constraint_results(cs)
        target = constraint_target_of(c)
        val = constraint_value_of(c)
        i = findfirst(p -> p.first === target, out)
        if isnothing(i)
            push!(out, target => Any[val])
            continue
        end
        @argcheck(target in PIPELINE_ACCUMULATING_TARGETS,
                  ArgumentError("two constraint steps write the :$target routing target, which holds one value; the second would silently replace the first. Drop one of the steps, or combine them into the single value the field expects."))
        push!(out[i].second, val)
    end
    return Pair{Symbol, Any}[p.first => (if length(p.second) == 1
                                             p.second[1]
                                         else
                                             accumulate_constraint_values(Val(p.first), p.second)
                                         end) for p in out]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace the configuration of an optimisation step with the computed slots of the Pipeline Context, immediately before the step runs.

This function is the pipeline half of the routing. It turns the slots into a flat sequence of [routing targets](@ref PIPELINE_ROUTING_TARGETS). To do this, it finds which halves of the uncertainty pair are present, which result types the `constraints` slot holds, and how many of each. Then it gives each target to [`pipe_route`](@ref), which finds the field that receives the target. The optimiser owns that choice, so a field rename is a local change.

[`unroutable_target`](@ref) handles a target that an optimiser has no field for. `:pe` and `:cle` pass without a change, and every other target raises an error. So a naive optimiser or a meta-optimiser accepts a computed prior that it can use, and refuses an uncertainty set that it cannot use.

# Algorithm

 1. Return `opt` unchanged when the `prior`, `phylogeny`, `uncertainty` and `constraints` slots are all `nothing`.
 2. Route the `prior` slot to `:pe`.
 3. Route the `phylogeny` slot to `:cle` when it is an [`AbstractClusteringResult`](@ref). A phylogeny result of another type has no `:cle` target, so this step skips it.
 4. Route the mean half of the `uncertainty` slot to `:mu_ucs`, and its covariance half to `:sigma_ucs`, when the half is present.
 5. Route each `target => value` pair of [`constraint_targets`](@ref) to its target.
 6. Return the rebuilt optimiser `opt′`.

# Arguments

  - `opt`: The optimisation step estimator.
  - `ctx`: The pipeline context.

# Returns

  - `opt′`: The estimator that runs. It is `opt` itself when no slot applies, and a rebuilt copy otherwise.

# Related

  - [`pipe_route`](@ref)
  - [`PIPELINE_ROUTING_TARGETS`](@ref)
  - [`constraint_targets`](@ref)
  - [`fit`](@ref)
"""
function inject_context(opt::OptimisationEstimator, ctx::PipelineContext)
    if isnothing(ctx.prior) &&
       isnothing(ctx.phylogeny) &&
       isnothing(ctx.uncertainty) &&
       isnothing(ctx.constraints)
        return opt
    end
    if !isnothing(ctx.prior)
        opt = pipe_route(opt, Val(:pe), ctx.prior)
    end
    #! A phylogeny result that is not a clustering structure has no :cle target; it reaches
    #! the optimiser as constraint results instead, so there is nothing to route here.
    if isa(ctx.phylogeny, AbstractClusteringResult)
        opt = pipe_route(opt, Val(:cle), ctx.phylogeny)
    end
    unc = ctx.uncertainty
    if !isnothing(unc)
        if !isnothing(unc.mu)
            opt = pipe_route(opt, Val(:mu_ucs), unc.mu)
        end
        if !isnothing(unc.sigma)
            opt = pipe_route(opt, Val(:sigma_ucs), unc.sigma)
        end
    end
    for (target, v) in constraint_targets(ctx.constraints)
        opt = pipe_route(opt, Val(target), v)
    end
    return opt
end
"""
    maybe_inject_step(est, ::PipelineContext) = est
    maybe_inject_step(opt::OptimisationEstimator, ctx::PipelineContext)
    maybe_inject_step(ps::PipelineStep, ctx::PipelineContext)

Return the step that runs, with the Pipeline Context injected when the step is an optimiser.

A step that is not an optimiser runs unchanged. An optimiser takes the context through [`inject_context`](@ref). For a [`PipelineStep`](@ref) that wraps an optimiser, the function builds a new `PipelineStep` around the injected optimiser, with the same `reads`, `writes` and `target`.

# Arguments

  - `est`: A step estimator.
  - `opt`: An optimisation step estimator.
  - `ps`: A [`PipelineStep`](@ref). A `PipelineStep` that wraps no optimiser runs unchanged.
  - `ctx`: The pipeline context.

# Returns

  - `est`: The step estimator, unchanged.
  - `opt′`: The optimiser with the configuration that the context replaced.
  - `ps′`: The pipeline step with the injected optimiser.
"""
maybe_inject_step(est, ::PipelineContext) = est
function maybe_inject_step(opt::OptimisationEstimator, ctx::PipelineContext)
    return inject_context(opt, ctx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply the injection rules to a precomputed optimisation result in the optimisation step.

The predict-only fold of a mixed [`TimeDependent`](@ref) schedule puts a result in the optimisation step. A result is solved, so it has no configuration to replace. It follows the rule of [`inject_context`](@ref) for an optimiser without a field for the target. The computed `prior` and `phylogeny` slots pass, because the result has its own. A computed uncertainty set or constraint must not disappear without a solve, so a populated `uncertainty` or `constraints` slot stops the fit.

# Arguments

  - `res`: The precomputed optimisation result.
  - `ctx`: The pipeline context.

# Validation

  - `isnothing(ctx.uncertainty)`. Raises an `ArgumentError`.
  - `isnothing(ctx.constraints)`. Raises an `ArgumentError`.

# Returns

  - `res`, unchanged.

# Related

  - [`inject_context`](@ref)
  - [`run_step`](@ref)
"""
function maybe_inject_step(res::NonFiniteAllocationOptimisationResult, ctx::PipelineContext)
    @argcheck(isnothing(ctx.uncertainty),
              ArgumentError("cannot route uncertainty sets into a $(Base.typename(typeof(res)).wrapper): a precomputed optimisation result is already solved, so a computed uncertainty set would be silently dropped"))
    @argcheck(isnothing(ctx.constraints),
              ArgumentError("cannot route constraint results into a $(Base.typename(typeof(res)).wrapper): a precomputed optimisation result is already solved, so computed constraints would be silently dropped"))
    return res
end
function maybe_inject_step(ps::PipelineStep, ctx::PipelineContext)
    if isa(ps.est, OptimisationEstimator)
        return PipelineStep(; est = inject_context(ps.est, ctx), reads = ps.reads,
                            writes = ps.writes, target = ps.target)
    end
    return ps
end
"""
    StatsAPI.fit(pipe::Pipeline, data::Prices_RR) -> PipelineResult

Fit a [`Pipeline`](@ref) on price data or returns data.

`fit` has no folds, so each [`TimeDependent`](@ref) schedule step resolves to its explicit `default`, see [`reset_time_dependent_estimator`](@ref). A schedule with no `default` raises a [`TimeDependentDefaultError`](@ref). Backtest such a pipeline with [`cross_val_predict`](@ref), whose folds resolve the schedule. Inside a fold loop the reset changes nothing, because the loop first replaces each schedule with its value for the fold.

# Algorithm

 1. When the pipeline has a schedule step, reset each schedule to its `default`.
 2. Fill the slot of the input type, giving `ctx`. A [`PricesResult`](@ref) fills `prices`, and a [`ReturnsResult`](@ref) fills `returns`. So a pipeline for returns data has no price steps, because a price step raises an `IsNothingError` when the `prices` slot is empty.
 3. For each step in order, inject `ctx` into an optimisation step with [`maybe_inject_step`](@ref). Then run the step with [`run_step`](@ref), giving its fitted result and the new `ctx`.
 4. Return the step names, the fitted results in step order and the last `ctx` as a [`PipelineResult`](@ref).

# Arguments

  - `pipe`: The pipeline.
  - `data`: The input data, a [`PricesResult`](@ref) or a [`ReturnsResult`](@ref).

# Returns

  - `res::PipelineResult`: The named fitted result of each step, and the last context.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 4),
                     [100.0 101.0; 102.0 103.0; 101.0 104.0; 103.0 102.0], [\"A\", \"B\"]);

julia> pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()));

julia> res = fit(pipe, PricesResult(; X = X));

julia> res.w
2-element Vector{Float64}:
 0.5
 0.5
```

# Related

  - [`Pipeline`](@ref)
  - [`PipelineResult`](@ref)
  - [`run_step`](@ref)
  - [`inject_context`](@ref)
"""
function StatsAPI.fit(pipe::Pipeline, data::Prices_RR)::PipelineResult
    if is_time_dependent(pipe)
        pipe = reset_time_dependent_estimator(pipe)
    end
    ctx = if isa(data, AbstractPricesResult)
        PipelineContext(; prices = data)
    else
        PipelineContext(; returns = data)
    end
    fitted = Vector{Any}(undef, length(pipe.steps))
    for (i, est) in enumerate(pipe.steps)
        step = maybe_inject_step(est, ctx)
        fitted[i], ctx = run_step(step, ctx)
    end
    return PipelineResult(pipe.names, Tuple(fitted), ctx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that a test window and the training window come from the same ingestion.

The last weights use the asset axis of the training window. A test window with a different asset set, or with the same set from a different universe statement, misaligns the weights and the returns. This check reports that case with an error that names both universes. Without the check, the case shows as a dimension mismatch inside the risk calculation.

The check reads two things. Equality of `nx` checks the asset axis. [`check_asset_panel`](@ref) binds the asset axis of an [`AssetPanel`](@ref) to the `nx` of the result that holds it, so equality of `nx` also compares the axes of the panels. Parity of panel presence checks where the universe was stated. A window with a panel and a fitted context without one, or the reverse, do not come from one ingestion.

Two other parts of the library hold the rest of the alignment. The ingestion layer fixes the asset axis before the split, and [`port_opt_view`](@ref) slices it. So each window of each fold carries every asset, and no policy of the layer drops a row or a column. An asset that is present in both windows and not investable in one is a Held Gap, which `filter_held_gaps` reads from the weights and the returns under the strictness policy.

Two cases remain, and the error message names both. The first is returns data built outside the ingestion layer. The second is a step of the caller that changes the asset set.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).
  - `rd`: The returns of the test window, after the fitted steps.

# Validation

  - When the fitted context has no `returns` slot, the check passes.
  - `rd.nx == train.nx`, where `train` is the `returns` slot of the fitted context. Raises an `ArgumentError`.
  - Both `rd` and `train` carry an [`AssetPanel`](@ref), or neither does. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
  - [`PriceIngestion`](@ref)
  - [`price_ingestion`](@ref)
  - [`AssetPanel`](@ref)
"""
function assert_universe_aligned(res::PipelineResult, rd::AbstractReturnsResult)::Nothing
    train = res.ctx.returns
    if isnothing(train)
        return nothing
    end
    @argcheck(rd.nx == train.nx,
              ArgumentError("the pipeline's fitted steps produced a test-window universe $(rd.nx) that differs from the training universe $(train.nx), so the weights and the test returns would not be aligned. The ingestion layer fixes the asset axis before the split, so this reaches two situations only: returns data built outside it, and a step of your own that changes the asset set. Build the returns data with price_ingestion(PriceIngestion(), X)."))
    @argcheck(isnothing(rd.pnl) == isnothing(train.pnl),
              ArgumentError("the pipeline's fitted steps produced a test window that $(isnothing(rd.pnl) ? "states no universe" : "states a universe") while the training window $(isnothing(train.pnl) ? "states none" : "states one"), so the two did not come from one ingestion. A ReturnsResult the ingestion layer built always carries an Asset Panel, so pnl === nothing on one of them says that one was built outside it."))
    return nothing
end
"""
    apply_fitted_step(fitted, data) -> data′

Apply one fitted step of a pipeline to a data window for a prediction.

A preprocessing step changes the window at its own data level. A price-level fitted object, an [`AbstractPricesPreprocessingResult`](@ref) or an [`AbstractPricesPreprocessingEstimator`](@ref), changes a price window. A returns-level fitted object changes a returns window. [`PricesToReturns`](@ref) converts a price window to returns.

A fitted object of the other data level returns the window unchanged. So a pipeline that was fitted on prices can predict on a returns window, and [`assert_universe_aligned`](@ref) then checks the asset axis of that window. The result of every other step, such as a prior, a phylogeny, an uncertainty set, a constraint or an optimisation result, returns the window unchanged. A nested [`PipelineResult`](@ref) applies its own steps with [`apply_fitted_steps`](@ref).

# Arguments

  - `fitted`: The fitted result of one step, from a [`PipelineResult`](@ref).
  - `data`: The data window, an [`AbstractPricesResult`](@ref) or an [`AbstractReturnsResult`](@ref).

# Returns

  - `data′`: The changed data window, or `data` itself.

# Related

  - [`apply_fitted_steps`](@ref)
  - [`apply_preprocessing`](@ref)
  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
"""
function apply_fitted_step(::Any, data::Prices_RR)
    return data
end
function apply_fitted_step(f::PricesToReturns, pr::AbstractPricesResult)
    return apply_preprocessing(f, pr)
end
apply_fitted_step(::PricesToReturns, rd::AbstractReturnsResult) = rd
function apply_fitted_step(f::Union{<:AbstractPricesPreprocessingResult,
                                    <:AbstractPricesPreprocessingEstimator},
                           pr::AbstractPricesResult)
    return apply_preprocessing(f, pr)
end
function apply_fitted_step(::Union{<:AbstractPricesPreprocessingResult,
                                   <:AbstractPricesPreprocessingEstimator},
                           rd::AbstractReturnsResult)
    return rd
end
function apply_fitted_step(f::Union{<:AbstractReturnsPreprocessingResult,
                                    <:AbstractReturnsPreprocessingEstimator},
                           rd::AbstractReturnsResult)
    return apply_preprocessing(f, rd)
end
function apply_fitted_step(::Union{<:AbstractReturnsPreprocessingResult,
                                   <:AbstractReturnsPreprocessingEstimator},
                           pr::AbstractPricesResult)
    return pr
end
function apply_fitted_step(f::PipelineResult, data::Prices_RR)
    return apply_fitted_steps(f.results, data)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply the fitted steps of a pipeline to a data window, in step order.

# Algorithm

 1. For each fitted result `f` in `results`, in step order, replace `data` with `apply_fitted_step(f, data)`, see [`apply_fitted_step`](@ref).
 2. Return the last `data`.

# Arguments

  - `results`: The fitted result of each step, from a [`PipelineResult`](@ref).
  - `data`: The data window.

# Returns

  - `data′`: The changed data window. It is returns data when the steps include a [`PricesToReturns`](@ref) step.

# Related

  - [`apply_fitted_step`](@ref)
  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
"""
function apply_fitted_steps(results::Tuple, data::Prices_RR)
    for f in results
        data = apply_fitted_step(f, data)
    end
    return data
end
"""
    predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idx = Colon(), cols = Colon()) -> PredictionResult

    predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idxs::VecVecInt, cols = Colon()) -> Vector{<:PredictionResult}

    predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idx = Colon(), cols = Colon()) -> PredictionResult

    predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idxs::VecVecInt, cols = Colon()) -> Vector{<:PredictionResult}

Apply a fitted pipeline to a new data window, and return the [`PredictionResult`](@ref) that the weight-level functions read.

`test_idx` selects the rows of the window, and `cols` selects its asset columns. The fitted preprocessing steps change the window in step order. They use the universe, the imputation parameters and the returns conversion of the training window, so no statistic of the test window changes the window. The weight-level `predict` then reads the window, so the scorers and the risk measures apply without a change.

A vector of index vectors predicts on each window and returns one result per window. The cross-validation functions read this shape.

# Algorithm

 1. Read the optimisation result `opt` of the pipeline.
 2. Take the rows `test_idx` and the columns `cols` of `data` with [`port_opt_view`](@ref), giving the window. Returns data with `:` for both is the window as it is.
 3. Apply the fitted steps to the window with [`apply_fitted_steps`](@ref), giving `rd`. For price data, check that `rd` is returns data.
 4. Check the asset axis of `rd` with [`assert_universe_aligned`](@ref).
 5. Return `predict(opt, rd)`, with the keyword arguments.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).
  - `data`: Price data or returns data that contains the window, a [`PricesResult`](@ref) or a [`ReturnsResult`](@ref).
  - `test_idx`: The rows of `data` in the window, as integer indices or `:` for all rows. Price data also takes a vector of timestamps.
  - `test_idxs`: Several such windows, as a vector of index vectors.
  - `cols`: The columns of `data` in the window, as integer indices or `:` for all assets. The columns must give the training universe. So a subset of the columns needs a fit on the same subset, as `fit_and_predict(pipe, data; cols)` does.

# Keyword Arguments

  - `wd`, `hwd`, `fa`, `store_weight_path`, `strict`, `w_prev`: The keywords of [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), passed on.

# Validation

  - `!isnothing(res.ctx.opt)`. Raises an `IsNothingError`.
  - For price data, the fitted steps convert the window to returns, so the pipeline contains a [`PricesToReturns`](@ref) step. Raises an `ArgumentError`.
  - The window passes [`assert_universe_aligned`](@ref).

# Returns

  - `pred::PredictionResult`: The weight-level prediction on the changed window, or one such result per window when several are given.

# Related

  - [`PipelineResult`](@ref)
  - [`apply_fitted_steps`](@ref)
  - [`port_opt_view`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
"""
function StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idx = Colon(), cols = Colon();
                          wd::Option{<:AbstractWeightDrift} = nothing,
                          hwd::Option{<:AbstractWeightDrift} = wd,
                          fa::Option{<:AbstractFeeAmortisation} = nothing,
                          store_weight_path::Bool = false, strict::Bool = false,
                          w_prev::Option{<:VecNum_VecVecNum} = nothing)
    opt = res.ctx.opt
    @argcheck(!isnothing(opt),
              IsNothingError("the pipeline produced no optimisation result; add a terminal optimisation step before predicting"))
    pr = port_opt_view(data, test_idx, cols)
    rd = apply_fitted_steps(res.results, pr)
    @argcheck(isa(rd, AbstractReturnsResult),
              ArgumentError("the pipeline's fitted steps do not convert price-level data to returns; predicting on a $(Base.typename(typeof(data)).wrapper) requires a PricesToReturns step"))
    assert_universe_aligned(res, rd)
    return StatsAPI.predict(opt, rd; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict,
                            w_prev = w_prev)
end
function StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idxs::VecVecInt, cols = Colon(); kwargs...)
    return [StatsAPI.predict(res, data, test_idx, cols; kwargs...)
            for test_idx in test_idxs]
end
function StatsAPI.predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idx = Colon(), cols = Colon();
                          wd::Option{<:AbstractWeightDrift} = nothing,
                          hwd::Option{<:AbstractWeightDrift} = wd,
                          fa::Option{<:AbstractFeeAmortisation} = nothing,
                          store_weight_path::Bool = false, strict::Bool = false,
                          w_prev::Option{<:VecNum_VecVecNum} = nothing)
    opt = res.ctx.opt
    @argcheck(!isnothing(opt),
              IsNothingError("the pipeline produced no optimisation result; add a terminal optimisation step before predicting"))
    rd = if isa(test_idx, Colon) && isa(cols, Colon)
        data
    else
        port_opt_view(data, test_idx, cols)
    end
    rd = apply_fitted_steps(res.results, rd)
    assert_universe_aligned(res, rd)
    return StatsAPI.predict(opt, rd; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict,
                            w_prev = w_prev)
end
function StatsAPI.predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idxs::VecVecInt, cols = Colon(); kwargs...)
    return [StatsAPI.predict(res, data, test_idx, cols; kwargs...)
            for test_idx in test_idxs]
end
function fit_and_predict(res::PipelineResult, data::Prices_RR; test_idx::VecInt_VecVecInt,
                         cols = :, wd::Option{<:AbstractWeightDrift} = nothing,
                         hwd::Option{<:AbstractWeightDrift} = wd,
                         fa::Option{<:AbstractFeeAmortisation} = nothing,
                         store_weight_path::Bool = false, strict::Bool = false,
                         w_prev::Option{<:VecNum_VecVecNum} = nothing, kwargs...)
    return StatsAPI.predict(res, data, test_idx, cols; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict,
                            w_prev = w_prev)
end
function fit_and_predict(pipe::Pipeline, data::Prices_RR;
                         train_idx::Option{<:VecInt} = nothing, test_idx::VecInt_VecVecInt,
                         cols = :, wd::Option{<:AbstractWeightDrift} = nothing,
                         hwd::Option{<:AbstractWeightDrift} = wd,
                         fa::Option{<:AbstractFeeAmortisation} = nothing,
                         store_weight_path::Bool = false, strict::Bool = false,
                         w_prev::Option{<:VecNum_VecVecNum} = nothing)
    res = pipeline_fold_fit(pipe, data, train_idx, cols)
    return StatsAPI.predict(res, data, test_idx, cols; wd = wd, hwd = hwd, fa = fa,
                            store_weight_path = store_weight_path, strict = strict,
                            w_prev = w_prev)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the test window that the [`TrainTestSplit`](@ref) step of a fitted pipeline keeps, or `nothing` when the pipeline has no split.

The function reads the steps of the outer pipeline only. [`assert_split_position`](@ref) keeps a split there.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).

# Returns

  - The `test` window of the [`TrainTestSplitResult`](@ref), or `nothing`.

# Related

  - [`fit_predict`](@ref)
  - [`TrainTestSplitResult`](@ref)
"""
function holdout_window(res::PipelineResult)
    i = findfirst(r -> isa(r, TrainTestSplitResult), getfield(res, :results))
    return isnothing(i) ? nothing : getfield(res, :results)[i].test
end
"""
    fit_predict(pipe::Pipeline, data::Prices_RR) -> PredictionResult

Fit a pipeline on `data`, and predict with it at once.

When the pipeline starts with a [`TrainTestSplit`](@ref), the prediction uses the test window of the split, which no fitted step saw. Otherwise the prediction uses `data` itself, in sample. So one call with a split fits on the training rows and scores on the test rows.

# Algorithm

 1. Fit `pipe` on `data` with [`fit`](@ref), giving `res`.
 2. Read the test window of the split with [`holdout_window`](@ref), giving `test`.
 3. Predict with `res` on `test`, or on `data` when `test` is `nothing`.

# Arguments

  - `pipe`: The pipeline.
  - `data::Prices_RR`: Price data or returns data.

# Returns

  - `pred::PredictionResult`: The prediction on the test window when the pipeline has a split, and on `data` otherwise.

# Related

  - [`predict(res::PipelineResult, data::Prices_RR)`](@ref)
  - [`Pipeline`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`PredictionResult`](@ref)
"""
function fit_predict(pipe::Pipeline, data::Prices_RR)
    res = StatsAPI.fit(pipe, data)
    test = holdout_window(res)
    return StatsAPI.predict(res, isnothing(test) ? data : test)
end
function run_step(p::Pipeline, ctx::PipelineContext)
    data = if :prices in pipe_reads(p)
        require_slot(ctx, :prices, p)
        ctx.prices
    else
        require_slot(ctx, :returns, p)
        ctx.returns
    end
    res = StatsAPI.fit(p, data)
    slot = pipe_writes(p)
    return res, set_slot(ctx, slot, getproperty(getfield(res, :ctx), slot))
end
function optimise(::Pipeline, args...; kwargs...)
    return throw(ArgumentError("a Pipeline is a workflow, not an OptimisationEstimator: fit it with fit(pipeline, data). Wrapping a Pipeline inside a meta-optimiser is not supported."))
end

export Pipeline, PipelineResult
public has_split, assert_no_holdout, assert_split_position, holdout_window
