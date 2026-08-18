"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that a [`TrainTestSplit`](@ref) appears only as the first step of a [`Pipeline`](@ref), and never inside a nested one.

The holdout exists to keep the test window away from every fitted step. A stateful step fitted *before* the split — a [`MissingDataFilter`](@ref) choosing the universe, an [`Imputer`](@ref) computing fill values — would have read the held-out rows, so its fitted state leaks test data into the training workflow. Position one is the only place that cannot happen, and a nested pipeline is never step one of itself.

## Validation

  - At most one `TrainTestSplit`, and only at index 1.
  - No `TrainTestSplit` inside a nested `Pipeline` or a [`PipelineStep`](@ref).

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

Validate that an optimisation step, if present, is the last step of a [`Pipeline`](@ref).

A pipeline's optimiser writes the terminal `:opt` slot — the workflow's output. Nothing is derived from `:opt`, so a step running *after* an optimiser could only strand those weights: a later data or estimator step would leave `:opt` computed on a since-changed context, and no later step reads `:opt` to catch it. Pinning the optimiser last keeps `:opt` genuinely terminal, which is also what lets [`PIPELINE_INVALIDATES`](@ref) omit it from the invalidatable slots. A terminal optimiser is optional (a prior-only pipeline is legal); when absent the rule is vacuous.

## Validation

  - No step writes `:opt` unless it is the final step. A nested [`Pipeline`](@ref) reports the slot its own last step writes and is validated at its own construction, so a non-terminal optimiser hidden inside one is caught there.

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

Validate that every constraint step of a [`Pipeline`](@ref) resolves to exactly one [routing target](@ref PIPELINE_ROUTING_TARGETS).

Runs [`resolve_constraint_target`](@ref) on each constraint step, which is the same call [`run_constraint_step`](@ref) makes when the step runs. Doing it here moves three failures from the fold loop to the constructor: a family that computes nothing for the `constraints` slot and is therefore not a step, a family that names several targets and was not told which, and a declared target that belongs to another family.

## Validation

  - Each constraint step's family declares at least one target (see [`pipe_constraint_targets`](@ref)).
  - A family declaring several has a [`PipelineStep`](@ref) `target` naming one of them.

# Arguments

  - `ests`: The step estimators.

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

The [routing targets](@ref PIPELINE_ROUTING_TARGETS) a step is *known at construction* to produce.

An uncertainty-set step qualifies. It must declare which parameters it bounds through its [`PipelineStep`](@ref) wrapper, and that declaration is a field of the step rather than a property of a computed result, so the targets it will write are known before anything runs.

A constraint step qualifies for the same reason, one step removed: its target is declared by its *family* through [`pipe_constraint_targets`](@ref), and where the family names several, by the step's own `target` field. Both are known before anything runs, and [`run_constraint_step`](@ref) resolves the destination from the same declaration, so the target checked here is the target the step will write.

Everything else returns an empty tuple. A callable step writing `:constraints` declares no family, and a precomputed result carried in by the pipeline input names its target only by its type.

# Arguments

  - `est`: A step estimator.

# Returns

  - A tuple of routing targets, empty when nothing is statically known.

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

Reject at construction a pipeline whose terminal optimiser cannot receive a target an earlier step will write.

Without this, an unroutable uncertainty set is discovered by [`inject_context`](@ref) at injection time — which, under [`cross_val_predict`](@ref), is after the fold loop has already fitted every earlier step of the first fold. The check asks the optimiser directly via [`pipe_accepts`](@ref), so it stays honest as optimisers gain or lose fields.

It is deliberately structural: it establishes that the optimiser *family* can receive the target at all, not that this particular configuration will accept the value. A [`JuMPOptimiser`](@ref) always accepts `:mu_ucs`, but one carrying a non-[`ArithmeticReturn`](@ref) estimator still fails at injection — that condition belongs to [`pipe_route`](@ref) and is not duplicated here.

Skipped when the terminal step is a [`TimeDependent`](@ref) schedule or a precomputed result, since the optimiser is then not known until the fold loop resolves it.

# Arguments

  - `ests`: The step estimators, optimisation step last (see [`assert_opt_last`](@ref)).

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

Return the first element that repeats an earlier one, for a name-uniqueness error
that names the offending token without dumping the whole collection (ADR 0026 boundary
discipline). Only ever called on the failing path.

# Arguments

  - `xs`: A collection of names.

# Returns

  - `seen::Set{String}`: A collection of names.

# Related

  - [`Pipeline`](@ref)
"""
function first_duplicate(xs)
    seen = Set{eltype(xs)}()
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

A reified end-to-end portfolio workflow: an ordered list of steps executed left-to-right over a [`PipelineContext`](@ref).

Steps are ordinary estimators — preprocessing, prior, phylogeny, uncertainty-set, constraint-generation, and optimisation estimators, nested `Pipeline`s, or [`PipelineStep`](@ref) wrappers — mapped to context slots by their family via [`pipe_writes`](@ref)/[`pipe_reads`](@ref). Fitting a pipeline with [`fit`](@ref) walks the steps in order; computed slots override the terminal optimiser's internal configuration (see [`inject_context`](@ref)), and absent steps fall back to whatever the optimiser computes internally, so every stage is optional.

A terminal optimiser is not required: a prior-only pipeline is legal; prediction is what needs weights.

See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Pipeline(; steps::Union{<:Tuple, <:AbstractVector}) -> Pipeline

Steps are given in execution order. Each element is either a step estimator or a `"name" => estimator` pair; unnamed steps are auto-named from the slot they write (`"prior"`), suffixed in order of appearance when a slot repeats (`"prices_1"`, `"prices_2"`).

## Validation

  - `!isempty(steps)`.
  - Every step must be steppable ([`pipe_writes`](@ref) must be defined for it).
  - Every slot a step reads must be written by an earlier step or fillable by the pipeline input (`prices` or `returns`).
  - No step may write a slot that invalidates a slot an earlier step already wrote (see [`PIPELINE_INVALIDATES`](@ref)). A step that rewrites `:returns` after a prior, phylogeny, uncertainty, or constraint step would leave that result computed on a stale asset universe.
  - An optimisation step, if present, must be the last step (see [`assert_opt_last`](@ref)): it writes the terminal `:opt` slot, and no step may run after it.
  - Step names must be unique.

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
    Step names, aligned with `steps`.
    """
    names
    """
    The step estimators, in execution order.
    """
    steps
    function Pipeline(names::Tuple{Vararg{String}}, steps::Tuple)
        @argcheck(!isempty(steps), IsEmptyError("steps cannot be empty"))
        @argcheck(length(names) == length(steps), DimensionMismatch)
        @argcheck(allunique(names),
                  ArgumentError("pipeline step names must be unique; the name $(repr(first_duplicate(names))) is repeated among the $(length(names)) steps"))
        return new{typeof(names), typeof(steps)}(names, steps)
    end
end
function Pipeline(; steps::Union{<:Tuple, <:AbstractVector})::Pipeline
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
    return Pipeline(Tuple(names), Tuple(ests))
end
pipe_writes(p::Pipeline) = pipe_writes(p.steps[end])
pipe_reads(p::Pipeline) = pipe_reads(p.steps[1])
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether a step is, or contains, a [`TrainTestSplit`](@ref).

A nested [`Pipeline`](@ref) is searched recursively: a split hidden inside one would be fitted on data an outer step had already touched, which is exactly what pinning it to the first position prevents. The same recursion answers whether a whole pipeline carries a holdout, which is what the cross-validation entry points check before running.

# Related

  - [`assert_split_position`](@ref)
  - [`assert_no_holdout`](@ref)
  - [`TrainTestSplit`](@ref)
"""
has_split(::Any) = false
has_split(::TrainTestSplit) = true
has_split(p::Pipeline) = any(has_split, p.steps)
has_split(ps::PipelineStep) = has_split(ps.est)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reject a [`Pipeline`](@ref) carrying a [`TrainTestSplit`](@ref) from the cross-validation machinery.

A holdout split and a cross-validator are two evaluation protocols, and cross-validation already defines the train/test windows of every fold. A split left in the pipeline would shave a second, redundant holdout off each fold's training window and stash a test window nobody reads — a silent loss of training data. One protocol per call: this throws instead.

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

Deliberately unsupported: a [`Pipeline`](@ref) cannot be sub-selected by asset view.

Meta-optimisers (`NestedClustered`, `Stacking`, `SubsetResampling`) build asset sub-portfolios by taking a `port_opt_view` of their inner estimator. A pipeline's asset universe is *fitted state* — the missing-data filter decides it from the training window — so an asset view taken before fitting is not well defined. Wrapping a `Pipeline` in a meta-optimiser is therefore unsupported in v1 (ADR 0028, "Future expansion"); a meta-optimiser may still be the *optimisation step of* a pipeline.

# Related

  - [`Pipeline`](@ref)
  - [`optimise(::Pipeline)`](@ref)
"""
function port_opt_view(::Pipeline, args...; kwargs...)
    return throw(ArgumentError("a Pipeline cannot be sub-selected with port_opt_view: its asset universe is fitted state, so wrapping a Pipeline inside a meta-optimiser is unsupported (ADR 0028). A meta-optimiser may be used as the optimisation step of a Pipeline instead."))
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`Pipeline`](@ref).

Carries the fitted per-step results (named, in step order), and the final [`PipelineContext`](@ref) whose slots hold the computed data, prior, phylogeny, uncertainty, constraints, and terminal optimisation result.

Step results are accessed by name with `getindex` (`res["prior"]`) or by position through the `results` field (`res.results[2]`); integer indexing keeps the package-wide length-1 container semantics. The `w` property forwards to the terminal optimisation result's weights (`res.ctx.opt.w`) and throws a [`PropertyPathError`](@ref) when the pipeline produced no optimisation result.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Pipeline`](@ref)
  - [`AbstractPipelineResult`](@ref)
  - [`fit`](@ref)
"""
@concrete struct PipelineResult <: AbstractPipelineResult
    """
    Step names, aligned with `results`.
    """
    names
    """
    Fitted per-step results, in step order.
    """
    results
    """
    The final [`PipelineContext`](@ref).
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

Iterate the elements of the `constraints` slot uniformly.

# Arguments

  - `x`: `nothing`, a single [`AbstractConstraintResult`](@ref), or a vector of them.

# Returns

  - An iterable of constraint results (empty for `nothing`).

# Related

  - [`constraint_targets`](@ref)
"""
constraint_results(::Nothing) = ()
constraint_results(c::AbstractConstraintResult) = (c,)
constraint_results(c::AbstractVector{<:AbstractConstraintResult}) = c
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [routing target](@ref PIPELINE_ROUTING_TARGETS) a constraint result names by its type alone, or `nothing`.

Four result types name exactly one optimiser field, so a value of one of those types places itself: a [`WeightBounds`](@ref) can only be `:wb`, a [`LinearConstraint`](@ref) only `:lcse`, a phylogeny constraint result only `:ple`, a [`RiskBudget`](@ref) only `:rkb`. Everything else answers `nothing`, and needs a target carried alongside it — see [`TargetedConstraint`](@ref).

This is the *only* type-driven half of the fan-out, and it is also what decides whether a step's value needs a wrapper at all: [`add_constraint_result`](@ref) wraps exactly when the value cannot name its own destination, so the `constraints` slot holds a bare result wherever it can.

# Arguments

  - `c`: A constraint result.

# Returns

  - `target::Union{Nothing, Symbol}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref), or `nothing`.

# Related

  - [`constraint_target_of`](@ref)
  - [`TargetedConstraint`](@ref)
"""
implicit_constraint_target(::WeightBounds) = :wb
implicit_constraint_target(::LinearConstraint) = :lcse
implicit_constraint_target(::AbstractPhylogenyConstraintResult) = :ple
implicit_constraint_target(::RiskBudget) = :rkb
implicit_constraint_target(::Any) = nothing
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [routing target](@ref PIPELINE_ROUTING_TARGETS) one element of the `constraints` slot lands in.

An element a constraint step could not place by type carries its target — [`run_constraint_step`](@ref) paired the two — and it is read straight off. Everything else is placed by [`implicit_constraint_target`](@ref).

Two cases throw. A [`Threshold`](@ref) names six optimiser fields, so its type cannot place it; the error names the declaration that would. A result of any other unplaceable type has no target at all, and is rejected here rather than at an optimiser.

# Arguments

  - `c`: One element of the `constraints` slot.

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

The value one element of the `constraints` slot delivers, with the routing wrapper removed.

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

Combine the several values that reached one [accumulating](@ref PIPELINE_ACCUMULATING_TARGETS) routing target.

The default packs them into a vector in write order, which is the shape every field holding one result per estimator expects.

`:cte` is the exception, and it is what this seam exists for. Its field takes a vector of [`CentralityConstraint`](@ref) *estimators*, and [`centrality_constraints`](@ref) appends every row of every estimator into **one** [`LinearConstraint`](@ref). Separate steps therefore merge rather than pack, so *n* centrality steps in a [`Pipeline`](@ref) reach the optimiser with the value one `cte` field holding *n* estimators would have produced.

Only ever called with more than one value; a single value is unwrapped by [`constraint_targets`](@ref) before it gets here.

# Arguments

  - `::Val{target}`: The routing target the values reached.
  - `vals`: The values, in write order.

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

Fan the `constraints` slot out into [routing targets](@ref PIPELINE_ROUTING_TARGETS).

Each element is placed by [`constraint_target_of`](@ref) and unwrapped by [`constraint_value_of`](@ref). Several results reaching one [accumulating](@ref PIPELINE_ACCUMULATING_TARGETS) target are combined by [`accumulate_constraint_values`](@ref) — packed into a vector in write order, or, for `:cte`, merged into the one constraint that holds all their rows. A group of one is unwrapped, matching the scalar-or-vector shape those fields accept everywhere else. A second result reaching any other target is refused, because that field holds one value and the second would silently replace the first.

# Arguments

  - `cs`: The `constraints` slot.

# Returns

  - A vector of `target => value` pairs, in the order the results were written.

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

Override an optimisation step's internal configuration with the computed slots of the pipeline context, immediately before the step runs.

This is the pipeline-owned half of the injection seam. It resolves everything that depends on the *slots* — which halves of the uncertainty pair are populated, which result types the `constraints` slot holds, how many of each — into a flat sequence of [routing targets](@ref PIPELINE_ROUTING_TARGETS), then hands each one to [`pipe_route`](@ref) without knowing where it lands. Which optimiser field receives a target is the optimiser's business, so a field rename is a local edit rather than a break here.

Targets an optimiser has no home for are handled by [`unroutable_target`](@ref): `:pe` and `:cle` pass by, everything else throws rather than being silently dropped. This is why a naive or meta-optimiser accepts a computed prior it can use while still rejecting an uncertainty set it cannot.

# Arguments

  - `opt`: The optimisation step estimator.
  - `ctx`: The pipeline context.

# Returns

  - `opt′`: The (possibly rebuilt) estimator actually run.

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

Either return the step estimator unchanged, inject the context into the optimiser, or inject the context into the optimiser and create a pipeline step.

# Arguments

  - `est`: A step estimator.
  - `opt`: An optimisation step estimator.
  - `ps`: A [`PipelineStep`](@ref) wrapping an optimisation step estimator.
  - `ctx`: The pipeline context.

# Returns

  - `est′`: The step estimator to run.
  - `opt`: The optimiser with its configuration overridden by the context.
  - `ps`: The pipeline step with its optimiser overridden by the context.
"""
maybe_inject_step(est, ::PipelineContext) = est
function maybe_inject_step(opt::OptimisationEstimator, ctx::PipelineContext)
    return inject_context(opt, ctx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Injection rules for a precomputed optimisation result standing in the optimisation step — the predict-only fold of a mixed [`TimeDependent`](@ref) schedule.

A result is already solved, so it has no configuration to override; this reuses the non-injectable pattern of [`inject_context`](@ref): computed `prior` and `phylogeny` slots pass by (the result was fitted with its own), but populated `uncertainty` or `constraints` slots throw an `ArgumentError` rather than being silently dropped — a computed constraint that never reaches a solve is a fail-closed error, not a no-op.

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

Fit a [`Pipeline`](@ref) on price- or returns-level data.

The context slot matching the input type is filled (`PricesResult` → `prices`, `ReturnsResult` → `returns`, so passing returns-level data skips the price stages), then the steps run left-to-right via [`run_step`](@ref). Immediately before an optimisation step runs, the computed slots override its internal configuration via [`inject_context`](@ref).

`fit` is a fold-less entry point, so [`TimeDependent`](@ref) schedule steps are inert here: each resolves to its explicit `default` (see [`reset_time_dependent_estimator`](@ref)) before the steps run, and a schedule with no `default` throws a [`TimeDependentDefaultError`](@ref) — backtest the pipeline with [`cross_val_predict`](@ref), whose folds the schedule resolves against. Inside a fold loop this reset is a no-op, because the loop swaps every schedule for its per-fold value first.

# Arguments

  - `pipe`: The pipeline.
  - `data`: The input data ([`PricesResult`](@ref) or [`ReturnsResult`](@ref)).

# Returns

  - `res::PipelineResult`: Named per-step fitted results and the final context.

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

Assert that replaying a pipeline's fitted steps on a test window reproduces the training asset universe.

The terminal weights are indexed by the *training* universe, so a test window whose transformed returns carry a different asset set (or a different asset order) would silently misalign weights and returns. This is the failure the fit/apply contract exists to prevent, so it is reported as an error naming both universes rather than surfacing as a dimension mismatch inside the risk calculation.

The usual cause is relying on [`PricesToReturns`](@ref) alone to define the universe: it is stateless, and the underlying [`prices_to_returns`](@ref) drops assets that are entirely missing in the window being converted, which differs between train and test. Pin the universe with a [`MissingDataFilter`](@ref) step, and fill the remaining gaps with an [`Imputer`](@ref) step, before converting.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).
  - `rd`: The transformed test-window returns.

# Returns

  - `nothing`.

# Related

  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`Imputer`](@ref)
"""
function assert_universe_aligned(res::PipelineResult, rd::AbstractReturnsResult)::Nothing
    train = res.ctx.returns
    if isnothing(train)
        return nothing
    end
    @argcheck(rd.nx == train.nx,
              ArgumentError("the pipeline's fitted steps produced a test-window universe $(rd.nx) that differs from the training universe $(train.nx), so the weights and the test returns would not be aligned. PricesToReturns is stateless and drops assets that are entirely missing in the window it converts; pin the universe with a MissingDataFilter step (and an Imputer step to fill the remaining gaps) before converting to returns."))
    return nothing
end
"""
    apply_fitted_step(fitted, data) -> data′

Replay one fitted pipeline step on a data window during prediction.

Preprocessing steps transform the window at the data level they apply to: price-level fitted objects ([`AbstractPricesPreprocessingResult`](@ref), [`AbstractPricesPreprocessingEstimator`](@ref)) transform price-level windows, returns-level ones transform returns-level windows, and [`PricesToReturns`](@ref) converts the window from prices to returns. A fitted object whose data level does not match the current window passes it through unchanged — mirroring fit, where such a step cannot affect the data that reaches the optimiser. Non-preprocessing fitted results (priors, phylogeny, uncertainty, constraints, optimisation) pass the window through untouched, and a nested [`PipelineResult`](@ref) replays its own steps recursively.

# Arguments

  - `fitted`: A fitted per-step result from a [`PipelineResult`](@ref).
  - `data`: The current data window ([`AbstractPricesResult`](@ref) or [`AbstractReturnsResult`](@ref)).

# Returns

  - `data′`: The transformed (or untouched) data window.

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

Replay the fitted preprocessing steps of a pipeline on a data window, in step order.

# Arguments

  - `results`: The fitted per-step results of a [`PipelineResult`](@ref).
  - `data`: The data window to transform.

# Returns

  - `data′`: The transformed data window (returns-level when the steps include a [`PricesToReturns`](@ref) conversion).

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
                          test_idxs::VecVecInt, cols = Colon()) -> PredictionResult

    predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idx = Colon(), cols = Colon()) -> PredictionResult

Apply a fitted pipeline to an unseen data test_idx and produce the same [`PredictionResult`](@ref) the weights-level machinery consumes.

The `test_idx` selects observation rows of `data` (integer indices, timestamps, or `:` for all rows). The test_idx is transformed by replaying the fitted preprocessing steps in step order — the *training* universe subset, the *training* imputation parameters, then the returns conversion — so no statistics of the test test_idx leak into the transformation. The result is then handed to the existing weights-level `predict`, so scorers and risk measures carry over untouched.

Price-level data requires the pipeline to contain a [`PricesToReturns`](@ref) step; a pipeline that produced no optimisation result cannot predict.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).
  - `data`: Price- or returns-level data containing the test_idx ([`PricesResult`](@ref) or [`ReturnsResult`](@ref)).
  - `test_idx`: Observation test_idx into the rows of `data`. Integer indices, timestamps, or `:` (all rows).

# Returns

  - `pred::PredictionResult`: The weights-level prediction on the transformed test_idx.

# Related

  - [`PipelineResult`](@ref)
  - [`apply_fitted_steps`](@ref)
  - [`port_opt_view`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
"""
function StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idx = Colon(), cols = Colon())
    opt = res.ctx.opt
    @argcheck(!isnothing(opt),
              IsNothingError("the pipeline produced no optimisation result; add a terminal optimisation step before predicting"))
    pr = port_opt_view(data, test_idx, cols)
    rd = apply_fitted_steps(res.results, pr)
    @argcheck(isa(rd, AbstractReturnsResult),
              ArgumentError("the pipeline's fitted steps do not convert price-level data to returns; predicting on a $(Base.typename(typeof(data)).wrapper) requires a PricesToReturns step"))
    assert_universe_aligned(res, rd)
    return StatsAPI.predict(opt, rd)
end
function StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult,
                          test_idxs::VecVecInt, cols = Colon())
    return [StatsAPI.predict(res, data, test_idx, cols) for test_idx in test_idxs]
end
function StatsAPI.predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idx = Colon(), cols = Colon())
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
    return StatsAPI.predict(opt, rd)
end
function StatsAPI.predict(res::PipelineResult, data::AbstractReturnsResult,
                          test_idxs::VecVecInt, cols = Colon())
    return [StatsAPI.predict(res, data, test_idx, cols) for test_idx in test_idxs]
end
function fit_and_predict(res::PipelineResult, data::AbstractReturnsResult;
                         test_idx::VecInt_VecVecInt, cols = :, kwargs...)
    opt = res.ctx.opt
    @argcheck(!isnothing(opt),
              IsNothingError("the pipeline produced no optimisation result; add a terminal optimisation step before predicting"))
    return StatsAPI.predict(res, data, test_idx, cols)
end
function fit_and_predict(pipe::Pipeline, data::Prices_RR; train_idx::VecInt,
                         test_idx::VecInt_VecVecInt, cols = :)
    data_train = pipeline_data_view(data, train_idx, cols)
    #! Maybe we should define a port_opt_view for pipelines?
    # if !isa(cols, Colon)
    #     opt = port_opt_view(pipe, cols)
    # end
    res = StatsAPI.fit(pipe, data_train)
    return StatsAPI.predict(res, data, test_idx, cols)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the held-out window stashed by a pipeline's [`TrainTestSplit`](@ref) step, or `nothing` when it has none.

# Related

  - [`fit_predict`](@ref)
  - [`TrainTestSplitResult`](@ref)
"""
function holdout_window(res::PipelineResult)
    i = findfirst(r -> isa(r, TrainTestSplitResult), getfield(res, :results))
    return isnothing(i) ? nothing : getfield(res, :results)[i].test
end
"""
    fit_predict(opt::Pipeline, data::Prices_RR)

Fit pipeline estimator `opt` on data `data` and immediately produce a
[`PredictionResult`](@ref).

The prediction is made on `data` itself — *in-sample* — unless the pipeline begins with a [`TrainTestSplit`](@ref), in which case it is made on the held-out window that step reserved and no fitted step has seen. That is the one-line holdout evaluation: fit on the training rows, score on the test rows.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `data::Prices_RR`: Price- or returns-level data.

# Returns

  - [`PredictionResult`](@ref): On the held-out window when the pipeline splits, on `data` otherwise.

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
