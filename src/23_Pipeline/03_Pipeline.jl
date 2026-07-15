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
                  ArgumentError("pipeline step names must be unique, got $(collect(names))"))
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
    slots = Symbol[pipe_writes(e) for e in ests]
    avail = Set{Symbol}((:prices, :returns))
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
    i = findfirst(==(name), getfield(pr, :names))
    @argcheck(!isnothing(i),
              ArgumentError("no pipeline step named $(repr(name)); available steps: $(collect(getfield(pr, :names)))"))
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

  - [`inject_config`](@ref)
"""
constraint_results(::Nothing) = ()
constraint_results(c::AbstractConstraintResult) = (c,)
constraint_results(c::AbstractVector{<:AbstractConstraintResult}) = c
"""
    inject_config(cfg::Union{<:JuMPOptimiser, <:HierarchicalOptimiser}, ctx::PipelineContext)

Override an optimiser configuration's internal estimators with the computed slots of the pipeline context.

Routing:

  - `prior` slot → `pe` (both configurations; the result-in-place-of-estimator idiom, cf. `prior(::AbstractPriorResult)`).
  - `phylogeny` slot → `cle` on [`HierarchicalOptimiser`](@ref) when it is a hierarchical clustering result; ignored by [`JuMPOptimiser`](@ref), whose phylogeny information enters as constraint results.
  - `uncertainty` slot, mean half → `ret.ucs` on [`JuMPOptimiser`](@ref) (requires [`ArithmeticReturn`](@ref)); unroutable into a [`HierarchicalOptimiser`](@ref). The covariance half is routed separately into the risk measure by [`inject_sigma_ucs`](@ref).
  - `constraints` slot elements, by result type: [`WeightBounds`](@ref) → `wb`; [`LinearConstraint`](@ref) → `lcse` and phylogeny constraint results → `ple` ([`JuMPOptimiser`](@ref) only).

Unroutable elements throw an `ArgumentError` naming the offending type.

# Arguments

  - `cfg`: The optimiser configuration.
  - `ctx`: The pipeline context.

# Returns

  - `cfg′`: The updated configuration.

# Related

  - [`inject_context`](@ref)
  - [`inject_sigma_ucs`](@ref)
"""
function inject_config(cfg::JuMPOptimiser, ctx::PipelineContext)
    if !isnothing(ctx.prior)
        cfg = Accessors.set(cfg, Accessors.PropertyLens{:pe}(), ctx.prior)
    end
    unc = ctx.uncertainty
    if !isnothing(unc) && !isnothing(unc.mu)
        @argcheck(isa(cfg.ret, ArithmeticReturn),
                  ArgumentError("cannot route a mean uncertainty set into a $(Base.typename(typeof(cfg.ret)).wrapper); expected returns uncertainty requires an ArithmeticReturn return estimator"))
        ret = Accessors.set(cfg.ret, Accessors.PropertyLens{:ucs}(), unc.mu)
        cfg = Accessors.set(cfg, Accessors.PropertyLens{:ret}(), ret)
    end
    lcs = LinearConstraint[]
    ples = AbstractPhylogenyConstraintResult[]
    for c in constraint_results(ctx.constraints)
        if isa(c, WeightBounds)
            cfg = Accessors.set(cfg, Accessors.PropertyLens{:wb}(), c)
        elseif isa(c, LinearConstraint)
            push!(lcs, c)
        elseif isa(c, AbstractPhylogenyConstraintResult)
            push!(ples, c)
        else
            throw(ArgumentError("cannot route a $(Base.typename(typeof(c)).wrapper) constraint result into a JuMPOptimiser; supported: WeightBounds, LinearConstraint, and phylogeny constraint results"))
        end
    end
    if !isempty(lcs)
        cfg = Accessors.set(cfg, Accessors.PropertyLens{:lcse}(),
                            length(lcs) == 1 ? lcs[1] : lcs)
    end
    if !isempty(ples)
        cfg = Accessors.set(cfg, Accessors.PropertyLens{:ple}(),
                            length(ples) == 1 ? ples[1] : identity.(ples))
    end
    return cfg
end
function inject_config(cfg::HierarchicalOptimiser, ctx::PipelineContext)
    if !isnothing(ctx.prior)
        cfg = Accessors.set(cfg, Accessors.PropertyLens{:pe}(), ctx.prior)
    end
    if !isnothing(ctx.phylogeny) && isa(ctx.phylogeny, AbstractClusteringResult)
        cfg = Accessors.set(cfg, Accessors.PropertyLens{:cle}(), ctx.phylogeny)
    end
    unc = ctx.uncertainty
    if !isnothing(unc) && !isnothing(unc.mu)
        throw(ArgumentError("cannot route a mean uncertainty set into a HierarchicalOptimiser; uncertainty sets require a JuMP-based optimiser"))
    end
    for c in constraint_results(ctx.constraints)
        if isa(c, WeightBounds)
            cfg = Accessors.set(cfg, Accessors.PropertyLens{:wb}(), c)
        else
            throw(ArgumentError("cannot route a $(Base.typename(typeof(c)).wrapper) constraint result into a HierarchicalOptimiser; supported: WeightBounds"))
        end
    end
    return cfg
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Route a covariance uncertainty set into the [`UncertaintySetVariance`](@ref) risk measure(s) of an optimisation estimator.

The optimiser must expose a risk-measure field `r` containing at least one [`UncertaintySetVariance`](@ref) (possibly inside a vector); each one's `ucs` is replaced with `sig`. Anything else throws an `ArgumentError`.

# Arguments

  - `opt`: The optimisation estimator.
  - `sig`: The covariance uncertainty set result.

# Returns

  - `opt′`: The updated estimator.

# Related

  - [`inject_context`](@ref)
  - [`inject_config`](@ref)
"""
function inject_sigma_ucs(opt::OptimisationEstimator, sig::AbstractUncertaintySetResult)
    @argcheck(hasproperty(opt, :r),
              ArgumentError("cannot route a covariance uncertainty set into a $(Base.typename(typeof(opt)).wrapper): it has no risk-measure field"))
    replace_usv(x) =
        if isa(x, UncertaintySetVariance)
            Accessors.set(x, Accessors.PropertyLens{:ucs}(), sig)
        else
            x
        end
    r = opt.r
    newr = isa(r, AbstractVector) ? identity.([replace_usv(x) for x in r]) : replace_usv(r)
    found = if isa(newr, AbstractVector)
        any(x -> isa(x, UncertaintySetVariance), newr)
    else
        isa(newr, UncertaintySetVariance)
    end
    @argcheck(found,
              ArgumentError("cannot route a covariance uncertainty set into a $(Base.typename(typeof(opt)).wrapper): no UncertaintySetVariance risk measure in its r field"))
    return Accessors.set(opt, Accessors.PropertyLens{:r}(), newr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Override an optimisation step's internal configuration with the computed slots of the pipeline context, immediately before the step runs.

Estimators carrying an `opt` configuration field ([`JuMPOptimiser`](@ref) or [`HierarchicalOptimiser`](@ref)) are rebuilt via [`inject_config`](@ref), and a covariance uncertainty set is routed into their risk measures via [`inject_sigma_ucs`](@ref). Estimators without an injectable configuration (naive and meta-optimisers) ignore the `prior` and `phylogeny` slots — they compute internally — but populated `uncertainty` or `constraints` slots throw an `ArgumentError` rather than being silently dropped.

# Arguments

  - `opt`: The optimisation step estimator.
  - `ctx`: The pipeline context.

# Returns

  - `opt′`: The (possibly rebuilt) estimator actually run.

# Related

  - [`inject_config`](@ref)
  - [`inject_sigma_ucs`](@ref)
  - [`fit`](@ref)
"""
function inject_context(opt::OptimisationEstimator, ctx::PipelineContext)
    if isnothing(ctx.prior) &&
       isnothing(ctx.phylogeny) &&
       isnothing(ctx.uncertainty) &&
       isnothing(ctx.constraints)
        return opt
    end
    cfg = hasproperty(opt, :opt) ? opt.opt : nothing
    if isa(cfg, Union{<:JuMPOptimiser, <:HierarchicalOptimiser})
        opt = Accessors.set(opt, Accessors.PropertyLens{:opt}(), inject_config(cfg, ctx))
        unc = ctx.uncertainty
        if !isnothing(unc) && !isnothing(unc.sigma)
            opt = inject_sigma_ucs(opt, unc.sigma)
        end
        return opt
    end
    @argcheck(isnothing(ctx.uncertainty),
              ArgumentError("cannot route uncertainty sets into a $(Base.typename(typeof(opt)).wrapper): it has no injectable optimiser configuration"))
    @argcheck(isnothing(ctx.constraints),
              ArgumentError("cannot route constraint results into a $(Base.typename(typeof(opt)).wrapper): it has no injectable optimiser configuration"))
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
                     [100.0 101.0; 102.0 103.0; 101.0 104.0; 103.0 102.0], ["A", "B"]);

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

export Pipeline, PipelineResult, fit
public has_split, assert_no_holdout, assert_split_position, holdout_window
