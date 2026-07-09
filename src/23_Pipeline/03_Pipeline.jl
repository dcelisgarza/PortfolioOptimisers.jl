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
    slots = Symbol[pipe_writes(e) for e in ests]
    avail = Set{Symbol}((:prices, :returns))
    written = Dict{Symbol, Any}()
    for (e, slot) in zip(ests, slots)
        for r in pipe_reads(e)
            @argcheck(r in avail,
                      ArgumentError("a $(typeof(e)) step reads the :$r slot, which no earlier step writes and the pipeline input cannot fill"))
        end
        for inv in get(PIPELINE_INVALIDATES, slot, ())
            @argcheck(!haskey(written, inv),
                      ArgumentError("a $(typeof(e)) step writes the :$slot slot, invalidating the :$inv slot written by an earlier $(typeof(written[inv])) step; the stale :$inv result would no longer match the assets of the new :$slot data. Move the $(typeof(e)) step before the $(typeof(written[inv])) step, or drop one of them."))
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
                  ArgumentError("cannot route a mean uncertainty set into a $(typeof(cfg.ret)); expected returns uncertainty requires an ArithmeticReturn return estimator"))
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
            throw(ArgumentError("cannot route a $(typeof(c)) constraint result into a JuMPOptimiser; supported: WeightBounds, LinearConstraint, and phylogeny constraint results"))
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
            throw(ArgumentError("cannot route a $(typeof(c)) constraint result into a HierarchicalOptimiser; supported: WeightBounds"))
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
              ArgumentError("cannot route a covariance uncertainty set into a $(typeof(opt)): it has no risk-measure field"))
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
              ArgumentError("cannot route a covariance uncertainty set into a $(typeof(opt)): no UncertaintySetVariance risk measure in its r field"))
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
              ArgumentError("cannot route uncertainty sets into a $(typeof(opt)): it has no injectable optimiser configuration"))
    @argcheck(isnothing(ctx.constraints),
              ArgumentError("cannot route constraint results into a $(typeof(opt)): it has no injectable optimiser configuration"))
    return opt
end
maybe_inject_step(est, ::PipelineContext) = est
function maybe_inject_step(opt::OptimisationEstimator, ctx::PipelineContext)
    return inject_context(opt, ctx)
end
function maybe_inject_step(ps::PipelineStep, ctx::PipelineContext)
    if isa(ps.est, OptimisationEstimator)
        return PipelineStep(; est = inject_context(ps.est, ctx), reads = ps.reads,
                            writes = ps.writes, target = ps.target)
    end
    return ps
end
"""
    StatsAPI.fit(pipe::Pipeline, data::Union{<:AbstractPricesResult, <:AbstractReturnsResult}) -> PipelineResult

Fit a [`Pipeline`](@ref) on price- or returns-level data.

The context slot matching the input type is filled (`PricesResult` → `prices`, `ReturnsResult` → `returns`, so passing returns-level data skips the price stages), then the steps run left-to-right via [`run_step`](@ref). Immediately before an optimisation step runs, the computed slots override its internal configuration via [`inject_context`](@ref).

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

julia> res = PortfolioOptimisers.fit(pipe, PricesResult(; X = X));

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
function StatsAPI.fit(pipe::Pipeline,
                      data::Union{<:AbstractPricesResult, <:AbstractReturnsResult})::PipelineResult
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
    return throw(ArgumentError("a Pipeline is a workflow, not an OptimisationEstimator: fit it with PortfolioOptimisers.fit(pipeline, data). Wrapping a Pipeline inside a meta-optimiser is not supported."))
end

export Pipeline, PipelineResult
public fit
