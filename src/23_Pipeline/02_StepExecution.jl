"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that the [`PipelineContext`](@ref) slot `slot` is populated before step `est` runs.

# Arguments

  - `ctx`: The pipeline context.
  - `slot`: The required slot, one of [`PIPELINE_SLOTS`](@ref).
  - `est`: The step about to run, used in the error message.

# Returns

  - `nothing`.

# Related

  - [`run_step`](@ref)
  - [`PipelineContext`](@ref)
"""
function require_slot(ctx::PipelineContext, slot::Symbol, est)::Nothing
    @argcheck(!isnothing(getproperty(ctx, slot)),
              IsNothingError("the :$slot slot of the pipeline context must be populated before a $(typeof(est)) step can run; add an earlier step that writes :$slot or provide it as the pipeline input"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`PipelineContext`](@ref) with slot `slot` set to `val` and every other slot unchanged.

# Arguments

  - `ctx`: The pipeline context.
  - `slot`: The slot to write, one of [`PIPELINE_SLOTS`](@ref).
  - `val`: The value to write.

# Returns

  - `ctx::PipelineContext`: The updated context.

# Related

  - [`run_step`](@ref)
  - [`PipelineContext`](@ref)
"""
function set_slot(ctx::PipelineContext, slot::Symbol, val)::PipelineContext
    return Accessors.set(ctx, Accessors.PropertyLens{slot}(), val)
end
"""
    run_step(est, ctx::PipelineContext) -> (fitted, ctx′)

Execute one pipeline step: fit `est` on the [`PipelineContext`](@ref) slots it reads and return the fitted object together with a new context whose written slot is updated.

Each estimator family dispatches to its native verb — [`prior`](@ref) for prior estimators, [`clusterise`](@ref)/[`phylogeny_matrix`](@ref) for phylogeny estimators, [`optimise`](@ref) for optimisation estimators, [`fit_preprocessing`](@ref)/[`apply_preprocessing`](@ref) for preprocessing estimators. The fitted object is what [`apply_preprocessing`](@ref) later uses to transform unseen data windows; for non-preprocessing steps it is the step's ordinary result.

Estimators whose family is not steppable throw an `ArgumentError` directing the caller to [`PipelineStep`](@ref).

# Arguments

  - `est`: The step estimator (or a [`PipelineStep`](@ref) wrapper).
  - `ctx`: The pipeline context.

# Returns

  - `(fitted, ctx′)`: The fitted object and the updated context.

# Related

  - [`apply_preprocessing`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`PipelineContext`](@ref)
  - [`PipelineStep`](@ref)
"""
function run_step(est, ::PipelineContext)
    return throw(ArgumentError("a $(typeof(est)) is not steppable; wrap it in a PipelineStep to declare its reads/writes explicitly"))
end
function run_step(pe::AbstractPriorEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, pe)
    pr = prior(pe, ctx.returns)
    return pr, set_slot(ctx, :prior, pr)
end
function run_step(cle::AbstractClustersEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, cle)
    res = clusterise(cle, ctx.returns)
    return res, set_slot(ctx, :phylogeny, res)
end
function run_step(ne::AbstractNetworkEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, ne)
    res = phylogeny_matrix(ne, ctx.returns)
    return res, set_slot(ctx, :phylogeny, res)
end
function run_step(opt::OptimisationEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, opt)
    res = optimise(opt, ctx.returns)
    return res, set_slot(ctx, :opt, res)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Execute a precomputed optimisation result as the optimisation step: there is nothing to solve, so the result is written to the `opt` slot as-is and the fold predicts with its weights.

This is how a *mixed* [`TimeDependent`](@ref) schedule runs its result entries: the fold-loop swap ([`update_time_dependent_estimator`](@ref)) replaces the schedule step with entry `i`, which may be an estimator (the fold optimises) or a result (the fold only predicts). A result cannot consume computed context slots — see [`maybe_inject_step`](@ref) for the fail-closed injection rules.

# Related

  - [`run_step`](@ref)
  - [`maybe_inject_step`](@ref)
  - [`TD_OptE_Opt`](@ref)
"""
function run_step(res::NonFiniteAllocationOptimisationResult, ctx::PipelineContext)
    return res, set_slot(ctx, :opt, res)
end
function run_step(est::AbstractPricesPreprocessingEstimator, ctx::PipelineContext)
    require_slot(ctx, :prices, est)
    res = fit_preprocessing(est, ctx.prices)
    return res, set_slot(ctx, :prices, apply_preprocessing(res, ctx.prices))
end
function run_step(est::AbstractReturnsPreprocessingEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, est)
    res = fit_preprocessing(est, ctx.returns)
    return res, set_slot(ctx, :returns, apply_preprocessing(res, ctx.returns))
end
function run_step(ps::PipelineStep, ctx::PipelineContext)
    for r in ps.reads
        require_slot(ctx, r, ps.est)
    end
    if isa(ps.est, Function)
        val = ps.est(ctx)
        return val, set_slot(ctx, ps.writes, val)
    end
    if isa(ps.est, AbstractUncertaintySetEstimator)
        return run_uncertainty_step(ps.est, ps.target, ctx)
    end
    if isa(ps.est, AbstractConstraintEstimator)
        return run_constraint_step(ps.est, ps.target, ctx)
    end
    return run_step(ps.est, ctx)
end
function run_step(ue::AbstractUncertaintySetEstimator, ::PipelineContext)
    return throw(ArgumentError("a $(typeof(ue)) step must declare which parameter it bounds; wrap it in a PipelineStep with target = :mu, target = :sigma, or target = :both"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Execute an uncertainty-set step pinned to a target and merge its result into the `uncertainty` slot.

The target comes from the [`PipelineStep`](@ref) wrapper:

  - `:mu` computes [`mu_ucs`](@ref) and fills the mean half.
  - `:sigma` computes [`sigma_ucs`](@ref) and fills the covariance half.
  - `:both` computes [`ucs`](@ref), which derives *both* halves from one fit — sharing the prior and, for the sampling algorithms, the simulation draws — and is therefore cheaper than the two narrowed calls.

A narrowed step fills its half of the [`PipelineUncertaintySets`](@ref) pair and leaves the other untouched, so separate `:mu` and `:sigma` steps compose. Every populated half must reach the optimiser: each becomes its own [routing target](@ref PIPELINE_ROUTING_TARGETS) — `:mu_ucs` and `:sigma_ucs` — and neither is [optional](@ref PIPELINE_OPTIONAL_TARGETS), so a set that cannot be routed is rejected rather than dropped and `:both` requires an optimiser with an [`ArithmeticReturn`](@ref) *and* an [`UncertaintySetVariance`](@ref) risk measure.

# Arguments

  - `ue`: The uncertainty-set estimator.
  - `target`: `:mu`, `:sigma`, or `:both`; anything else throws an `ArgumentError`.
  - `ctx`: The pipeline context; requires the `returns` slot.

# Returns

  - `(res, ctx′)`: The computed result and the updated context. For `:both`, `res` is the `(mu, sigma)` [`PipelineUncertaintySets`](@ref) pair; otherwise it is the single [`AbstractUncertaintySetResult`](@ref).

# Related

  - [`run_step`](@ref)
  - [`PipelineStep`](@ref)
  - [`PipelineUncertaintySets`](@ref)
  - [`ucs`](@ref)
"""
function run_uncertainty_step(ue::AbstractUncertaintySetEstimator, target::Option{Symbol},
                              ctx::PipelineContext)
    @argcheck(target in PIPELINE_STEP_TARGETS,
              ArgumentError("the PipelineStep target of a $(typeof(ue)) step must be :mu, :sigma, or :both, got $(repr(target))"))
    require_slot(ctx, :returns, ue)
    cur = ctx.uncertainty
    res, pair = if target == :mu
        r = mu_ucs(ue, ctx.returns.X, ctx.returns.F)
        r, PipelineUncertaintySets(; mu = r, sigma = isnothing(cur) ? nothing : cur.sigma)
    elseif target == :sigma
        r = sigma_ucs(ue, ctx.returns.X, ctx.returns.F)
        r, PipelineUncertaintySets(; mu = isnothing(cur) ? nothing : cur.mu, sigma = r)
    else
        mu_set, sigma_set = ucs(ue, ctx.returns.X, ctx.returns.F)
        p = PipelineUncertaintySets(; mu = mu_set, sigma = sigma_set)
        p, p
    end
    return res, set_slot(ctx, :uncertainty, pair)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the [`UniverseSets`](@ref) a constraint-generation step needs from the universe names of the context's `returns` slot.

Every axis the returns declare is declared here: `nx` always, and `nf` whenever the returns carry factors. The factor axis is what an [`ExposureConstraintEstimator`](@ref) step resolves its names against, and taking it from `rd.nf` is what makes it agree with the loadings by construction — the columns of `rr.M` are the factors the regression was fitted on, which are the columns of `rd.F`.

Constraint estimators referencing groups beyond the plain universe names cannot be satisfied by this minimal set; precompute their result instead, or wrap a callable in a [`PipelineStep`](@ref) that supplies richer sets.

# Arguments

  - `ctx`: The pipeline context; requires the `returns` slot.
  - `est`: The step about to run, used in the error message.

# Returns

  - `sets::UniverseSets`: Universe sets whose `nx` entry holds the asset names, plus an `nf` entry holding the factor names when the returns carry them.

# Related

  - [`run_step`](@ref)
  - [`UniverseSets`](@ref)
"""
function pipeline_asset_sets(ctx::PipelineContext, est)::UniverseSets
    require_slot(ctx, :returns, est)
    nf = ctx.returns.nf
    dict = if isnothing(nf)
        Dict("nx" => ctx.returns.nx)
    else
        Dict("nx" => ctx.returns.nx, "nf" => nf)
    end
    return UniverseSets(; dict = dict)
end
"""
    add_constraint_result(ctx::PipelineContext, res::AbstractConstraintResult) -> PipelineContext
    add_constraint_result(ctx::PipelineContext, res::AbstractVector) -> PipelineContext
    add_constraint_result(ctx::PipelineContext, ::Nothing) -> PipelineContext

Append a constraint result to the `constraints` slot of the context.

The slot accumulates: the first result is stored as-is, later results widen it into a `Vector{AbstractConstraintResult}` preserving step order.

Two shapes constraint generation can return are absorbed rather than rejected, because both are ordinary outcomes of a step rather than errors:

  - `nothing`, which is what [`linear_constraints`](@ref) returns when every row was dropped — a non-`strict` run whose names were all unknown, or a re-basis the loadings annihilated. The step contributed no constraint, so the slot is left untouched; the slot's job is to carry constraints, not to re-diagnose a condition generation already decided was recoverable.
  - a vector, which is what a step wrapping a vector of estimators returns. Its elements are appended individually, so they reach [`constraint_targets`](@ref) as siblings of every other step's result rather than as a nested vector it has no case for.

# Arguments

  - `ctx`: The pipeline context.
  - `target`: The [routing target](@ref PIPELINE_ROUTING_TARGETS) the value must land in. The value is paired with it as a [`TargetedConstraint`](@ref) only when its own type does not already name that target (see [`implicit_constraint_target`](@ref)).
  - `res`: The computed value to append, `nothing`, or a vector of either.

# Returns

  - `ctx′::PipelineContext`: The updated context.

# Related

  - [`run_constraint_step`](@ref)
  - [`TargetedConstraint`](@ref)
  - [`PipelineContext`](@ref)
  - [`constraint_targets`](@ref)
"""
function add_constraint_result(ctx::PipelineContext,
                               res::AbstractConstraintResult)::PipelineContext
    cur = ctx.constraints
    val = if isnothing(cur)
        res
    elseif isa(cur, AbstractVector)
        AbstractConstraintResult[cur; res]
    else
        AbstractConstraintResult[cur, res]
    end
    return set_slot(ctx, :constraints, val)
end
function add_constraint_result(ctx::PipelineContext, target::Symbol, res)::PipelineContext
    #! A value whose own type names the target needs no wrapper, so the slot keeps the
    #! bare result wherever it can and a reader of the slot sees what generation returned.
    tagged = if implicit_constraint_target(res) === target
        res
    else
        TargetedConstraint(target, res)
    end
    return add_constraint_result(ctx, tagged)
end
function add_constraint_result(ctx::PipelineContext, ::Symbol, ::Nothing)::PipelineContext
    return ctx
end
function add_constraint_result(ctx::PipelineContext, target::Symbol,
                               res::AbstractVector)::PipelineContext
    for r in res
        ctx = add_constraint_result(ctx, target, r)
    end
    return ctx
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the value a constraint step contributes to the `constraints` slot.

One method per constraint family, each calling that family's constraint-generation verb. The value is paired with its [routing target](@ref PIPELINE_ROUTING_TARGETS) by [`run_constraint_step`](@ref); this method decides only *what* is computed, never where it lands.

# Arguments

  - `ce`: The constraint estimator.
  - `ctx`: The pipeline context.

# Returns

  - The computed value, `nothing`, or a vector of either.

# Related

  - [`run_constraint_step`](@ref)
  - [`pipe_constraint_targets`](@ref)
  - [`pipeline_asset_sets`](@ref)
"""
function constraint_step_value(ce::WeightBoundsEstimator, ctx::PipelineContext)
    return weight_bounds_constraints(ce, pipeline_asset_sets(ctx, ce))
end
function constraint_step_value(ce::LinearConstraintEstimator, ctx::PipelineContext)
    return linear_constraints(ce, pipeline_asset_sets(ctx, ce))
end
function constraint_step_value(ce::ThresholdEstimator, ctx::PipelineContext)
    return threshold_constraints(ce, pipeline_asset_sets(ctx, ce))
end
function constraint_step_value(ce::RiskBudgetEstimator, ctx::PipelineContext)
    return risk_budget_constraints(ce, pipeline_asset_sets(ctx, ce))
end
function constraint_step_value(ce::AssetSetsMatrixEstimator, ctx::PipelineContext)
    return asset_sets_matrix(ce, pipeline_asset_sets(ctx, ce))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the [routing target](@ref PIPELINE_ROUTING_TARGETS) a constraint step writes.

The family declares its targets through [`pipe_constraint_targets`](@ref) and the step's [`PipelineStep`](@ref) wrapper supplies `target`. The three outcomes are the three tuple lengths that declaration can have — no target means the family is not a step, one means the step needs no annotation, several mean it must carry one.

# Arguments

  - `ce`: The constraint estimator.
  - `target`: The step's declared target, or `nothing` when it carries no annotation.

# Returns

  - `target::Symbol`: The resolved routing target.

# Related

  - [`pipe_constraint_targets`](@ref)
  - [`run_constraint_step`](@ref)
"""
function resolve_constraint_target(ce::AbstractConstraintEstimator,
                                   target::Option{Symbol})::Symbol
    ts = pipe_constraint_targets(ce)
    name = Base.typename(typeof(ce)).wrapper
    @argcheck(!isempty(ts),
              ArgumentError("a $name computes no value for the constraints slot, so it is not a pipeline step; precompute its result and pass it to the optimiser, or wrap a callable in a PipelineStep that writes :constraints"))
    if isnothing(target)
        @argcheck(length(ts) == 1,
                  ArgumentError("a $name step must declare which routing target it writes, because its result names $(length(ts)) of them; wrap it in a PipelineStep with target = one of $ts"))
        return ts[1]
    end
    @argcheck(target in ts,
              ArgumentError("the PipelineStep target of a $name step must be one of $ts, got :$target"))
    return target
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Execute a constraint step and append its value, paired with its [routing target](@ref PIPELINE_ROUTING_TARGETS), to the `constraints` slot.

Every constraint family that computes a value is a step, and every value it computes has a declared destination — that is the whole of the rule, and [`pipe_constraint_targets`](@ref) is where it is written down. The destination travels with the value as a [`TargetedConstraint`](@ref) rather than being re-derived from the result type at injection, so a family whose result type names no unique field (a [`Threshold`](@ref), a centrality [`LinearConstraint`](@ref), an asset-sets matrix) routes as cleanly as one whose does.

# Arguments

  - `ce`: The constraint estimator.
  - `target`: The [`PipelineStep`](@ref) target, or `nothing` for an unwrapped step.
  - `ctx`: The pipeline context.

# Returns

  - `(res, ctx′)`: The computed value and the updated context. `res` is the bare value, so what a fitted pipeline reports is the constraint result itself rather than the routing wrapper.

# Related

  - [`run_step`](@ref)
  - [`resolve_constraint_target`](@ref)
  - [`constraint_step_value`](@ref)
  - [`add_constraint_result`](@ref)
"""
function run_constraint_step(ce::AbstractConstraintEstimator, target::Option{Symbol},
                             ctx::PipelineContext)
    target = resolve_constraint_target(ce, target)
    res = constraint_step_value(ce, ctx)
    return res, add_constraint_result(ctx, target, res)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute an [`ExposureConstraintEstimator`](@ref) step's value: re-base its rows through the loadings of the *pipeline's* prior into an asset-space [`LinearConstraint`](@ref).

This is the one constraint step that reads a computed slot other than `:returns`. The basis is `ctx.prior.rr`, so a prior step must come earlier; a prior that carries no regression makes [`constraint_space_basis`](@ref) throw, which is the intended failure — see [`ExposureConstraintEstimator`](@ref).

`ctx.returns` is passed as well, so a space that names a regression **estimator** refits the loadings here rather than throwing. That is the one arrangement in which a prior carrying no factor block still admits a factor mandate — see [`FactorSpace`](@ref).

!!! warning

    The constraint is **pinned to the pipeline's prior**. Its rows were projected through the loadings this step saw, and a downstream optimiser that refits its own prior does not re-project them. Passing the estimator to the optimiser's `lcse` field instead recomputes the projection with the optimiser's own prior, per fold, which is what a cross-validated factor mandate needs.

# Related

  - [`run_constraint_step`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`linear_constraints`](@ref)
  - [`pipeline_asset_sets`](@ref)
"""
function constraint_step_value(ce::ExposureConstraintEstimator, ctx::PipelineContext)
    require_slot(ctx, :prior, ce)
    return linear_constraints(ce, pipeline_asset_sets(ctx, ce); rr = ctx.prior.rr,
                              rd = ctx.returns)
end
function constraint_step_value(ce::AbstractPhylogenyConstraintEstimator,
                               ctx::PipelineContext)
    require_slot(ctx, :returns, ce)
    return phylogeny_constraints(ce, ctx.returns)
end
function constraint_step_value(ce::AbstractCentralityConstraint, ctx::PipelineContext)
    require_slot(ctx, :returns, ce)
    return centrality_constraints(ce, ctx.returns)
end
function run_step(ce::AbstractConstraintEstimator, ctx::PipelineContext)
    return run_constraint_step(ce, nothing, ctx)
end
pipe_reads(::PricesToReturns) = (:prices,)
pipe_writes(::PricesToReturns) = :returns
function run_step(ptr::PricesToReturns, ctx::PipelineContext)
    require_slot(ctx, :prices, ptr)
    return ptr, set_slot(ctx, :returns, apply_preprocessing(ptr, ctx.prices))
end
"""
    pipe_writes(::TrainTestSplit) = :split
    pipe_reads(::TrainTestSplit) = ()

A [`TrainTestSplit`](@ref) narrows whichever data slot the pipeline input filled, so the slot it writes is not a property of its type.

`:split` is a sentinel, deliberately *not* a member of [`PIPELINE_SLOTS`](@ref): it names the step (`pipe.names` reads `"split"`), invalidates nothing, and satisfies nothing. That is sound only because a split is pinned to the first position of a [`Pipeline`](@ref), where both data slots are already available from the input and no derived slot exists to invalidate. Which data slot is actually rewritten — `:prices` or `:returns` — is decided at run time by [`run_step`](@ref).

# Related

  - [`TrainTestSplit`](@ref)
  - [`assert_split_position`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
"""
pipe_writes(::TrainTestSplit) = :split
pipe_reads(::TrainTestSplit) = ()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Execute a [`TrainTestSplit`](@ref) step: replace the input data slot with the training window, and return the fitted [`TrainTestSplitResult`](@ref) holding both windows.

The split runs at whichever level the pipeline input supplied — `prices` when the pipeline was fed price-level data, `returns` otherwise — so the same estimator serves both. Every later step therefore fits on the training window alone, which is the whole point of pinning the split to the first position.

# Related

  - [`TrainTestSplit`](@ref)
  - [`fit_predict`](@ref)
  - [`run_step`](@ref)
"""
function run_step(tts::TrainTestSplit, ctx::PipelineContext)
    slot = !isnothing(ctx.prices) ? :prices : :returns
    require_slot(ctx, slot, tts)
    res = fit_preprocessing(tts, getproperty(ctx, slot))
    return res, set_slot(ctx, slot, res.train)
end
