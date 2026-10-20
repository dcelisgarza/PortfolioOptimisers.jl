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

A step that reads the returns reads them from the source that its own reader states in its `x_src` field, as the same reader does in the slot of an optimiser. A [`PipelineStep`](@ref) wrapper holds no source.

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
    res = clusterise(cle, ctx.returns; rd = ctx.returns)
    return res, set_slot(ctx, :phylogeny, res)
end
function run_step(ne::AbstractNetworkEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, ne)
    res = phylogeny_matrix(ne, ctx.returns; rd = ctx.returns)
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
    return run_step(ps.est, ps, ctx)
end
# The estimator of the step selects the method. A callable writes its value to the slot the
# step names, a set or a constraint estimator reads the target of the step, and every other
# estimator runs as a bare step.
function run_step(f::Function, ps::PipelineStep, ctx::PipelineContext)
    val = f(ctx)
    return val, set_slot(ctx, ps.writes, val)
end
function run_step(ue::AbstractUncertaintySetEstimator, ps::PipelineStep,
                  ctx::PipelineContext)
    return run_uncertainty_step(ue, ps.target, ctx)
end
function run_step(ce::AbstractConstraintEstimator, ps::PipelineStep, ctx::PipelineContext)
    return run_constraint_step(ce, ps.target, ctx)
end
function run_step(est, ::PipelineStep, ctx::PipelineContext)
    return run_step(est, ctx)
end
function run_step(ue::AbstractUncertaintySetEstimator, ::PipelineContext)
    return throw(ArgumentError("a $(typeof(ue)) step must declare which parameter it bounds; wrap it in a PipelineStep with target = :mu, target = :sigma, or target = :both"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the slot an uncertainty-set step fits from, as the arguments its verb takes.

An estimator that reads a prior result — [`reads_prior_result`](@ref) answers `true` — is fitted from the `prior` slot, and never touches the returns; every other estimator is fitted from the `returns` slot, whose `X` and `F` its verb takes. The estimator reads the source that its own prior states, [`returns_source`](@ref) of `ue.pe`, as the verb over a [`ReturnsResult`](@ref) reads it. The observations of a Shared Series gap that the fit reads are dropped first with [`shared_series_view`](@ref): a gap of a vector benchmark under `ExcessSource()`, and a gap of the factor returns when the prior of the estimator reads them. When the optimiser receives the set, the `returns` slot already has the gaps of its group dropped, see [`run_group_step`](@ref), so a raw prior in an excess group reads the same observations as the other readers of the group. The asset returns are then read with [`benchmark_relative_returns`](@ref) under the same source. The required slot is checked here, so a missing one is refused with an error that names it before any fit. The predicate reads the type, so the branch folds.

# Arguments

  - `ue`: The uncertainty-set estimator.
  - `ctx`: The pipeline context.

# Validation

  - The slot the estimator reads is filled, through [`require_slot`](@ref).
  - The rules of [`benchmark_relative_returns`](@ref) and [`shared_series_view`](@ref).

# Returns

  - `src::Tuple`: `(ctx.prior,)` for a prior-reading estimator, `(X, F)` otherwise, where `X` is the asset returns from the source of `ue.pe` and both hold the observations that the fit keeps.

# Related

  - [`run_uncertainty_step`](@ref)
  - [`reads_prior_result`](@ref)
  - [`require_slot`](@ref)
"""
function uncertainty_step_source(ue::AbstractUncertaintySetEstimator, ctx::PipelineContext)
    if reads_prior_result(ue)
        require_slot(ctx, :prior, ue)
        return (ctx.prior,)
    end
    require_slot(ctx, :returns, ue)
    x_src = returns_source(ue.pe)
    rd = shared_series_view(ctx.returns, x_src, reads_factor_returns(ue.pe))
    return (benchmark_relative_returns(rd, x_src), rd.F)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Execute an uncertainty-set step pinned to a target and merge its result into the `uncertainty` slot.

The target comes from the [`PipelineStep`](@ref) wrapper:

  - `:mu` computes [`mu_ucs`](@ref) and fills the mean half.
  - `:sigma` computes [`sigma_ucs`](@ref) and fills the covariance half.
  - `:both` computes [`ucs`](@ref), which derives *both* halves from one fit — sharing the prior and, for the sampling algorithms, the simulation draws — and is therefore cheaper than the two narrowed calls.

A narrowed step fills its half of the [`PipelineUncertaintySets`](@ref) pair and leaves the other untouched, so separate `:mu` and `:sigma` steps compose. Every populated half must reach the optimiser: each becomes its own [routing target](@ref PIPELINE_ROUTING_TARGETS) — `:mu_ucs` and `:sigma_ucs` — and neither is [optional](@ref PIPELINE_OPTIONAL_TARGETS), so a set that cannot be routed is rejected rather than dropped and `:both` requires an optimiser with an [`ArithmeticReturn`](@ref) *and* an [`UncertaintySetVariance`](@ref) risk measure.

Which slot the step reads is decided by [`reads_prior_result`](@ref). An estimator with a prior of its own is fitted from the `returns` slot. One that reads a prior result — an [`AbstractPriorUncertaintySetEstimator`](@ref), or a returns-data estimator with `pe = nothing` — is fitted from the `prior` slot instead, so a prior step must come earlier; the returns are not read at all, so a pipeline that writes `:prior` from a precomputed result needs no `:returns` slot for such a step.

# Arguments

  - `ue`: The uncertainty-set estimator.
  - `target`: `:mu`, `:sigma`, or `:both`; anything else throws an `ArgumentError`.
  - `ctx`: The pipeline context; requires the `returns` slot, or the `prior` slot for an estimator that reads a prior result.

# Returns

  - `(res, ctx′)`: The computed result and the updated context. For `:both`, `res` is the `(mu, sigma)` [`PipelineUncertaintySets`](@ref) pair; otherwise it is the single [`AbstractUncertaintySetResult`](@ref).

# Related

  - [`run_step`](@ref)
  - [`PipelineStep`](@ref)
  - [`PipelineUncertaintySets`](@ref)
  - [`uncertainty_step_source`](@ref)
  - [`reads_prior_result`](@ref)
  - [`AbstractPriorUncertaintySetEstimator`](@ref)
  - [`ucs`](@ref)
"""
function run_uncertainty_step(ue::AbstractUncertaintySetEstimator, target::Option{Symbol},
                              ctx::PipelineContext)
    @argcheck(target in (:mu, :sigma, :both),
              ArgumentError("the PipelineStep target of a $(typeof(ue)) step must be :mu, :sigma, or :both, got $(repr(target))"))
    src = uncertainty_step_source(ue, ctx)
    cur = ctx.uncertainty
    res, pair = if target == :mu
        r = mu_ucs(ue, src...)
        r, PipelineUncertaintySets(; mu = r, sigma = isnothing(cur) ? nothing : cur.sigma)
    elseif target == :sigma
        r = sigma_ucs(ue, src...)
        r, PipelineUncertaintySets(; mu = isnothing(cur) ? nothing : cur.mu, sigma = r)
    else
        mu_set, sigma_set = ucs(ue, src...)
        p = PipelineUncertaintySets(; mu = mu_set, sigma = sigma_set)
        p, p
    end
    return res, set_slot(ctx, :uncertainty, pair)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the [`UniverseSets`](@ref) a constraint-generation step needs from the universe names of the context's `returns` slot.

Every axis the returns declare is declared here: `nx` always, and `nf` whenever the returns carry factors. `nf` is the default `tfkey`, the **time-series** factor axis, and that is the right axis because the returns carry `F` and nothing else: a cross-sectional model's factors are the exposures its own fit was built from, and no returns result carries them. An [`ExposureConstraintEstimator`](@ref) step resolves its names against this axis, and taking it from `rd.nf` is what makes it agree with the loadings by construction — the columns of `rr.M` are the factors the regression was fitted on, which are the columns of `rd.F`. A step whose loadings are a [`CrossSectionalFactorModel`](@ref) therefore finds no `cfkey` entry here and says so through [`factor_universe`](@ref); such a mandate needs sets a caller supplies.

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

A family that reads the returns reads them from the source that its own reader states, as it does in the slot of an optimiser. The phylogeny and centrality families pass `ctx.returns` to their verb, which reads the `x_src` of the estimator. An exposure constraint refits through the regression of its [`FactorSpace`](@ref), which states its own source.

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
    return resolve_constraint_target(target, ts, name)
end
# An unstated target is the one target the estimator names, and a stated one must be among
# them.
function resolve_constraint_target(::Nothing, ts, name)::Symbol
    @argcheck(length(ts) == 1,
              ArgumentError("a $name step must declare which routing target it writes, because its result names $(length(ts)) of them; wrap it in a PipelineStep with target = one of $ts"))
    return ts[1]
end
function resolve_constraint_target(target::Symbol, ts, name)::Symbol
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

`ctx.returns` is passed as well, so a space that names a regression **estimator** refits the loadings here rather than throwing. That is the one arrangement in which a prior carrying no factor block still admits a factor mandate — see [`FactorSpace`](@ref). The refit reads the returns from the source that the regression of the space states, as it does in the `lcse` field of an optimiser.

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
    return phylogeny_constraints(ce, ctx.returns; rd = ctx.returns)
end
function constraint_step_value(ce::AbstractCentralityConstraint, ctx::PipelineContext)
    require_slot(ctx, :returns, ce)
    return centrality_constraints(ce, ctx.returns; rd = ctx.returns)
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
"""
    reader_frames(ps::PipelineStep, path::AbstractString, refit::Bool) -> ReturnsFrames

Return the frames that the estimator of a [`PipelineStep`](@ref) adds to the group of a [`Pipeline`](@ref).

The wrapper holds no source. The estimator that it wraps states its own, so the walk enters the field `est`.

# Arguments

  - `ps`: The step wrapper.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of `ps.est`, at the path `"<path>.est"`.

# Related

  - [`reader_frames`](@ref)
  - [`pipeline_frames`](@ref)
"""
function reader_frames(ps::PipelineStep, path::AbstractString, refit::Bool)
    return reader_frames(ps.est, join_path(path, :est), refit)
end
"""
    step_prior(step) -> Option{<:AbstractPriorEstimator}

Return the prior estimator that a step of a [`Pipeline`](@ref) fits into the `prior` slot, or `nothing`.

The answer sees through a [`PipelineStep`](@ref) and an [`Online`](@ref) wrapper. A nested `Pipeline` answers for its last step. A callable and a precomputed Prior Result state no source, so they answer `nothing`.

Internal machinery — not part of the user-facing API.

# Arguments

  - `step`: A step of a `Pipeline`.

# Returns

  - `pe::Option{<:AbstractPriorEstimator}`: The prior estimator of the step, or `nothing`.

# Related

  - [`pipeline_frames`](@ref)
  - [`assert_pipeline_data_benchmark`](@ref)
"""
step_prior(::Any) = nothing
step_prior(pe::AbstractPriorEstimator) = pe
step_prior(ps::PipelineStep) = step_prior(ps.est)
step_prior(o::Online) = step_prior(o.est)
"""
    pipe_step_targets(step) -> Tuple

Return the [routing targets](@ref PIPELINE_ROUTING_TARGETS) that the value of a step of a [`Pipeline`](@ref) fills in the last optimiser.

[`inject_context`](@ref) puts the `prior` slot into `pe`, and a clustering result in the `phylogeny` slot into `cle`. Every other target comes from [`pipe_required_targets`](@ref). A network step writes a phylogeny matrix, which reaches no target.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. A step that writes `:prior` fills `:pe`.
 2. A step that writes `:phylogeny` fills `:cle` when it is a clustering estimator or a clustering result, and nothing otherwise.
 3. Every other step fills the targets that [`pipe_required_targets`](@ref) gives.

# Arguments

  - `step`: A step of a `Pipeline`.

# Returns

  - `targets::Tuple`: The routing targets of the value of the step.

# Related

  - [`pipe_target_paths`](@ref)
  - [`pipeline_frames`](@ref)
"""
function pipe_step_targets(step)
    slot = pipe_writes(step)
    return if slot === :prior
        (:pe,)
    elseif slot === :phylogeny
        est = isa(step, PipelineStep) ? step.est : step
        if isa(est, Union{<:AbstractClustersEstimator, <:AbstractClusteringResult})
            (:cle,)
        else
            ()
        end
    else
        pipe_required_targets(step)
    end
end
"""
    pipe_target_paths(x, ::Val{target}) -> Vector{String}

Return the paths, from the optimiser `x`, of the slots that a value at a [routing target](@ref PIPELINE_ROUTING_TARGETS) replaces.

The walk follows [`pipe_route`](@ref). A target goes into the field of the same name, or into the configuration that [`@pipe_delegates`](@ref) names. `:mu_ucs` goes into `ret.ucs` of a [`JuMPOptimiser`](@ref), and `:sigma_ucs` goes into the `ucs` of each [`UncertaintySetVariance`](@ref) in the field `r`. The frame walk of a [`Pipeline`](@ref) leaves these slots out of the group of the optimiser, because the value of the step replaces the reader that they hold.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. `:sigma_ucs` on a value with a field `r`: the path `"r.ucs"`, or `"r[i].ucs"` for each element `i` of a vector that is an `UncertaintySetVariance`.
 2. A target with a field of the same name in `x`: the name of the field.
 3. Otherwise the paths in the configuration of `x`, after its field name, see [`pipe_config_paths`](@ref).

# Arguments

  - `x`: The optimiser or the optimiser configuration.
  - `::Val{target}`: The routing target.

# Returns

  - `paths::Vector{String}`: The paths of the replaced slots. It is empty when `x` has no slot for the target.

# Related

  - [`pipe_step_targets`](@ref)
  - [`pipe_route`](@ref)
  - [`pipeline_frames`](@ref)
"""
function pipe_target_paths(x, t::Val{target})::Vector{String} where {target}
    return hasfield(typeof(x), target) ? [string(target)] : pipe_config_paths(x, t)
end
function pipe_target_paths(::JuMPOptimiser, ::Val{:mu_ucs})::Vector{String}
    return ["ret.ucs"]
end
function pipe_target_paths(x, t::Val{:sigma_ucs})::Vector{String}
    if !hasfield(typeof(x), :r)
        return pipe_config_paths(x, t)
    end
    r = getfield(x, :r)
    return if isa(r, UncertaintySetVariance)
        ["r.ucs"]
    elseif isa(r, AbstractVector)
        [string("r[", i, "].ucs")
         for (i, y) in enumerate(r) if isa(y, UncertaintySetVariance)]
    else
        String[]
    end
end
"""
    pipe_config_paths(x, t::Val) -> Vector{String}

Return the paths of [`pipe_target_paths`](@ref) in the configuration of `x`, after the name of the field that holds it, or an empty vector when `x` delegates to no configuration.

Internal machinery — not part of the user-facing API.

# Arguments

  - `x`: The optimiser.
  - `t`: The routing target.

# Returns

  - `paths::Vector{String}`: The paths from `x`.

# Related

  - [`pipe_target_paths`](@ref)
  - [`pipe_config_field`](@ref)
"""
function pipe_config_paths(x, t::Val)::Vector{String}
    f = pipe_config_field(x)
    return if isnothing(f)
        String[]
    else
        [join_path(string(f), p) for p in pipe_target_paths(getfield(x, f), t)]
    end
end
"""
    pipe_prior_estimator(x)

Return the prior that the optimiser `x` holds in the field that receives the `:pe` routing target, or `nothing` when it has none.

The walk follows [`pipe_route`](@ref): the field `pe` of `x`, or of the configuration that [`@pipe_delegates`](@ref) names.

Internal machinery — not part of the user-facing API.

# Arguments

  - `x`: The optimiser or the optimiser configuration.

# Returns

  - `pe`: The prior estimator, the Prior Result or the schedule in the field, or `nothing`.

# Related

  - [`refits_regression`](@ref)
  - [`pipeline_frames`](@ref)
"""
function pipe_prior_estimator(x)
    if hasfield(typeof(x), :pe)
        return getfield(x, :pe)
    end
    f = pipe_config_field(x)
    return isnothing(f) ? nothing : pipe_prior_estimator(getfield(x, f))
end
"""
    cut_frames(x, path::AbstractString, refit::Bool, cut::AbstractVector{<:AbstractString}) -> ReturnsFrames

Return the frames that `x` adds to a group, without the slots at the paths in `cut`.

The walk of a [`Pipeline`](@ref) calls this function on its last optimiser. A step value replaces each slot in `cut`, so the reader that the slot holds never runs. The function enters only the values on the way to a cut slot, field by field and element by element, and gives every other value to [`reader_frames`](@ref). The optimiser itself is at the empty path, and it always enters, because a head has no method of `reader_frames`.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. A value at a path in `cut` adds nothing.
 2. A value at a path that no cut path starts below adds `reader_frames(x, path, refit)`.
 3. Otherwise join the frames of each element of a vector, or of each field, with [`cut_children`](@ref).

# Arguments

  - $(arg_dict[:frame_x])
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])
  - `cut`: The paths of the replaced slots, from [`pipe_target_paths`](@ref).

# Returns

  - `fr::ReturnsFrames`: The frames of `x` without the cut slots.

# Related

  - [`pipeline_frames`](@ref)
  - [`reader_frames`](@ref)
  - [`merge_frames`](@ref)
"""
function cut_frames(x, path::AbstractString, refit::Bool,
                    cut::AbstractVector{<:AbstractString})::ReturnsFrames
    if path in cut
        return ReturnsFrames()
    end
    below = any(c -> isempty(path) ||
                     startswith(c, string(path, ".")) ||
                     startswith(c, string(path, "[")), cut)
    if !below
        return reader_frames(x, path, refit)
    end
    return merge_frames(cut_children(x, path, refit, cut)...)
end
"""
    cut_children(x, path::AbstractString, refit::Bool, cut::AbstractVector{<:AbstractString})
    cut_children(x::AbstractVector, path::AbstractString, refit::Bool,
                 cut::AbstractVector{<:AbstractString})

Return the frames of each field of `x`, or of each element of a vector, through [`cut_frames`](@ref).

[`cut_frames`](@ref) calls this function on a value on the way to a cut slot. A field is at the path that [`join_path`](@ref) gives, and an element is at `"<path>[i]"`, as [`reader_frames`](@ref) names it.

Internal machinery — not part of the user-facing API.

# Arguments

  - $(arg_dict[:frame_x])
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])
  - `cut`: The paths of the replaced slots.

# Returns

  - `fs::Vector{ReturnsFrames}`: The frames of each field or element, in order.

# Related

  - [`cut_frames`](@ref)
"""
function cut_children(x, path::AbstractString, refit::Bool,
                      cut::AbstractVector{<:AbstractString})
    return ReturnsFrames[cut_frames(getfield(x, f), join_path(path, f), refit, cut)
                         for f in fieldnames(typeof(x))]
end
function cut_children(x::AbstractVector, path::AbstractString, refit::Bool,
                      cut::AbstractVector{<:AbstractString})
    return ReturnsFrames[cut_frames(y, string(path, "[", i, "]"), refit, cut)
                         for (i, y) in enumerate(x)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the optimiser of the last step of a [`Pipeline`](@ref), or `nothing` when the pipeline has none that a construction check can read.

The checks that compare the earlier steps with the optimiser read it here: [`assert_routable`](@ref), [`assert_pipeline_data_benchmark`](@ref) and [`pipeline_frames`](@ref).

# Algorithm

 1. Read the optimiser from the last step, without its [`PipelineStep`](@ref) wrapper.
 2. Return `nothing` when the pipeline has one step, or when the optimiser is not an [`OptimisationEstimator`](@ref). A [`TimeDependent`](@ref) schedule and a precomputed result are such steps, and the fold loop resolves them later. The type is checked first, because [`fit`](@ref) also reads the optimiser for the gap drop of the group, see [`pipeline_gap_sources`](@ref), and in a fold of a mixed schedule the last step is a precomputed result, whose written slot [`pipe_writes`](@ref) cannot infer.
 3. Return the optimiser when its step writes `:opt`, and `nothing` otherwise.

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Returns

  - `opt::Option{<:OptimisationEstimator}`: The optimiser, or `nothing`.

# Related

  - [`assert_routable`](@ref)
  - [`assert_pipeline_data_benchmark`](@ref)
  - [`pipeline_frames`](@ref)
"""
function pipe_terminal_optimiser(ests)
    n = length(ests)
    opt = isa(ests[n], PipelineStep) ? ests[n].est : ests[n]
    if n < 2 || !isa(opt, OptimisationEstimator)
        return nothing
    end
    return pipe_writes(ests[n]) === :opt ? opt : nothing
end
"""
    pipeline_frames(ests) -> ReturnsFrames

Return the frames of the group that the steps of a [`Pipeline`](@ref) join: the group of its last optimiser.

[`inject_context`](@ref) puts the value of each step into a slot of the last optimiser, so the step reads the returns for that optimiser. The steps whose values the optimiser receives join its group, and the readers that those values replace leave it. A network step writes a phylogeny matrix that no slot receives, so it stays out. A prior step decides `refit` for the optimiser when the optimiser receives its value, because that prior then replaces the prior of the optimiser. A step reads the `prior` slot, so the last prior step decides `refit` for the steps. With no prior step a step reads a Prior Result that the input gave, which reads as raw. [`group_frames`](@ref) of a `Pipeline` calls this function.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. Read the optimiser `opt` of the last step with [`pipe_terminal_optimiser`](@ref). Answer that the group holds no reader when there is none.
 2. For each earlier step, keep the targets of [`pipe_step_targets`](@ref) that `opt` accepts ([`pipe_accepts`](@ref)). The value of a step with no such target does not reach `opt`.
 3. Find `refit` for the steps and for `opt` with [`pipeline_refits`](@ref).
 4. Walk each step that `opt` receives with [`reader_frames`](@ref), at the path `"steps[i]"`.
 5. Walk `opt` without the slots that the steps replace with [`terminal_frames`](@ref), and join every part with [`merge_frames`](@ref).

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Returns

  - `fr::ReturnsFrames`: The frames of the group of the last optimiser, with the steps that it receives.

# Related

  - [`group_frames`](@ref)
  - [`warn_raw_readers`](@ref)
  - [`pipeline_refits`](@ref)
  - [`terminal_frames`](@ref)
  - [`pipe_step_targets`](@ref)
"""
function pipeline_frames(ests)::ReturnsFrames
    opt = pipe_terminal_optimiser(ests)
    if isnothing(opt)
        return ReturnsFrames()
    end
    n = length(ests)
    received = pipeline_received(ests, opt)
    step_refit, opt_refit = pipeline_refits(ests, received, opt)
    fs = ReturnsFrames[reader_frames(ests[i], string("steps[", i, "]"), step_refit)
                       for i in 1:(n - 1) if !isempty(received[i])]
    push!(fs, terminal_frames(ests[n], n, opt, received, opt_refit))
    return merge_frames(fs...)
end
"""
    pipeline_received(ests, opt) -> Vector{Vector{Symbol}}

Return, for each step of a [`Pipeline`](@ref) before the last, the routing targets of the step that the last optimiser `opt` accepts.

A step with no such target does not reach `opt`, so it stays out of the group of `opt`.

Internal machinery — not part of the user-facing API.

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).
  - `opt`: The optimiser of the last step.

# Returns

  - `received::Vector{Vector{Symbol}}`: The targets that `opt` receives from each earlier step.

# Related

  - [`pipeline_frames`](@ref)
  - [`pipeline_gap_sources`](@ref)
  - [`pipe_step_targets`](@ref)
  - [`pipe_accepts`](@ref)
"""
function pipeline_received(ests, opt)::Vector{Vector{Symbol}}
    return [Symbol[t for t in pipe_step_targets(ests[i]) if pipe_accepts(opt, Val(t))]
            for i in 1:(length(ests) - 1)]
end
"""
    pipeline_gap_sources(ests) -> Vector{Option{ExcessSource}}

Return, for each step of a [`Pipeline`](@ref), the source under which the Pipeline drops a gap of a vector benchmark before the step runs.

The steps whose values the last optimiser receives join its group, see [`pipeline_frames`](@ref). A head drops a gap of a vector benchmark when any reader of its group states [`ExcessSource`](@ref), see [`group_source`](@ref), so every reader of the head pairs the same observations. The Pipeline applies the same rule: each step of the group, and the optimiser, reads the returns data with the gaps of the group dropped. A raw reader of the group loses the observations too. The rule decides which observations a step reads, and not its frame: each reader still reads the frame that its own `x_src` states. A step outside the group gets `nothing`, so it drops by the rule of its own reader alone.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. Read the optimiser `opt` of the last step with [`pipe_terminal_optimiser`](@ref). Answer `nothing` for every step when there is none.
 2. Find the source of the group with [`group_source`](@ref) of [`pipeline_frames`](@ref), giving `gsrc`.
 3. Answer `gsrc` for each step that `opt` receives ([`pipeline_received`](@ref)) and for the last step, and `nothing` for every other step.

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Returns

  - `gsrcs::Vector{Option{ExcessSource}}`: One source for each step.

# Related

  - [`run_group_step`](@ref)
  - [`pipeline_frames`](@ref)
  - [`group_source`](@ref)
"""
function pipeline_gap_sources(ests)::Vector{Option{ExcessSource}}
    n = length(ests)
    gsrcs = Vector{Option{ExcessSource}}(nothing, n)
    opt = pipe_terminal_optimiser(ests)
    if isnothing(opt)
        return gsrcs
    end
    gsrc = group_source(pipeline_frames(ests))
    received = pipeline_received(ests, opt)
    for i in 1:(n - 1)
        if !isempty(received[i])
            gsrcs[i] = gsrc
        end
    end
    gsrcs[n] = gsrc
    return gsrcs
end
"""
    run_group_step(step, ctx::PipelineContext, gsrc::Nothing) -> (fitted, ctx′)
    run_group_step(step, ctx::PipelineContext, gsrc::ExcessSource) -> (fitted, ctx′)

Run one step of a [`Pipeline`](@ref) with the gaps of the group of its optimiser dropped from the returns data that the step reads.

The step reads the `returns` slot with each observation at which a vector benchmark has a gap dropped, through [`shared_series_view`](@ref) under `gsrc`. The returns data is then put back in the slot, so the next step, and the context of the [`PipelineResult`](@ref), hold the returns data that the caller gave. A step of the group never writes the `returns` slot, because the optimiser receives no Data Slot. The reader of the step then drops a gap of the factor returns, and a gap of the benchmark under its own source, as it does in a head. Both drops are idempotent on a view with no gap left.

Internal machinery — not part of the user-facing API.

# Algorithm

The method that Julia selects is the algorithm.

 1. `gsrc` is `nothing`: run the step with [`run_step`](@ref).
 2. `gsrc` is [`ExcessSource`](@ref): view the `returns` slot with [`group_returns_view`](@ref), run the step on that context, and put the returns data back.

# Arguments

  - `step`: The step, after [`maybe_inject_step`](@ref).
  - `ctx`: The pipeline context.
  - `gsrc`: The source of the gap drop of the step, see [`pipeline_gap_sources`](@ref).

# Returns

  - `(fitted, ctx′)`: The fitted object and the updated context.

# Related

  - [`pipeline_gap_sources`](@ref)
  - [`run_step`](@ref)
  - [`shared_series_view`](@ref)
"""
function run_group_step(step, ctx::PipelineContext, ::Nothing)
    return run_step(step, ctx)
end
function run_group_step(step, ctx::PipelineContext, gsrc::ExcessSource)
    rd = ctx.returns
    fitted, ctx = run_step(step, set_slot(ctx, :returns, group_returns_view(rd, gsrc)))
    return fitted, set_slot(ctx, :returns, rd)
end
"""
    group_returns_view(rd::ReturnsResult, gsrc::ExcessSource) -> ReturnsResult
    group_returns_view(rd, gsrc::ExcessSource)

Return the returns data that a step of the group of a [`Pipeline`](@ref) reads: `rd` with each gap of a vector benchmark dropped, see [`shared_series_view`](@ref).

A slot that holds no [`ReturnsResult`](@ref), such as an empty slot before a step that reads prices, is returned unchanged.

Internal machinery — not part of the user-facing API.

# Related

  - [`run_group_step`](@ref)
  - [`shared_series_view`](@ref)
"""
function group_returns_view(rd::ReturnsResult, gsrc::ExcessSource)
    return shared_series_view(rd, gsrc, false)
end
function group_returns_view(rd, ::ExcessSource)
    return rd
end
"""
    pipeline_refits(ests, received, opt) -> Tuple{Bool, Bool}

Return `refit` for the steps of a [`Pipeline`](@ref) and `refit` for its last optimiser, see [`refits_regression`](@ref).

A step reads the `prior` slot, so the last prior step decides `refit` for the steps. With no prior step a step reads a Prior Result that the input gave, which reads as raw. A prior step decides `refit` for the optimiser when the optimiser receives its value, because that prior then replaces the prior of the optimiser.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. Find the last prior step with [`step_prior`](@ref). `refit` for the steps is `refits_regression` of its prior, or `false` when there is none.
 2. Find the last step whose targets in `received` hold `:pe`. `refit` for `opt` is `refits_regression` of its prior, or of the prior of `opt` ([`pipe_prior_estimator`](@ref)) when there is none.

# Arguments

  - `ests`: The step estimators, with the optimisation step last.
  - `received`: The routing targets that `opt` receives from each earlier step.
  - `opt`: The optimiser of the last step.

# Returns

  - `(step_refit, opt_refit)::Tuple{Bool, Bool}`: `refit` for the steps and for `opt`.

# Related

  - [`pipeline_frames`](@ref)
  - [`refits_regression`](@ref)
"""
function pipeline_refits(ests, received, opt)::Tuple{Bool, Bool}
    n = length(ests)
    kp = findlast(i -> !isnothing(step_prior(ests[i])), 1:(n - 1))
    kpe = findlast(i -> :pe in received[i], 1:(n - 1))
    step_refit = !isnothing(kp) && refits_regression(step_prior(ests[kp]))
    opt_refit = refits_regression(if isnothing(kpe)
                                      pipe_prior_estimator(opt)
                                  else
                                      step_prior(ests[kpe])
                                  end)
    return step_refit, opt_refit
end
"""
    terminal_frames(step, n::Integer, opt, received, refit::Bool) -> ReturnsFrames

Return the frames of the last optimiser of a [`Pipeline`](@ref), without the slots that the earlier steps replace, at the path of the last step.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. Find the paths of the slots that the earlier steps replace with [`pipe_target_paths`](@ref).
 2. When there are none, take [`group_frames`](@ref) of `opt`. Otherwise walk `opt` without those slots with [`cut_frames`](@ref).
 3. Put the path of the last step, `"steps[n]"` or `"steps[n].est"` for a [`PipelineStep`](@ref), before each raw path.

# Arguments

  - `step`: The last step, `opt` or its wrapper.
  - `n`: The position of the last step.
  - `opt`: The optimiser of the last step.
  - `received`: The routing targets that `opt` receives from each earlier step.
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of `opt`, with paths from the pipeline.

# Related

  - [`pipeline_frames`](@ref)
  - [`cut_frames`](@ref)
"""
function terminal_frames(step, n::Integer, opt, received, refit::Bool)::ReturnsFrames
    cut = String[p for r in received for t in r for p in pipe_target_paths(opt, Val(t))]
    fo = isempty(cut) ? group_frames(opt) : cut_frames(opt, "", refit, cut)
    head = string("steps[", n, "]", isa(step, PipelineStep) ? ".est" : "")
    return ReturnsFrames(fo.excess, String[join_path(head, p) for p in fo.raw])
end
"""
    pipe_data_benchmark_path(x) -> Option{<:String}

Return the path of the first [`DataBenchmarkTracking`](@ref) in a measure that reads the prior of the optimiser `x`, or `nothing`.

Every tracking form compares `pr.X * w` with its benchmark series, so the prior that the measure reads decides its frame. The measures that read the prior of a head are the tracking errors `tr` of its configuration and the risk measures `r`, `ri` and `ro` of the head. The walk follows the path of the `:pe` routing target through [`pipe_config_field`](@ref), so it reads the fields that hold the prior and the fields beside them. It does not enter an inner optimiser or a fallback, because each one checks its own prior.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. For each of the fields `tr`, `r`, `ri` and `ro` of `x`, in order, return the path that [`data_benchmark_path`](@ref) finds below it, after the name of the field.
 2. Otherwise return the answer for the configuration of `x`, after the name of its field. An `x` with no configuration answers for `nothing`, which is `nothing`.

# Arguments

  - `x`: The optimiser or the optimiser configuration.

# Returns

  - `path::Option{<:String}`: The path of the data benchmark from `x`, or `nothing`.

# Related

  - [`assert_pipeline_data_benchmark`](@ref)
  - [`data_benchmark_path`](@ref)
"""
function pipe_data_benchmark_path(x)::Option{<:String}
    for f in filter(f -> hasfield(typeof(x), f), (:tr, :r, :ri, :ro))
        p = data_benchmark_path(getfield(x, f))
        if !isnothing(p)
            return tree_path(f, p)
        end
    end
    c = pipe_config_field(x)
    p = pipe_data_benchmark_path(isnothing(c) ? nothing : getfield(x, c))
    return isnothing(p) ? nothing : tree_path(something(c), p)
end
pipe_data_benchmark_path(::Nothing) = nothing
"""
    assert_pipeline_data_benchmark(ests) -> Nothing

Refuse at construction a [`Pipeline`](@ref) whose prior step states `x_src = ExcessSource()` when the optimiser that receives the prior tracks a [`DataBenchmarkTracking`](@ref).

The optimiser receives the Prior Result of the step in its field `pe`. In the Excess Frame `pr.X` holds `X - B`, so a data benchmark gives the series `(X - B) * w - B`, and the benchmark is subtracted two times. The constructor of the optimiser refuses this pair for its own prior. It cannot see the source of the step, because the step value replaces its prior only when the pipeline runs. So the `Pipeline` constructor refuses it. The configuration alone decides the refusal. A prior step that holds a precomputed Prior Result, or a callable, states no source, so it reads as raw and passes.

Internal machinery — not part of the user-facing API.

# Algorithm

 1. Read the optimiser `opt` of the last step with [`pipe_terminal_optimiser`](@ref). Stop when there is none, or when it has no field for the `:pe` routing target.
 2. Find the last prior step, whose Prior Result the `prior` slot holds when `opt` runs, with [`step_prior`](@ref). Stop when there is none, or when its source is not `ExcessSource()`.
 3. Find a data benchmark in a measure of `opt` that reads its prior with [`pipe_data_benchmark_path`](@ref).

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Validation

  - The last prior step does not state `x_src = ExcessSource()`, or the optimiser holds no [`DataBenchmarkTracking`](@ref) in a measure that reads its prior. An `ArgumentError` that names the step, the field and the two remedies is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`DataBenchmarkTracking`](@ref)
  - [`ExcessSource`](@ref)
  - [`pipe_data_benchmark_path`](@ref)
  - [`assert_data_benchmark_prior`](@ref)
"""
function assert_pipeline_data_benchmark(ests)::Nothing
    opt = pipe_terminal_optimiser(ests)
    if isnothing(opt) || !pipe_accepts(opt, Val(:pe))
        return nothing
    end
    n = length(ests)
    k = findlast(i -> !isnothing(step_prior(ests[i])), 1:(n - 1))
    if isnothing(k) || !isa(returns_source(step_prior(ests[k])), ExcessSource)
        return nothing
    end
    path = pipe_data_benchmark_path(opt)
    @argcheck(isnothing(path),
              ArgumentError("step $k of $n fits a prior that states `x_src = ExcessSource()`, and the `$(something(path, ""))` of the terminal `$(nameof(typeof(opt)))`, which reads that prior, tracks the benchmark `B` of the returns data with `DataBenchmarkTracking`. Every tracking form compares `pr.X * w` with its benchmark series, so the series would be `(X - B) * w - B`, and the benchmark would be subtracted two times. State `x_src = DataSource()` or `nothing` on the prior of the step to track `X * w - B`, or replace the data benchmark with a plain risk measure on the excess prior. The two models differ: for a vector `B` the plain measure reads `X * w - B * sum(w)`, which equals `X * w - B` only when `sum(w) == 1`."))
    return nothing
end
