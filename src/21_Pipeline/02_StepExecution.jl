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

A step that reads the returns, see [`step_reads_returns`](@ref), fits under the `brt` of its [`PipelineStep`](@ref) wrapper, so its verb reads the returns in excess of the benchmark when the wrapper says so. An unwrapped step fits on the raw returns.

# Arguments

  - `est`: The step estimator (or a [`PipelineStep`](@ref) wrapper).
  - `ctx`: The pipeline context.
  - $(arg_dict[:brt_read]) Only the methods for a prior, a clustering and a network estimator take it.

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
function run_step(pe::AbstractPriorEstimator, ctx::PipelineContext; brt::Bool = false)
    require_slot(ctx, :returns, pe)
    pr = prior(pe, ctx.returns; brt = brt)
    return pr, set_slot(ctx, :prior, pr)
end
function run_step(cle::AbstractClustersEstimator, ctx::PipelineContext; brt::Bool = false)
    require_slot(ctx, :returns, cle)
    res = clusterise(cle, ctx.returns; rd = ctx.returns, x_src = :data, brt = brt)
    return res, set_slot(ctx, :phylogeny, res)
end
function run_step(ne::AbstractNetworkEstimator, ctx::PipelineContext; brt::Bool = false)
    require_slot(ctx, :returns, ne)
    res = phylogeny_matrix(ne, ctx.returns; rd = ctx.returns, x_src = :data, brt = brt)
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
        return run_uncertainty_step(ps.est, ps.target, ctx; brt = ps.brt)
    end
    if isa(ps.est, AbstractConstraintEstimator)
        return run_constraint_step(ps.est, ps.target, ctx; brt = ps.brt)
    end
    return run_step_under(ps.est, ctx, ps.brt)
end
"""
    run_step_under(est, ctx::PipelineContext, brt::Bool)

Run the estimator of a [`PipelineStep`](@ref) under the `brt` of the step.

A prior, a clustering or a network estimator fits under `brt` through [`run_step`](@ref). Every other estimator runs as a bare step. Such an estimator is a preprocessing step or an optimiser, which [`assert_step_brt`](@ref) gives `brt = false`. An uncertainty-set step and a constraint step do not come here, because [`run_step`](@ref) passes `brt` to their own runners.

Internal machinery — not part of the user-facing API.

# Arguments

  - `est`: The estimator that the step wraps.
  - `ctx`: The pipeline context.
  - $(arg_dict[:brt_read])

# Returns

  - `(fitted, ctx′)`: The fitted object and the updated context, as [`run_step`](@ref) returns them.

# Related

  - [`run_step`](@ref)
  - [`assert_step_brt`](@ref)
"""
run_step_under(est, ctx::PipelineContext, ::Bool) = run_step(est, ctx)
function run_step_under(est::Union{<:AbstractPriorEstimator, <:AbstractPhylogenyEstimator},
                        ctx::PipelineContext, brt::Bool)
    return run_step(est, ctx; brt = brt)
end
function run_step(ue::AbstractUncertaintySetEstimator, ::PipelineContext)
    return throw(ArgumentError("a $(typeof(ue)) step must declare which parameter it bounds; wrap it in a PipelineStep with target = :mu, target = :sigma, or target = :both"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the slot an uncertainty-set step fits from, as the arguments its verb takes.

An estimator that reads a prior result — [`reads_prior_result`](@ref) answers `true` — is fitted from the `prior` slot, and never touches the returns; every other estimator is fitted from the `returns` slot, whose `X` and `F` its verb takes. The asset returns are read with [`benchmark_relative_returns`](@ref) under the `brt` of the step, as the optimiser reads them for its own uncertainty set. The required slot is checked here, so a missing one is refused with an error that names it before any fit. The predicate reads the type, so the branch folds.

# Arguments

  - `ue`: The uncertainty-set estimator.
  - `ctx`: The pipeline context.
  - $(arg_dict[:brt_read]) A prior-reading estimator does not read it.

# Validation

  - The slot the estimator reads is filled, through [`require_slot`](@ref).
  - The rules of [`benchmark_relative_returns`](@ref).

# Returns

  - `src::Tuple`: `(ctx.prior,)` for a prior-reading estimator, `(X, ctx.returns.F)` otherwise, where `X` is the asset returns in the frame of `brt`.

# Related

  - [`run_uncertainty_step`](@ref)
  - [`reads_prior_result`](@ref)
  - [`require_slot`](@ref)
"""
function uncertainty_step_source(ue::AbstractUncertaintySetEstimator, ctx::PipelineContext,
                                 brt::Bool = false)
    if reads_prior_result(ue)
        require_slot(ctx, :prior, ue)
        return (ctx.prior,)
    end
    require_slot(ctx, :returns, ue)
    return (benchmark_relative_returns(ctx.returns, brt), ctx.returns.F)
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
  - $(arg_dict[:brt_read]) It is the `brt` of the [`PipelineStep`](@ref), and [`uncertainty_step_source`](@ref) reads it.

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
                              ctx::PipelineContext; brt::Bool = false)
    @argcheck(target in (:mu, :sigma, :both),
              ArgumentError("the PipelineStep target of a $(typeof(ue)) step must be :mu, :sigma, or :both, got $(repr(target))"))
    src = uncertainty_step_source(ue, ctx, brt)
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

A family that reads the returns, see [`step_reads_returns`](@ref), takes the keyword `brt` and reads them in that frame, as the optimiser does with its own `brt`. The phylogeny and centrality families read `ctx.returns` as the optimiser does under `x_src = :data`.

# Arguments

  - `ce`: The constraint estimator.
  - `ctx`: The pipeline context.
  - $(arg_dict[:brt_read]) Only the phylogeny, centrality and exposure methods take it.

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
  - $(arg_dict[:brt_read]) It is the `brt` of the [`PipelineStep`](@ref). Only a family that reads the returns, see [`step_reads_returns`](@ref), receives it.

# Returns

  - `(res, ctx′)`: The computed value and the updated context. `res` is the bare value, so what a fitted pipeline reports is the constraint result itself rather than the routing wrapper.

# Related

  - [`run_step`](@ref)
  - [`resolve_constraint_target`](@ref)
  - [`constraint_step_value`](@ref)
  - [`add_constraint_result`](@ref)
"""
function run_constraint_step(ce::AbstractConstraintEstimator, target::Option{Symbol},
                             ctx::PipelineContext; brt::Bool = false)
    target = resolve_constraint_target(ce, target)
    #! The predicate reads the type, so the branch folds; a family that reads no returns
    #! takes no `brt`, and `assert_step_brt` gave its step `brt = false`.
    res = if step_reads_returns(ce)
        constraint_step_value(ce, ctx; brt = brt)
    else
        constraint_step_value(ce, ctx)
    end
    return res, add_constraint_result(ctx, target, res)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute an [`ExposureConstraintEstimator`](@ref) step's value: re-base its rows through the loadings of the *pipeline's* prior into an asset-space [`LinearConstraint`](@ref).

This is the one constraint step that reads a computed slot other than `:returns`. The basis is `ctx.prior.rr`, so a prior step must come earlier; a prior that carries no regression makes [`constraint_space_basis`](@ref) throw, which is the intended failure — see [`ExposureConstraintEstimator`](@ref).

`ctx.returns` is passed as well, so a space that names a regression **estimator** refits the loadings here rather than throwing. That is the one arrangement in which a prior carrying no factor block still admits a factor mandate — see [`FactorSpace`](@ref). The refit reads the returns under the `brt` of the step, as the optimiser refits under its own `brt`.

!!! warning

    The constraint is **pinned to the pipeline's prior**. Its rows were projected through the loadings this step saw, and a downstream optimiser that refits its own prior does not re-project them. Passing the estimator to the optimiser's `lcse` field instead recomputes the projection with the optimiser's own prior, per fold, which is what a cross-validated factor mandate needs.

# Related

  - [`run_constraint_step`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`linear_constraints`](@ref)
  - [`pipeline_asset_sets`](@ref)
"""
function constraint_step_value(ce::ExposureConstraintEstimator, ctx::PipelineContext;
                               brt::Bool = false)
    require_slot(ctx, :prior, ce)
    return linear_constraints(ce, pipeline_asset_sets(ctx, ce); rr = ctx.prior.rr,
                              rd = ctx.returns, brt = brt)
end
function constraint_step_value(ce::AbstractPhylogenyConstraintEstimator,
                               ctx::PipelineContext; brt::Bool = false)
    require_slot(ctx, :returns, ce)
    return phylogeny_constraints(ce, ctx.returns; rd = ctx.returns, x_src = :data,
                                 brt = brt)
end
function constraint_step_value(ce::AbstractCentralityConstraint, ctx::PipelineContext;
                               brt::Bool = false)
    require_slot(ctx, :returns, ce)
    return centrality_constraints(ce, ctx.returns; rd = ctx.returns, x_src = :data,
                                  brt = brt)
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
    step_reads_returns(est) -> Bool

Return `true` when a step of a [`Pipeline`](@ref) reads the asset returns of the `returns` slot to compute its value.

Such a step reads the returns in a frame: the raw `X`, or the excess `X - B` under `brt`. A [`PipelineStep`](@ref) states the frame in its `brt`, see [`assert_step_brt`](@ref). The answer reads the type alone, so the constructor of the step can use it.

  - A prior estimator, and an [`Online`](@ref) declaration of one, reads the returns.
  - A clustering estimator and a network estimator read the returns.
  - An uncertainty-set estimator reads the returns when it fits its own prior, which is when [`reads_prior_result`](@ref) answers `false`. Otherwise it reads the `prior` slot, whose frame the prior step states.
  - A phylogeny constraint estimator and a centrality constraint estimator read the returns.
  - An [`ExposureConstraintEstimator`](@ref) reads the returns when its [`FactorSpace`](@ref) names a regression estimator, which refits the loadings when the prior carries none. Otherwise it reads the loadings of the prior.
  - Every other estimator, and a callable, reads no returns through the wrapper.

Internal machinery — not part of the user-facing API.

# Arguments

  - `est`: The estimator or the callable that the step wraps.

# Returns

  - `flag::Bool`: `true` when `est` reads the returns.

# Related

  - [`assert_step_brt`](@ref)
  - [`step_brt`](@ref)
  - [`benchmark_relative_returns`](@ref)
"""
step_reads_returns(::Any) = false
function step_reads_returns(::Union{<:AbstractPriorEstimator, <:AbstractPhylogenyEstimator,
                                    <:AbstractPhylogenyConstraintEstimator,
                                    <:AbstractCentralityConstraint})
    return true
end
step_reads_returns(ue::AbstractUncertaintySetEstimator) = !reads_prior_result(ue)
function step_reads_returns(::ExposureConstraintEstimator{<:Any,
                                                          <:FactorSpace{<:AbstractRegressionEstimator}})
    return true
end
step_reads_returns(o::Online) = step_reads_returns(o.est)
"""
    assert_step_brt(est, brt::Bool)

Refuse `brt = true` on a [`PipelineStep`](@ref) whose estimator reads no returns.

A step that reads the returns, see [`step_reads_returns`](@ref), takes `brt = true` and reads them in excess of the benchmark. Every other estimator and every callable takes `brt = false` alone, because the wrapper passes `brt` to no such step. The constructor of `PipelineStep` calls this check.

Internal machinery — not part of the user-facing API.

# Arguments

  - `est`: The estimator or the callable that the step wraps.
  - $(arg_dict[:brt_read])

# Validation

  - `brt` is `false`, or `step_reads_returns(est)` is `true`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`PipelineStep`](@ref)
  - [`step_reads_returns`](@ref)
  - [`assert_benchmark_frame`](@ref)
"""
function assert_step_brt(est, brt::Bool)::Nothing
    @argcheck(!brt || step_reads_returns(est),
              ArgumentError("brt = true fits a step on the returns in excess of the benchmark, and this PipelineStep wraps a $(Base.typename(typeof(est)).wrapper), which reads no returns: the wrapper passes brt to a prior, a clustering or network estimator, an uncertainty set that fits its own prior, a phylogeny or centrality constraint, and an exposure constraint whose FactorSpace refits its loadings. Set the brt of the optimiser to optimise over the excess returns, or wrap an estimator that reads the returns."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the `brt` of the optimiser configuration that receives a computed prior.

[`pipe_route`](@ref) puts the `:pe` routing target into the field `pe` of the optimiser, or of the configuration that [`@pipe_delegates`](@ref) names. The `brt` beside that field states the frame that the optimiser reads its prior in. Under `brt` every other reader of the optimiser subtracts the benchmark: the clustering under `x_src = :data`, the uncertainty sets, and the phylogeny, centrality and exposure constraints. So the prior must hold the moments of the excess returns `X - B`, and the value of every step that the optimiser receives must come from them. This function follows the path of `pipe_route`.

# Algorithm

 1. When `x` has a field `pe`, return the field `brt` of `x`, or `false` when `x` has no such field.
 2. Else, when `x` delegates to a configuration ([`pipe_config_field`](@ref)), return the answer for that configuration.
 3. Else return `false`.

# Arguments

  - `x`: The optimiser or the optimiser configuration.

# Returns

  - `brt::Bool`: The frame of the prior that `x` reads.

# Related

  - [`assert_benchmark_frame`](@ref)
  - [`pipe_route`](@ref)
  - [`benchmark_relative_returns`](@ref)
"""
function pipe_prior_brt(x)::Bool
    if hasfield(typeof(x), :pe)
        return hasfield(typeof(x), :brt) && getfield(x, :brt)
    end
    f = pipe_config_field(x)
    return isnothing(f) ? false : pipe_prior_brt(getfield(x, f))
end
"""
    step_brt(step) -> Option{Bool}

Return the frame of the returns that a step of a [`Pipeline`](@ref) reads to compute its value.

The answer is `true` when the step reads the returns in excess of the benchmark, and `false` when it reads the raw returns. A bare estimator that reads the returns, see [`step_reads_returns`](@ref), reads the raw returns, and a [`PipelineStep`](@ref) reads them under its `brt`. A nested `Pipeline` states the frame of its last step. A step that reads no returns, such as a callable or a weight-bounds estimator, states no frame, so the answer is `nothing`, and [`assert_benchmark_frame`](@ref) does not check it.

Internal machinery — not part of the user-facing API.

# Related

  - [`assert_benchmark_frame`](@ref)
  - [`step_reads_returns`](@ref)
  - [`PipelineStep`](@ref)
"""
step_brt(est) = step_reads_returns(est) ? false : nothing
function step_brt(ps::PipelineStep)
    return if isa(ps.est, Pipeline)
        step_brt(ps.est)
    elseif step_reads_returns(ps.est)
        ps.brt
    else
        nothing
    end
end
"""
    pipe_receives_step(opt, step) -> Bool

Return `true` when the optimiser `opt` receives the value of a step of a [`Pipeline`](@ref).

[`inject_context`](@ref) routes the value of a step into the optimiser. Two routing targets are [optional](@ref PIPELINE_OPTIONAL_TARGETS): an optimiser with no field for `:pe` or `:cle` ignores a prior or a clustering. A network estimator writes a phylogeny matrix, which `inject_context` routes nowhere. The value of every other step that reads the returns, an uncertainty set or a constraint, must reach the optimiser, or the routing raises.

Internal machinery — not part of the user-facing API.

# Arguments

  - `opt`: The optimiser of the last step, see [`pipe_terminal_optimiser`](@ref).
  - `step`: The step, a bare estimator, a [`PipelineStep`](@ref) or a nested `Pipeline`.

# Returns

  - `flag::Bool`: `true` when `opt` receives the value of `step`.

# Related

  - [`assert_benchmark_frame`](@ref)
  - [`pipe_accepts`](@ref)
"""
pipe_receives_step(opt, ::Any) = true
pipe_receives_step(opt, ps::PipelineStep) = pipe_receives_step(opt, ps.est)
pipe_receives_step(opt, o::Online) = pipe_receives_step(opt, o.est)
pipe_receives_step(opt, ::AbstractPriorEstimator) = pipe_accepts(opt, Val(:pe))
pipe_receives_step(opt, ::AbstractClustersEstimator) = pipe_accepts(opt, Val(:cle))
pipe_receives_step(opt, ::AbstractNetworkEstimator) = false
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the optimiser of the last step of a [`Pipeline`](@ref), or `nothing` when the pipeline has none that a construction check can read.

The checks that compare the earlier steps with the optimiser read it here: [`assert_routable`](@ref) and [`assert_benchmark_frame`](@ref).

# Algorithm

 1. Return `nothing` when the pipeline has one step, or when its last step writes a slot other than `:opt`.
 2. Read the optimiser from the last step, without its [`PipelineStep`](@ref) wrapper.
 3. Return it when it is an [`OptimisationEstimator`](@ref). Otherwise return `nothing`. A [`TimeDependent`](@ref) schedule and a precomputed result are such steps, and the fold loop resolves them later.

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Returns

  - `opt::Option{<:OptimisationEstimator}`: The optimiser, or `nothing`.

# Related

  - [`assert_routable`](@ref)
  - [`assert_benchmark_frame`](@ref)
"""
function pipe_terminal_optimiser(ests)
    n = length(ests)
    if n < 2 || !(pipe_writes(ests[n]) === :opt)
        return nothing
    end
    opt = isa(ests[n], PipelineStep) ? ests[n].est : ests[n]
    return isa(opt, OptimisationEstimator) ? opt : nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse at construction a pipeline with a step that reads the returns in a frame other than the frame of the optimiser that receives the value of the step.

[`inject_context`](@ref) puts the value of a step into the last optimiser: a prior into `pe`, a clustering into `cle`, an uncertainty set into `mu_ucs` or `sigma_ucs`, and a constraint into its routing target. The optimiser uses each value unchanged, so the `brt` of the optimiser never reaches it. Under `brt = true` every other reader of the optimiser reads the excess returns `X - B`: the clustering under `x_src = :data`, the exposure regression, the uncertainty sets and the loadings of the risk budgeting. A value computed from the raw `X` then mixes two frames, and no error shows it. So the `brt` of each step that reads the returns, see [`PipelineStep`](@ref), must equal the `brt` of the optimiser. The configuration alone decides this, so the check runs when the pipeline is built.

A clustering step and the optimiser read the same frame, because the prior of the optimiser is in the frame of its `brt` too: a prior step passes this check, and an optimiser fits its own prior under its `brt`. A network step writes a phylogeny matrix that the optimiser does not receive, so it is not checked.

# Algorithm

 1. Read the optimiser `opt` of the last step with [`pipe_terminal_optimiser`](@ref). Stop when there is none.
 2. Read the frame of the optimiser with [`pipe_prior_brt`](@ref).
 3. Read the frame of each earlier step with [`step_brt`](@ref), and compare the two. Skip a step that states no frame, and a step whose value `opt` does not receive ([`pipe_receives_step`](@ref)).

# Arguments

  - `ests`: The step estimators, with the optimisation step last, see [`assert_opt_last`](@ref).

# Validation

  - Each step that reads the returns, and whose value the optimiser receives, has the `brt` of the optimiser. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`PipelineStep`](@ref)
  - [`pipe_prior_brt`](@ref)
  - [`step_brt`](@ref)
  - [`pipe_receives_step`](@ref)
  - [`assert_routable`](@ref)
  - [`benchmark_relative_returns`](@ref)
"""
function assert_benchmark_frame(ests)::Nothing
    opt = pipe_terminal_optimiser(ests)
    if isnothing(opt)
        return nothing
    end
    brt = pipe_prior_brt(opt)
    n = length(ests)
    for i in 1:(n - 1)
        sbrt = step_brt(ests[i])
        if isnothing(sbrt) || !pipe_receives_step(opt, ests[i])
            continue
        end
        @argcheck(sbrt == brt,
                  ArgumentError("step $i of $n reads the returns with brt = $sbrt, and the terminal $(Base.typename(typeof(opt)).wrapper) reads them with brt = $brt; the injected value of the step and the other readers of the optimiser would read the returns in two frames, the raw X and the excess X - B. Wrap the step as PipelineStep(; est = <estimator>, writes = :$(pipe_writes(ests[i])), brt = $brt), or set the brt of the optimiser to $sbrt."))
    end
    return nothing
end
