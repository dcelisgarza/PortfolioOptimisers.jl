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
    res = phylogeny_matrix(ne, ctx.returns.X)
    return res, set_slot(ctx, :phylogeny, res)
end
function run_step(opt::OptimisationEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, opt)
    res = optimise(opt, ctx.returns)
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

A narrowed step fills its half of the [`PipelineUncertaintySets`](@ref) pair and leaves the other untouched, so separate `:mu` and `:sigma` steps compose. Every populated half must reach the optimiser: [`inject_config`](@ref) and [`inject_sigma_ucs`](@ref) reject a set they cannot route rather than dropping it, so `:both` requires an optimiser with an [`ArithmeticReturn`](@ref) *and* an [`UncertaintySetVariance`](@ref) risk measure.

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
    @argcheck(target in (:mu, :sigma, :both),
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

Build the [`AssetSets`](@ref) a constraint-generation step needs from the asset names of the context's `returns` slot.

Constraint estimators referencing groups beyond the plain asset names cannot be satisfied by this minimal set; precompute their result instead, or wrap a callable in a [`PipelineStep`](@ref) that supplies richer sets.

# Arguments

  - `ctx`: The pipeline context; requires the `returns` slot.
  - `est`: The step about to run, used in the error message.

# Returns

  - `sets::AssetSets`: Asset sets whose `nx` entry holds the asset names.

# Related

  - [`run_step`](@ref)
  - [`AssetSets`](@ref)
"""
function pipeline_asset_sets(ctx::PipelineContext, est)::AssetSets
    require_slot(ctx, :returns, est)
    return AssetSets(; dict = Dict("nx" => ctx.returns.nx))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Append a constraint result to the `constraints` slot of the context.

The slot accumulates: the first result is stored as-is, later results widen it into a `Vector{AbstractConstraintResult}` preserving step order.

# Arguments

  - `ctx`: The pipeline context.
  - `res`: The constraint result to append.

# Returns

  - `ctx′::PipelineContext`: The updated context.

# Related

  - [`run_step`](@ref)
  - [`PipelineContext`](@ref)
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
function run_step(ce::WeightBoundsEstimator, ctx::PipelineContext)
    res = weight_bounds_constraints(ce, pipeline_asset_sets(ctx, ce))
    return res, add_constraint_result(ctx, res)
end
function run_step(ce::LinearConstraintEstimator, ctx::PipelineContext)
    res = linear_constraints(ce, pipeline_asset_sets(ctx, ce))
    return res, add_constraint_result(ctx, res)
end
function run_step(ce::ThresholdEstimator, ctx::PipelineContext)
    res = threshold_constraints(ce, pipeline_asset_sets(ctx, ce))
    return res, add_constraint_result(ctx, res)
end
function run_step(ce::RiskBudgetEstimator, ctx::PipelineContext)
    res = risk_budget_constraints(ce, pipeline_asset_sets(ctx, ce))
    return res, add_constraint_result(ctx, res)
end
function run_step(ce::AbstractPhylogenyConstraintEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, ce)
    res = phylogeny_constraints(ce, ctx.returns.X)
    return res, add_constraint_result(ctx, res)
end
function run_step(ce::AbstractConstraintEstimator, ::PipelineContext)
    return throw(ArgumentError("a $(typeof(ce)) is not supported as a bare pipeline step; precompute its result and pass it to the optimiser, or wrap a callable in a PipelineStep that writes :constraints"))
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
