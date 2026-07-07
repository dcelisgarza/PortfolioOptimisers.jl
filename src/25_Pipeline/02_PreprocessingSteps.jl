"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `x` counts as a missing observation in price-level data.

Price-level data stores absent observations either as `missing` or as `NaN` (the two conventions [`prices_to_returns`](@ref) already unifies).

# Arguments

  - `x`: The value to test.

# Returns

  - `flag::Bool`: `true` when `x` is `missing` or a `NaN` number.

# Related

  - [`MissingDataFilter`](@ref)
  - [`Imputer`](@ref)
"""
function is_missing_value(x)::Bool
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
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

Each estimator family dispatches to its native verb — [`prior`](@ref) for prior estimators, [`clusterise`](@ref)/[`phylogeny_matrix`](@ref) for phylogeny estimators, [`optimise`](@ref) for optimisation estimators, [`fit_step`](@ref)/[`apply_step`](@ref) for preprocessing estimators. The fitted object is what [`apply_step`](@ref) later uses to transform unseen data windows; for non-preprocessing steps it is the step's ordinary result.

Estimators whose family is not steppable throw an `ArgumentError` directing the caller to [`PipelineStep`](@ref).

# Arguments

  - `est`: The step estimator (or a [`PipelineStep`](@ref) wrapper).
  - `ctx`: The pipeline context.

# Returns

  - `(fitted, ctx′)`: The fitted object and the updated context.

# Related

  - [`apply_step`](@ref)
  - [`fit_step`](@ref)
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
    res = fit_step(est, ctx.prices)
    return res, set_slot(ctx, :prices, apply_step(res, ctx.prices))
end
function run_step(est::AbstractReturnsPreprocessingEstimator, ctx::PipelineContext)
    require_slot(ctx, :returns, est)
    res = fit_step(est, ctx.returns)
    return res, set_slot(ctx, :returns, apply_step(res, ctx.returns))
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
    return throw(ArgumentError("a $(typeof(ue)) step must declare whether it bounds the mean or the covariance; wrap it in a PipelineStep with target = :mu or target = :sigma"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Execute an uncertainty-set step pinned to a target and merge its result into the `uncertainty` slot.

The target comes from the [`PipelineStep`](@ref) wrapper: `:mu` computes [`mu_ucs`](@ref), `:sigma` computes [`sigma_ucs`](@ref). The result fills its half of the [`PipelineUncertaintySets`](@ref) pair, leaving the other half untouched, so separate mu and sigma steps compose.

# Arguments

  - `ue`: The uncertainty-set estimator.
  - `target`: `:mu` or `:sigma`; anything else throws an `ArgumentError`.
  - `ctx`: The pipeline context; requires the `returns` slot.

# Returns

  - `(res, ctx′)`: The computed [`AbstractUncertaintySetResult`](@ref) and the updated context.

# Related

  - [`run_step`](@ref)
  - [`PipelineStep`](@ref)
  - [`PipelineUncertaintySets`](@ref)
"""
function run_uncertainty_step(ue::AbstractUncertaintySetEstimator, target::Option{Symbol},
                              ctx::PipelineContext)
    @argcheck(target in (:mu, :sigma),
              ArgumentError("the PipelineStep target of a $(typeof(ue)) step must be :mu or :sigma, got $(repr(target))"))
    require_slot(ctx, :returns, ue)
    cur = ctx.uncertainty
    res, pair = if target == :mu
        r = mu_ucs(ue, ctx.returns.X, ctx.returns.F)
        r, PipelineUncertaintySets(; mu = r, sigma = isnothing(cur) ? nothing : cur.sigma)
    else
        r = sigma_ucs(ue, ctx.returns.X, ctx.returns.F)
        r, PipelineUncertaintySets(; mu = isnothing(cur) ? nothing : cur.mu, sigma = r)
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
function run_step(ce::AbstractConstraintEstimator, ::PipelineContext)
    return throw(ArgumentError("a $(typeof(ce)) is not supported as a bare pipeline step; precompute its result and pass it to the optimiser, or wrap a callable in a PipelineStep that writes :constraints"))
end
"""
    fit_step(est::AbstractPreprocessingEstimator, data) -> fitted

Fit a preprocessing estimator on a data window and return the fitted object consumed by [`apply_step`](@ref).

The fitted object carries whatever state the transformation needs to be replayed consistently on unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators return themselves.

# Interfaces

Concrete preprocessing estimators must implement:

  - `fit_step(est::MyPreprocessing, data) -> fitted`: Compute the fitted state from the training window.
  - `apply_step(fitted, data) -> data′`: Transform a data window with the fitted state.

# Arguments

  - `est`: The preprocessing estimator.
  - `data`: The training data window ([`PricesResult`](@ref) or [`ReturnsResult`](@ref) depending on the estimator's level).

# Returns

  - `fitted`: The fitted object, typically an [`AbstractPreprocessingResult`](@ref) or the estimator itself when stateless.

# Related

  - [`apply_step`](@ref)
  - [`run_step`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
"""
function fit_step(est::AbstractPreprocessingEstimator, data)
    return throw(ArgumentError("fit_step is not implemented for $(typeof(est)); implement fit_step(est, data) and apply_step(fitted, data)"))
end
"""
    apply_step(fitted, data) -> data′

Transform a data window with a fitted preprocessing object.

Applying the fitted object produced by [`fit_step`](@ref) on the training window to an unseen (test) window replays the *same* transformation — the same asset universe, the same imputation parameters — so train and test data stay consistent and no information flows from test to train.

# Arguments

  - `fitted`: The fitted object returned by [`fit_step`](@ref) (an [`AbstractPreprocessingResult`](@ref), or a stateless estimator).
  - `data`: The data window to transform.

# Returns

  - `data′`: The transformed data window.

# Related

  - [`fit_step`](@ref)
  - [`run_step`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
function apply_step(fitted::Union{<:AbstractPreprocessingEstimator,
                                  <:AbstractPreprocessingResult}, data)
    return throw(ArgumentError("apply_step is not implemented for $(typeof(fitted))"))
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator converting price-level data into returns-level data.

`PricesToReturns` is the pipeline-step form of [`prices_to_returns`](@ref): it consumes the `prices` slot ([`PricesResult`](@ref)) and writes the `returns` slot ([`ReturnsResult`](@ref)). It is stateless — applying it to any window simply runs the conversion — so its fitted object is the estimator itself.

Missing-data filtering is deliberately *not* part of this estimator (the corresponding [`prices_to_returns`](@ref) keywords are held at their permissive defaults); use [`MissingDataFilter`](@ref) and [`Imputer`](@ref) as separate, independently tunable steps.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesToReturns(;
        ret_method::Symbol = :simple,
        padding::Bool = false,
        collapse_args::Tuple = (),
        map_func::Option{<:Function} = nothing,
        join_method::Symbol = :outer,
    ) -> PricesToReturns

Keywords correspond to the struct's fields.

## Validation

  - `ret_method in (:simple, :log)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], ["A", "B"]);

julia> pr = PricesResult(; X = X);

julia> rr = PortfolioOptimisers.apply_step(PricesToReturns(), pr);

julia> size(rr.X)
(2, 2)

julia> rr.nx
2-element Vector{String}:
 "A"
 "B"
```

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`prices_to_returns`](@ref)
  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
@concrete struct PricesToReturns <: AbstractPreprocessingEstimator
    """
    Return calculation method (`:simple` or `:log`).
    """
    ret_method
    """
    Whether to pad missing values in the returns calculation.
    """
    padding
    """
    Arguments for collapsing the time series (e.g. to lower frequency).
    """
    collapse_args
    """
    Optional function applied to the data before the returns calculation.
    """
    map_func
    """
    How asset, factor, and benchmark data are joined (`:outer`, `:inner`, etc.).
    """
    join_method
    function PricesToReturns(ret_method::Symbol, padding::Bool, collapse_args::Tuple,
                             map_func::Option{<:Function}, join_method::Symbol)
        @argcheck(ret_method in (:simple, :log),
                  ArgumentError("ret_method must be :simple or :log, got :$ret_method"))
        return new{typeof(ret_method), typeof(padding), typeof(collapse_args),
                   typeof(map_func), typeof(join_method)}(ret_method, padding,
                                                          collapse_args, map_func,
                                                          join_method)
    end
end
function PricesToReturns(; ret_method::Symbol = :simple, padding::Bool = false,
                         collapse_args::Tuple = (), map_func::Option{<:Function} = nothing,
                         join_method::Symbol = :outer)::PricesToReturns
    return PricesToReturns(ret_method, padding, collapse_args, map_func, join_method)
end
pipe_reads(::PricesToReturns) = (:prices,)
pipe_writes(::PricesToReturns) = :returns
function fit_step(ptr::PricesToReturns, ::PricesResult)
    return ptr
end
function apply_step(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(pr.X, pr.F; B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                             ret_method = ptr.ret_method, padding = ptr.padding,
                             collapse_args = ptr.collapse_args, map_func = ptr.map_func,
                             join_method = ptr.join_method)
end
function run_step(ptr::PricesToReturns, ctx::PipelineContext)
    require_slot(ctx, :prices, ptr)
    return ptr, set_slot(ctx, :returns, apply_step(ptr, ctx.prices))
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator dropping assets and observations with excessive missing data from price-level data.

The *asset universe is fitted state*: the training window decides which assets survive (per-column missing fraction at most `col_thr`), and applying the fitted result to an unseen window subsets it to that same universe — so train weights and test returns always refer to the same assets. Observation (row) filtering is window-local: rows whose missing fraction across the surviving assets exceeds `row_thr` are dropped from whichever window is being transformed.

This estimator supersedes the `missing_col_percent`/`missing_row_percent` keywords of [`prices_to_returns`](@ref) inside pipelines, making the thresholds independently tunable. Only the asset series `X` (and the matching implied volatility columns) participate; factor and benchmark series pass through unchanged.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilter(;
        col_thr::Number = 1.0,
        row_thr::Number = 1.0,
    ) -> MissingDataFilter

Keywords correspond to the struct's fields.

## Validation

  - `0 < col_thr <= 1`.
  - `0 < row_thr <= 1`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100.0 NaN; 102.0 NaN; 104.0 105.0],
                     ["A", "B"]);

julia> pr = PricesResult(; X = X);

julia> res = PortfolioOptimisers.fit_step(MissingDataFilter(; col_thr = 0.5), pr);

julia> res.nx
1-element Vector{Symbol}:
 :A
```

# Related

  - [`MissingDataFilterResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`Imputer`](@ref)
  - [`PricesResult`](@ref)
"""
@concrete struct MissingDataFilter <: AbstractPricesPreprocessingEstimator
    """
    Maximum allowed fraction `(0, 1]` of missing observations per asset column; assets above it are dropped from the universe at fit time.
    """
    col_thr
    """
    Maximum allowed fraction `(0, 1]` of missing assets per observation row; rows above it are dropped from the window being transformed.
    """
    row_thr
    function MissingDataFilter(col_thr::Number, row_thr::Number)
        @argcheck(zero(col_thr) < col_thr <= one(col_thr), DomainError)
        @argcheck(zero(row_thr) < row_thr <= one(row_thr), DomainError)
        return new{typeof(col_thr), typeof(row_thr)}(col_thr, row_thr)
    end
end
function MissingDataFilter(; col_thr::Number = 1.0,
                           row_thr::Number = 1.0)::MissingDataFilter
    return MissingDataFilter(col_thr, row_thr)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`MissingDataFilter`](@ref).

Carries the asset universe selected on the training window plus the row threshold needed to transform further windows. Produced by [`fit_step`](@ref), consumed by [`apply_step`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`MissingDataFilter`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
@concrete struct MissingDataFilterResult <: AbstractPreprocessingResult
    """
    Names of the assets that survived the training window (the fitted universe).
    """
    nx
    """
    Maximum allowed fraction `(0, 1]` of missing assets per observation row.
    """
    row_thr
end
function fit_step(mdf::MissingDataFilter, pr::PricesResult)::MissingDataFilterResult
    vals = values(pr.X)
    frac = vec(count(is_missing_value, vals; dims = 1)) / size(vals, 1)
    keep = frac .<= mdf.col_thr
    @argcheck(any(keep),
              IsEmptyError("MissingDataFilter with col_thr = $(mdf.col_thr) drops every asset in the training window"))
    return MissingDataFilterResult(TimeSeries.colnames(pr.X)[keep], mdf.row_thr)
end
function apply_step(res::MissingDataFilterResult, pr::PricesResult)::PricesResult
    cols = findall(in(res.nx), TimeSeries.colnames(pr.X))
    @argcheck(!isempty(cols),
              IsEmptyError("none of the fitted universe assets $(res.nx) are present in the data window"))
    vals = values(pr.X)[:, cols]
    rows = findall(vec(count(is_missing_value, vals; dims = 2)) .<=
                   length(cols) * res.row_thr)
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X)[rows], vals[rows, :],
                             TimeSeries.colnames(pr.X)[cols])
    iv, ivpa = if isnothing(pr.iv)
        nothing, pr.ivpa
    else
        ivv = values(pr.iv)[:, cols]
        ivm = TimeSeries.TimeArray(TimeSeries.timestamp(pr.iv), ivv,
                                   TimeSeries.colnames(pr.iv)[cols])
        ivm, isa(pr.ivpa, VecNum) ? pr.ivpa[cols] : pr.ivpa
    end
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = iv, ivpa = ivpa)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator imputing missing price observations from per-asset statistics fitted on the training window.

The *imputation parameters are fitted state*: each asset's fill value is computed from the training window's observed (non-missing) prices with the configured [`VectorToScalarMeasure`](@ref), and applying the fitted result to an unseen window fills that window's missing observations with the *training* values — never with statistics of the window being transformed, which is exactly the leakage the pipeline exists to prevent.

Assets with no observed values in the training window get no fill value and are left untouched at apply time; combine with [`MissingDataFilter`](@ref) to drop them instead.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Imputer(;
        stat::VectorToScalarMeasure = MedianValue(),
    ) -> Imputer

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100.0 1.0; NaN 3.0; 104.0 5.0],
                     ["A", "B"]);

julia> pr = PricesResult(; X = X);

julia> res = PortfolioOptimisers.fit_step(Imputer(), pr);

julia> pv = PortfolioOptimisers.apply_step(res, pr);

julia> values(pv.X)[2, 1]
102.0
```

# Related

  - [`ImputerResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`VectorToScalarMeasure`](@ref)
"""
@concrete struct Imputer <: AbstractPricesPreprocessingEstimator
    """
    Reducer computing an asset's fill value from its observed training prices ([`VectorToScalarMeasure`](@ref)).
    """
    stat
    function Imputer(stat::VectorToScalarMeasure)
        return new{typeof(stat)}(stat)
    end
end
function Imputer(; stat::VectorToScalarMeasure = MedianValue())::Imputer
    return Imputer(stat)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of an [`Imputer`](@ref).

Carries the per-asset fill values computed on the training window. Produced by [`fit_step`](@ref), consumed by [`apply_step`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Imputer`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
@concrete struct ImputerResult <: AbstractPreprocessingResult
    """
    Names of the assets with a fitted fill value.
    """
    nx
    """
    Fill values, aligned with `nx`.
    """
    v
end
function fit_step(imp::Imputer, pr::PricesResult)::ImputerResult
    names = TimeSeries.colnames(pr.X)
    vals = values(pr.X)
    keep = Vector{Int}(undef, 0)
    v = Vector{Any}(undef, 0)
    for i in axes(vals, 2)
        obs = identity.([x for x in view(vals, :, i) if !is_missing_value(x)])
        if isempty(obs)
            continue
        end
        push!(keep, i)
        push!(v, vec_to_real_measure(imp.stat, obs))
    end
    return ImputerResult(names[keep], identity.(v))
end
function apply_step(res::ImputerResult, pr::PricesResult)::PricesResult
    names = TimeSeries.colnames(pr.X)
    vals = copy(values(pr.X))
    for (name, fill_val) in zip(res.nx, res.v)
        j = findfirst(==(name), names)
        if isnothing(j)
            continue
        end
        for i in axes(vals, 1)
            if is_missing_value(vals[i, j])
                vals[i, j] = fill_val
            end
        end
    end
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X), vals, names)
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa)
end

export PricesToReturns, MissingDataFilter, MissingDataFilterResult, Imputer, ImputerResult
