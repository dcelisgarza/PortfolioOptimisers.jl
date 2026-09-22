"""
$(DocStringExtensions.TYPEDEF)

The log-wealth regret of a strategy against a comparator over one sequence of rows, with the test of its per-row difference.

`LogWealthRegretResult` is what [`log_wealth_regret`](@ref) returns. The regret is the gap in log terminal wealth between the comparator and the strategy, positive when the comparator wins; the per-row difference series carries a Newey–West test of equal expected log growth, in the shape of [`CovarianceForecastComparisonResult`](@ref). Negative regret is expected on many sequences and is not a defect: a causal strategy that reads the market's structure can beat a constant portfolio, and a Hindsight Comparator is a ceiling only over the class it was chosen from.

The Result also states the comparator's side of a dynamic-regret bound: `path_length` is the summed Euclidean distance between the comparator's consecutive targets, ``P_T``, which is zero for a constant comparator and grows with every switch a [`HindsightSplit`](@ref) comparator makes, and `cumulative` is the running regret, the cumulative sum of `difference`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogWealthRegretResult(
        regret, regret_per_period, difference, cumulative, variance, z, p, lags,
        n_periods, wealth_a, wealth_b, path_length
    ) -> LogWealthRegretResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result, so [`log_wealth_regret`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`log_wealth_regret`](@ref)
  - [`CovarianceForecastComparisonResult`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`HindsightSplit`](@ref)
  - [`budgeted_hindsight_path`](@ref)
"""
@concrete struct LogWealthRegretResult <: AbstractResult
    """
    Log-wealth regret, ``\\sum_t \\log(1 + r_{b,t}) - \\sum_t \\log(1 + r_{a,t})``: the comparator's log terminal wealth less the strategy's, positive when the comparator wins.
    """
    regret
    """
    The regret divided by the number of periods, which is the mean of `difference`.
    """
    regret_per_period
    """
    Per-row series ``\\log(1 + r_{b,t}) - \\log(1 + r_{a,t})``, the comparator's log growth less the strategy's at every period.
    """
    difference
    """
    Running regret, the cumulative sum of `difference`: its last entry is `regret`.
    """
    cumulative
    """
    Newey–West long-run variance of `difference`, at `lags` lags.
    """
    variance
    """
    Diebold–Mariano–West statistic of `difference`, asymptotically standard normal under equal expected log growth when the comparator did not read the rows.
    """
    z
    """
    Two-sided p-value of the statistic against the standard normal. It is `NaN` when the two series coincide, because their difference has no variance.
    """
    p
    """
    Number of lags of the Bartlett kernel.
    """
    lags
    """
    Number of periods compared.
    """
    n_periods
    """
    Terminal wealth of the strategy from a unit start, ``\\prod_t (1 + r_{a,t})``.
    """
    wealth_a
    """
    Terminal wealth of the comparator from a unit start, ``\\prod_t (1 + r_{b,t})``.
    """
    wealth_b
    """
    Path length of the comparator, ``P_T = \\sum_{t \\geq 2} \\lVert u_t - u_{t-1} \\rVert_2`` over its consecutive targets, read off the comparator's folds and embedded by asset name on the union of their universes. `NaN` when the comparator is one prediction result, which holds one target and no path.
    """
    path_length
end
"""
    regret_series(pred::PredictionResult)
    regret_series(pred::MultiPeriodPredictionResult)

The realised return series and the timestamps a prediction result was scored on, as [`log_wealth_regret`](@ref) reads them.

The series is the one [`performance_summary`](@ref) reads: the portfolio returns [`predict`](@ref) stored, net of the fee the fold settled, drifted where a Weight Drift ran; a population of paths reads its first path. A multi-period result reads its stacked returns and timestamps.

# Arguments

  - `pred`: The prediction result.

# Returns

  - `ret::VecNum`: The realised return series.
  - `ts::Option{<:AbstractVector}`: The timestamps of the rows, or `nothing` when the fold carried none.

# Related

  - [`log_wealth_regret`](@ref)
  - [`performance_summary`](@ref)
"""
function regret_series(pred::PredictionResult)
    rd = pred.rd
    return isa(rd.X, VecVecNum) ? first(rd.X) : rd.X, rd.ts
end
function regret_series(pred::MultiPeriodPredictionResult)
    mrd = pred.mrd
    return isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X, mrd.ts
end
"""
    first_member(w::VecNum)
    first_member(w::VecVecNum)

The one weight vector of a fold, or the first member's under a population.

A population result carries one vector per member, and the evaluation readers of this file read its first path, as [`regret_series`](@ref) and [`performance_summary`](@ref) do; a single vector is its own first member.

# Arguments

  - `w`: Weights of a fold, one vector or one per member.

# Returns

  - `w::VecNum`: The vector read.

# Related

  - [`fold_target`](@ref)
  - [`regret_series`](@ref)
"""
function first_member(w::VecNum)
    return w
end
function first_member(w::VecVecNum)
    return first(w)
end
"""
    fold_target(pred::PredictionResult)

The weights a fold started from, and, through [`fold_held`](@ref), the weights it held after its last observation, on the fold's own asset names.

`fold_target` is the fold's target on a fold that carries no Held Weights record, and the record's `w0` — its own target on a solved fold, the previous weights it was handed on a failed one, as [`held_start_weights`](@ref) names them — on a fold that carries one. `fold_held` is the same target with no record, because a fold without a drift holds its target to the end, and the record's `w` with one, the drifted holding the next fold trades from. Both read the first member of a population, and both are viewed at the fold's Investable Mask, so they sit on `pred.rd.nx`, the names of the fold's own universe, which is what [`stacked_fold_weights`](@ref) embeds by.

# Arguments

  - `pred`: One fold's prediction result.

# Returns

  - `w::VecNum`: The weights, one entry per name in `pred.rd.nx`.

# Related

  - [`stacked_fold_weights`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`held_start_weights`](@ref)
  - [`result_investable_mask`](@ref)
  - [`investable_weights_view`](@ref)
"""
function fold_target(pred::PredictionResult)
    hw = pred.hw
    w = first_member(isnothing(hw) ? pred.res.w : hw.w0)
    return investable_weights_view(result_investable_mask(pred.res), w)
end
"""
    fold_held(pred::PredictionResult)

The weights a fold held after its last observation, on the fold's own asset names. [`fold_target`](@ref) documents the pair.

# Related

  - [`fold_target`](@ref)
  - [`held_path_turnover`](@ref)
"""
function fold_held(pred::PredictionResult)
    hw = pred.hw
    w = first_member(isnothing(hw) ? pred.res.w : hw.w)
    return investable_weights_view(result_investable_mask(pred.res), w)
end
"""
    stacked_fold_weights(pred::MultiPeriodPredictionResult, read) -> MatNum

One weight vector per fold, read by `read` and embedded by asset name on the union of the folds' universes.

The folds of a multi-period result need not share a universe: a [`Pipeline`](@ref) with an Asset Selector fits each fold on the assets it selected, so a top-1 selector's fold carries one name and one weight. A path over the folds is stated on one axis, so this reader takes the union of every fold's `rd.nx` in order of first appearance, and writes each fold's weights into the columns of its own names, zero elsewhere. A fold whose weights and names disagree in length is refused, because the embedding would misplace them.

# Arguments

  - `pred`: The multi-period prediction result.
  - `read`: The per-fold reader, [`fold_target`](@ref) or [`fold_held`](@ref).

# Validation

  - Every fold's weights have one entry per name of its `rd.nx`.

# Returns

  - `W::MatNum`: `folds × assets` weights on the union universe, one row per fold in split order.

# Related

  - [`fold_target`](@ref)
  - [`comparator_path_length`](@ref)
  - [`held_path_turnover`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function stacked_fold_weights(pred::MultiPeriodPredictionResult, read)
    folds = pred.pred
    ws = [read(f) for f in folds]
    names = unique(mapreduce(f -> collect(f.rd.nx), vcat, folds))
    pos = Dict(n => i for (i, n) in enumerate(names))
    Tf = mapreduce(eltype, promote_type, ws)
    W = zeros(Tf, length(folds), length(names))
    for (i, f) in enumerate(folds)
        nx = f.rd.nx
        @argcheck(length(ws[i]) == length(nx),
                  DimensionMismatch("fold $i carries $(length(ws[i])) weights over $(length(nx)) asset names, so its weights cannot be embedded by name"))
        W[i, [pos[n] for n in nx]] = ws[i]
    end
    return W
end
"""
    comparator_path_length(b::PredictionResult, ::Type{Tf})
    comparator_path_length(b::MultiPeriodPredictionResult, ::Type{Tf})

The path length ``P_T = \\sum_{t \\geq 2} \\lVert u_t - u_{t-1} \\rVert_2`` of a comparator's targets, as [`log_wealth_regret`](@ref) reports it.

A multi-period result's targets are one per fold, read through [`fold_target`](@ref) and stacked by [`stacked_fold_weights`](@ref), and the path length sums the Euclidean distance between consecutive folds' targets; at `test_size = 1` a fold is a row and ``u_t`` the comparator's target on row ``t``. A single prediction result holds one target and no path, and so does a multi-period result of one fold: both answer `NaN`, in the number type `Tf` the regret is stated in.

# Arguments

  - `b`: The comparator's prediction result.
  - `Tf`: Number type of the answer when the comparator has no path.

# Returns

  - `P::Number`: The path length, or `NaN`.

# Related

  - [`log_wealth_regret`](@ref)
  - [`LogWealthRegretResult`](@ref)
  - [`stacked_fold_weights`](@ref)
  - [`HindsightSplit`](@ref)
"""
function comparator_path_length(::PredictionResult, ::Type{Tf}) where {Tf}
    return Tf(NaN)
end
function comparator_path_length(b::MultiPeriodPredictionResult, ::Type{Tf}) where {Tf}
    W = stacked_fold_weights(b, fold_target)
    F = size(W, 1)
    if F < 2
        return Tf(NaN)
    end
    return sum(LinearAlgebra.norm(view(W, i, :) - view(W, i - 1, :)) for i in 2:F)
end
"""
    log_wealth_regret(a::PredRes_MultiPredRes, b::PredRes_MultiPredRes; lags::Integer = 0) -> LogWealthRegretResult

The log-wealth regret of strategy `a` against comparator `b` over one and the same sequence of rows.

Regret is defined over one sequence, so the verb refuses unless the two prediction results carry the same timestamps. Each series is read exactly as it was scored — net of the fee the fold settled, drifted where a Weight Drift ran — so the comparator's fee policy is the caller's, and a fee-free comparator against a fee-paying strategy measures the fee as regret. The comparator is any prediction result over the rows: the same estimator run causally through `cross_val_predict` with the strategy's `cv`, or a Hindsight Comparator, an estimator fit on the rows it is scored on and predicted in sample, `predict(optimise(est, rd_test), rd_test)`. The best constant rebalanced portfolio in hindsight is [`BestConstantRebalancedPortfolio`](@ref), or [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref) when it is bounded; the best stock in hindsight is a [`ScoreSelector`](@ref) under [`RankRule`](@ref)`(; best = 1)` and [`MeanReturn`](@ref)`(; flag = true)` composed with [`EqualWeighted`](@ref) in a [`Pipeline`](@ref).

# Mathematical definition

```math
\\begin{align}
\\delta_t &= \\log\\left(1 + r_{b,t}\\right) - \\log\\left(1 + r_{a,t}\\right)\\,, \\\\
R &= \\sum_{t=1}^{T} \\delta_t\\,, \\quad \\bar{\\delta} = \\frac{R}{T}\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} \\left(1 - \\frac{k}{\\ell + 1}\\right) \\hat{\\gamma}_k\\,, \\\\
z &= \\frac{\\sqrt{T}\\, \\bar{\\delta}}{\\sqrt{\\hat{\\omega}^2}}\\,.
\\end{align}
```

Where:

  - ``r_{a,t}``, ``r_{b,t}``: Realised return of the strategy and of the comparator at period ``t``, as scored.
  - ``\\delta_t``: Per-row difference in log growth.
  - ``R``: Log-wealth regret, positive when the comparator wins.
  - ``\\bar{\\delta}``: Regret per period.
  - ``T``: Number of periods.
  - ``\\hat{\\omega}^2``: Long-run variance of ``\\delta_t``, the Bartlett-kernel estimate with ``\\ell`` lags.
  - ``\\hat{\\gamma}_k``: Sample autocovariance of ``\\delta_t`` at lag ``k``.
  - ``\\ell``: Number of lags, ``0`` by default because the rows of a walk-forward at `test_size = 1` do not overlap.
  - ``z``: Diebold–Mariano–West statistic.

Under equal expected log growth, ``z`` is asymptotically standard normal, and the two-sided `p` reads it against that law. The test is exact only for a comparator that did not read the rows it is scored on; against a Hindsight Comparator it is optimistic by construction, because the comparator was chosen on the very sequence the difference is tested over, and the `p` then overstates the evidence that the comparator is better. Two identical series have ``\\bar{\\delta} = 0`` and ``\\hat{\\omega}^2 = 0``, so their statistic and `p` are `NaN`. Negative regret is expected on many sequences and is not a defect.

The Result also carries the comparator's **path length** ``P_T = \\sum_{t \\geq 2} \\lVert u_t - u_{t-1} \\rVert_2`` over its consecutive targets, and the **running regret**, the cumulative sum of ``\\delta_t``. Dynamic regret is regret against a comparator that moves, and its bounds are stated in ``P_T``, so the two numbers are read together: a static comparator has ``P_T = 0`` and its regret is the static regret; a moving one buys its lower regret with a path the bound charges for. The three comparators the literature states dynamic regret against are the same estimator through three schemes, on the three-row example ``x_1 = (1.2, 0.98)``, ``x_2 = (0.9, 1.1)``, ``x_3 = (1.3, 0.7)`` of price relatives:

  - **Static**: the best constant rebalanced portfolio over all three rows, `predict(optimise(est, rd), rd)`, is all-in on the first asset, wealth ``1.2 \\cdot 0.9 \\cdot 1.3 = 1.404``, ``P_T = 0``. `est` is [`BestConstantRebalancedPortfolio`](@ref) or [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref).
  - **Be-the-leader**: `cross_val_predict(est, rd, HindsightSplit())`, the best constant rebalanced portfolio over the rows through ``t`` played on row ``t``: ``u_1 = (1, 0)``, ``u_2 \\approx (0.523, 0.477)``, ``u_3 = (1, 0)``, wealth ``1.2 \\cdot 0.995 \\cdot 1.3 \\approx 1.553``, ``P_T \\approx 1.35``.
  - **Per-period minimiser**: the top-1 [`ScoreSelector`](@ref) under [`MeanReturn`](@ref)`(; flag = true)` composed with [`EqualWeighted`](@ref) through `HindsightSplit(; prefix = false)`, one-hot on each row's best asset, wealth ``1.2 \\cdot 1.1 \\cdot 1.3 = 1.716``, ``P_T = 2 \\sqrt{2}``.

The path length reads the comparator's per-fold targets, embedded by asset name on the union of the folds' universes through [`stacked_fold_weights`](@ref), and it is `NaN` when the comparator is one prediction result, which holds one target and no path. The bound reads the other way too: the regret at a **given** budget ``L`` is the gap to the best sequence whose path length is at most ``L``, which neither be-the-leader nor the per-period minimiser is, and [`budgeted_hindsight_path`](@ref) builds that sequence as one fold per row for this verb to read.

No scorer and no summary column ship for regret. A hyperparameter search ranks on [`MeanReturn`](@ref)`(; flag = true)`, which is log wealth per period and orders candidates as regret against any fixed comparator would; the performance summary reads one series and holds no comparator.

# Arguments

  - `a`: The strategy's prediction result.
  - `b`: The comparator's prediction result.
  - `lags`: Number of lags of the Bartlett kernel.

# Validation

  - `a` and `b` carry the same timestamps and the same number of rows. An `ArgumentError` is thrown otherwise: two runs over different rows compare nothing.
  - `0 <= lags < n_periods`. A `DomainError` is thrown otherwise.

# Returns

  - `reg::LogWealthRegretResult`: The regret, its per-period form, the per-row difference, its running sum and its test, the two terminal wealths, and the comparator's path length.

# Related

  - [`LogWealthRegretResult`](@ref)
  - [`HindsightSplit`](@ref)
  - [`budgeted_hindsight_path`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`newey_west_variance`](@ref)
  - [`performance_summary`](@ref)
  - [`cross_val_predict`](@ref)

# References

  - $(ref_dict[:lihoi2014])
"""
function log_wealth_regret(a::PredRes_MultiPredRes, b::PredRes_MultiPredRes;
                           lags::Integer = 0)
    ra, tsa = regret_series(a)
    rb, tsb = regret_series(b)
    @argcheck(isequal(tsa, tsb) && length(ra) == length(rb),
              ArgumentError("the two prediction results do not share their rows: regret is defined over one sequence, so both must be scored over the same rows, with the same timestamps. Run both through `cross_val_predict` with the same `rd` and `cv`, or fit the comparator on the rows the strategy was scored on and predict it over them."))
    T = length(ra)
    @argcheck(0 <= lags < T, DomainError(lags, "`lags` must lie in [0, n_periods)"))
    la = log1p.(ra)
    lb = log1p.(rb)
    d = lb .- la
    # The running regret is summed once, so its last entry is the regret and the per-period
    # form is that entry over the rows, as the Result's fields state.
    cumulative = cumsum(d)
    regret = last(cumulative)
    md = regret / T
    v = newey_west_variance(d, lags)
    z = sqrt(T) * md / sqrt(v)
    p = 2 * Distributions.ccdf(Distributions.Normal(), abs(z))
    return LogWealthRegretResult(regret, md, d, cumulative, v, z, p, lags, T, exp(sum(la)),
                                 exp(sum(lb)), comparator_path_length(b, typeof(regret)))
end

"""
    path_norm_epigraph!(model::JuMP.Model, t, x, p::Number)

Write the epigraph ``t \\geq \\lVert x \\rVert_p`` of one step of a comparator path into a bare model.

The cone is the norm's own: the second-order cone at `p = 2`, the norm-one cone at `p = 1`, the norm-infinity cone at `p = Inf`, and otherwise the power-cone form ``r_j^{1/p} t^{1 - 1/p} \\geq \\lvert x_j \\rvert``, ``\\sum_j r_j = t``, one scalar row per entry. It is the epigraph [`norm_ball_dual_norm_epigraph!`](@ref) writes, without the model registry a bare programme has no use for.

# Arguments

  - `model`: The bare model.
  - `t`: The epigraph variable.
  - `x`: The affine expression whose norm is bounded, one entry per asset.
  - `p`: The norm order, `1 <= p`.

# Returns

  - `nothing`.

# Related

  - [`path_budget_constraint!`](@ref)
  - [`norm_ball_dual_norm_epigraph!`](@ref)
"""
function path_norm_epigraph!(model::JuMP.Model, t, x, p::Number)::Nothing
    if p == 2
        JuMP.@constraint(model, [t; x] in JuMP.SecondOrderCone())
    elseif isone(p)
        JuMP.@constraint(model, [t; x] in JuMP.MOI.NormOneCone(1 + length(x)))
    elseif isinf(p)
        JuMP.@constraint(model, [t; x] in JuMP.MOI.NormInfinityCone(1 + length(x)))
    else
        p_inv = inv(p)
        r = JuMP.@variable(model, [1:length(x)])
        for j in eachindex(x)
            JuMP.@constraint(model, [r[j], t, x[j]] in JuMP.MOI.PowerCone(p_inv))
        end
        JuMP.@constraint(model, sum(r) - t == 0)
    end
    return nothing
end
"""
    path_row_constraints!(model::JuMP.Model, u, z, x, wb::WeightBounds)

Write one row of the budgeted path programme: the budget `Σ u = 1`, the resolved bounds `lb ≤ u ≤ ub` where they are finite, and the exponential cone `z ≤ log⟨u, x⟩`.

# Arguments

  - `model`: The bare model.
  - `u`: The row's allocation variables, one per asset.
  - `z`: The row's log-wealth variable.
  - `x`: The row's price relatives.
  - $(arg_dict[:wb])

# Returns

  - `nothing`.

# Related

  - [`budgeted_hindsight_path`](@ref)
  - [`path_budget_constraint!`](@ref)
"""
function path_row_constraints!(model::JuMP.Model, u, z, x, wb::WeightBounds)::Nothing
    JuMP.@constraint(model, sum(u) == 1)
    if w_finite_flag(wb.lb)
        JuMP.@constraint(model, u ⊖ wb.lb >= 0)
    end
    if w_finite_flag(wb.ub)
        JuMP.@constraint(model, u ⊖ wb.ub <= 0)
    end
    JuMP.@constraint(model, [z, 1, LinearAlgebra.dot(x, u)] in JuMP.MOI.ExponentialCone())
    return nothing
end
"""
    path_budget_constraint!(model::JuMP.Model, u, L::Number, p::Number)

Write the path-length budget of the budgeted path programme: one epigraph variable per step through [`path_norm_epigraph!`](@ref), and their sum at most `L`. A path of one row has no step and writes nothing.

# Arguments

  - `model`: The bare model.
  - `u`: The `T × N` allocation variables, one row per period.
  - `L`: The path-length budget.
  - `p`: The norm order.

# Returns

  - `nothing`.

# Related

  - [`budgeted_hindsight_path`](@ref)
  - [`path_row_constraints!`](@ref)
"""
function path_budget_constraint!(model::JuMP.Model, u, L::Number, p::Number)::Nothing
    T = size(u, 1)
    if T < 2
        return nothing
    end
    s = JuMP.@variable(model, [1:(T - 1)])
    for t in 2:T
        path_norm_epigraph!(model, s[t - 1], view(u, t, :) - view(u, t - 1, :), p)
    end
    JuMP.@constraint(model, sum(s) <= L)
    return nothing
end
"""
    path_weight_bounds(wb::WeightBoundsEstimator, sets::UniverseSets, N::Integer, strict::Bool, datatype::DataType)
    path_weight_bounds(wb::WeightBoundsEstimator, sets::Nothing, N::Integer, strict::Bool, datatype::DataType)
    path_weight_bounds(wb::Option{<:WeightBounds}, sets::Option{<:UniverseSets}, N::Integer, strict::Bool, datatype::DataType)

Resolve the weight bounds a budgeted path is solved on, through [`weight_bounds_constraints`](@ref), and refuse by name a [`WeightBoundsEstimator`](@ref) handed no `sets`, which it cannot be resolved without.

# Arguments

  - $(arg_dict[:wb])
  - $(arg_dict[:sets])
  - `N`: Number of assets.
  - $(arg_dict[:strict])
  - `datatype`: Number type of the resolved bounds.

# Validation

  - A `WeightBoundsEstimator` needs `sets`. An `IsNothingError` is thrown otherwise.

# Returns

  - `wb::WeightBounds`: The resolved bounds.

# Related

  - [`budgeted_hindsight_path`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function path_weight_bounds(wb::WeightBoundsEstimator, sets::UniverseSets, N::Integer,
                            strict::Bool, datatype::DataType)
    return weight_bounds_constraints(wb, sets; N = N, strict = strict, datatype = datatype)
end
function path_weight_bounds(::WeightBoundsEstimator, ::Nothing, ::Integer, ::Bool,
                            ::DataType)
    return throw(IsNothingError("`sets` cannot be nothing when `wb` is a WeightBoundsEstimator"))
end
function path_weight_bounds(wb::Option{<:WeightBounds}, sets::Option{<:UniverseSets},
                            N::Integer, strict::Bool, datatype::DataType)
    return weight_bounds_constraints(wb, sets; N = N, strict = strict, datatype = datatype)
end
"""
    budgeted_hindsight_path(rd::ReturnsResult, L::Number; p::Number = 2,
                            wb::Option{<:WbE_Wb} = WeightBounds(),
                            sets::Option{<:UniverseSets} = nothing, strict::Bool = false,
                            slv::Slv_VecSlv) -> MultiPeriodPredictionResult

The best comparator sequence under a path-length budget: the path of per-row allocations whose log wealth over `rd` is the largest among those whose summed step length is at most `L`.

Dynamic regret at a budget is the gap to this path, the comparator of Zinkevich's (2003) Definition 7, and neither per-row comparator a [`HindsightSplit`](@ref) builds is it: be-the-leader is one point at its own path length, and the per-period minimiser is the unbudgeted limit. The verb solves one concave programme over the whole panel, `T × N` variables, so its answer is a path and not one allocation; it is returned as a multi-period prediction result of one fold per row, which [`log_wealth_regret`](@ref) reads unchanged as the comparator, reporting the path's Euclidean length beside the regret.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{u}_1, \\ldots, \\boldsymbol{u}_T}{\\max} &\\quad \\sum_{t=1}^{T} \\log \\left\\langle \\boldsymbol{u}_t, \\boldsymbol{x}_t \\right\\rangle \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{u}_t = 1\\,, \\quad \\boldsymbol{l} \\leq \\boldsymbol{u}_t \\leq \\boldsymbol{h}\\,, \\\\
&\\quad \\sum_{t=2}^{T} \\lVert \\boldsymbol{u}_t - \\boldsymbol{u}_{t-1} \\rVert_p \\leq L\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{u}_t``: The comparator's allocation on row ``t``.
  - ``\\boldsymbol{x}_t``: The price relatives of row ``t``, one plus the returns.
  - ``\\boldsymbol{l}``, ``\\boldsymbol{h}``: The resolved weight bounds, the simplex by default.
  - ``L``: The path-length budget.
  - ``p``: The norm order of the budget.

Each row is one exponential cone on the resolved bounds through [`path_row_constraints!`](@ref), and each step of the path is one norm cone of order `p` through [`path_budget_constraint!`](@ref). At `L = 0` the path is constant and the answer is the best constant rebalanced portfolio in hindsight, [`BestConstantRebalancedPortfolio`](@ref) under the same bounds; at any `L` that the per-period minimiser's own path length does not exceed, the budget is slack and the answer is that minimiser, one-hot on each row's best asset under the simplex bounds. Between the two, the path spends the budget where a switch buys the most log wealth, and the regret against it is the regret at that budget. The Lagrangian form, a penalty on the path length in place of the budget, is the same programme with one term moved and is not built.

The budget is stated in the norm of order `p`, and [`LogWealthRegretResult`](@ref) reports the path length in the Euclidean norm whatever `p` is, so the two numbers agree at `p = 2` alone. A comparator fit on the rows it is scored on is a Hindsight Comparator, and the test of [`log_wealth_regret`](@ref) against it is optimistic by construction.

# Arguments

  - $(arg_dict[:rd]) Its returns matrix is the panel the path is solved over, and its rows are the folds of the answer.
  - `L`: The path-length budget, `0 <= L`.
  - `p`: The norm order of the budget, `1 <= p`.
  - $(arg_dict[:wb])
  - $(arg_dict[:sets])
  - $(arg_dict[:strict])
  - $(arg_dict[:slv])

# Validation

  - `!isnothing(rd.X)`, and every entry of `rd.X` is finite: the programme reads every row, so a missing return has no price relative to read. An `ArgumentError` is thrown on a non-finite entry.
  - `0 <= L` and `1 <= p`. A `DomainError` is thrown otherwise.
  - A vector `slv` is not empty. An `IsEmptyError` is thrown otherwise.
  - A `WeightBoundsEstimator` `wb` comes with `sets`. An `IsNothingError` is thrown otherwise, by [`path_weight_bounds`](@ref).

# Returns

  - `path::MultiPeriodPredictionResult`: One fold per row of `rd`, each a [`PredictionResult`](@ref) whose [`NaiveOptimisationResult`](@ref) holds that row's allocation of the path on the resolved bounds, predicted over that row alone. A programme that no solver of `slv` solved carries `NaN` allocations and an [`OptimisationFailure`](@ref) naming the trials, on every fold, as a failed fold of a walk-forward does.

# Related

  - [`log_wealth_regret`](@ref)
  - [`LogWealthRegretResult`](@ref)
  - [`HindsightSplit`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`path_weight_bounds`](@ref)
  - [`path_row_constraints!`](@ref)
  - [`path_budget_constraint!`](@ref)

# References

  - $(ref_dict[:zinkevich2003])
"""
function budgeted_hindsight_path(rd::ReturnsResult, L::Number; p::Number = 2,
                                 wb::Option{<:WbE_Wb} = WeightBounds(),
                                 sets::Option{<:UniverseSets} = nothing,
                                 strict::Bool = false, slv::Slv_VecSlv)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    @argcheck(all(isfinite, rd.X),
              ArgumentError("every entry of `rd.X` must be finite: the budgeted path reads every row of the panel, so a missing return has no price relative to read."))
    @argcheck(L >= zero(L), DomainError(L, "`L` must be non-negative"))
    @argcheck(p >= one(p), DomainError(p, "`p` must be at least one"))
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
    end
    X = one(eltype(rd.X)) .+ rd.X
    T, N = size(X)
    wb = path_weight_bounds(wb, sets, N, strict, eltype(X))
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    u = JuMP.@variable(model, [1:T, 1:N])
    z = JuMP.@variable(model, [1:T])
    # One scalar row per fold rather than a JuMP container, which JET reads through the
    # container closure and reports a builtin call on.
    for t in 1:T
        path_row_constraints!(model, view(u, t, :), z[t], view(X, t, :), wb)
    end
    path_budget_constraint!(model, u, L, p)
    JuMP.@objective(model, Max, sum(z))
    res = optimise_JuMP_model!(model, slv)
    W, retcode = if res.success
        JuMP.value.(u), OptimisationSuccess()
    else
        fill(eltype(X)(NaN), T, N), OptimisationFailure(; res = res.trials)
    end
    folds = [predict(NaiveOptimisationResult(; pr = nothing, wb = wb, retcode = retcode,
                                             w = W[t, :], fb = nothing),
                     port_opt_view(rd, t:t, :)) for t in 1:T]
    return MultiPeriodPredictionResult(; pred = folds)
end

export log_wealth_regret, LogWealthRegretResult, budgeted_hindsight_path
