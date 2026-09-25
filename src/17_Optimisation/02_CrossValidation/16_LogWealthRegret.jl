"""
$(DocStringExtensions.TYPEDEF)

Holds the log-wealth regret of a strategy against a comparator, and the test of its per-row difference.

[`log_wealth_regret`](@ref) returns it, over one sequence of rows. The regret is the comparator's log terminal wealth less the strategy's, so it is positive when the comparator wins. The per-row difference carries a Newey–West test of equal expected log growth, in the shape of [`CovarianceForecastComparisonResult`](@ref). A negative regret is common and is not a defect: a causal strategy that reads the market's structure can beat a constant portfolio, and a Hindsight Comparator is best only over the class it was chosen from.

Two fields state the comparator's side of a dynamic-regret bound. `path_length` is ``P_T``, the summed Euclidean distance between the comparator's consecutive targets: zero for a constant comparator, and larger with every switch that a [`HindsightSplit`](@ref) comparator makes. `cumulative` is the running regret.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogWealthRegretResult(
        regret, regret_per_period, difference, cumulative, variance, z, p, lags,
        n_periods, wealth_a, wealth_b, path_length
    ) -> LogWealthRegretResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result: [`log_wealth_regret`](@ref) builds it and a caller reads it. It has no keyword constructor and validates nothing of its own.

# Related

  - [`log_wealth_regret`](@ref)
  - [`CovarianceForecastComparisonResult`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`HindsightSplit`](@ref)
  - [`BudgetedHindsightPath`](@ref)
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
    Two-sided p-value of the statistic against the standard normal. It is `NaN` when the two series coincide, because their difference has no variance, and zero when the difference is a non-zero constant.
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

The series is the one [`performance_summary`](@ref) reads: the portfolio returns that [`predict`](@ref) stored, net of the fee the fold settled, and drifted where a Weight Drift ran. A population of paths gives its first path. A multi-period result gives its stacked returns and timestamps.

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

A population result carries one vector per member, and the path readers of the regret take the first, as [`regret_series`](@ref) and [`performance_summary`](@ref) take the first path. A single vector is its own first member.

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

On a fold with no Held Weights record, `fold_target` is the fold's target. On a fold with one, it is the record's `w0`, the start that [`held_start_weights`](@ref) names: the fold's own target when it solved, and the weights it was handed when it failed. `fold_held` is the target on a fold with no record, because a fold without a drift holds its target to the end. On a fold with a record it is the record's `w`, the drifted holding that the next fold trades from.

Both read the first member of a population. Both are viewed at the fold's Investable Mask, so they sit on `pred.rd.nx`, the names of the fold's own universe, which [`stacked_fold_weights`](@ref) embeds by.

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

The folds of a multi-period result need not share a universe. A [`Pipeline`](@ref) with an Asset Selector fits each fold on the assets it selected, so the fold of a top-1 selector carries one name and one weight. A path over the folds is stated on one axis, so each fold's weights go into the columns of its own names on the union axis, and every other entry of its row is zero.

# Algorithm

 1. Read each fold's weights through `read`, giving `ws`.
 2. Take the union of the folds' `rd.nx` in order of first appearance, giving `names`, and map each name to its column, giving `pos`.
 3. Promote the element types of `ws`, giving `Tf`, and allocate `W`, a `folds × assets` matrix of zeros of type `Tf`.
 4. For each fold `i`, check the length of `ws[i]` against its `rd.nx`, and write each weight into row `i` of `W`, at the column that `pos` gives its name.

# Arguments

  - `pred`: The multi-period prediction result.
  - `read`: The per-fold reader, [`fold_target`](@ref) or [`fold_held`](@ref).

# Validation

  - Every fold's weights have one entry per name of its `rd.nx`. A `DimensionMismatch` is thrown otherwise, because the embedding would misplace them.

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
        # `enumerate`, not `zip(nx, ws[i])`: the zip of two untyped values gives JET a false
        # report inside `Base.Iterators` (issue #1341).
        for (j, wn) in enumerate(ws[i])
            W[i, pos[nx[j]]] = wn
        end
    end
    return W
end
"""
    comparator_path_length(b::PredictionResult, ::Type{Tf})
    comparator_path_length(b::MultiPeriodPredictionResult, ::Type{Tf})

The path length of a comparator's targets, as [`log_wealth_regret`](@ref) reports it.

A multi-period result has one target per fold, so the path is over the folds. At `test_size = 1` a fold is a row, and ``\\boldsymbol{u}_t`` is the comparator's target on row ``t``. A single prediction result holds one target and no path, and so does a multi-period result of one fold. Both answer `NaN`, in the number type `Tf` of the regret.

# Mathematical definition

```math
\\begin{align}
P_T &= \\sum_{t=2}^{T} \\lVert \\boldsymbol{u}_t - \\boldsymbol{u}_{t-1} \\rVert_2\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_T_path])
  - ``\\boldsymbol{u}_t``: Comparator's target on fold ``t``, embedded by asset name on the union of the folds' universes, with a zero at every asset outside the fold's own universe.
  - ``T``: Number of folds.

# Algorithm

 1. Read one target per fold through [`fold_target`](@ref), and embed the targets on one axis through [`stacked_fold_weights`](@ref), giving `W`, one row per fold.
 2. When `W` has fewer than two rows, return `NaN` of type `Tf`.
 3. Sum the Euclidean norms of the differences of consecutive rows of `W`.

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

Regret is defined over one sequence, so the verb refuses two prediction results that do not carry the same timestamps. Each series is read as it was scored: net of the fee the fold settled, and drifted where a Weight Drift ran. The comparator's fee policy is therefore the caller's, and a fee-free comparator against a fee-paying strategy measures the fee as regret.

The comparator is any prediction result over the rows. It can be the same estimator run causally through `cross_val_predict` with the strategy's `cv`. It can also be a Hindsight Comparator, an estimator fit on the rows it is scored on and predicted over them, `predict(optimise(est, rd_test), rd_test)`. Two Hindsight Comparators are common:

  - The best constant rebalanced portfolio in hindsight: [`BestConstantRebalancedPortfolio`](@ref) with no solver, or [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref) with a solver, which is exact under any bound.
  - The best stock in hindsight: a [`ScoreSelector`](@ref) under [`RankRule`](@ref)`(; best = 1)` and [`MeanReturn`](@ref)`(; flag = true)`, composed with [`EqualWeighted`](@ref) in a [`Pipeline`](@ref).

No scorer and no summary column ship for regret. A hyperparameter search ranks on [`MeanReturn`](@ref)`(; flag = true)`, the log wealth per period, which orders candidates as the regret against any fixed comparator orders them. The performance summary reads one series and holds no comparator.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= \\log\\left(1 + r_{b,t}\\right) - \\log\\left(1 + r_{a,t}\\right)\\,, \\\\
R &= \\sum_{t=1}^{T} \\delta_t\\,, \\quad \\bar{\\delta} = \\frac{R}{T}\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{\\ell} \\left(1 - \\frac{k}{\\ell + 1}\\right) \\hat{\\gamma}_k\\,, \\\\
z &= \\frac{\\sqrt{T}\\, \\bar{\\delta}}{\\sqrt{\\hat{\\omega}^2}}\\,, \\\\
P_T &= \\sum_{t=2}^{T} \\lVert \\boldsymbol{u}_t - \\boldsymbol{u}_{t-1} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``r_{a,t}``, ``r_{b,t}``: Realised return of the strategy and of the comparator at period ``t``, as scored.
  - ``\\delta_t``: Per-row difference in log growth.
  - ``R``: Log-wealth regret, positive when the comparator wins.
  - ``\\bar{\\delta}``: Regret per period.
  - ``T``: Number of periods.
  - ``\\hat{\\omega}^2``: Long-run variance of ``\\delta_t``, the Bartlett-kernel estimate with ``\\ell`` lags.
  - ``\\hat{\\gamma}_k``: Sample autocovariance of ``\\delta_t`` at lag ``k``, about ``\\bar{\\delta}`` and divided by ``T``.
  - ``\\ell``: Number of lags, ``0`` by default because the rows of a walk-forward at `test_size = 1` do not overlap.
  - ``z``: Diebold–Mariano–West statistic.
  - $(math_dict[:P_T_path])
  - ``\\boldsymbol{u}_t``: Comparator's target on fold ``t``, embedded by asset name on the union of the folds' universes.

Under equal expected log growth, ``z`` is asymptotically standard normal, and the two-sided `p` reads it against that law. The test is exact only for a comparator that did not read the rows it is scored on. Against a Hindsight Comparator it is optimistic by construction: the comparator was chosen on the sequence that the difference is tested over, so `p` overstates the evidence that the comparator is better.

Three degenerate series have a stated answer:

  - Two identical series have ``\\bar{\\delta} = 0`` and ``\\hat{\\omega}^2 = 0``, so ``z`` and `p` are `NaN`.
  - A difference that is a non-zero constant has ``\\hat{\\omega}^2 = 0``, so ``z`` is infinite and `p` is zero. In floating point ``\\hat{\\omega}^2`` is round-off of about ``10^{-33}``, so ``z`` is a very large finite number.
  - A return of ``-1`` is a total loss. Its log growth is ``-\\infty``, so ``R`` is infinite, and ``z`` and `p` are `NaN`.

Dynamic regret is regret against a comparator that moves. Its bounds are stated in ``P_T``, so ``R`` and ``P_T`` are read together: a static comparator has ``P_T = 0`` and its regret is the static regret, and a moving comparator buys its lower regret with a path that the bound charges for. ``P_T`` is `NaN` when the comparator is one prediction result, which holds one target and no path. The literature states dynamic regret against three comparators, and each is the same estimator through its own scheme. On the three-row example ``\\boldsymbol{x}_1 = (1.2, 0.98)``, ``\\boldsymbol{x}_2 = (0.9, 1.1)``, ``\\boldsymbol{x}_3 = (1.3, 0.7)`` of price relatives:

  - **Static**: the best constant rebalanced portfolio over all three rows, `predict(optimise(est, rd), rd)`, is all-in on the first asset. Its wealth is ``1.2 \\cdot 0.9 \\cdot 1.3 = 1.404``, and ``P_T = 0``. `est` is [`BestConstantRebalancedPortfolio`](@ref), or [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref).
  - **Be-the-leader**: `cross_val_predict(est, rd, HindsightSplit())` plays on row ``t`` the best constant rebalanced portfolio over the rows through ``t``. The targets are ``\\boldsymbol{u}_1 = (1, 0)``, ``\\boldsymbol{u}_2 = (23/44, 21/44) \\approx (0.523, 0.477)`` and ``\\boldsymbol{u}_3 = (1, 0)``. The wealth is ``1.2 \\cdot 0.995 \\cdot 1.3 \\approx 1.553``, and ``P_T = 2 \\sqrt{2} \\cdot 21/44 \\approx 1.350``. The first fold trains on one row, so [`MeanRisk`](@ref) needs a prior that fits one row, which [`HindsightSplit`](@ref) states.
  - **Per-period minimiser**: the top-1 [`ScoreSelector`](@ref) under [`MeanReturn`](@ref)`(; flag = true)`, composed with [`EqualWeighted`](@ref) through `HindsightSplit(; prefix = false)`, is one-hot on each row's best asset. Its wealth is ``1.2 \\cdot 1.1 \\cdot 1.3 = 1.716``, and ``P_T = 2 \\sqrt{2}``.

The bound also reads the other way. The regret at a given budget ``L`` is the gap to the best sequence whose path length is at most ``L``. Neither be-the-leader nor the per-period minimiser is that sequence. [`BudgetedHindsightPath`](@ref) fits it, and its prediction result has one fold per row for this verb to read.

# Algorithm

 1. Read the strategy's series and timestamps through [`regret_series`](@ref), giving `ra` and `tsa`, and the comparator's, giving `rb` and `tsb`.
 2. Check that the two series share their timestamps and their length, giving `T`.
 3. Check `lags`, and check that no return is below ``-1``.
 4. Take the log growth of each series, giving `la` and `lb`, and their difference `d`.
 5. Sum `d` cumulatively, giving `cumulative`. Its last entry is `regret`, and `regret / T` is `md`.
 6. Estimate the long-run variance of `d` through [`newey_west_variance`](@ref), giving `v`.
 7. Form the statistic `z` from `md` and `v`, and its two-sided p-value `p`.
 8. Read the comparator's path length through [`comparator_path_length`](@ref).

# Arguments

  - `a`: The strategy's prediction result.
  - `b`: The comparator's prediction result.
  - `lags`: Number of lags of the Bartlett kernel.

# Validation

  - `a` and `b` carry the same timestamps and the same number of rows. An `ArgumentError` is thrown otherwise: two runs over different rows compare nothing.
  - `0 <= lags < n_periods`. A `DomainError` is thrown otherwise.
  - No return of either series is below ``-1``. A `DomainError` is thrown otherwise, because a negative wealth has no log. A `NaN` return passes, and its `NaN` reaches the regret.

# Returns

  - `reg::LogWealthRegretResult`: The regret, its per-period form, the per-row difference, its running sum and its test, the two terminal wealths, and the comparator's path length.

# Related

  - [`LogWealthRegretResult`](@ref)
  - [`HindsightSplit`](@ref)
  - [`BudgetedHindsightPath`](@ref)
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
    @argcheck(!any(<(-1), ra) && !any(<(-1), rb),
              DomainError(minimum(filter(<(-1), vcat(ra, rb))),
                          "a return below -1 leaves a negative wealth, whose log is not defined, so the log-wealth regret cannot read it"))
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

Writes the epigraph ``t \\geq \\lVert \\boldsymbol{x} \\rVert_p`` of one step of a comparator path into a bare model.

The cone is the norm's own at `p = 2`, `p = 1` and `p = Inf`. Every other order takes the power-cone form, one row per entry. It is the epigraph that [`norm_ball_dual_norm_epigraph!`](@ref) writes, without the model registry, which a bare programme does not use.

# JuMP formulation

## Variables

  - `t`: read. It is the caller's epigraph variable.
  - `r`: created at an order other than `1`, `2` and `Inf`, an anonymous vector of one variable per entry of ``\\boldsymbol{x}``.

## Constraints

Every row is registered under no name.

  - At ``p = 2``, the second-order cone row ``t \\geq \\lVert \\boldsymbol{x} \\rVert_2``.
  - At ``p = 1``, the norm-one cone row ``t \\geq \\lVert \\boldsymbol{x} \\rVert_1``.
  - At ``p = \\infty``, the norm-infinity cone row ``t \\geq \\lVert \\boldsymbol{x} \\rVert_\\infty``.
  - At any other ``p``, one power cone row per entry, ``r_j^{1/p}\\, t^{1 - 1/p} \\geq \\lvert x_j \\rvert`` with ``r_j \\geq 0`` and ``t \\geq 0``, and the row ``\\sum_j r_j - t = 0``.

Where:

  - ``t``: Epigraph variable of the step.
  - ``\\boldsymbol{x}``: Affine expression whose norm is bounded, with entries ``x_j``.
  - $(math_dict[:p_norm_order])
  - ``r_j``: Auxiliary variable of entry ``j``.

The power-cone rows are exact. Each row gives ``r_j \\geq \\lvert x_j \\rvert^p / t^{p-1}``, and the sum of the rows gives ``t^p \\geq \\sum_j \\lvert x_j \\rvert^p``, which is ``t \\geq \\lVert \\boldsymbol{x} \\rVert_p``.

# Arguments

  - $(arg_dict[:model])
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
    row_bound_view(b::Number, m::AbstractVector{Bool})
    row_bound_view(b::VecNum, m::AbstractVector{Bool})

One side of a resolved weight bound on the investable assets of a row.

A scalar bound applies to every asset, so it passes through. A vector bound is viewed at the row's mask.

# Arguments

  - `b`: One side of the resolved bound, a number or one entry per asset.
  - `m`: The row's Investable Mask.

# Returns

  - `b`: The scalar bound, or the view of the vector bound at `m`.

# Related

  - [`path_row_constraints!`](@ref)
"""
function row_bound_view(b::Number, ::AbstractVector{Bool})
    return b
end
function row_bound_view(b::VecNum, m::AbstractVector{Bool})
    return view(b, m)
end
"""
    path_row_constraints!(model::JuMP.Model, u, z, x, m::AbstractVector{Bool}, wb::WeightBounds)

Writes one row of the budgeted path programme on the row's investable assets.

The budget, the finite bounds and the log wealth of the row read the assets that `m` keeps. The allocation of every other asset is fixed at zero, because a row with no price relative for an asset cannot hold it.

# JuMP formulation

## Variables

  - `u`: read. It is the row's allocations, one per asset. Every entry outside `m` is fixed at zero with `JuMP.fix`, which sets the variable's bounds and adds no row.
  - `z`: read. It is the row's log-wealth variable.

## Constraints

Every row is registered under no name, and each reads the assets in ``m`` alone.

  - ``\\sum_{i \\in m} u_i = 1``.
  - ``u_i - l_i \\geq 0`` for each ``i \\in m``, when an entry of ``\\boldsymbol{l}`` is finite.
  - ``u_i - h_i \\leq 0`` for each ``i \\in m``, when an entry of ``\\boldsymbol{h}`` is finite.
  - The exponential cone row ``\\left(z, 1, \\sum_{i \\in m} x_i u_i\\right) \\in \\mathcal{K}_{\\exp}``, which holds when ``z \\leq \\log \\sum_{i \\in m} x_i u_i``.

Where:

  - ``u_i``: Allocation of asset ``i`` on the row.
  - ``z``: Log-wealth variable of the row.
  - ``x_i``: Price relative of asset ``i`` on the row, one plus its return.
  - ``m``: Set of the assets in the row's Investable Mask.
  - ``l_i``, ``h_i``: Resolved lower and upper weight bounds of asset ``i``, the entries of ``\\boldsymbol{l}`` and ``\\boldsymbol{h}``.
  - ``\\mathcal{K}_{\\exp} = \\mathrm{cl}\\,\\{(a, b, c) : b\\, e^{a / b} \\leq c,\\, b > 0\\}``: Exponential cone.

# Arguments

  - $(arg_dict[:model])
  - `u`: The row's allocation variables, one per asset.
  - `z`: The row's log-wealth variable.
  - `x`: The row's price relatives, finite where `m` holds.
  - `m`: The row's Investable Mask, the Coverage Universe of the one-row window.
  - $(arg_dict[:wb])

# Returns

  - `nothing`.

# Related

  - [`BudgetedHindsightPath`](@ref)
  - [`path_budget_constraint!`](@ref)
  - [`row_bound_view`](@ref)
"""
function path_row_constraints!(model::JuMP.Model, u, z, x, m::AbstractVector{Bool},
                               wb::WeightBounds)::Nothing
    ui = view(u, m)
    JuMP.@constraint(model, sum(ui) == 1)
    if w_finite_flag(wb.lb)
        JuMP.@constraint(model, ui ⊖ row_bound_view(wb.lb, m) >= 0)
    end
    if w_finite_flag(wb.ub)
        JuMP.@constraint(model, ui ⊖ row_bound_view(wb.ub, m) <= 0)
    end
    JuMP.@constraint(model,
                     [z, 1, LinearAlgebra.dot(view(x, m), ui)] in
                     JuMP.MOI.ExponentialCone())
    for i in findall(!, m)
        JuMP.fix(u[i], 0; force = true)
    end
    return nothing
end
"""
    path_budget_constraint!(model::JuMP.Model, u, L::Number, p::Number)

Writes the path-length budget of the budgeted path programme.

Each step of the path takes one epigraph variable through [`path_norm_epigraph!`](@ref), and the sum of these variables is at most `L`. A path of one row has no step, so the function writes nothing.

# JuMP formulation

## Variables

  - `u`: read. It is the `T × N` allocation variables, one row per period.
  - `s`: created, an anonymous vector of ``T - 1`` step lengths.
  - `d`: created, an anonymous ``(T - 1) \\times N`` matrix of steps.

## Constraints

Every row is registered under no name.

  - ``\\boldsymbol{d}_{t-1} = \\boldsymbol{u}_t - \\boldsymbol{u}_{t-1}``, for ``t = 2, \\ldots, T``, one row per entry.
  - The rows of [`path_norm_epigraph!`](@ref) on ``s_{t-1} \\geq \\lVert \\boldsymbol{d}_{t-1} \\rVert_p``, for ``t = 2, \\ldots, T``.
  - ``\\sum_{t=1}^{T-1} s_t \\leq L``.

Where:

  - ``\\boldsymbol{u}_t``: Allocation of period ``t``, one entry per asset.
  - ``\\boldsymbol{d}_t``: Step from period ``t`` to period ``t + 1``.
  - ``s_t``: Length of step ``t`` in the norm of order ``p``.
  - ``L``: Path-length budget.
  - $(math_dict[:p_norm_order])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:model])
  - `u`: The `T × N` allocation variables, one row per period.
  - `L`: The path-length budget.
  - `p`: The norm order.

# Returns

  - `nothing`.

# Related

  - [`BudgetedHindsightPath`](@ref)
  - [`path_row_constraints!`](@ref)
"""
function path_budget_constraint!(model::JuMP.Model, u, L::Number, p::Number)::Nothing
    T = size(u, 1)
    if T < 2
        return nothing
    end
    s = JuMP.@variable(model, [1:(T - 1)])
    d = JuMP.@variable(model, [1:(T - 1), 1:size(u, 2)])
    for t in 2:T
        JuMP.@constraint(model, view(d, t - 1, :) .== view(u, t, :) - view(u, t - 1, :))
        path_norm_epigraph!(model, s[t - 1], view(d, t - 1, :), p)
    end
    JuMP.@constraint(model, sum(s) <= L)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Fits the path of per-row allocations with the largest log wealth whose summed step length is at most `L`.

Dynamic regret at a budget is the gap to this path, the comparator of Zinkevich's (2003) Definition 7. Neither per-row comparator that a [`HindsightSplit`](@ref) builds is this path: be-the-leader is one point at its own path length, and the per-period minimiser is the limit with no budget. The fit is one concave programme over the whole panel. The Lagrangian form, which puts a penalty on the path length in place of the budget, is the same programme with one term moved, and the library does not build it.

The estimator follows the Hindsight Comparator rule of [`log_wealth_regret`](@ref), as every other comparator does. [`optimise`](@ref) returns a [`BudgetedHindsightPathResult`](@ref), which holds the path. [`predict`](@ref) over the same rows returns one fold per row, and it refuses any other rows. So `predict(optimise(est, rd_test), rd_test)` is a prediction result that the regret verb reads unchanged, and the verb reports its Euclidean path length beside the regret.

Each row's universe is the Coverage Universe of its one-row window, as [`coverage_mask`](@ref) derives it: an asset is in the row when its return is finite and the Asset Panel's active mask is `true` there. A static panel, or no panel, reads finiteness alone. A [`HindsightSplit`](@ref)`(; prefix = false)` fold fits on the same universe, so the two per-row comparators agree on what a row can hold. The result carries the masks, so `predict` views each row at its own mask and names no Held Gap. A row with no asset in its universe is refused, because its budget of one cannot be met.

The budget is stated in the norm of order `p`, but [`LogWealthRegretResult`](@ref) reports the path length in the Euclidean norm. The two numbers agree at `p = 2` alone. A tight budget on a long panel puts most step cones at their apex, where an interior-point solver can stall short of its tolerance and report insufficient progress. A first-order solver solves such a programme, so on a panel of many rows the robust `slv` is a solver vector with a first-order solver as the fallback. The fit reads the rows it is scored on, so the test of [`log_wealth_regret`](@ref) against it is optimistic by construction.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{u}_1, \\ldots, \\boldsymbol{u}_T}{\\max} &\\quad \\sum_{t=1}^{T} \\log \\sum_{i \\in \\mathcal{M}_t} u_{t,i}\\, x_{t,i} \\\\
\\textrm{s.t.} &\\quad \\sum_{i \\in \\mathcal{M}_t} u_{t,i} = 1\\,, \\quad l_i \\leq u_{t,i} \\leq h_i \\;\\; \\forall i \\in \\mathcal{M}_t\\,, \\quad u_{t,i} = 0 \\;\\; \\forall i \\notin \\mathcal{M}_t\\,, \\\\
&\\quad \\sum_{t=2}^{T} \\lVert \\boldsymbol{u}_t - \\boldsymbol{u}_{t-1} \\rVert_p \\leq L\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{u}_t``: Comparator's allocation on row ``t``, with entries ``u_{t,i}``.
  - $(math_dict[:x_t_rel])
  - ``x_{t,i}``: Entry ``i`` of ``\\boldsymbol{x}_t``, one plus the return of asset ``i`` on row ``t``.
  - ``\\mathcal{M}_t``: Coverage Universe of row ``t``, the assets that the row can hold.
  - ``l_i``, ``h_i``: Resolved lower and upper weight bounds of asset ``i``, zero and one by default.
  - ``L``: Path-length budget.
  - $(math_dict[:p_norm_order])
  - $(math_dict[:T])

Three consequences follow from the programme:

  - At ``L = 0`` the path is constant. It holds only the assets that every row covers, and it is the best constant rebalanced portfolio in hindsight over them, as [`BestConstantRebalancedPortfolio`](@ref) fits it under the same bounds. When no asset is in every row's universe, the programme is infeasible.
  - When ``L`` is at least the path length of the per-period minimiser, the budget is slack. Under the default bounds the answer is then that minimiser, one-hot on each row's best asset, when each row has one best asset.
  - An asset that delists inside the panel is outside every later row's universe. The path therefore sells it on its last row at any budget, and the forced step counts towards the path length like any other step.

Between the two ends the path spends the budget where a switch buys the most log wealth, and the regret against it is the regret at that budget.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BudgetedHindsightPath(;
        L::Number,
        slv::Slv_VecSlv,
        p::Number = 2,
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:UniverseSets} = nothing,
        strict::Bool = false,
        fb::Option{<:BudgetedHindsightPath} = nothing
    ) -> BudgetedHindsightPath

Keywords correspond to the struct's fields. `L` and `slv` are required: a budget has no default, and the fit is a programme.

## Validation

  - `0 <= L` and `1 <= p`. A `DomainError` is thrown otherwise.
  - A vector `slv` is not empty. An `IsEmptyError` is thrown otherwise.
  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`. An `IsNothingError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> BudgetedHindsightPath(; L = 5 * sqrt(2), slv = Solver(; solver = nothing))
BudgetedHindsightPath
       L ┼ Float64: 7.0710678118654755
       p ┼ Int64: 2
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    sets ┼ nothing
  strict ┼ Bool: false
     slv ┼ Solver
         │          name ┼ String: ""
         │        solver ┼ nothing
         │      settings ┼ nothing
         │     check_sol ┼ @NamedTuple{}: NamedTuple()
         │   add_bridges ┴ Bool: true
      fb ┴ nothing
```

# Related

  - [`BudgetedHindsightPathResult`](@ref)
  - [`optimise`](@ref)
  - [`predict`](@ref)
  - [`log_wealth_regret`](@ref)
  - [`HindsightSplit`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:zinkevich2003])
"""
@propagatable @concrete struct BudgetedHindsightPath <: OptimisationEstimator
    """
    The path-length budget, `0 <= L`.
    """
    L
    """
    The norm order of the budget, `1 <= p`.
    """
    p
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:slv])
    """
    slv
    """
    Fallback estimator, another `BudgetedHindsightPath` that [`optimise`](@ref) runs when the fit fails, or `nothing`.
    """
    fb
    function BudgetedHindsightPath(L::Number, p::Number, wb::Option{<:WbE_Wb},
                                   sets::Option{<:UniverseSets}, strict::Bool,
                                   slv::Slv_VecSlv, fb::Option{<:BudgetedHindsightPath})
        @argcheck(L >= zero(L), DomainError(L, "`L` must be non-negative"))
        @argcheck(p >= one(p), DomainError(p, "`p` must be at least one"))
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        return new{typeof(L), typeof(p), typeof(wb), typeof(sets), typeof(strict),
                   typeof(slv), typeof(fb)}(L, p, wb, sets, strict, slv, fb)
    end
end
function BudgetedHindsightPath(; L::Number, slv::Slv_VecSlv, p::Number = 2,
                               wb::Option{<:WbE_Wb} = WeightBounds(),
                               sets::Option{<:UniverseSets} = nothing, strict::Bool = false,
                               fb::Option{<:BudgetedHindsightPath} = nothing)::BudgetedHindsightPath
    return BudgetedHindsightPath(L, p, wb, sets, strict, slv, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds a budgeted hindsight path, one allocation per row of the panel it was solved over.

[`BudgetedHindsightPath`](@ref) fits it. It also carries the rows' Investable Masks, the resolved bounds, the rows' names and clock, and the return code. The path is bound to its rows, so [`predict`](@ref) answers one fold per row over those rows alone, and refuses a returns result whose names, clock or row count differ. A fit that no solver solved carries `NaN` on every investable entry, and an [`OptimisationFailure`](@ref) that names the trials.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BudgetedHindsightPathResult(;
        w::MatNum,
        wb::WeightBounds,
        retcode::OptimisationReturnCode,
        imsk::Option{<:BitMatrix} = nothing,
        nx::Option{<:AbstractVector} = nothing,
        ts::Option{<:AbstractVector} = nothing,
        fb::Option{<:FbChain} = nothing
    ) -> BudgetedHindsightPathResult

Keywords correspond to the struct's fields. The type is a Result: [`optimise`](@ref) builds it and a caller reads it. It validates nothing of its own.

# Related

  - [`BudgetedHindsightPath`](@ref)
  - [`predict`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`log_wealth_regret`](@ref)
"""
@concrete struct BudgetedHindsightPathResult <: OptimisationResult
    """
    The path, `rows × assets`, one allocation per row on the full universe, with an exact zero at every non-investable entry and `NaN` on a failed fit.
    """
    w
    """
    The resolved weight bounds the path was solved on.
    """
    wb
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    The rows' Investable Masks, `rows × assets`, each row the Coverage Universe of its one-row window, or `nothing` when every row covers every asset.
    """
    imsk
    """
    The asset names of the panel the path was solved over, or `nothing` when it carried none.
    """
    nx
    """
    The clock of the panel the path was solved over, or `nothing` when it carried none.
    """
    ts
    """
    The fallback chain [`optimise`](@ref) walked to reach this result, or `nothing` when the first fit answered.
    """
    fb
    function BudgetedHindsightPathResult(w::MatNum, wb::WeightBounds,
                                         retcode::OptimisationReturnCode,
                                         imsk::Option{<:BitMatrix},
                                         nx::Option{<:AbstractVector},
                                         ts::Option{<:AbstractVector},
                                         fb::Option{<:FbChain})
        return new{typeof(w), typeof(wb), typeof(retcode), typeof(imsk), typeof(nx),
                   typeof(ts), typeof(fb)}(w, wb, retcode, imsk, nx, ts, fb)
    end
end
function BudgetedHindsightPathResult(; w::MatNum, wb::WeightBounds,
                                     retcode::OptimisationReturnCode,
                                     imsk::Option{<:BitMatrix} = nothing,
                                     nx::Option{<:AbstractVector} = nothing,
                                     ts::Option{<:AbstractVector} = nothing,
                                     fb::Option{<:FbChain} = nothing)::BudgetedHindsightPathResult
    return BudgetedHindsightPathResult(w, wb, retcode, imsk, nx, ts, fb)
end
"""
    factory(res::BudgetedHindsightPathResult, fb::Option{<:FbChain})

Rebuilds a budgeted path result with the fallback chain `fb` that [`optimise`](@ref) walked, and keeps every other field.

# Arguments

  - `res`: The budgeted path result.
  - `fb`: The fallback chain, or `nothing`.

# Returns

  - `res::BudgetedHindsightPathResult`: A new result with `fb` in place of the old chain.

# Related

  - [`BudgetedHindsightPathResult`](@ref)
  - [`FbChain`](@ref)
"""
function factory(res::BudgetedHindsightPathResult, fb::Option{<:FbChain})
    return BudgetedHindsightPathResult(res.w, res.wb, res.retcode, res.imsk, res.nx, res.ts,
                                       fb)
end
"""
    row_coverage_mask(rd::ReturnsResult, t::Integer) -> Option{BitVector}

The Coverage Universe of row `t` of `rd` alone, through [`coverage_mask`](@ref) on the one-row view.

# Arguments

  - $(arg_dict[:rd])
  - `t`: The row.

# Validation

  - The row has at least one asset in its Coverage Universe. An `IsEmptyError` that names the row is thrown otherwise, through [`row_universe_error`](@ref). Any other error of [`coverage_mask`](@ref) passes through as it is.

# Returns

  - `msk::Option{BitVector}`: The row's mask, one entry per asset, or `nothing` when the row covers every asset.

# Related

  - [`path_row_masks`](@ref)
  - [`coverage_mask`](@ref)
"""
function row_coverage_mask(rd::ReturnsResult, t::Integer)
    rdt = port_opt_view(rd, t:t, :)
    try
        return coverage_mask(rdt.X, rdt.pnl)
    catch err
        throw(row_universe_error(err, t))
    end
end
"""
    row_universe_error(err::IsEmptyError, t::Integer) -> IsEmptyError
    row_universe_error(err::Exception, t::Integer) -> Exception

The error that [`row_coverage_mask`](@ref) throws for row `t`.

The `IsEmptyError` of an empty Coverage Universe is restated to name the row. Any other error is returned as it is.

# Related

  - [`row_coverage_mask`](@ref)
"""
function row_universe_error(::IsEmptyError, t::Integer)
    return IsEmptyError("row $t of `rd.X` has no asset in its Coverage Universe, so the path cannot hold its budget of one there: every return is non-finite or every asset is inactive on that row.")
end
row_universe_error(err::Exception, ::Integer) = err
"""
    path_row_masks(rd::ReturnsResult) -> Option{BitMatrix}

The Investable Mask of every row of a budgeted path.

Row `t` is the Coverage Universe of the one-row window `t`: `true` at every asset whose return is finite and whose Asset Panel active mask is `true` on that row. A delisted or unlisted asset is therefore out of the row, as it is out of the fit of a `HindsightSplit(; prefix = false)` fold.

# Algorithm

 1. Start from `msk`, a `rows × assets` matrix of `true`.
 2. For each row `t`, derive the row's Coverage Universe through [`row_coverage_mask`](@ref), giving `cm`, and write it into row `t` of `msk` when it is not `nothing`.
 3. Return `nothing` when every entry of `msk` is `true`, and `msk` otherwise.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - Every row has at least one asset in its Coverage Universe. An `IsEmptyError` naming the row is thrown otherwise, because the row's budget of one cannot be met.

# Returns

  - `msk::Option{BitMatrix}`: The masks, `rows × assets`, or `nothing`.

# Related

  - [`BudgetedHindsightPath`](@ref)
  - [`row_coverage_mask`](@ref)
  - [`path_row_constraints!`](@ref)
"""
function path_row_masks(rd::ReturnsResult)
    T, N = size(rd.X)
    msk = trues(T, N)
    for t in 1:T
        cm = row_coverage_mask(rd, t)
        if !isnothing(cm)
            msk[t, :] = cm
        end
    end
    return all(msk) ? nothing : msk
end

"""
    _optimise(est::BudgetedHindsightPath, rd::ReturnsResult; dims::Int = 1, kwargs...) -> BudgetedHindsightPathResult

Solves the budgeted path programme of [`BudgetedHindsightPath`](@ref) over the rows of `rd`.

[`optimise`](@ref) is the door, and it walks `est.fb` on a failure. The value of a variable fixed at zero is the solver's zero, so the fit writes an exact zero at every entry outside a row's mask.

# Algorithm

 1. Check that `rd.X` is not `nothing` and that `dims` is `1`.
 2. Add one to the returns, giving the price relatives `X`, of size `T × N`.
 3. Derive the rows' Investable Masks through [`path_row_masks`](@ref), giving `imsk`, and `msk`, which is all `true` when `imsk` is `nothing`.
 4. Resolve the weight bounds through [`weight_bounds_constraints`](@ref), giving `wb`.
 5. Make a bare model with the allocation variables `u` and the log-wealth variables `z`.
 6. Write each row through [`path_row_constraints!`](@ref), and the budget through [`path_budget_constraint!`](@ref).
 7. Set the objective, and solve the model through [`optimise_JuMP_model!`](@ref), giving `res`.
 8. Read the path `W` from `u` and `OptimisationSuccess` on a success. On a failure, fill `W` with `NaN` and make an `OptimisationFailure` that names the trials.
 9. Set every entry of `W` outside `msk` to zero.

# JuMP formulation

## Variables

  - `u`: created, an anonymous `T × N` matrix of allocations, one row per period.
  - `z`: created, an anonymous vector of ``T`` log-wealth variables, one per period.

## Constraints

The rows of [`path_row_constraints!`](@ref), one call per period, which give ``z_t \\leq \\log \\sum_{i \\in \\mathcal{M}_t} u_{t,i}\\, x_{t,i}``, and the rows of [`path_budget_constraint!`](@ref). The function registers no row itself.

## Objective

  - `Max`: ``\\sum_{t=1}^{T} z_t``, the log wealth of the path.

Where:

  - ``z_t``: Log-wealth variable of period ``t``.
  - ``u_{t,i}``: Allocation of asset ``i`` in period ``t``.
  - ``x_{t,i}``: Price relative of asset ``i`` in period ``t``, one plus its return.
  - ``\\mathcal{M}_t``: Coverage Universe of period ``t``.
  - $(math_dict[:T])

# Arguments

  - `est`: The estimator.
  - $(arg_dict[:rd]) Its returns matrix is the panel the path is solved over.
  - `dims`: Must be `1`. A `ReturnsResult` is always observations × assets, so `dims == 2` throws `ConflictingArgumentError`.
  - `kwargs`: Ignored.

# Validation

  - `!isnothing(rd.X)`. An `IsNothingError` is thrown otherwise.
  - Every row has at least one asset in its Coverage Universe, through [`path_row_masks`](@ref).

# Returns

  - `res::BudgetedHindsightPathResult`: The path, or the failure.

# Related

  - [`BudgetedHindsightPath`](@ref)
  - [`BudgetedHindsightPathResult`](@ref)
  - [`path_row_masks`](@ref)
  - [`path_row_constraints!`](@ref)
  - [`path_budget_constraint!`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
function _optimise(est::BudgetedHindsightPath, rd::ReturnsResult; dims::Int = 1, kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    assert_returns_result_dims(dims)
    X = one(eltype(rd.X)) .+ rd.X
    T, N = size(X)
    imsk = path_row_masks(rd)
    msk = isnothing(imsk) ? trues(T, N) : imsk
    wb = weight_bounds_constraints(est.wb, est.sets; N = N, strict = est.strict,
                                   datatype = eltype(X))
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    u = JuMP.@variable(model, [1:T, 1:N])
    z = JuMP.@variable(model, [1:T])
    # One scalar row per period rather than a JuMP container, which JET reads through the
    # container closure and reports a builtin call on.
    for t in 1:T
        path_row_constraints!(model, view(u, t, :), z[t], view(X, t, :), view(msk, t, :),
                              wb)
    end
    path_budget_constraint!(model, u, est.L, est.p)
    JuMP.@objective(model, Max, sum(z))
    res = optimise_JuMP_model!(model, est.slv)
    W, retcode = if res.success
        JuMP.value.(u), OptimisationSuccess()
    else
        fill(eltype(X)(NaN), T, N), OptimisationFailure(; res = res.trials)
    end
    # A fixed variable's value is the solver's zero; the gap's is exact.
    W[.!msk] .= zero(eltype(W))
    return BudgetedHindsightPathResult(; w = W, wb = wb, retcode = retcode, imsk = imsk,
                                       nx = rd.nx, ts = rd.ts)
end
"""
    predict(res::BudgetedHindsightPathResult, rd::ReturnsResult) -> MultiPeriodPredictionResult

The prediction result of a budgeted path over the rows it was solved on: one fold per row, each a [`PredictionResult`](@ref) whose [`NaiveOptimisationResult`](@ref) holds that row's allocation on the resolved bounds, with the row's Investable Mask when the row has a missing return, predicted over that row alone.

The path is bound to its rows, so `rd` must be the panel the path was solved over: the same names, the same clock and the same row count. `predict(optimise(est, rd_test), rd_test)` is then the Hindsight Comparator rule of [`log_wealth_regret`](@ref), and the regret verb reads the folds unchanged.

# Arguments

  - `res`: The fit.
  - $(arg_dict[:rd]) The panel the path was solved over.

# Validation

  - `rd` carries the names, the clock and the row count of `res`. An `ArgumentError` is thrown otherwise: a path over other rows is another fit.

# Returns

  - `pred::MultiPeriodPredictionResult`: One fold per row.

# Related

  - [`BudgetedHindsightPath`](@ref)
  - [`BudgetedHindsightPathResult`](@ref)
  - [`log_wealth_regret`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function StatsAPI.predict(res::BudgetedHindsightPathResult, rd::ReturnsResult)
    assert_path_rows(res, rd)
    return MultiPeriodPredictionResult(;
                                       pred = [path_row_fold(res, rd, t)
                                               for t in axes(res.w, 1)])
end
"""
    assert_path_rows(res::BudgetedHindsightPathResult, rd::ReturnsResult)

Refuse a returns result that is not the panel a budgeted path was solved over: the names, the clock and the row count must be the fit's, because the path is bound to its rows. An `ArgumentError` is thrown otherwise.

# Related

  - [`predict`](@ref)
  - [`BudgetedHindsightPathResult`](@ref)
"""
function assert_path_rows(res::BudgetedHindsightPathResult, rd::ReturnsResult)::Nothing
    same = !isnothing(rd.X) &&
           size(rd.X, 1) == size(res.w, 1) &&
           isequal(rd.nx, res.nx) &&
           isequal(rd.ts, res.ts)
    @argcheck(same,
              ArgumentError("the path was solved over other rows: `predict` reads the panel the `BudgetedHindsightPath` was fit on, with the same asset names, timestamps and row count, because the path is bound to its rows. Fit the estimator on the rows to score."))
    return nothing
end
"""
    path_row_fold(res::BudgetedHindsightPathResult, rd::ReturnsResult, t::Integer) -> PredictionResult

One fold of a budgeted path's prediction result: row `t`'s allocation as a [`NaiveOptimisationResult`](@ref) on the resolved bounds, predicted over row `t` of `rd` alone.

A row with a gap carries its Investable Mask, so `predict` views the row and the weights at the mask and never reads the gap. A row with no gap carries `nothing`, as a fit on the full universe does.

# Related

  - [`predict`](@ref)
  - [`BudgetedHindsightPathResult`](@ref)
"""
function path_row_fold(res::BudgetedHindsightPathResult, rd::ReturnsResult, t::Integer)
    mt = isnothing(res.imsk) || all(view(res.imsk, t, :)) ? nothing : res.imsk[t, :]
    wt = isnothing(mt) ? res.w[t, :] : res.w[t, mt]
    return predict(NaiveOptimisationResult(; pr = nothing, wb = res.wb,
                                           retcode = res.retcode, w = wt, imsk = mt,
                                           fb = nothing), port_opt_view(rd, t:t, :))
end

export log_wealth_regret, LogWealthRegretResult, BudgetedHindsightPath,
       BudgetedHindsightPathResult
