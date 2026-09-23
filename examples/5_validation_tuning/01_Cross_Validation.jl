#=
```@meta
Description = "Cross-validation in PortfolioOptimisers.jl: walk-forward, K-fold and combinatorial splitters, the metrics they compute and their plots."
```

# Cross validation

Cross-validation scores a model on data that the fit did not see. This example shows the
cross-validation schemes of PortfolioOptimisers.jl, how to run them on a portfolio
optimisation, and the measures and plots you can compute from their results.

You can use cross-validation on its own to score an estimator, or inside a hyperparameter
search. [`NestedClustered`](@ref) and [`Stacking`](@ref) also take a scheme in their `cv`
field, wrapped in an [`OptimisationCrossValidation`](@ref). They then fit the outer estimator
on the out-of-sample returns of the inner estimators.
=#
using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
#=
## 1. Setting up

We use five years of daily data, so that all the schemes have enough rows for their
training windows and their test windows.

Cross-validation fits the estimator again on the training rows of every fold. Its fields must
therefore be estimators, such as a prior estimator, and not results computed beforehand as in
the earlier examples. `cross_val_predict` throws an error for a JuMP optimiser with a
precomputed prior.
=#

using CSV, TimeSeries, DataFrames, Clarabel, Statistics

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 5):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.70),
              check_sol = (; allow_local = true, allow_almost = true))];

#=
We use a [`MeanRisk`](@ref) estimator. Cross-validation works for every optimisation estimator
except the finite allocations, [`DiscreteAllocation`](@ref) and [`GreedyAllocation`](@ref), and
it also works for an estimator that computes a Pareto frontier.
=#

mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

#=
## 2. Cross validation
### 2.1 KFold

The simplest scheme is K-fold. It cuts the data into K consecutive folds, trains on K - 1 of
them and tests on the fold that is left. It does this K times, and each fold is the test set
once.

You can make the [`KFold`](@ref) indices without an optimisation. With 5 folds, a fold is one
year of 252 rows.
=#

kfold = KFold(; n = 5)

#=
To show the splits, we make them with [`split`](@ref). `cross_val_predict` makes them itself,
so you do not need this step.
=#

kfold_res = split(kfold, rd)

show(kfold_res.train_idx)
show(kfold_res.test_idx)

#=
We run the cross-validation.
=#
kfold_pred = cross_val_predict(mr, rd, kfold)

#=
Four plots show the cumulative returns, the weight distribution, the turnover and the score
of every fold.
=#

using StatsPlots, GraphRecipes
# The cumulative returns of the portfolio over all the KFold test windows.
plot_portfolio_cumulative_returns(kfold_pred)
# The distribution of the weight of every asset over the folds.
plot_weight_stability(kfold_pred)
# The turnover of the portfolio between consecutive folds.
plot_turnover(kfold_pred)
# The score of every fold, here the second moment, which is the variance.
plot_cv_scores(LowOrderMoment(; alg = SecondMoment()), kfold_pred)

#=
The result is a [`MultiPeriodPredictionResult`](@ref). Its field `pred` is a vector of
[`PredictionResult`](@ref) values, one per fold. Every [`PredictionResult`](@ref) keeps the
optimisation result of its training fold in `res`, and a [`PredictionReturnsResult`](@ref) of
the optimised portfolio on its test fold in `rd`.

You can index `pred` to get one fold. The property `mrd` joins the returns of all the folds
into one [`PredictionReturnsResult`](@ref), and the property `res` gives the vector of the
optimisation results of the folds. This K-fold has no purge and no embargo, and the test folds
cover all the rows once. We compare the timestamps of `mrd` with the timestamps of the returns.
=#

println("isequal(kfold_pred.mrd.ts, rd.ts) = $(isequal(kfold_pred.mrd.ts, rd.ts))")

#=
You can also compute risk measures on the predicted returns. A prediction holds only the return
series of the portfolio. A measure that needs the weights or data per asset cannot score it,
and `expected_risk` throws an error for such a measure. These measures cannot score a
prediction: [`StandardDeviation`](@ref), [`Variance`](@ref), [`UncertaintySetVariance`](@ref),
[`NegativeSkewness`](@ref), [`TurnoverRiskMeasure`](@ref), [`TrackingRiskMeasure`](@ref) with
[`WeightsTracking`](@ref), [`EqualRisk`](@ref), [`ExpectedReturn`](@ref) and
[`ExpectedReturnRiskRatio`](@ref). Neither can `RiskTrackingRiskMeasure`,
`VarianceSkewKurtosis`, a `ValueatRisk` with `DistributionValueatRisk`, a moment measure whose
target `mu` is a vector per asset, or a ratio that uses one of these. Most of them have a
replacement.

  - For the variance, use [`LowOrderMoment`](@ref) with `alg = SecondMoment()`. For the
    standard deviation, use it with `alg = SecondMoment(; alg2 = SOCRiskExpr())`.
  - For [`NegativeSkewness`](@ref), use [`HighOrderMoment`](@ref) or [`Skewness`](@ref).
  - For [`ExpectedReturn`](@ref) and [`ExpectedReturnRiskRatio`](@ref), use
    [`MeanReturn`](@ref) and [`MeanReturnRiskRatio`](@ref).

We compute the variance.
=#

println("KFold(5) prediction variance = $(expected_risk(LowOrderMoment(; alg = SecondMoment()), kfold_pred))")

#=
### 2.2 Combinatorial

[`CombinatorialCrossValidation`](@ref) cuts the rows into `n_folds` consecutive folds. Every
choice of `n_test_folds` of them is the test set of one split, and the other folds are its
training set. A purge and an embargo remove the training rows next to the test folds. The test
folds of the splits then join into paths, and a path covers all the rows once. The scheme needs
one fit per split, many more than K-fold, and it gives many paths instead of one.

[`optimal_number_folds`](@ref) chooses `n_folds` and `n_test_folds` from the number of rows, a
target size of the training set and a target number of paths. The keywords `train_size_w` and
`n_test_paths_w` set the weight of each target. The cell names the third argument
`target_test_size`, but it is the target number of paths.
=#

T = size(rd.X, 1)
target_train_size = 200
target_test_size = 70
n_folds, n_test_folds = optimal_number_folds(T, target_train_size, target_test_size)
cfold = CombinatorialCrossValidation(; n_folds = n_folds, n_test_folds = n_test_folds)

#=
We make the splits.
=#

cfold_res = split(cfold, rd)

#=
The scheme uses 13 folds with 11 test folds, so it has `binomial(13, 11) = 78` splits. A
split trains on only 2 folds, about 194 rows, and tests on the other 11. The 78 splits give
`78 × 11 = 858` test folds, which join into 66 paths.

`cfold_res.path_ids` has one row per test fold of a split and one column per split. Every
entry is the number of the path that the test fold belongs to.
=#

cfold_res.path_ids

#=
We run the cross-validation. The result has one prediction per path.
=#

cfold_pred = cross_val_predict(mr, rd, cfold)

#=
Every one of the 66 paths is an out-of-sample prediction. To score the model, you pick one path
or summarise them. The median path is a common choice, because outliers do not move it. You
can pick a path with your own function or with a subtype of [`PredictionScorer`](@ref).
[`NearestQuantilePrediction`](@ref) computes a measure on all the paths, finds a quantile of the
values, by default the median, and returns the first path whose value is nearest to it. It
leaves out paths whose optimisation failed or whose measure is not finite.

We use the mean return over the variance as the measure. The cell names the scorer
`sharpe_scorer`, but this ratio divides by the variance, not by the standard deviation.
=#

sharpe_scorer = NearestQuantilePrediction(;
                                          r = MeanReturnRiskRatio(;
                                                                  rk = LowOrderMoment(;
                                                                                      alg = SecondMoment())))

#=
A scorer is a callable object. It takes the population of paths and returns the path it picks.
The field `id` of that path is its position in the population.
=#

median_pred_max_sharpe = sharpe_scorer(cfold_pred)

#=
We compare the path with the entry `id` of `cfold_pred.pred`.
=#
median_pred_max_sharpe === cfold_pred.pred[median_pred_max_sharpe.id]

#=
As for K-fold, the scheme has no purge and no embargo, so a path covers all the rows. The
cell compares the timestamps of the path with those of the returns.
=#
isequal(median_pred_max_sharpe.mrd.ts, rd.ts)

#=
We compute the ratio for all the paths and find the path nearest to the median by hand, to
compare it with the pick of the scorer. This comparison leaves out no path. It matches the
scorer only when no path failed.
=#

sharpe_ratios = expected_risk(MeanReturnRiskRatio(;
                                                  rk = LowOrderMoment(;
                                                                      alg = SecondMoment())),
                              cfold_pred)
argmin(abs.(sharpe_ratios .- median(sharpe_ratios))) == median_pred_max_sharpe.id

#=
The next plot shows the distribution of the weight of each asset over the paths. A wide
distribution means that the weights depend on which folds the model trains on.
=#

plot_weight_stability(cfold_pred)

#=
`plot_cv_scores` draws the ratio of every path. A wide spread means that
the ratio depends on which folds form the path.
=#

plot_cv_scores(MeanReturnRiskRatio(; rk = LowOrderMoment(; alg = SecondMoment())),
               cfold_pred)

#=
Any measure that scores a return series works in the scorer. We now pick the median path by
variance.
=#

variance_scorer = NearestQuantilePrediction(; r = LowOrderMoment(; alg = SecondMoment()))
median_pred_min_variance = variance_scorer(cfold_pred)

#=
The `id` of this path also indexes it in `cfold_pred.pred`.
=#
median_pred_min_variance === cfold_pred.pred[median_pred_min_variance.id]

#=
We compare its timestamps with those of the returns once more.
=#
isequal(median_pred_min_variance.mrd.ts, rd.ts)

#=
### 2.3 WalkForward

There are two walk-forward schemes, [`IndexWalkForward`](@ref) and [`DateWalkForward`](@ref).
The first sizes its windows in rows. The second sizes them in dates, and you can use Julia's
`Dates` module to put the window boundaries on chosen dates.

A walk-forward trains on the past and tests on the rows that follow, which is how you use a
model in practice. The folds run in time order, so a fold can use the weights of the fold
before it. A non-fixed [`Turnover`](@ref), a `WeightsTracking` or a `TurnoverRiskMeasure` takes
those previous weights as its reference.

#### 2.3.1 IndexWalkForward

[`IndexWalkForward`](@ref) is the simpler of the two, so we start with it. We train on one
year, 252 rows, and test on the next quarter, 63 rows. There is no purge, and a test window
starts on the row after its training window. The first test window starts at row 253, and the
test windows together cover the returns from there to the end.
=#

idx_walk_forward = IndexWalkForward(252, round(Int, 252 / 4))
idx_walk_forward_res = split(idx_walk_forward, rd)
show(idx_walk_forward_res.train_idx)
show(idx_walk_forward_res.test_idx)

#=
We run the walk-forward.
=#
idx_walkforward_pred = cross_val_predict(mr, rd, idx_walk_forward)

#=
We compare its timestamps with the timestamps of the returns from row 253 on.
=#

isequal(idx_walkforward_pred.mrd.ts, rd.ts[253:end])

# The cumulative return over all the test windows, joined in time order.

plot_portfolio_cumulative_returns(idx_walkforward_pred)

# The weights of every fold as a stacked bar.

plot_composition(idx_walkforward_pred)

# The turnover at every rebalance, without a turnover constraint.

plot_turnover(idx_walkforward_pred)

#=
Every column of the table holds the weights of one fold.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(idx_walkforward_pred.res, :w)),
                            Symbol.(1:16))); formatters = [resfmt])

#=
Some assets change weight by more than 20 % between two consecutive folds. Four tools limit
this when their reference weights are not fixed: a turnover constraint, a turnover fee, a
turnover risk measure and a tracking of the previous weights. We use a turnover constraint
that lets no asset's weight change by more than 2 % from one fold to the next. Equal weights
are the reference of the first fold. A [`Turnover`](@ref) has `fixed = false` by default, and
a fold then takes the weights of the fold before as its reference.
=#
N = size(rd.X, 2)
tn = Turnover(; w = range(; start = 1 / N, stop = 1 / N, length = N), val = 0.02)

#=
We add the constraint to the optimiser and run the walk-forward again.
=#
mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tn = tn))
idx_tn_walkforward_pred = cross_val_predict(mr, rd, idx_walk_forward)

#=
The table gives the weights again, so you can compare every column with the one before it.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(idx_tn_walkforward_pred.res, :w)),
                            Symbol.(1:16))); formatters = [resfmt])

#=
The constraint limits the change of each asset, not the total. The turnover of a rebalance is
the sum of the absolute changes over the 20 assets, so it can be larger than 2 %.
=#

# The cumulative returns with the turnover constraint.
plot_portfolio_cumulative_returns(idx_tn_walkforward_pred)
# The turnover at every rebalance.
plot_turnover(idx_tn_walkforward_pred)
# The distribution of the weight of every asset over the folds.
plot_weight_stability(idx_tn_walkforward_pred)

#=
#### 2.3.2 DateWalkForward

[`DateWalkForward`](@ref) works like [`IndexWalkForward`](@ref), but you give the windows in
dates. Use it to align the windows with calendar periods, such as fiscal years or quarters.

We use `lastdayofmonth` from `Dates` to define an adjuster. It moves the dates of a range to
the last day of their month, and drops a month end that comes after the last date of the
range.
=#

function ldm(x)
    val = lastdayofmonth.(x)
    while !isempty(val)
        if val[end] > x[end]
            val = val[1:(end - 1)]
        else
            break
        end
    end
    return val
end;

#=
The first argument, the training size, is an integer count of steps of the date range, or a
`Period` or `CompoundPeriod` from `Dates`. The second argument, the test size, is an integer
count of steps. `period` sets the step of the date range, and `adjuster` changes the range.
Here the step is one month and the adjuster moves the dates to month ends, so we train on 12
months and test on 3.

The scheme maps each date of the range to a row. A date that is a timestamp maps to its own
row. A date between two timestamps maps to the earlier one when `previous = true`, and to the
later one when `previous = false`. A month end can fall on a weekend, so we set
`previous = true`. A test window then starts on the last trading day of a month.
=#

date_walk_forward = DateWalkForward(12, 3; period = Month(1), adjuster = ldm,
                                    previous = true)

#=
We look at the splits of the new scheme.
=#

date_walk_forward_res = split(date_walk_forward, rd)
show(date_walk_forward_res.train_idx)
show(date_walk_forward_res.test_idx)

#=
We run the new scheme with the turnover constraint.
=#

date_tn_walkforward_pred = cross_val_predict(mr, rd, date_walk_forward)

#=
The date walk-forward has 15 folds, and the table has one column for each.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(date_tn_walkforward_pred.res, :w)),
                            Symbol.(1:15))); formatters = [resfmt])

#=
The two plots show the cumulative returns and the turnover over the month-end windows.
=#

# The cumulative returns over the month-end windows.
plot_portfolio_cumulative_returns(date_tn_walkforward_pred)
# The turnover at every month-end rebalance.
plot_turnover(date_tn_walkforward_pred)

#=
The date walk-forward uses different windows from the index walk-forward, so its weights
differ.
The training windows of the two schemes cover almost the same dates, and the turnover
constraint limits the change of an asset to 2 % in both.

[`MultipleRandomised`](@ref) is one more scheme. It runs a walk-forward on random subsets of the
assets, and section 5.2 of the pipelines example uses it.
=#
