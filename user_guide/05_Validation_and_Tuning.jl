#=
```@meta
Description = "Walk-forward and combinatorial cross-validation, and cross-validated hyperparameter search, for any PortfolioOptimisers.jl optimiser."
```

# Validation and tuning

Cross-validation scores a strategy on data that it was not fitted to, and a search chooses the
parameters of the strategy by that score. `PortfolioOptimisers.jl` has cross-validation splitters,
and a cross-validated search over parameters, that work with any optimiser. This page shows the
minimal path. For the other splitters and searches, see the
[validation and tuning examples](../examples/5_validation_tuning/01_Cross_Validation.md).
=#

using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

#=
## 1. Cross-validation

A cross-validation splitter divides the dates into folds. [`KFold`](@ref) is the simplest. It makes
`n` folds of consecutive dates, and each fold is the test fold once, while the other folds train the
optimiser. [`cross_val_predict`](@ref) fits the optimiser on the training folds of each split,
predicts the returns of the portfolio on the test fold, and joins the predictions. So you can score
the strategy on data that it did not see.
=#

kfold = KFold(; n = 3)
pred = cross_val_predict(mr, rd, kfold)

#=
We compute the risk of the joined predictions with [`expected_risk`](@ref). The measure here is
the variance of the predicted returns.
=#

cv_risk = expected_risk(LowOrderMoment(; alg = SecondMoment()), pred)

#=
[`CombinatorialCrossValidation`](@ref) takes every combination of a number of test folds out of
the folds, and joins the test folds into several paths through the dates. You get several
out-of-sample paths in place of one, for more fits. See
[Cross Validation](../examples/5_validation_tuning/01_Cross_Validation.md). A walk-forward can also
update one estimator from fold to fold instead of refitting it on each fold, with
[`OnlineIndexWalkForward`](@ref). See [The online walk-forward](09_Online_Walk_Forward.md).

## 2. Hyperparameter tuning

[`GridSearchCrossValidation`](@ref) tries every combination of a grid of parameters, and keeps the
one with the best score on the test folds. The grid is a list of `"path" => values` pairs. The
path is a string that names a field inside the estimator, and the library turns it into a lens of
[Accessors.jl](https://github.com/JuliaObjects/Accessors.jl). A scoring measure, such as
[`MeanReturnRiskRatio`](@ref), ranks the candidates. [`search_cross_validation`](@ref) runs the
search and returns the tuned estimator in its `opt` field. We tune the strength of the L1 penalty
of our `MeanRisk`.
=#

score = MeanReturnRiskRatio(; rk = LowOrderMoment(; alg = SecondMoment()))
grid = [["opt.l1" => [0.001, 0.01, 0.05]]]

gs_res = search_cross_validation(mr, GridSearchCrossValidation(grid; r = score), rd)

#=
We optimise the tuned estimator on the full sample, with the same call as any other estimator.
=#

res_tuned = optimise(gs_res.opt, rd)

#=
[`RandomisedSearchCrossValidation`](@ref) draws candidates from the grid, or from distributions,
instead of trying every one. It costs less on a large grid. See
[Hyperparameter Tuning](../examples/5_validation_tuning/02_Hyperparameter_Tuning.md).

## 3. Time-dependent inputs

Under cross-validation, each fold is a separate optimisation over its own dates. An input that
defines the problem, such as a constraint, a prior, a risk measure, an objective or the fallback
optimiser, can change from fold to fold. Put a vector with one entry for each fold, or a function
of the fold's context, in a [`TimeDependent`](@ref), which this page calls a schedule. Then store
the schedule in the field that it changes. The fold loop uses entry `i` for fold `i`. An input that
controls how the problem runs, such as a solver or a random number generator, stays the same on
every fold. We tighten the cap on the weight of each asset as a walk-forward moves forward.
=#

wf = IndexWalkForward(126, 42)
n = n_splits(wf, rd)
bounds = [WeightBounds(; lb = 0.0, ub = ub) for ub in range(0.35, 0.2, n)]
caps = TimeDependent(bounds)
mr_caps = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = caps))
pred_caps = cross_val_predict(mr_caps, rd, wf)

#=
An [`OnlinePortfolioSelection`](@ref) takes a schedule of its allocation set, the set of weights
that its rule can hold, in the same way. The loop puts entry `i` in place before it processes the
rows of fold `i`. So the weights at each row of fold `i`, and the weights that the fold reports, lie
inside the set of entry `i`. Each entry is a whole allocation set. To change one bound and keep the
others, write the whole set in each entry, or build it in a function of the fold's context.
=#

caps_online = TimeDependent([BoundedAllocationSet(; wb = wb) for wb in bounds])
ops_caps = OnlinePortfolioSelection(; alg = ExponentiatedGradient(), set = caps_online)
pred_ops = cross_val_predict(ops_caps, rd, OnlineIndexWalkForward(126, 42))

#=
The entries of a schedule can be whole optimisers, so that the strategy changes from fold to fold.
You can pass such a schedule to [`cross_val_predict`](@ref) as the optimiser. Outside a fold there
is then no fixed optimiser, so the schedule names the optimiser that [`optimise`](@ref) runs there,
in its `default` keyword.
=#

iv = InverseVolatility()
strategies = TimeDependent([isodd(i) ? mr : iv for i in 1:n]; default = mr)
pred_switch = cross_val_predict(strategies, rd, wf)

#=
Outside a fold loop, `optimise` does not read the entries of a schedule. It uses the schedule's
`default`, or else the value that the constructor gives the field. For schedules in the fields of
a meta-optimiser, functions that read the data of the fold, and schedules that hold computed
results, see
[Time Dependent Constraints](../examples/5_validation_tuning/04_Time_Dependent_Constraints.md)
and [Time Dependent Optimisers](../examples/5_validation_tuning/06_Time_Dependent_Optimisers.md).

## 4. Cross-validation scores

[`plot_cv_scores`](@ref) plots the out-of-sample score of each fold, so you can see how much the
score of the strategy changes from fold to fold.
=#

plot_cv_scores(LowOrderMoment(; alg = SecondMoment()), pred)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow guide page split from the validation/tuning examples. KFold(3) cross_val_predict
#src   + GridSearchCrossValidation over a tiny l1 grid; small folds/grid to keep the page fast.
#src   Verified on kaimon (f102cae9): KFold(3) pred risk 0.0001001; GridSearch picks best l1=0.05.
#src - GOTCHA: `l1` must be strictly > 0 (0 < l1), so a grid including 0.0 throws DomainError. Use
#src   strictly-positive values; to compare "no regularisation" run a separate unregularised fit.
#src   The grid path is a string lens ("opt.l1") via Accessors — same mechanism as the examples.
