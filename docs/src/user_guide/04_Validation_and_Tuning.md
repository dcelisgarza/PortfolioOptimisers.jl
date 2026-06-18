The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/04_Validation_and_Tuning.jl"
```

# Validation and tuning

Before trusting a strategy you want to know how it behaves on data it was not fitted to, and you
want its hyperparameters chosen by that out-of-sample performance rather than by hand.
`PortfolioOptimisers.jl` provides cross-validation splitters and cross-validated parameter
search that work with *any* optimiser. This page shows the minimal path; for the full menu of
splitters and search strategies see the
[validation & tuning examples](../examples/5_validation_tuning/01_Cross_Validation.md).

````@example 04_Validation_and_Tuning
using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
````

## 1. Cross-validation

A cross-validation splitter partitions the timeline into training and testing folds.
[`KFold`](@ref) is the simplest — `n` contiguous folds, each held out in turn.
[`cross_val_predict`](@ref) fits the optimiser on each training fold and stitches the
out-of-sample predictions back together, so you can score the strategy on data it never saw.

````@example 04_Validation_and_Tuning
kfold = KFold(; n = 3)
pred = cross_val_predict(mr, rd, kfold)
````

The stitched predictions carry an out-of-sample realised risk, here the second moment of the
predicted returns via [`expected_risk`](@ref).

````@example 04_Validation_and_Tuning
cv_risk = expected_risk(LowOrderMoment(; alg = SecondMoment()), pred)
````

For a more exhaustive evaluation, [`CombinatorialCrossValidation`](@ref) scores every
train/test fold combination — heavier, but a fuller picture. See
[Cross Validation](../examples/5_validation_tuning/01_Cross_Validation.md).

## 2. Hyperparameter tuning

[`GridSearchCrossValidation`](@ref) searches a parameter grid and keeps the combination that
scores best on the test folds. The grid is a list of `"path" => values` pairs, where the path is
a string lens into the estimator (parsed by [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl));
a scoring rule like [`MeanReturnRiskRatio`](@ref) ranks the candidates.
[`search_cross_validation`](@ref) runs the search and returns the tuned estimator in its `opt`
field. Here we tune the L1 regularisation strength of our `MeanRisk`.

````@example 04_Validation_and_Tuning
score = MeanReturnRiskRatio(; rk = LowOrderMoment(; alg = SecondMoment()))
grid = [["opt.l1" => [0.001, 0.01, 0.05]]]

gs_res = search_cross_validation(mr, GridSearchCrossValidation(grid; r = score), rd)
````

The tuned estimator optimises like any other.

````@example 04_Validation_and_Tuning
res_tuned = optimise(gs_res.opt, rd)
````

[`RandomisedSearchCrossValidation`](@ref) samples the grid (or distributions) instead of
enumerating it — cheaper for large spaces; see
[Hyperparameter Tuning](../examples/5_validation_tuning/02_Hyperparameter_Tuning.md).

## 3. Cross-validation scores

[`plot_cv_scores`](@ref) visualises the per-fold out-of-sample scores — a quick read on how
stable the strategy is across the timeline.

````@example 04_Validation_and_Tuning
plot_cv_scores(LowOrderMoment(; alg = SecondMoment()), pred)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
