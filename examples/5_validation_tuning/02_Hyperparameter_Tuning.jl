#=
```@meta
Description = "Hyperparameter tuning in PortfolioOptimisers.jl: grid and randomised search over any estimator, scored on cross-validation folds."
```

# [Hyperparameter tuning](@id example-hyperparameter-tuning)

Hyperparameter tuning chooses the parameters of an estimator by how they score on the test
folds of a cross-validation. This example shows the two searches of PortfolioOptimisers.jl, a
grid search and a randomised search.
=#
using PortfolioOptimisers, PrettyTables
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

We use the same five years of daily data as the cross-validation example.

As in the cross-validation example, the fields of the estimator you tune must be estimators,
not precomputed results. The search fits each candidate again on each fold.
=#

using CSV, TimeSeries, DataFrames, Clarabel, Statistics, StableRNGs, Distributions

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 5):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

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
## 2. Hyperparameter tuning

We tune a [`Stacking`](@ref) estimator, and the search works the same way for the other
optimisation estimators, such as [`NestedClustered`](@ref). A key of the grid is a string with
the property path to the parameter, such as `"opti[1].opt.l2.val"`. The search parses the
string into an [`Accessors.jl`](https://juliaobjects.github.io/Accessors.jl/stable/) lens, an object
that returns a copy of the estimator with that one field changed.

A search needs a risk measure, `r`, to score every fold, and a cross-validation scheme, `cv`,
which defaults to [`KFold`](@ref). It also accepts a [`WalkForwardEstimator`](@ref), a
[`CombinatorialCrossValidation`](@ref) and a [`MultipleRandomised`](@ref). Under a
combinatorial scheme, it scores each path instead of each fold.
=#

opt = JuMPOptimiser(; slv = slv)
#! `l2` is `nothing` by default, and the key `l2.val` needs an `L2Regularisation` in it.
optl2 = JuMPOptimiser(; slv = slv, l2 = L2Regularisation())
r = MeanReturnRiskRatio(; rk = LowOrderMoment(; alg = SecondMoment()))
st = Stacking(; opti = [MeanRisk(; opt = optl2), RiskBudgeting(; opt = opt)],
              opto = MeanRisk(; opt = opt))

#=
### 2.1 Grid cross validation search

[`GridSearchCrossValidation`](@ref) tries every point of a parameter grid with
[`search_cross_validation`](@ref), and keeps the point with the best mean test score.

A grid is a vector of pairs. The first item of a pair is the key of a parameter, and the second
is a vector of the values to try. A dictionary of the same pairs also works. The search takes
the product of the values in a grid. A `Dict` has no fixed order, which matters to a
randomised search, as section 2.2 explains.

You can give a vector of grids. The search expands each grid on its own and joins the results
into one list of candidates.

A literal vector of grids whose values have different types has an abstract element type, and
the constructor throws a `MethodError` for it. [`concrete_typed_array`](@ref) narrows the
element type to the union of the types of the grids, which the constructor accepts. A vector
of dictionaries needs the same call.

We search four grids, and the search keeps the best candidate over all of them.
=#

p = concrete_typed_array([["opti[2].opt.l1" =>
                               range(; start = 0.0005, stop = 0.0008, length = 3),
                           "opti[1].opt.l2.val" =>
                               range(; start = 0.0004, stop = 0.0007, length = 3)],
                          ["opti[1].opt.l2.val" =>
                               range(; start = 0.0004, stop = 0.0007, length = 3)],
                          ["opti[2].opt.l1" =>
                               range(; start = 0.0009, stop = 0.0012, length = 3)],
                          ["opti[2]" => [MeanRisk(; opt = opt, obj = MaximumUtility()),
                                         MeanRisk(; opt = opt, obj = MaximumRatio())]]])
gs_cv = GridSearchCrossValidation(p; r = r)
#=
We run the search. The four grids give `3 × 3 + 3 + 3 + 2 = 17` candidates. The result has
these fields.

  - `opt` is the best estimator.
  - `test_scores` has one row per fold and one column per candidate.
  - `train_scores` is `nothing` unless you set `train_score = true`.
  - `lens_grid` and `val_grid` give the keys and the values of every candidate.
  - `idx` is the index of the best candidate.

The best candidate has the highest mean test score. The search negates a measure for which a
lower value is better, such as a risk measure, and keeps a ratio as it is. Here the measure is
a ratio, and the highest mean ratio wins.
=#
gs_res1 = search_cross_validation(st, gs_cv, rd)

#=
The table gives the keys and the values of the best candidate. The loop after it prints every
value next to the same field read from the chosen estimator.
=#

pretty_table(DataFrame("Lens" => gs_res1.lens_grid[gs_res1.idx],
                       "Value" => collect(gs_res1.val_grid[gs_res1.idx])))

for (lens, val) in zip(gs_res1.lens_grid[gs_res1.idx], gs_res1.val_grid[gs_res1.idx])
    println("$(lpad("lens:", 12)) $lens\n$(lpad("val:", 12)) $val\n$(lpad("Field value:", 12)) $(lens(gs_res1.opt))\n")
end

#=
We optimise the best estimator on the full data and plot its weights.
=#

using StatsPlots, GraphRecipes

res_gs1 = optimise(gs_res1.opt, rd)
plot_composition(res_gs1, rd)

#=
### 2.2 Randomised cross validation search

[`RandomisedSearchCrossValidation`](@ref) draws `n_iter` trials from the grid, and then runs a
grid search over the trials with [`search_cross_validation`](@ref). It takes the same grids as
[`GridSearchCrossValidation`](@ref). A parameter can also take a
[`Distributions.Distribution`](https://juliastats.org/Distributions.jl/latest/types/#Distributions)
in place of a vector of values.

A trial is one candidate. It sets every parameter of one grid at the same time, so the search
fits `n_iter` candidates, however many parameters it tunes.

  - If every value is a vector, the search draws `n_iter` distinct candidates of the grid
    search, each with the same probability. It keeps them in the order of the grid search. If
    the grid holds `n_iter` candidates or fewer, the search takes all of them.
  - If a value is a distribution, each trial picks one grid with equal probability. Then it
    draws one value from each distribution and one entry from each vector of that grid. Two
    trials can then be the same.

The search draws from one random number generator, `rng`, trial by trial and parameter by
parameter. The order of the grids and of the parameters therefore changes the trials. To repeat
a search, keep that order fixed. Use an `OrderedDict` from
[`OrderedCollections`](https://juliacollections.github.io/OrderedCollections.jl/stable/) or a vector
for each grid.

#### 2.2.1 Sampling from a predefined parameter space

We sample from the same grids as before, with `n_iter = 8`. Every value is in a vector, so the
search draws 8 of the 17 candidates of the grid search, without replacement.
=#
rs_cv1 = RandomisedSearchCrossValidation(p; rng = StableRNG(42), r = r, n_iter = 8)

#=
The search fits 8 candidates, where the grid search fitted 17.
=#
rs_res1 = search_cross_validation(st, rs_cv1, rd)

#=
We read the best candidate of the randomised search in the same way.
=#

pretty_table(DataFrame("Lens" => rs_res1.lens_grid[rs_res1.idx],
                       "Value" => collect(rs_res1.val_grid[rs_res1.idx])))

for (lens, val) in zip(rs_res1.lens_grid[rs_res1.idx], rs_res1.val_grid[rs_res1.idx])
    println("$(lpad("lens:", 12)) $lens\n$(lpad("val:", 12)) $val\n$(lpad("Field value:", 12)) $(lens(rs_res1.opt))\n")
end

res_rs1 = optimise(rs_res1.opt, rd)
plot_composition(res_rs1, rd)

#=
#### 2.2.2 Sampling from a distribution

We now mix vectors and distributions, with `n_iter = 10`. This grid needs no
[`concrete_typed_array`](@ref). A distribution among the values makes the literal a vector of
vectors of `Pair{String, Any}`, which the constructor of [`RandomisedSearchCrossValidation`](@ref)
accepts. The constructor checks that every value is a vector or a distribution. A grid of vectors
alone still needs `concrete_typed_array`.
=#
p = [["opti[2].opt.l1" => range(; start = 0.0005, stop = 0.0008, length = 3),
      "opti[1].opt.l2.val" => LogUniform(0.0003, 0.1)],
     ["opti[1].opt.l2.val" => LogUniform(0.001, 0.1)],
     ["opti[2].opt.l1" => range(; start = 0.0009, stop = 0.0012, length = 3)],
     ["opti[2]" => [MeanRisk(; opt = opt, obj = MaximumUtility()),
                    MeanRisk(; opt = opt, obj = MaximumRatio())]]]

rs_cv2 = RandomisedSearchCrossValidation(p; rng = StableRNG(42), r = r, n_iter = 10)

#=
Two grids hold a distribution, so each of the 10 trials picks one of the four grids with
equal probability. A trial of the first grid draws one value of `opti[1].opt.l2.val` from its
distribution and one of the three values of `opti[2].opt.l1`. A trial of the fourth grid picks
one of the two estimators. That makes 10 candidates. Two trials of the fourth grid can pick the
same estimator, and then two candidates are the same.
=#
rs_res2 = search_cross_validation(st, rs_cv2, rd)

#=
We read off the best candidate of the second search.
=#

pretty_table(DataFrame("Lens" => rs_res2.lens_grid[rs_res2.idx],
                       "Value" => collect(rs_res2.val_grid[rs_res2.idx])))

for (lens, val) in zip(rs_res2.lens_grid[rs_res2.idx], rs_res2.val_grid[rs_res2.idx])
    println("$(lpad("lens:", 12)) $lens\n$(lpad("val:", 12)) $val\n$(lpad("Field value:", 12)) $(lens(rs_res2.opt))\n")
end

res_rs2 = optimise(rs_res2.opt, rd)
plot_composition(res_rs2, rd)

#=
The plot puts the weights of the three best portfolios side by side: the grid search, the
randomised search over vectors, and the randomised search with distributions.
=#

plot_stacked_bar_composition([res_gs1, res_rs1, res_rs2], rd)

#=
You can tune any estimator that cross-validation accepts. You can also tune a
[`Pipeline`](@ref), and the pipelines example tunes an asset filter and a gap fill.
=#
