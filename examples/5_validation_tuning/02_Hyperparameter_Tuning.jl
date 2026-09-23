#=
```@meta
Description = "Hyperparameter tuning in PortfolioOptimisers.jl: grid and randomised search over any estimator, scored on cross-validation folds."
```

# Hyperparameter tuning

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
string into an [`Accessors.jl`](https://github.com/JuliaObjects/Accessors.jl) lens, an object
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

[`RandomisedSearchCrossValidation`](@ref) samples values from the grid, and then runs a grid
search over the samples with [`search_cross_validation`](@ref). It takes the same grids as
[`GridSearchCrossValidation`](@ref). A parameter can also take a
[`Distributions.Distribution`](https://juliastats.org/Distributions.jl/latest/types/#Distributions)
in place of a vector of values.

The search samples every parameter on its own, and `n_iter` sets the number of samples.

  - From a distribution, it draws `n_iter` values.
  - From a vector, it draws `min(n_iter, length(v))` values. It draws without replacement,
    unless the grid also holds a distribution. Then it draws with replacement, and it can draw
    a value twice.

Then it takes the product of the samples in each grid, as the grid search does. A vector of
grids works as it does for [`GridSearchCrossValidation`](@ref).

The search draws from one random number generator, `rng`, grid by grid and parameter by
parameter. The order of the grids and of the parameters therefore changes the samples. To repeat a
search, keep that order fixed. Use an `OrderedDict` from
[`OrderedCollections`](https://github.com/JuliaCollections/OrderedCollections.jl) or a vector
for each grid.

#### 2.2.1 Sampling from a predefined parameter space

We sample from the same grids as before, with `n_iter = 2`. Every value is in a vector. The
search draws two values for each parameter, without replacement.
=#
rs_cv1 = RandomisedSearchCrossValidation(p; rng = StableRNG(42), r = r, n_iter = 2)

#=
The first grid has two parameters and gives `2 × 2 = 4` candidates. The other three grids
give 2 candidates each, for `4 + 2 + 2 + 2 = 10` in total.
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

We now mix vectors and distributions, with `n_iter = 5`. This grid needs no
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

rs_cv2 = RandomisedSearchCrossValidation(p; rng = StableRNG(42), r = r, n_iter = 5)

#=
The first grid holds a distribution. The search draws 5 values from it, and 3 values with
replacement from the vector of three. The second grid gives 5 values from its distribution.
The third and fourth grids hold only vectors and give their 3 and 2 values. That makes
`3 × 5 + 5 + 3 + 2 = 25` candidates. A draw with replacement can repeat a value, and then
two candidates are the same.
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
