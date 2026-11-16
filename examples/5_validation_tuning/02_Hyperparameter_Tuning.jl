#=
```@meta
Description = "Hyperparameter tuning in PortfolioOptimisers.jl: grid and randomised search over any estimator, scored on cross-validation folds."
```

# [Hyperparameter tuning](@id example-hyperparameter-tuning)

Hyperparameter tuning chooses the parameters of an estimator by how they score on the test
folds of a cross-validation. This example shows the two searches of PortfolioOptimisers.jl, a
grid search and a randomised search. Then it shows a search that scores with many risk measures
and many search scorers, and keeps the predictions that it scored.
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

A literal vector of grids whose values have different types has the element type `Vector`. The
constructor accepts it, checks that each element is a grid, and narrows the element type to
the union of the types of the grids. A vector of dictionaries works in the same way.

We search four grids. `n_best = 3` keeps the three best candidates over all of them, in rank
order. The default `n_best = 1` keeps the best one alone.
=#

p = [["opti[2].opt.l1" => range(; start = 0.0005, stop = 0.0008, length = 3),
      "opti[1].opt.l2.val" => range(; start = 0.0004, stop = 0.0007, length = 3)],
     ["opti[1].opt.l2.val" => range(; start = 0.0004, stop = 0.0007, length = 3)],
     ["opti[2].opt.l1" => range(; start = 0.0009, stop = 0.0012, length = 3)],
     ["opti[2]" => [MeanRisk(; opt = opt, obj = MaximumUtility()),
                    MeanRisk(; opt = opt, obj = MaximumRatio())]]]
gs_cv = GridSearchCrossValidation(p; r = r, n_best = 3)
#=
We run the search. The four grids give `3 × 3 + 3 + 3 + 2 = 17` candidates. The result has
these fields.

  - `opt` is the best estimator.
  - `test_scores` has one row per fold and one column per candidate.
  - `train_scores` is `nothing` unless you set `train_score = true`.
  - `lens_grid` and `val_grid` give the keys and the values of every candidate.
  - `idx` is the index of the best candidate.
  - `ranked` holds the `n_best` best estimators, best first, so `ranked[1]` is `opt`.
  - `rank_idx` holds the column of each one in `test_scores`, and `rank_scores` holds its mean
    test score.
  - `path_ids` gives the path of each row of `test_scores`. It is `nothing` here, because the
    default scheme has one path.
  - `rank_predictions` is `nothing` unless you set `rank_prediction = true`. Section 2.3 shows
    it.

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
The search keeps configurations, not fitted results. The table gives the column, the mean test
score and the standard deviation of the test scores over the folds of each kept candidate. We
optimise each kept candidate on the full data, as we did for `opt`.
=#

pretty_table(DataFrame("Rank" => eachindex(gs_res1.ranked), "Column" => gs_res1.rank_idx,
                       "Mean score" => gs_res1.rank_scores,
                       "Std score" =>
                           vec(std(gs_res1.test_scores[:, gs_res1.rank_idx]; dims = 1))))

res_ranked = [optimise(cand, rd) for cand in gs_res1.ranked]
plot_composition(res_ranked[2], rd)

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

We now mix vectors and distributions, with `n_iter = 10`. The constructor of
[`RandomisedSearchCrossValidation`](@ref) checks that every value is a vector or a distribution.
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
### 2.3 Many risk measures and many search scorers

A search can score each fold with a vector of risk measures in `r`, and order the candidates
with a vector of search scorers in `scorer`. We score with a ratio and with a risk measure. A
higher ratio is better, and a lower risk is better. The search orients each measure before it
combines them. It negates a measure for which a lower value is better, and multiplies each value
by the scale of its measure. So every score is higher-is-better, and two measures of mixed
orientation can share one vector.

We search a grid of nine candidates. The ratio divides the mean return by the
[`ConditionalValueatRisk`](@ref), so the two measures have values of about the same size.
=#

p_mm = [["opti[2].opt.l1" => [0.0005, 0.001, 0.002],
         "opti[1].opt.l2.val" => [0.0005, 0.005, 0.05]]]
r_mm = [MeanReturnRiskRatio(; rk = ConditionalValueatRisk()), ConditionalValueatRisk()]

#=
#### 2.3.1 One score for each fold

The field `r_sca` combines the measures in each fold. Its default `nothing` gives the sum, so the
score of a fold is the ratio minus the CVaR. `test_scores` is one matrix, as for one measure, and
the search gives one ranking.
=#

mm_sum = search_cross_validation(st, GridSearchCrossValidation(p_mm; r = r_mm), rd);
size(mm_sum.test_scores), mm_sum.idx

#=
#### 2.3.2 One ranking for each measure

[`EachValue`](@ref) in `r_sca` does not combine the measures. The search keeps one score matrix
for each measure, and ranks the candidates by each measure apart, as the multi-metric search of
sklearn does. `opt`, `idx`, `ranked`, `rank_idx` and `rank_scores` then hold one entry for each
measure, in the order of `r`, and so does `rank_predictions` under `rank_prediction = true`.
=#

mm_each = search_cross_validation(st,
                                  GridSearchCrossValidation(p_mm; r = r_mm,
                                                            r_sca = EachValue()), rd);
pretty_table(DataFrame("Measure" => ["Ratio", "CVaR"], "Column" => mm_each.idx,
                       "Mean score" => [s[1] for s in mm_each.rank_scores]))

#=
The search takes no refit, where sklearn takes a `refit` argument. A refit is a call to
[`optimise`](@ref). We fit the winner of each measure on the full data and compare the weights.
=#

res_each = [optimise(o, rd) for o in mm_each.opt]
plot_stacked_bar_composition(res_each, rd)

#=
#### 2.3.3 Combine the measures in each fold, or once for each candidate

A reduction in `r_sca` combines the measures in each fold, and then the search scorer reduces the
folds. [`PerCandidate`](@ref)`(sca)` takes the two stages in the other order. The search scorer
reduces the folds of each measure, and then `sca` combines the values of each candidate. For the
sum and [`HighestMeanScore`](@ref) the two orders give the same value, because the mean of a sum
is the sum of the means. For a scorer that is not linear in the folds, the two orders differ.

A search scorer can be a function. It reads the score matrix, with one row per fold and one column
per candidate, and returns one value per column, higher is better. `worst_fold` scores a candidate
by its worst fold.
=#

worst_fold(X) = vec(minimum(X; dims = 1))

mm_fold = search_cross_validation(st,
                                  GridSearchCrossValidation(p_mm; r = r_mm,
                                                            scorer = worst_fold), rd);
mm_cand = search_cross_validation(st,
                                  GridSearchCrossValidation(p_mm; r = r_mm,
                                                            r_sca = PerCandidate(),
                                                            scorer = worst_fold), rd);
pretty_table(DataFrame("r_sca" => ["nothing", "PerCandidate()"],
                       "Column" => [mm_fold.idx, mm_cand.idx],
                       "Score" => [mm_fold.rank_scores[1], mm_cand.rank_scores[1]]))

#=
The first search keeps the candidate whose worst fold has the highest sum of the two scores. The
second search keeps the candidate with the highest sum of its worst ratio and its worst CVaR. The
two worst values can come from different folds. Under `PerCandidate`, `test_scores` holds one
matrix for each measure, as under `EachValue()`, but the search gives one ranking.

#### 2.3.4 Many search scorers, and the predictions of the kept candidates

`scorer` takes a vector of search scorers. The field `s_sca` combines the values of the scorers
of each candidate. Its default `nothing` gives the sum, and each built-in scorer holds a `scale`,
its weight in the sum. Here each candidate scores the sum of its mean and of its worst fold.
=#

mm_ssum = search_cross_validation(st,
                                  GridSearchCrossValidation(p_mm; r = r_mm,
                                                            scorer = [HighestMeanScore(),
                                                                      worst_fold]), rd);
mm_ssum.idx, mm_ssum.rank_scores

#=
`s_sca = EachValue()` lets each scorer rank the candidates alone. Under `EachValue()` in both
`r_sca` and `s_sca`, the search keeps a `K × L` matrix of rankings, with row `k` for measure `k`
and column `l` for scorer `l`. We keep the two best candidates of each ranking.

`rank_prediction = true` also keeps the prediction that the search scored for each kept
candidate. A prediction holds one optimisation Result for each fold, so the switch costs memory
for every candidate of the grid until the search ends. Its default is `false`.
=#

mm_kl = search_cross_validation(st,
                                GridSearchCrossValidation(p_mm; r = r_mm,
                                                          r_sca = EachValue(),
                                                          scorer = [HighestMeanScore(),
                                                                    worst_fold],
                                                          s_sca = EachValue(), n_best = 2,
                                                          rank_prediction = true), rd);
pretty_table(DataFrame("Measure" => ["Ratio", "CVaR"], "Mean" => mm_kl.idx[:, 1],
                       "Worst fold" => mm_kl.idx[:, 2]))

#=
`mm_kl.opt[k, l]` is the winner of measure `k` under scorer `l`. We fit the winner of the CVaR
under the worst fold on the full data.
=#

res_kl = optimise(mm_kl.opt[2, 2], rd)
plot_composition(res_kl, rd)

#=
`mm_kl.rank_predictions[k, l][j]` is the prediction of the candidate `mm_kl.ranked[k, l][j]`. It
is the object that the search scored, with no copy and no refit. The risk of each fold of the
prediction, with the sign of the search, gives the column of the candidate in `test_scores`. The
CVaR is a risk measure, so its sign is negative.
=#

pred_kl = mm_kl.rank_predictions[2, 2][1]
pretty_table(DataFrame("Fold" => eachindex(pred_kl.pred),
                       "-CVaR of the prediction" =>
                           [-expected_risk(r_mm[2], p) for p in pred_kl.pred],
                       "test_scores" => mm_kl.test_scores[2][:, mm_kl.rank_idx[2, 2][1]]))

#=
A kept prediction goes into the same functions as the output of [`cross_val_predict`](@ref).
[`plot_cv_scores`](@ref) plots the CVaR of each fold of the winner.
=#

plot_cv_scores(r_mm[2], pred_kl)

#=
You can tune any estimator that cross-validation accepts. You can also tune a
[`Pipeline`](@ref), and the pipelines example tunes an asset filter and a gap fill.
=#
