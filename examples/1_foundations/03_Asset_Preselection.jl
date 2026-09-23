#=
```@meta
Description = "Asset pre-selection in PortfolioOptimisers.jl: drop constant, low-ranked and redundant assets before optimising, and why it helps."
```

# Asset pre-selection

Some assets in a universe add nothing to the optimisation. A constant column, such as a name that
never traded, carries no information. Some assets rank too low, for example when you want the twenty
names with the lowest risk and the universe has two hundred. Some are redundant, because they move
so closely with another asset that keeping both gives the optimiser the same information twice. The
optimiser then treats one common factor as two separate sources of risk.

An asset selector removes such assets from the universe, based on the returns. It is an ordinary
preprocessing estimator and needs no pipeline. A [`Pipeline`](@ref) runs it through
[`fit_preprocessing`](@ref) and [`apply_preprocessing`](@ref), as it runs a prior estimator through
[`prior`](@ref). The library has three kinds:

  - [`CompleteAssetSelector`](@ref) drops every asset whose column holds a `missing` or a `NaN`.
  - [`ScoreSelector`](@ref) scores every asset with a risk measure and keeps the assets a rule
    admits. [`ZeroVarianceFilter`](@ref) is a named case of it.
  - [`RedundancySelector`](@ref) drops assets whose returns repeat the information of other
    assets.

A selector chooses its universe on the training window, and that universe is the result of the fit.
Applying the fitted result to a later window keeps the same universe and does not choose again. A
selector that chose again on each window would pick its assets with the returns it is about to be
scored on, which is look-ahead bias.

!!! tip "When to reach for this"
    Reach for a selector when the choice of universe is part of your model and not given to you.
    Run the selectors for degenerate columns, `ZeroVarianceFilter` and `CompleteAssetSelector`,
    on every universe. Selection by score, `ScoreSelector`, tests the hypothesis that a screen on
    a risk measure improves the results out of sample. Pruning redundant assets,
    `RedundancySelector`, helps when a correlation matrix close to singular makes the optimiser
    unstable. The screen and the pruning have parameters, and you tune them inside
    cross-validation, as section 7 shows, never on the full sample.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics, StatsAPI

resfmt = (v, i, j) -> begin
    if isa(v, Number) && !isa(v, Integer)
        return "$(round(v * 100, digits = 3)) %"
    else
        return v
    end
end;

#=
## 1. The data

We take twenty S&P 500 names over the last 1000 trading days. The correlations of real data matter
in section 5, where two algorithms that sound the same give different answers.
=#

using CSV, TimeSeries

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)
rd = prices_to_returns(X[(end - 1000):end])
size(rd.X), rd.nx

#=
## 2. Score an asset with a risk measure

`ScoreSelector` scores asset `i` with a risk measure evaluated on the returns of that asset alone,
`score(view(X, :, i))`. The library has no separate family of scores, because the risk measures
already cover what you would rank assets on: variance and semi-variance, VaR and CVaR, the drawdown
measures, and [`MeanReturn`](@ref).

Two traits decide how a measure scores. [`supports_precomputed_returns`](@ref) says whether a
measure can take a single return series, and [`bigger_is_better`](@ref) says which end of the
ranking is the best. We print both traits for five measures.
=#

measures = ["SCM()" => SCM(), "ConditionalValueatRisk()" => ConditionalValueatRisk(),
            "MaximumDrawdown()" => MaximumDrawdown(), "MeanReturn()" => MeanReturn(),
            "Variance()" => Variance()]

pretty_table(DataFrame(; measure = first.(measures),
                       scoreable = [PortfolioOptimisers.supports_precomputed_returns(r)
                                    for (_, r) in measures],
                       bigger_is_better = [PortfolioOptimisers.bigger_is_better(r)
                                           for (_, r) in measures]))

#=
A smaller `ConditionalValueatRisk` is better and a bigger `MeanReturn` is better. A rule that asks
for the `best` assets takes the right end for each measure, and you do not state which end it is.
We score every asset with both measures.
=#

cvar = PortfolioOptimisers.asset_scores(ConditionalValueatRisk(), rd.X)
mu = PortfolioOptimisers.asset_scores(MeanReturn(), rd.X)
pretty_table(sort(DataFrame(; asset = rd.nx, cvar = cvar, mean_return = mu), :cvar);
             formatters = [resfmt])

#=
### 2.1 `Variance` cannot score a single asset

One measure of the family does not work as a score. [`Variance`](@ref) is a
[`WeightsInput`](@ref) measure. It takes the weights of a portfolio and a covariance matrix, not a
return series. It has no meaning for one asset alone, so `ScoreSelector` throws an error when you
construct it with `Variance`.
=#

try
    ScoreSelector(; score = Variance(), rule = ThresholdRule(; lo = 0.0))
catch e
    println(e.msg)
end

#=
The error names the replacement, `SCM()`. It is the second central moment, an alias for
[`LowOrderMoment`](@ref) with [`SecondMoment`](@ref) and [`FullMoment`](@ref). It computes the
same quantity from a return series, and it can score an asset. We print its scores for three
assets beside the variances that `var` computes.
=#

PortfolioOptimisers.asset_scores(SCM(), rd.X)[1:3],
var(rd.X[:, 1:3]; dims = 1, corrected = false)

#=
## 3. Rules: what to do with the scores

A [selection rule](@ref PortfolioOptimisers.AbstractSelectionRule) turns the scores into a mask of
the assets to keep. There are two kinds of rule, and they treat the scores differently.

A literal rule compares the scores with fixed values. [`ThresholdRule`](@ref) compares each score
with the bounds `(lo, hi)`, both optional and both exclusive, and ignores `bigger_is_better`. A
filter for zero variance must drop the assets of low variance, and a rule that kept the better
assets by the trait would keep those assets instead.

An ordinal rule compares the scores with each other. [`RankRule`](@ref) and
[`QuantileRule`](@ref) sort the assets from best to worst by `bigger_is_better`, and take a count or
a fraction of assets from each end. A rule takes counts, not positions, and it can take the two
ends at once. `action = :drop` inverts the selection, so the rule drops the assets it names and
keeps the rest. With it you can drop the worst five without knowing how many assets there are.
=#

rules = ["RankRule(best = 5)" => RankRule(; best = 5),
         "RankRule(worst = 5)" => RankRule(; worst = 5),
         "RankRule(worst = 3, action = :drop)" => RankRule(; worst = 3, action = :drop),
         "RankRule(best = 2, worst = 2)" => RankRule(; best = 2, worst = 2),
         "QuantileRule(best = 0.25)" => QuantileRule(; best = 0.25)]

rows = map(rules) do (label, rule)
    kept = fit_preprocessing(ScoreSelector(; score = ConditionalValueatRisk(), rule = rule),
                             rd).nx
    return (; rule = label, n = length(kept), kept = join(kept, ", "))
end
pretty_table(DataFrame(rows))

#=
The direction of the ranking changes with the measure. `ConditionalValueatRisk` is better when
smaller, so `best = 5` returns the five defensive names. `MeanReturn` is better when bigger, so
the same rule returns the five growth names. You pass no flag for the direction.
=#

DataFrame(; rule = ["best = 5", "worst = 5"],
          by_cvar = [join(fit_preprocessing(ScoreSelector(;
                                                          score = ConditionalValueatRisk(),
                                                          rule = RankRule(; best = 5)), rd).nx,
                          ", "),
                     join(fit_preprocessing(ScoreSelector(;
                                                          score = ConditionalValueatRisk(),
                                                          rule = RankRule(; worst = 5)),
                                            rd).nx, ", ")],
          by_mean_return = [join(fit_preprocessing(ScoreSelector(; score = MeanReturn(),
                                                                 rule = RankRule(;
                                                                                 best = 5)),
                                                   rd).nx, ", "),
                            join(fit_preprocessing(ScoreSelector(; score = MeanReturn(),
                                                                 rule = RankRule(;
                                                                                 worst = 5)),
                                                   rd).nx, ", ")])

#=
### 3.1 Degenerate columns

[`ZeroVarianceFilter`](@ref) is `ScoreSelector(; score = SCM(), rule = ThresholdRule(; lo = tol))`
under its own name. A constant column has zero variance, adds nothing to a portfolio, and makes a
covariance matrix singular. We set the returns of one asset to zero, and the filter drops it.

[`CompleteAssetSelector`](@ref) drops the assets with missing values in the returns. Use it when a
pipeline receives a `ReturnsResult` directly, so the [`MissingDataFilter`](@ref) that works on
prices never runs.
=#

Xz = copy(rd.X)
Xz[:, 4] .= 0.0                                  # BBY stops trading
rd_z = ReturnsResult(; nx = rd.nx, X = Xz)
setdiff(rd.nx, fit_preprocessing(ZeroVarianceFilter(), rd_z).nx)

#=
`tol` defaults to `1e-12` and not `0`, because a column that moves by `1e-18` is constant for any
use. The bound is exclusive, so `tol = 0` still drops a column that is exactly constant.

## 4. Ties: if the scores cannot tell two assets apart, the rule keeps neither

When two assets have the same score, a rule keeps neither of them. A block of tied assets that
spans a cut of the ranking is dropped whole and never split. To split it, the rule would have to
break the tie by column index, and your portfolio would then depend on the order of the columns in
your CSV file, which is not a property of the data.

So `RankRule(; best = k)` can return fewer than `k` assets. To show it, we copy the column of the
asset with the second lowest variance over the column of the third, so that the two tie, and ask
for the best one, two and three assets.
=#

v = PortfolioOptimisers.asset_scores(SCM(), rd.X)
ord = sortperm(v)                                # ascending variance; lower is better
Xs = copy(rd.X)
Xs[:, ord[3]] = Xs[:, ord[2]]                    # make ranks 2 and 3 tie exactly
rd_tie = ReturnsResult(; nx = rd.nx, X = Xs)

tie_rows = map([1, 2, 3]) do k
    kept = fit_preprocessing(ScoreSelector(; score = SCM(), rule = RankRule(; best = k)),
                             rd_tie).nx
    return (; requested = k, returned = length(kept), kept = join(kept, ", "))
end
pretty_table(DataFrame(tie_rows))

#=
`best = 2` returns one asset, because ranks 2 and 3 tie and the cut falls between them, so the rule
drops the pair. `best = 3` returns three, because the pair now sits inside the cut and the rule
keeps it whole. A window whose scores are all equal selects nothing, and
[`fit_preprocessing`](@ref) throws an error rather than return an empty universe.

The redundancy selector treats ties in the same way. Two identical columns have a correlation of
one and the same score, so it keeps neither. We add a copy of AAPL and check whether either
stays.
=#

rd_dup = ReturnsResult(; nx = [rd.nx; "AAPL_copy"], X = hcat(rd.X, rd.X[:, 1]))
dup_kept = fit_preprocessing(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99),
                                                score = SCM()), rd_dup).nx
("AAPL" in dup_kept, "AAPL_copy" in dup_kept, length(dup_kept))

#=
## 5. Redundancy: two algorithms, two different answers

[`RedundancySelector`](@ref) has two parts. `alg` decides which assets are redundant, and `score`
decides which member of a redundant group stays. If you leave `score` as `nothing`, the
correlation algorithms keep the asset with the lowest summary correlation to the rest of the
universe, the least redundant one.

The two correlation algorithms sound as if they do the same thing, but they give different
answers.

[`PairwiseCorrelation`](@ref) is greedy and is the default. It visits the pairs of assets from the
most correlated to the least, and drops the worse asset of each pair, until no pair that remains
has a correlation above `t`. That is what the threshold states.

[`CorrelationComponents`](@ref) follows the correlations from one asset to the next. The assets
are the nodes of a graph, and each correlation above the threshold is an edge. Each connected
component of the graph keeps one asset. If `ρ(A,B) = 0.97` and `ρ(B,C) = 0.97` but
`ρ(A,C) = 0.10`, the three assets form one component, and the algorithm drops two of them, although
`A` and `C` are uncorrelated. We call such a sequence of edges a chain.

Chains form on real data too. We run both algorithms at the same threshold and print the largest
correlation between two assets the greedy algorithm keeps.
=#

greedy = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.65, absolute = true),
                            score = SCM())
comps = RedundancySelector(; alg = CorrelationComponents(; t = 0.65, absolute = true),
                           score = SCM())

kept_g = fit_preprocessing(greedy, rd).nx
kept_c = fit_preprocessing(comps, rd).nx

## the greedy guarantee, verified: no surviving pair exceeds t
gi = [findfirst(==(n), rd.nx) for n in kept_g]
sub = abs.(cor(rd.X[:, gi]))
max_surviving = maximum(sub[i, j] for j in axes(sub, 2) for i in (j + 1):size(sub, 1))

DataFrame(; algorithm = ["PairwiseCorrelation", "CorrelationComponents"],
          kept = [length(kept_g), length(kept_c)],
          max_surviving_abs_cor = [round(max_surviving; digits = 3), missing],
          extra_drops = ["—", join(setdiff(kept_g, kept_c), ", ")])

#=
`absolute = true` counts a correlation of `-0.9` as redundant too. That is usually what you want,
because two assets that move together carry the same information whatever the sign of their
correlation.

The largest correlation the greedy algorithm keeps is below the threshold, and the last column names
the assets that the components algorithm drops in addition. The greedy algorithm keeps every pair
below the threshold and keeps more assets. The components algorithm keeps one asset from each group
of correlated assets, and drops more. Choose the one whose rule you want.

### 5.1 Clustering as the grouping rule

[`ClusterGroups`](@ref) forms the groups with [`clusterise`](@ref), so any clustering method of
the library can define which assets are redundant: hierarchical linkage, DBHT, and the estimators
of the optimal number of clusters. It has no default rule for which asset stays, so it needs a
`score`, and the second cell prints the error you get without one.
=#

clustered = RedundancySelector(; alg = ClusterGroups(), score = SCM())
kept_cl = fit_preprocessing(clustered, rd).nx
length(kept_cl), join(kept_cl, ", ")

try
    RedundancySelector(; alg = ClusterGroups())            # no score
catch e
    println(e.msg)
end

#=
## 6. The universe is the result of the fit

Every selection above was fitted on one window, the whole sample. In a pipeline that window is the
training window, and the fitted result holds the selected universe. A prediction on a later window
keeps that universe. We fit a selector on the first 700 observations, apply it to the rest, and
compare the result with the universe the selector would choose on the rest alone.
=#

selector = ScoreSelector(; score = ConditionalValueatRisk(), rule = RankRule(; best = 10))
train = ReturnsResult(; nx = rd.nx, X = rd.X[1:700, :])
test = ReturnsResult(; nx = rd.nx, X = rd.X[701:end, :])

fitted = fit_preprocessing(selector, train)
replayed = apply_preprocessing(fitted, test)

## the *test* window's own ten lowest-CVaR names would have been different
would_have_chosen = fit_preprocessing(selector, test).nx

DataFrame(; universe = ["fitted on train", "replayed on test", "test window's own choice"],
          assets = [join(fitted.nx, ", "), join(replayed.nx, ", "),
                    join(would_have_chosen, ", ")])

#=
The universe applied to the test window is the one chosen on the training window, not the one the
test window would choose. The difference between the two is the look-ahead bias that fitting on
one window and applying to the next prevents.

### 6.1 The pipeline checks the order of its steps

A selector changes the assets of `:returns`. A step computed from `:returns` before the selector,
such as a prior, a phylogeny, an uncertainty set or a set of constraints, then describes assets
that are no longer in the universe. `Pipeline` throws an error for that order when you construct
it, and the prior with the wrong number of assets never reaches the optimiser.
=#

try
    Pipeline(; steps = (EmpiricalPrior(), ZeroVarianceFilter(), EqualWeighted()))
catch e
    println(e.msg)
end

#=
With the selector first, the pipeline accepts the other steps with no change to them.
=#

using Clarabel
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

pipe = Pipeline(;
                steps = ("select" => ScoreSelector(; score = ConditionalValueatRisk(),
                                                   rule = RankRule(; best = 10)),
                         EmpiricalPrior(),
                         "opt" => MeanRisk(; opt = JuMPOptimiser(; slv = slv))))
res = StatsAPI.fit(pipe, rd)
pretty_table(DataFrame(; asset = res.ctx.returns.nx, weight = res.w); formatters = [resfmt])

#=
The pipeline takes twenty assets and returns ten weights. `predict` on any window keeps the ten
assets of the fit. We predict on the whole sample and print the CVaR of the portfolio of ten
assets.
=#

pred = StatsAPI.predict(res, rd)
expected_risk(ConditionalValueatRisk(), pred)

#=
## 7. Selection is a parameter to tune

The number of assets to keep is a choice you cannot make by looking at the returns you will be
scored on. [`search_cross_validation`](@ref) fits the whole pipeline, the selector with it, on each
training window and scores it on the window held out, so no candidate sees the test window when it
chooses its universe.

A key of the grid names a step and one of its fields. `"select.rule"` replaces the whole rule of
the step named `"select"`.
=#

p = ["select.rule" => [RankRule(; best = 5), RankRule(; best = 10), RankRule(; best = 15)]]
gscv = GridSearchCrossValidation(p; cv = IndexWalkForward(500, 250),
                                 r = ConditionalValueatRisk())
tuned = search_cross_validation(pipe, gscv, rd)

pretty_table(DataFrame(; k = [v[1].best for v in tuned.val_grid],
                       mean_score = vec(mean(tuned.test_scores; dims = 1))))

#=
`tuned.opt` is the pipeline with the best score, selector included, and you can fit it on the full
sample. Its selector keeps this many assets:
=#

tuned.opt.steps[1].rule.best

#=
A count larger than the number of assets keeps every asset, so `best = 50` on a universe of 20
assets keeps 20 and throws no error, and the largest value of a grid never stops a search. Every
other degenerate case throws an error: a measure that cannot score an asset, a rule that selects
nothing, a score that is not finite, and a fitted asset that is missing from a test window.

## 8. Plot the two decisions

A score and a rule sort the assets along one axis and cut them at one place. The bar chart shows
the CVaR of each asset, and the blue bars are the ten assets that `RankRule(; best = 10)` keeps.
=#

using StatsPlots

perm = sortperm(cvar)
kept10 = fit_preprocessing(ScoreSelector(; score = ConditionalValueatRisk(),
                                         rule = RankRule(; best = 10)), rd).nx
colours = [n in kept10 ? :steelblue : :lightgray for n in rd.nx[perm]]

bar(1:length(perm), cvar[perm]; color = colours, legend = false,
    xticks = (1:length(perm), rd.nx[perm]), xrotation = 60, ylabel = "CVaR",
    title = "Per-asset CVaR — RankRule(best = 10) keeps the blue names")

#=
The redundancy decision depends on the threshold. We count the assets each algorithm keeps over a
range of thresholds. The gap between the two lines is the assets the components
algorithm drops through chains, and it closes at the thresholds where the graph has no chains.
=#

thrs = 0.5:0.025:0.85
n_greedy = [length(fit_preprocessing(RedundancySelector(;
                                                        alg = PairwiseCorrelation(; t = t,
                                                                                  absolute = true),
                                                        score = SCM()), rd).nx)
            for t in thrs]
n_comps = [length(fit_preprocessing(RedundancySelector(;
                                                       alg = CorrelationComponents(; t = t,
                                                                                   absolute = true),
                                                       score = SCM()), rd).nx)
           for t in thrs]

plot(thrs, n_greedy; label = "PairwiseCorrelation (greedy)", marker = :circle, lw = 2)
plot!(thrs, n_comps; label = "CorrelationComponents (transitive)", marker = :square, lw = 2)
hline!([length(rd.nx)]; label = "full universe", ls = :dash, color = :gray)
plot!(; xlabel = "|correlation| threshold", ylabel = "assets kept",
      title = "Chaining costs assets", legend = :bottomright)

#=
## Summary

  - An asset selector is a preprocessing estimator that works on returns. It chooses the universe
    on the training window, and applies that universe to later windows, so the selection uses no
    returns from the window it is scored on.
  - [`ScoreSelector`](@ref) scores each asset with any risk measure whose
    [`supports_precomputed_returns`](@ref) is `true`. [`bigger_is_better`](@ref) sets the direction
    of `best` and `worst`, so you pass no flag for it. [`Variance`](@ref) cannot score an asset,
    and `SCM()` replaces it.
  - [`ThresholdRule`](@ref) compares each score with fixed bounds and ignores the direction.
    [`RankRule`](@ref) and [`QuantileRule`](@ref) rank the assets and take counts or fractions
    from each end, so you can drop the worst five without knowing how many assets there are.
  - A rule keeps neither of two tied assets and never splits a tied block. `RankRule(; best = k)`
    can return fewer than `k` assets, and two identical columns both leave the universe.
  - [`PairwiseCorrelation`](@ref) keeps no pair above the threshold.
    [`CorrelationComponents`](@ref) follows chains of correlations and drops more assets. On the
    same returns the two keep different assets.
  - A pipeline throws an error at construction if a step changes `:returns` after a prior, a
    phylogeny, an uncertainty set or a constraint step, so a step never computes on a universe
    that no longer exists.
  - The thresholds and counts of a selection are parameters. Tune them with
    [`search_cross_validation`](@ref), never on the full sample.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New page; closes the asset-selection row in docs/adr/examples-coverage.md (1_foundations).
#src - Verified on the SP500 slice under Kaimon (20 assets × 1000 obs, docs env):
#src     * CVaR best-5  = JNJ, MRK, PEP, PG, WMT  (defensive);
#src       MeanReturn best-5 = AAPL, AMD, LLY, MSFT, RRC (growth). Orientation flip is real.
#src     * t = 0.65, absolute: greedy keeps 15 with max surviving |rho| = 0.643 < 0.65;
#src       components keeps 13, additionally dropping KO and XOM by chaining.
#src     * Tie demo is honest: duplicating the 2nd-lowest-variance column makes ranks 2 and 3
#src       tie, so best=2 returns 1 asset and best=3 returns 3. Do not "fix" this.
#src     * search_cross_validation over "select.rule" picks best = 10 on IndexWalkForward(500, 250).
#src - No new docs/Project.toml deps: CSV, TimeSeries, Clarabel, StatsPlots, PrettyTables,
#src   DataFrames all already present.
