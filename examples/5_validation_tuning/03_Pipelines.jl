#=
```@meta
Description = "Pipelines in PortfolioOptimisers.jl: fit the data cleaning, gap filling and asset filtering inside each cross-validation fold, and tune their parameters."
```

# Pipelines

The earlier examples start from a returns matrix. Several choices come before that matrix:
which assets have enough data to keep, how to fill the gaps in their prices, and whether the
returns are simple or logarithmic. Each choice has parameters, and so far the examples made
them once, on the full sample, before cross-validation started.

A choice made on the full sample uses the prices of the test windows. If a gap takes a fill
value computed from the whole price history, the prices of every test fold enter that value.
If you choose the assets by their missing data over all four years, the backtest keeps only
assets that you already know have a full history.

A [`Pipeline`](@ref) fits the whole workflow as one estimator. The workflow can clean the
prices, convert them to returns, estimate the prior and the phylogeny, make the constraints
and optimise. A pipeline fitted on a training window learns its preprocessing
state from that window, and applies that state unchanged to a test window. Cross-validation of
a pipeline splits the input rows, and every fold fits the cleaning steps again.

We build a pipeline one step at a time and show what each step changes. Then we tune the
preprocessing parameters under walk-forward cross-validation.
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

We use four years of daily data, so that walk-forward cross-validation has enough rows for
several folds. We start from prices, not returns, because the pipeline does the conversion.
=#

using CSV, TimeSeries, DataFrames, Clarabel, Statistics, StableRNGs

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 4):end]
X = X[Symbol.(["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "KO", "WMT", "PEP", "MRK"])...]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true));

#=
### 1.1 `PricesResult`, the price-level container

[`PricesResult`](@ref) is the price-level counterpart of [`ReturnsResult`](@ref). It stores
the asset prices and, when you have them, factor, benchmark and implied-volatility series. A
pipeline that starts at the price level takes one as its input.

So that the cleaning steps change something, we delete some prices. JNJ loses the first half of its
history, and AAPL and XOM lose one price each.
=#

vals = Matrix{Float64}(values(X))
vals[1:(end ÷ 2), 3] .= NaN    ## JNJ: missing for the first half of the sample
vals[10, 1] = NaN              ## AAPL: an isolated gap
vals[25, 5] = NaN              ## XOM: an isolated gap
Xm = TimeArray(timestamp(X), vals, colnames(X))
## `listing_span` finds the listing of each asset from its prices. A run of gaps at the start
## means that the asset is not listed yet, a run at the end means that it is delisted, and a
## gap in between is a gap inside the listing. [`price_ingestion`](@ref) makes the span for
## you. We call `listing_span` here because we built the prices by hand.
span = listing_span(vals)
pr = PricesResult(; X = Xm, span = span)

## How much is missing, per asset?
miss = DataFrame(; asset = string.(colnames(Xm)),
                 missing_frac = vec(count(isnan, vals; dims = 1)) ./ size(vals, 1))
pretty_table(miss; formatters = [resfmt])

#=
## 2. Building a pipeline one step at a time

A pipeline is an ordered list of steps, and a step is an ordinary estimator. Every step writes
one slot of the pipeline's context, such as `:prices`, `:returns`, `:prior` or `:opt`. The
kind of estimator decides the slot, so a step needs no wrapper. You can name a step with
`"name" => estimator`. A step with no name takes the name of its slot, with a suffix `_1`,
`_2` when two steps write the same slot.

### 2.1 A pipeline from prices to returns

[`PricesToReturns`](@ref) is the step form of [`prices_to_returns`](@ref). It learns nothing
when you fit it, so it runs the same conversion on any window.
=#

pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
pipe.names

#=
Fitting runs the steps from left to right. The result stores the fitted output of every step,
the context after the last step, and the portfolio weights in `res.w`.
=#

res = fit(pipe, pr)
pretty_table(DataFrame(; asset = res.ctx.returns.nx, weight = res.w); formatters = [resfmt])

#=
AAPL, JNJ and XOM get no weight. A gap in the prices reaches the returns as a missing value,
and the optimiser leaves out an asset whose returns have one.

We compare the number of price rows with the number of return rows. Cross-validation of a
pipeline sizes its windows in input rows, which are prices here, and not in returns.
=#

size(values(pr.X), 1), size(res.ctx.returns.X, 1)

#=
### 2.2 The choice of assets is fitted

[`MissingDataFilter`](@ref) drops every asset whose fraction of missing prices is above
`col_thr`. The training window decides which assets stay, and the fitted result stores their
names. Applied to another window, the fitted result selects the same assets there. The
training weights and the test returns then cover the same assets.

JNJ misses half of its history, so `col_thr = 0.4` drops it.
=#

pipe = Pipeline(;
                steps = ("filter" => MissingDataFilter(; col_thr = 0.4), PricesToReturns(),
                         EmpiricalPrior(), EqualWeighted()))
res = fit(pipe, pr)
res["filter"].nx

#=
With a higher threshold, JNJ stays. The threshold is a parameter, and section 4 tunes it
instead of fixing it by hand.

!!! note "The conversion keeps every asset"

    [`PricesToReturns`](@ref) learns nothing. It computes returns and deletes no observation
    and no asset. A gap reaches the returns as a missing value, and its row stays. The
    conversion therefore never gives a training window and a test window different assets.

    The choice of assets is fitted, and a [`MissingDataFilter`](@ref) step makes it. Put that
    step before the conversion. Use it when you want an asset removed. Use a
    [`PriceGapFill`](@ref) step when you want a gap inside a listing to take a price.
=#

res_lax = fit(Pipeline(;
                       steps = ("filter" => MissingDataFilter(; col_thr = 0.9),
                                PricesToReturns(), EmpiricalPrior(), EqualWeighted())), pr)
res_lax["filter"].nx

#=
### 2.3 Fill values are fitted

[`PriceGapFill`](@ref) gives a price to a gap inside the listing of an asset. It fills nothing
outside the listing. JNJ's missing first half comes before its first price. The step leaves
it empty and does not invent a history. The single gaps of AAPL and XOM are inside their
listings, and the step fills them.

The rule for the fill is the parameter `fill`. The default, [`CarriedPrice`](@ref), gives a
gap the last price before it, so a holding keeps its value over the gap. When you fit the
step, it records the last price of every asset in the field `v`. We fit the same step on two
windows and compare the two values for AAPL.
=#

pipe_imp = Pipeline(;
                    steps = ("gap_fill" => PriceGapFill(), PricesToReturns(),
                             EmpiricalPrior()))

res_train = fit(pipe_imp, PricesResult(; X = Xm[1:500], span = span[1:500, :]))
res_test = fit(pipe_imp, PricesResult(; X = Xm[501:end], span = span[501:end, :]))

j = findfirst(==(:AAPL), res_train["gap_fill"].nx)
k = findfirst(==(:AAPL), res_test["gap_fill"].nx)
fills = DataFrame(; window = ["train (1:500)", "test (501:end)"],
                  aapl_fill = [res_train["gap_fill"].v[j], res_test["gap_fill"].v[k]])
pretty_table(fills)

#=
Every window learns its own last price, so the two values differ. A pipeline fitted on the
training window stores the training value. When it predicts on a later window, it uses that
value only for a gap at the start of the window, before the window has a price of its own. A
later gap takes the last price of the window itself.

`PriceGapFill(; fill = MeanValue())` and `PriceGapFill(; fill = MedianValue())` give every gap
of an asset one constant, which the step computes from the training prices. A pipeline applies
the training constant to the test window. If you fit such a fill on the full sample before
the split, the constant includes the test prices, and every test score then uses information
from its own test window. Section 4 tunes the choice between the two.
=#

#=
## 3. The full workflow

We now build the full workflow. It filters the assets, fills the gaps, converts the prices to
returns, estimates the prior, makes the phylogeny constraints and the weight bounds, and ends
with a [`MeanRisk`](@ref) optimisation.

The pipeline connects the steps in two ways.

  - Every step writes the slot that its kind of estimator owns. A prior estimator writes
    `:prior`. A phylogeny estimator and a weight-bounds estimator write `:constraints`.
  - Before the optimisation step runs, the pipeline puts the computed slots into the
    optimiser. The prior replaces the optimiser's `pe`. A constraint result goes to the field
    of the optimiser that its kind names, such as `wb`, `lcse`, `cte`, `ple`, `rkb` or a
    threshold field.

A buy-in threshold can go to six different fields, so a [`ThresholdEstimator`](@ref) step
must name one. Wrap it in a [`PipelineStep`](@ref) with a `target`, for example
`target = :lt` for the long threshold or `target = :st` for the short threshold. If the
optimiser has no field for a constraint step, the `Pipeline` constructor throws an error.
Every step is optional. Without a prior step, the optimiser estimates its own prior, as it
does outside a pipeline.
=#

pipe = Pipeline(;
                steps = ("filter" => MissingDataFilter(; col_thr = 0.4),
                         "gap_fill" => PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                         SemiDefinitePhylogenyEstimator(),
                         "wb" => WeightBoundsEstimator(; lb = nothing, ub = 0.4),
                         MeanRisk(; opt = JuMPOptimiser(; slv = slv))))
pipe.names

res = fit(pipe, pr)
pretty_table(DataFrame(; asset = res.ctx.returns.nx, weight = res.w); formatters = [resfmt])

#=
We compare the largest weight with the upper bound of 0.4.
=#

maximum(res.w) <= 0.4 + 1e-8

#=
### 3.1 Predicting on an unseen window

[`predict`](@ref) applies the fitted preprocessing to a test window in step order. It selects
the training assets, fills the gaps with the rule it fitted, and converts the prices to
returns. Then it runs the ordinary prediction from the weights. Every scorer and risk measure
then works on the result as it does without a pipeline. The third argument gives the rows
of the input to predict on, here the prices after row 800.
=#

T = size(values(Xm), 1)
res_train = fit(pipe, PricesResult(; X = Xm[1:800], span = span[1:800, :]))
pred = PortfolioOptimisers.predict(res_train, pr, 801:T)
expected_risk(ConditionalValueatRisk(), pred)

#=
## 4. Tuning the whole workflow

[`search_cross_validation`](@ref) splits the input rows into windows of consecutive rows. For
every candidate, it fits the whole pipeline on the training window and scores it on the test
window. One grid can hold parameters of the preprocessing and of the optimiser, and the
preprocessing of a candidate never sees its test window.

A key of the grid names a step in one of three ways.

  - A step name with a property path, `"filter.col_thr"`, changes one field of that step.
  - A step name alone, `"gap_fill"`, or the position of the step as an integer key, replaces
    the whole step.
  - A property path from the pipeline, `"steps[1].col_thr"`, works as it does for an
    optimiser outside a pipeline. The path starts at `steps`.
=#

pipe = Pipeline(;
                steps = ("filter" => MissingDataFilter(; col_thr = 0.4),
                         "gap_fill" => PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                         "opt" => MeanRisk(; opt = JuMPOptimiser(; slv = slv))))

p = ["filter.col_thr" => [0.4, 0.9],
     "gap_fill" =>
         [PriceGapFill(; fill = MeanValue()), PriceGapFill(; fill = MedianValue())]]

gscv = GridSearchCrossValidation(p; cv = IndexWalkForward(500, 250),
                                 r = ConditionalValueatRisk())
tuned = search_cross_validation(pipe, gscv, pr)

## The mean test score of each candidate. The search negates a risk measure, so a bigger
## score is better.
scores = DataFrame(; candidate = 1:length(tuned.val_grid),
                   col_thr = [v[1] for v in tuned.val_grid],
                   gap_fill = [string(nameof(typeof(v[2].fill))) for v in tuned.val_grid],
                   mean_score = vec(mean(tuned.test_scores; dims = 1)))
pretty_table(scores)

#=
The two thresholds give the same score. In every training window, JNJ either fails the filter
or still has missing prices, and in both cases it gets no weight. The mean fill scores higher
than the median fill. The search keeps the first candidate with the highest mean score, and
`tuned.opt` is that pipeline, which we fit on the full sample.
=#

tuned.idx, tuned.opt.steps[1].col_thr, nameof(typeof(tuned.opt.steps[2].fill))

final = fit(tuned.opt, pr)
pretty_table(DataFrame(; asset = final.ctx.returns.nx, weight = final.w);
             formatters = [resfmt])

#=
### 4.1 Replacing a whole estimator

A grid value can be any object, so a grid can replace a whole step. Here the grid replaces the
prior estimator. The last step, [`EqualWeighted`](@ref), does not use the prior, so the two
candidates give the same weights and the same score, and the search keeps the first. To
compare two priors, end the pipeline with an optimiser that uses them.
=#

pipe_struct = Pipeline(;
                       steps = (MissingDataFilter(; col_thr = 0.4), PriceGapFill(),
                                PricesToReturns(), "prior" => EmpiricalPrior(),
                                EqualWeighted()))
p_struct = ["prior" => [EmpiricalPrior(),
                        EmpiricalPrior(;
                                       ce = PortfolioOptimisersCovariance(; ce = GerberCovariance()))]]
tuned_struct = search_cross_validation(pipe_struct,
                                       GridSearchCrossValidation(p_struct;
                                                                 cv = IndexWalkForward(500,
                                                                                       250),
                                                                 r = ConditionalValueatRisk()),
                                       pr)
tuned_struct.idx, vec(mean(tuned_struct.test_scores; dims = 1))

#=
### 4.2 Randomised search

[`RandomisedSearchCrossValidation`](@ref) samples candidates from the grid and then runs the
grid search on them, as it does outside a pipeline. When you give `seed`, the search seeds a
copy of `rng` with it.
=#

rscv = RandomisedSearchCrossValidation(p; cv = IndexWalkForward(500, 250),
                                       r = ConditionalValueatRisk(), rng = StableRNG(42),
                                       n_iter = 2, seed = 1)
tuned_rand = search_cross_validation(pipe, rscv, pr)
size(tuned_rand.test_scores)

#=
## 5. Cross-validation with many paths

The walk-forward search above gives one backtest path. [`CombinatorialCrossValidation`](@ref)
and [`MultipleRandomised`](@ref) give many paths, and with them a distribution of test results
instead of one number.

A pipeline that starts from prices runs both schemes. The combinatorial scheme joins groups
of rows that are not consecutive, and at the price level every join makes one return between
two prices that are not neighbours. We start from returns here and drop the price-level
cleaning, so every return comes from two consecutive prices.
=#

rd = prices_to_returns(X)
rpipe = Pipeline(;
                 steps = (EmpiricalPrior(),
                          MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; slv = slv))))

#=
### 5.1 Combinatorial paths

Every split trains on its training groups, which need not be consecutive, and predicts the
groups it holds out. The test groups of the splits join into paths. `expected_risk` over the
population gives one risk value per path. The spread of those values shows how much the
result depends on the path.
=#

comb = CombinatorialCrossValidation(; n_folds = 5, n_test_folds = 2)
pp = cross_val_predict(rpipe, rd, comb)
cvar = ConditionalValueatRisk()
pretty_table(DataFrame(; path = [p.id for p in pp.pred],
                       folds = [length(p.pred) for p in pp.pred],
                       cvar = expected_risk(cvar, pp)); formatters = [resfmt])

#=
### 5.2 Paths over random subsets of assets

[`MultipleRandomised`](@ref) draws a random subset of assets for every path and runs an inner
walk-forward on it. It draws from the assets with enough data in the window of that path. It
applies the subset to the input. The pipeline then fits from the start on each subset.
=#

mr = MultipleRandomised(IndexWalkForward(500, 250); subset_size = 6, n_subsets = 4,
                        rng = StableRNG(20240607))
pm = cross_val_predict(rpipe, rd, mr)
pretty_table(DataFrame(; path = [p.id for p in pm.pred],
                       folds = [length(p.pred) for p in pm.pred],
                       assets = [length(p.pred[1].res.w) for p in pm.pred]);
             formatters = [resfmt])

#=
## 6. What a pipeline does not do

  - A pipeline is not an optimisation estimator. `optimise` throws an error for a pipeline, so
    fit it with `fit`.
  - A meta-optimiser such as [`NestedClustered`](@ref), [`Stacking`](@ref) or
    `SubsetResampling` does not take a pipeline as an inner estimator, and its constructor
    throws an error. The other way round works, and a meta-optimiser can be the optimisation
    step of a pipeline.
  - A pipeline with no optimisation step is valid, for example one that fits only a prior.
    `predict` throws an error for it, because it has no weights.

We run two calls. `split` accepts price data for a combinatorial scheme, so the first call
returns a split and prints nothing. The second call prints the error that `optimise` gives for
a pipeline.
=#

try
    split(CombinatorialCrossValidation(), pr)
catch e
    println(e.msg)
end

try
    optimise(pipe, res.ctx.returns)
catch e
    println(e.msg)
end

#=
## 7. Summary

A [`Pipeline`](@ref) makes the data preparation part of the fitted model, so you can tune it.

  - A step is an ordinary estimator, and its kind decides the slot of the context it writes.
  - A preprocessing step learns its state on the training window, such as the assets to keep
    and the fill values. A prediction applies that state to the test window.
  - The pipeline puts the fitted prior and the fitted constraints into the optimiser, so the
    optimiser does not estimate them again.
  - Cross-validation splits the input rows. The pipeline fits its cleaning steps again in
    every fold, and no test window enters an asset filter or a fill constant.
  - One grid can hold the parameters of the preprocessing and of the optimiser, and a grid
    value can replace a whole step.

`docs/adr/0028-pipeline-workflow-estimator.md` gives the reasons for the design and lists the
features it leaves for later.
=#
