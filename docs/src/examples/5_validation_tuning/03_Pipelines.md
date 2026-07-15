The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/5_validation_tuning/03_Pipelines.jl"
```

# Pipelines

Every example so far has started from a returns matrix. But a returns matrix is already the
output of a series of decisions: which assets had enough data to keep, how the gaps in their
price history were filled, whether returns are simple or logarithmic. Those decisions have
hyperparameters, and until now they have been made *once*, on the full sample, before any
cross-validation loop began.

That is a leak. If the imputation fill value for an asset is computed from the whole price
history, then every "out-of-sample" test fold has already seen the future. If the surviving
asset universe is chosen by looking at missingness across all five years, the backtest is
quietly conditioned on knowing which assets survive.

A [`Pipeline`](@ref) fixes this by making the *entire workflow* — price cleaning,
prices-to-returns conversion, prior estimation, phylogeny, constraint generation, and
optimisation — the unit that gets fitted. Fit a pipeline on a training window and it learns
its preprocessing state there; predict on a test window and that state is *replayed*, never
recomputed. Cross-validate a pipeline and the split happens on the *input* rows, so the
cleaning steps live inside the fold.

This example builds a pipeline stage by stage, shows what each stage contributes, and then
tunes preprocessing hyperparameters jointly with optimiser hyperparameters under
walk-forward cross-validation.

````@example 03_Pipelines
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
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
nothing #hide
````

## 1. Setting up

We use four years of daily data so that walk-forward cross-validation has enough history for
several folds. Crucially, we start from **prices**, not returns — the pipeline is what turns
one into the other.

````@example 03_Pipelines
using CSV, TimeSeries, DataFrames, Clarabel, Statistics, StableRNGs

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 4):end]
X = X[Symbol.(["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "KO", "WMT", "PEP", "MRK"])...]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true));
nothing #hide
````

### 1.1 `PricesResult`, the price-level container

[`PricesResult`](@ref) is the prices-level mirror of [`ReturnsResult`](@ref): asset prices
plus optional factor, benchmark, and implied-volatility series. It is the input a pipeline
expects when it starts at the price level.

To make the cleaning steps do visible work, we punch some holes in the data: one asset loses
most of its early history, and two others lose isolated observations.

````@example 03_Pipelines
vals = Matrix{Float64}(values(X))
vals[1:(end ÷ 2), 3] .= NaN    ## JNJ: missing for the first half of the sample
vals[10, 1] = NaN              ## AAPL: an isolated gap
vals[25, 5] = NaN              ## XOM: an isolated gap
Xm = TimeArray(timestamp(X), vals, colnames(X))
pr = PricesResult(; X = Xm)

# How much is missing, per asset?
miss = DataFrame(; asset = string.(colnames(Xm)),
                 missing_frac = vec(count(isnan, vals; dims = 1)) ./ size(vals, 1))
pretty_table(miss; formatters = [resfmt])
````

## 2. Building a pipeline stage by stage

A pipeline is an ordered list of steps. Each step is an ordinary estimator; its *family*
decides which slot of the pipeline context it reads and writes, so there is no wrapper
ceremony. Steps may be named with `"name" => estimator`; unnamed steps are auto-named after
the slot they write.

### 2.1 The minimal pipeline: prices to returns

[`PricesToReturns`](@ref) is the step form of [`prices_to_returns`](@ref). It is stateless:
applying it to any window simply runs the conversion.

````@example 03_Pipelines
pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
pipe.names
````

Fitting walks the steps left to right. The result carries every step's fitted output, the
final context, and the terminal weights.

````@example 03_Pipelines
res = fit(pipe, pr)
pretty_table(DataFrame(; asset = res.ctx.returns.nx, weight = res.w); formatters = [resfmt])
````

Note the row count: `T` prices become `T - 1` returns. The pipeline tracks this contraction
for you, which matters when cross-validation windows are sized in *input* rows.

````@example 03_Pipelines
size(values(pr.X), 1), size(res.ctx.returns.X, 1)
````

### 2.2 Universe selection is fitted state

[`MissingDataFilter`](@ref) drops assets whose missing fraction exceeds `col_thr`. The
surviving universe is **fitted state**: the training window decides it, and applying the
fitted result to an unseen window subsets that window to the *same* assets. This is what
keeps train weights and test returns aligned.

With `col_thr = 0.4`, JNJ (missing half its history) is dropped.

````@example 03_Pipelines
pipe = Pipeline(;
                steps = ("filter" => MissingDataFilter(; col_thr = 0.4), PricesToReturns(),
                         EmpiricalPrior(), EqualWeighted()))
res = fit(pipe, pr)
res["filter"].nx
````

Raise the threshold and JNJ survives — the universe is a hyperparameter, and in §4 we will
tune it rather than guess it.

!!! warning "Pin the universe before converting to returns"

    [`PricesToReturns`](@ref) is *stateless*, and the underlying [`prices_to_returns`](@ref)
    silently drops assets that are entirely missing in the window it converts. A training
    window in which an asset has no history at all therefore produces fewer assets than a
    clean test window, and the fitted weights would not line up with the test returns.

    Any pipeline you intend to `predict` or cross-validate with should pin the universe with
    a [`MissingDataFilter`](@ref) step and fill the remaining gaps with an [`Imputer`](@ref)
    step *before* converting. If you forget, [`predict`](@ref) refuses to guess and reports
    the two universes it found.

````@example 03_Pipelines
res_lax = fit(Pipeline(;
                       steps = ("filter" => MissingDataFilter(; col_thr = 0.9),
                                PricesToReturns(), EmpiricalPrior(), EqualWeighted())), pr)
res_lax["filter"].nx
````

### 2.3 Imputation parameters are fitted state

[`Imputer`](@ref) fills gaps with a per-asset statistic computed on the training window. The
fill values are fitted state: a test window is filled with *training* statistics, never with
its own. This is the leakage-prevention exemplar.

To see why that matters, fit the same imputer on two different windows and compare what it
learns.

````@example 03_Pipelines
pipe_imp = Pipeline(; steps = ("impute" => Imputer(), PricesToReturns(), EmpiricalPrior()))

res_train = fit(pipe_imp, PricesResult(; X = Xm[1:500]))
res_test = fit(pipe_imp, PricesResult(; X = Xm[501:end]))

j = findfirst(==(:AAPL), res_train["impute"].nx)
k = findfirst(==(:AAPL), res_test["impute"].nx)
fills = DataFrame(; window = ["train (1:500)", "test (501:end)"],
                  aapl_fill = [res_train["impute"].v[j], res_test["impute"].v[k]])
pretty_table(fills)
````

The two windows disagree — AAPL's price level is very different across them. A pipeline
fitted on the training window carries the *train* number, and
[`predict`](@ref) replays exactly that number on the test window. Had we imputed on the full
sample before splitting, the fill would have been contaminated by the test period, and every
subsequent "out-of-sample" score would be optimistic.

The statistic itself is configurable — `Imputer(; stat = MeanValue())` versus
`MedianValue()` — and is another hyperparameter to tune in §4.

## 3. The full workflow

Now the whole chain from the ADR: prices → filter → impute → returns → prior → phylogeny
constraints → weight bounds → [`MeanRisk`](@ref).

Two things make this work without any plumbing:

- **Slot routing.** Each step writes the context slot its family owns. A prior estimator
    writes `:prior`; a phylogeny-constraint estimator writes `:constraints`.
- **Injection.** Immediately before the optimisation step runs, the computed slots
    override the optimiser's *internal* configuration. The pipeline's prior replaces the
    optimiser's default `pe`; the constraint results are routed by type into `wb`, `lcse`,
    and `ple`. Every stage is optional — an absent step simply lets the optimiser compute
    that quantity internally, exactly as it does today.

````@example 03_Pipelines
pipe = Pipeline(;
                steps = ("filter" => MissingDataFilter(; col_thr = 0.4),
                         "impute" => Imputer(), PricesToReturns(), EmpiricalPrior(),
                         SemiDefinitePhylogenyEstimator(),
                         "wb" => WeightBoundsEstimator(; ub = 0.4),
                         MeanRisk(; opt = JuMPOptimiser(; slv = slv))))
pipe.names

res = fit(pipe, pr)
pretty_table(DataFrame(; asset = res.ctx.returns.nx, weight = res.w); formatters = [resfmt])
````

The weight bound is respected, and the prior was computed **once** and shared with the
optimiser rather than recomputed inside it.

````@example 03_Pipelines
maximum(res.w) <= 0.4 + 1e-8
````

### 3.1 Predicting on an unseen window

[`predict`](@ref) replays the fitted preprocessing on a test window — universe subset, then
train-fitted imputation, then the returns conversion — and hands the result to the ordinary
weights-level prediction machinery. Scorers and risk measures carry over untouched.

````@example 03_Pipelines
T = size(values(Xm), 1)
res_train = fit(pipe, PricesResult(; X = Xm[1:800]))
pred = PortfolioOptimisers.predict(res_train, pr, 801:T)
expected_risk(ConditionalValueatRisk(), pred)
````

## 4. Tuning the whole workflow

This is the point of the whole exercise. [`search_cross_validation`](@ref) splits the
**input rows** into contiguous windows, and for each candidate fits the entire pipeline on
the training window and scores it on the test window. Preprocessing hyperparameters are
searched jointly with optimiser hyperparameters, and no candidate ever sees the test window
during preprocessing.

Lens keys address steps three ways:

- by **name** with a trailing property path — `"filter.col_thr"`;
- by **name** alone (or by integer position), which swaps the whole step — `"impute"`;
- by **raw property path**, exactly as for plain optimisers — `"steps[1].col_thr"`.

````@example 03_Pipelines
pipe = Pipeline(;
                steps = ("filter" => MissingDataFilter(; col_thr = 0.4),
                         "impute" => Imputer(), PricesToReturns(), EmpiricalPrior(),
                         "opt" => MeanRisk(; opt = JuMPOptimiser(; slv = slv))))

p = ["filter.col_thr" => [0.4, 0.9],
     "impute" => [Imputer(; stat = MeanValue()), Imputer(; stat = MedianValue())]]

gscv = GridSearchCrossValidation(p; cv = IndexWalkForward(500, 250),
                                 r = ConditionalValueatRisk())
tuned = search_cross_validation(pipe, gscv, pr)

# Mean test score per candidate (bigger is better after the sign convention).
scores = DataFrame(; candidate = 1:length(tuned.val_grid),
                   col_thr = [v[1] for v in tuned.val_grid],
                   imputer = [string(nameof(typeof(v[2].stat))) for v in tuned.val_grid],
                   mean_score = vec(mean(tuned.test_scores; dims = 1)))
pretty_table(scores)
````

`tuned.opt` is the winning *pipeline*, ready to fit on the full sample.

````@example 03_Pipelines
tuned.idx, tuned.opt.steps[1].col_thr, nameof(typeof(tuned.opt.steps[2].stat))

final = fit(tuned.opt, pr)
pretty_table(DataFrame(; asset = final.ctx.returns.nx, weight = final.w);
             formatters = [resfmt])
````

### 4.1 Structural search: swapping whole estimators

Because lens values are arbitrary objects, swapping an entire step is just another grid
value. Here we search over the *prior estimator* itself rather than one of its fields.

````@example 03_Pipelines
pipe_struct = Pipeline(;
                       steps = (MissingDataFilter(; col_thr = 0.4), Imputer(),
                                PricesToReturns(), "prior" => EmpiricalPrior(),
                                EqualWeighted()))
p_struct = ["prior" => [EmpiricalPrior(),
                        EmpiricalPrior(;
                                       ce = PortfolioOptimisersCovariance(;
                                                                          ce = GerberCovariance()))]]
tuned_struct = search_cross_validation(pipe_struct,
                                       GridSearchCrossValidation(p_struct;
                                                                 cv = IndexWalkForward(500,
                                                                                       250),
                                                                 r = ConditionalValueatRisk()),
                                       pr)
tuned_struct.idx, vec(mean(tuned_struct.test_scores; dims = 1))
````

### 4.2 Randomised search

[`RandomisedSearchCrossValidation`](@ref) samples the grid and then delegates to the grid
form, exactly as it does for plain optimisers.

````@example 03_Pipelines
rscv = RandomisedSearchCrossValidation(p; cv = IndexWalkForward(500, 250),
                                       r = ConditionalValueatRisk(), rng = StableRNG(42),
                                       n_iter = 2, seed = 1)
tuned_rand = search_cross_validation(pipe, rscv, pr)
size(tuned_rand.test_scores)
````

## 5. Many paths: combinatorial and asset-resampling cross-validation

The walk-forward above produces *one* backtest path. [`CombinatorialCrossValidation`](@ref) and
[`MultipleRandomised`](@ref) produce *many* — a distribution of out-of-sample outcomes rather
than a single number. There is one rule: they need contiguous input rows, which recombined
groups and resampled paths do not guarantee, so a pipeline that **starts from prices** — with a
[`PricesToReturns`](@ref) or any rolling, order-dependent step — rejects them (the
*rolling-window rule*). A **returns-level** pipeline has no such step, so it runs them exactly
as a plain optimiser would.

So we start from returns and drop the price-level cleaning:

````@example 03_Pipelines
rd = prices_to_returns(X)
rpipe = Pipeline(;
                 steps = (EmpiricalPrior(),
                          MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; slv = slv))))
````

### 5.1 Combinatorial paths

Each split trains on its (possibly non-contiguous) groups and predicts the held-out ones; the
per-split test groups recombine into paths. `expected_risk` over the population gives one
realised risk per path — the spread *is* the robustness picture the scheme exists to produce.

````@example 03_Pipelines
comb = CombinatorialCrossValidation(; n_folds = 5, n_test_folds = 2)
pp = cross_val_predict(rpipe, rd, comb)
cvar = ConditionalValueatRisk()
pretty_table(DataFrame(; path = [p.id for p in pp.pred],
                       folds = [length(p.pred) for p in pp.pred],
                       cvar = expected_risk(cvar, pp)); formatters = [resfmt])
````

### 5.2 Asset-resampling paths

[`MultipleRandomised`](@ref) draws a random asset subset per path and runs an inner
walk-forward over it. The subset is applied to the *input*, so the pipeline fits fresh on each
sub-universe — it never sub-selects fitted state (the restriction that used to rule this out).

````@example 03_Pipelines
mr = MultipleRandomised(IndexWalkForward(500, 250); subset_size = 6, n_subsets = 4,
                        rng = StableRNG(20240607))
pm = cross_val_predict(rpipe, rd, mr)
pretty_table(DataFrame(; path = [p.id for p in pm.pred],
                       folds = [length(p.pred) for p in pm.pred],
                       assets = [length(p.pred[1].res.w) for p in pm.pred]);
             formatters = [resfmt])
````

## 6. Boundaries

A few things a pipeline deliberately will not do, each with an explanatory error rather than
a silent wrong answer:

- **Combinatorial and multiple-randomised cross-validation are price-level-restricted.** A
    price-starting pipeline rejects them by the rolling-window rule of §5; run them on a
    returns-level pipeline, as above.
- **A pipeline cannot be wrapped in a meta-optimiser** (`NestedClustered`, `Stacking`,
    `SubsetResampling`). Those build asset sub-portfolios via an asset view of their inner
    estimator, and a pipeline's universe is fitted state, so the view is not well defined
    before fitting. The reverse *is* supported: a meta-optimiser makes a perfectly good
    optimisation step of a pipeline.
- **Predicting without weights** — a pipeline with no terminal optimisation step is legal
    (a prior-only pipeline is useful), but it has nothing to predict with.

````@example 03_Pipelines
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
````

## 7. Summary

A [`Pipeline`](@ref) turns an implicit, hand-tuned data-preparation prologue into an explicit,
fitted, tunable part of the model.

- Steps are ordinary estimators, routed to context slots by their family.
- Preprocessing steps have a fit/apply contract: the training window learns the universe
    and the imputation parameters, and unseen windows replay them.
- Computed slots are injected into the optimiser's configuration, so a prior is computed
    once and shared, and constraints are routed by result type.
- Cross-validation splits the *input* rows, so the cleaning steps are refitted inside every
    fold — which is precisely the leakage the pipeline exists to remove.
- Hyperparameters of the preprocessing and of the optimiser are searched in one grid, and
    whole estimators can be swapped as grid values.

See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale and the list of
deliberately deferred features.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
