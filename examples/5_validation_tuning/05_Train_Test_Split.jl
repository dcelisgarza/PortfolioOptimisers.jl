#=
# Train/test splitting

The cheapest honest question you can ask of a portfolio is: *does it survive data it has
never seen?* Cross-validation answers that many times over, on many folds, at many times the
cost. Before any of that, there is the **holdout split** — train on the first 80 % of history,
score on the last 20 %, once.

It is the protocol everyone reaches for first, and the one easiest to get quietly wrong. Not
because the split itself is hard, but because *everything upstream of it* has to respect it.
An imputation fill value computed from the whole price history has seen the test window. So
has a missing-data filter that chose the asset universe by looking at all five years. The
returns matrix you split was already contaminated before you split it.

[`TrainTestSplit`](@ref) (alias `TTS`) closes that hole by making the split a **step of the
pipeline** — the *first* step, before any other step has touched the data. Everything fitted
downstream sees the training window and nothing else, by construction rather than by
discipline.

This example covers the free function and its sizing rules, the pipeline step and the leakage
argument that pins it to first position, the payoff (`fit_predict` scoring on the held-out
window), the embargo, and why a pipeline carrying a holdout is refused by cross-validation.

See `docs/adr/0031-holdout-split-as-a-pipeline-step.md` for the design rationale.
=#
using PortfolioOptimisers, PrettyTables

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Setting up

Four years of daily prices for ten S&P 500 names. As in the pipelines example we start from
**prices**, not returns, because the whole point is that the split happens before the data is
cleaned and converted.
=#

using CSV, TimeSeries, DataFrames, Clarabel, Statistics, Dates

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 4):end]
X = X[Symbol.(["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "KO", "WMT", "PEP", "MRK"])...]
pr = PricesResult(; X = X)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true));

size(values(X))

#=
## 2. The free function

[`train_test_split`](@ref) cuts price- or returns-level data into two windows: the training
window is the **head** of the observations, the test window is the **tail**. Time-ordered, so
the test window is always the most recent data — a random shuffle would let the model train on
tomorrow to predict yesterday.

The windows are views, so nothing is copied.
=#

train, test = train_test_split(pr; test_size = 0.2)

DataFrame(; window = ["train", "test"],
          rows = [size(values(train.X), 1), size(values(test.X), 1)],
          from = [first(timestamp(train.X)), first(timestamp(test.X))],
          to = [last(timestamp(train.X)), last(timestamp(test.X))])

#=
### 2.1 Sizing: counts, fractions, and complements

A size is either an **`Integer` row count** or an **`AbstractFloat` fraction** of the
observations in `(0, 1)`. `test_size = 250` means the last 250 rows; `test_size = 0.25` means
the last quarter.

Give **one** side and the other is its complement — the two windows partition the data. Give
**neither** and the split is 75/25. The four spellings below all describe the same idea.
=#

N = size(values(X), 1)

spellings = ["train_test_split(pr)" => (nothing, nothing),
             "train_test_split(pr; test_size = 0.2)" => (nothing, 0.2),
             "train_test_split(pr; train_size = 0.8)" => (0.8, nothing),
             "train_test_split(pr; test_size = 202)" => (nothing, 202)]

sizes = DataFrame(; spelling = String[], train = Int[], test = Int[])
for (label, (tr_s, te_s)) in spellings
    a, b = train_test_split(pr; train_size = tr_s, test_size = te_s)
    push!(sizes, (label, size(values(a.X), 1), size(values(b.X), 1)))
end
sizes

#=
### 2.2 The embargo

Give **both** sizes and they need not cover the data. The head supplies the training rows, the
tail supplies the test rows, and anything in between is **embargoed** — it belongs to neither
window.

That gap is not waste, it is insurance. Financial features are autocorrelated: a return
computed over a 20-day window straddling the boundary leaks test information into the last
training row. Dropping the rows around the seam severs that link. (This is the same idea as
`KFold`'s `purged_size`/`embargo_size`, expressed for a single split.)

Windows that would *overlap* are rejected outright, as is any split that would leave a window
empty.
=#

tr_e, te_e = train_test_split(pr; train_size = 0.6, test_size = 0.2)

DataFrame(; window = ["train", "embargoed", "test"],
          rows = [size(values(tr_e.X), 1),
                  N - size(values(tr_e.X), 1) - size(values(te_e.X), 1),
                  size(values(te_e.X), 1)])

## Overlapping windows are a mistake, not a preference:
try
    train_test_split(pr; train_size = 0.9, test_size = 0.2)
catch e
    println(e.msg)
end

#=
## 3. Why the manual composition is a trap

Nothing stops you from writing this:

```julia
res  = fit(pipe, pr)          ## fitted on EVERYTHING
pred = predict(res, pr, test_window)
```

It runs. It produces a number. And that number is **not out-of-sample** — the pipeline's
missing-data filter chose its universe using the test rows, and its imputer computed fill
values from them. The call shape is identical to the honest one; only the data the fit saw
differs. Nothing warns you.

The fix is to make the split part of the workflow, so the fit *cannot* see the test rows.
=#

#=
## 4. The split as the first pipeline step

[`TrainTestSplit`](@ref) narrows whichever data slot the pipeline input filled — prices here —
and hands the training window to every step downstream. It also stashes the held-out window in
its fitted result, which is what makes the evaluation a one-liner later.

Note the auto-generated step name: `"split"`.
=#

pipe = Pipeline(;
                steps = (TrainTestSplit(; test_size = 0.2), MissingDataFilter(), Imputer(),
                         PricesToReturns(), EmpiricalPrior(),
                         MeanRisk(; r = Variance(),
                                  opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior()))))
pipe.names

#=
Fitting runs the steps left to right. Everything after the split saw 806 training prices, not
the 1009 in the sample.
=#

res = fit(pipe, pr)

split_res = res["split"]

DataFrame(;
          quantity = ["prices in the sample", "prices the workflow was fitted on",
                      "prices held out", "returns reaching the optimiser"],
          rows = [N, size(values(split_res.train.X), 1), size(values(split_res.test.X), 1),
                  size(res.ctx.returns.X, 1)])

#=
### 4.1 The position rule is enforced

A `TrainTestSplit` **must be the first step**. This is not tidiness — it is the entire safety
argument. A [`MissingDataFilter`](@ref) fitted before the split would choose the asset universe
using the held-out rows; an [`Imputer`](@ref) fitted before it would compute its fill values
from them. Their fitted state would carry test information into the training workflow, which is
exactly the leak the split exists to prevent.

So the constructor refuses, rather than letting you find out from a suspiciously good backtest.
=#

try
    Pipeline(;
             steps = (MissingDataFilter(), Imputer(), TrainTestSplit(; test_size = 0.2),
                      PricesToReturns(), EqualWeighted()))
catch e
    println(e.msg)
end

## The same applies to a split hidden inside a nested pipeline:
try
    inner = Pipeline(; steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns()))
    Pipeline(; steps = (inner, EqualWeighted()))
catch e
    println(e.msg)
end

#=
## 5. The payoff: `fit_predict` scores on the held-out window

For a pipeline carrying a split, [`fit_predict`](@ref) fits on the training rows and predicts
on the window the split reserved. That is the whole holdout protocol, in one call — and it is
out-of-sample by construction, because the split is what defined "sample".

For a pipeline *without* a split, `fit_predict` keeps its old meaning and stays in-sample. The
same call means different things only because the workflows genuinely differ.
=#

pred_test = fit_predict(pipe, pr)

## The in-sample counterpart: replay the fitted workflow on its own training window.
pred_train = predict(res, split_res.train)

#=
Both are scored on the *realised portfolio return series*, so the measures must be ones that can
read a bare return series: `SCM()` is the scoreable spelling of standard deviation
([`Variance`](@ref) and [`StandardDeviation`](@ref) consume portfolio weights instead — see the
asset pre-selection example, which hits the same wall).
=#

DataFrame(; window = ["train (in-sample)", "test (held out)"],
          observations = [length(pred_train.rd.X), length(pred_test.rd.X)],
          std = [expected_risk(SCM(), pred_train), expected_risk(SCM(), pred_test)],
          cvar = [expected_risk(ConditionalValueatRisk(), pred_train),
                  expected_risk(ConditionalValueatRisk(), pred_test)])

#=
The gap between the two rows is the number the whole exercise exists to produce. In-sample risk
is an estimate the optimiser *minimised*; held-out risk is an estimate it merely *inherited*.
The first is optimistic by construction. Only the second is evidence.
=#

## The weights are indexed by the training universe, and the test prediction is aligned to them.
DataFrame(; asset = res.ctx.returns.nx, weight = res.w)

#=
## 6. Replaying a fitted split on new data is a pass-through

A fitted holdout's rows are a fact about the *fitting* window. Replaying them on some unseen
window would be meaningless, so [`apply_preprocessing`](@ref) on a
[`TrainTestSplitResult`](@ref) passes the data straight through.

That is what keeps the ordinary workflow — fit on history, predict on genuinely new
observations — working. Every *other* fitted step still replays properly: the training asset
universe, the training fill values, the returns conversion. Only the split stands aside.
=#

## Pretend the last 40 rows are data that arrived after the model was built.
future = PricesResult(; X = X[(end - 40):end])
pred_future = predict(res, future)

DataFrame(; source = ["held-out window (fit_predict)", "fresh data (predict)"],
          observations = [length(pred_test.rd.X), length(pred_future.rd.X)],
          std = [expected_risk(SCM(), pred_test), expected_risk(SCM(), pred_future)])

#=
## 7. One evaluation protocol per call

A holdout and a cross-validator are two answers to the same question, and they do not compose.
Cross-validation already defines the train and test window of *every fold*; a split left in the
pipeline would shave a second, redundant holdout off each fold's training data and stash a test
window nobody ever reads.

That is silent loss of training data, so the library refuses it rather than quietly doing it.
=#

gscv = GridSearchCrossValidation(Dict("returns" =>
                                          [PricesToReturns(; ret_method = :simple),
                                           PricesToReturns(; ret_method = :log)]);
                                 cv = KFold(; n = 3), r = SCM())

try
    search_cross_validation(pipe, gscv, pr)
catch e
    println(e.msg)
end

#=
Drop the split and the same pipeline tunes happily — cross-validation supplies the discipline
the split was providing.
=#

pipe_cv = Pipeline(;
                   steps = (MissingDataFilter(), Imputer(), PricesToReturns(),
                            EmpiricalPrior(),
                            MeanRisk(; r = Variance(),
                                     opt = JuMPOptimiser(; slv = slv,
                                                         pe = EmpiricalPrior()))))
scv_res = search_cross_validation(pipe_cv, gscv, pr)
scv_res.idx

#=
## 8. Choosing between them

|                | Holdout (`TrainTestSplit`)              | Cross-validation                          |
|:---------------|:----------------------------------------|:------------------------------------------|
| Fits           | one                                     | one per fold                              |
| Answers        | *does this survive unseen data?*        | *how does this behave across regimes?*    |
| Test rows      | the most recent tail, once              | every row, in turn                        |
| Use it for     | a final, honest score on a locked model | selecting hyperparameters, estimating variance of the estimate |

The holdout is a **verdict**, not a search tool. Tune with cross-validation, and keep the
holdout untouched until the model is frozen — the moment you tune *against* the held-out score,
it stops being held out, and you are back to reporting an in-sample number with extra steps.

## Summary

  - [`train_test_split`](@ref) cuts data into a head (train) and a tail (test); one size given
    makes the other its complement, both given **embargoes** the rows between them.
  - [`TrainTestSplit`](@ref) makes that the **first step** of a [`Pipeline`](@ref), so every
    fitted step downstream — universe selection, imputation, prior, optimiser — sees the
    training window alone. The position rule is enforced at construction.
  - [`fit_predict`](@ref) on a split-bearing pipeline scores on the held-out window; the split's
    fitted result carries both windows (`res["split"].train`, `res["split"].test`).
  - Replaying a fitted split is a pass-through, so predicting on genuinely new data still works.
  - A pipeline carrying a holdout cannot also be cross-validated: one evaluation protocol per
    call.
=#
