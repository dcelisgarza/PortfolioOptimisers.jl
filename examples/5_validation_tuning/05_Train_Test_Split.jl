#=
```@meta
Description = "Train/test splitting in PortfolioOptimisers.jl: hold out the most recent rows, fit the whole workflow on the rest, and score the portfolio on the held-out rows."
```

# Train/test splitting

A holdout split trains on the first 80 % of the history and scores on the last 20 %.
Cross-validation scores many folds instead, at the cost of one fit per fold.

Every step before the optimiser must also respect the split. A fill value computed from the
whole price history includes the test window. So does a missing-data filter that chooses the
assets from all four years of data. If you split a returns matrix built that way, the test rows
have already entered the training data.

[`TrainTestSplit`](@ref), with the alias `TTS`, makes the split a step of a pipeline. It is
the first step and runs before any other step sees the data. Every fitted step after it sees
the training window and nothing else.

This example covers these subjects.

  - The function `train_test_split`, its sizes and the embargo.
  - The pipeline step, and why it must come first.
  - Scoring on the held-out window with `fit_predict`.
  - Why cross-validation throws an error for a pipeline with a split.

`docs/adr/0031-holdout-split-as-a-pipeline-step.md` gives the reasons for the design.
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

We use four years of daily prices for ten S&P 500 stocks. As in the pipelines example, we
start from prices, not returns, because the split must happen before the pipeline cleans and
converts the data.
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

[`train_test_split`](@ref) cuts price or returns data into two windows. The training window holds
the first rows and the test window the last rows. The split keeps the time order, and the
test window is always the most recent data. A random shuffle would let the model train on
later rows and test on earlier ones.
=#

train, test = train_test_split(pr; test_size = 0.2)

DataFrame(; window = ["train", "test"],
          rows = [size(values(train.X), 1), size(values(test.X), 1)],
          from = [first(timestamp(train.X)), first(timestamp(test.X))],
          to = [last(timestamp(train.X)), last(timestamp(test.X))])

#=
### 2.1 Sizes as counts or fractions

A size is an `Integer` count of rows, or an `AbstractFloat` fraction of the rows, strictly
between 0 and 1. `test_size = 250` means the last 250 rows, and `test_size = 0.25` means the
last quarter. A fraction rounds down to a whole number of rows.

If you give one size, the other window takes the remaining rows. If you give neither, the
training window takes 75 % of the rows. The cell prints the window sizes of four calls. A fraction
rounds down on the side you give, so `test_size = 0.2` and `train_size = 0.8` differ by one
row.
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

If you give both sizes, the two windows need not cover all the rows. The first rows go to the
training window, the last rows go to the test window, and the rows between them belong to
neither. Those rows are the embargo.

The embargo keeps the last training rows away from the first test rows. Many inputs to a model
are autocorrelated. A 20-day average return computed on a window that crosses the boundary
puts test information into the last training rows, and an embargo of 20 rows removes it.
[`KFold`](@ref) applies the same idea to many folds with its `purged_size` and `embargo_size`
fields.

`train_test_split` throws an error when the two windows overlap, and when a window is empty.
=#

tr_e, te_e = train_test_split(pr; train_size = 0.6, test_size = 0.2)

DataFrame(; window = ["train", "embargoed", "test"],
          rows = [size(values(tr_e.X), 1),
                  N - size(values(tr_e.X), 1) - size(values(te_e.X), 1),
                  size(values(te_e.X), 1)])

## Two windows that overlap throw an error.
try
    train_test_split(pr; train_size = 0.9, test_size = 0.2)
catch e
    println(e.msg)
end

#=
## 3. The mistake that a manual split allows

Nothing stops you from writing this:

```julia
res  = fit(pipe, pr)          ## fitted on EVERYTHING
pred = predict(res, pr, test_window)
```

The code runs and gives a number, but the number is not out of sample. The missing-data filter
of the pipeline used the test rows to choose its assets, and the price fill computed its values
from them. The call looks the same as a correct one. Only the rows that the fit saw differ, and
nothing warns you.

To prevent this, make the split part of the workflow, so that the fit cannot see the test rows.
=#

#=
## 4. The split as the first pipeline step

[`TrainTestSplit`](@ref) cuts the data slot that the pipeline input filled, here the prices,
and passes the training window to every later step. Its fitted result also keeps the held-out
window, which `fit_predict` uses in section 5.

The step takes the name `"split"` when you give it none. The filter and the gap fill both
write the prices slot, so they take the names `"prices_1"` and `"prices_2"`.
=#

pipe = Pipeline(;
                steps = (TrainTestSplit(; test_size = 0.2), MissingDataFilter(),
                         PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                         MeanRisk(; r = Variance(),
                                  opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior()))))
pipe.names

#=
We fit the pipeline and count the rows at each stage.
=#

res = fit(pipe, pr)

split_res = res["split"]

DataFrame(;
          quantity = ["prices in the sample", "prices the workflow was fitted on",
                      "prices held out", "returns reaching the optimiser"],
          rows = [N, size(values(split_res.train.X), 1), size(values(split_res.test.X), 1),
                  size(res.ctx.returns.X, 1)])

#=
### 4.1 The split must be the first step

A `TrainTestSplit` must be the first step, because then no step before it sees the test rows. A
[`MissingDataFilter`](@ref) fitted before the split would use the held-out rows to choose the
assets. A [`PriceGapFill`](@ref) fitted before it would compute its fill values from them.
Their fitted state would then depend on the test rows.

The `Pipeline` constructor therefore throws an error when a split is not the first step. It does the
same for a split inside a nested pipeline, even when the nested pipeline is the first step.
=#

try
    Pipeline(;
             steps = (MissingDataFilter(), PriceGapFill(),
                      TrainTestSplit(; test_size = 0.2), PricesToReturns(),
                      EqualWeighted()))
catch e
    println(e.msg)
end

## A split inside a nested pipeline also throws an error.
try
    inner = Pipeline(; steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns()))
    Pipeline(; steps = (inner, EqualWeighted()))
catch e
    println(e.msg)
end

#=
## 5. `fit_predict` scores on the held-out window

For a pipeline with a split, [`fit_predict`](@ref) fits on the training rows and predicts on
the rows that the split held out.

For a pipeline without a split, `fit_predict` predicts on the same data that it fits, so the
result is in sample.
=#

pred_test = fit_predict(pipe, pr)

## For comparison, we predict on the training window, in sample.
pred_train = predict(res, split_res.train)

#=
We score both predictions on the return series of the portfolio. `expected_risk` on a
prediction sees only that series. [`Variance`](@ref) and [`StandardDeviation`](@ref) need the
weights, so they throw an error here, as the asset pre-selection example shows for `Variance`.
`SCM()` is the second central moment of the return series, which is its variance.
=#

DataFrame(; window = ["train (in-sample)", "test (held out)"],
          observations = [length(pred_train.rd.X), length(pred_test.rd.X)],
          variance = [expected_risk(SCM(), pred_train), expected_risk(SCM(), pred_test)],
          cvar = [expected_risk(ConditionalValueatRisk(), pred_train),
                  expected_risk(ConditionalValueatRisk(), pred_test)])

#=
The difference between the two rows is what a holdout measures. The in-sample risk comes from
the rows on which the optimiser minimised the variance. The held-out risk comes from rows that
the fit never saw. Here both held-out values are lower than the in-sample values, so on this
window the in-sample risk did not understate the held-out risk. A holdout gives you one such
comparison, and a different window can give the opposite result.
=#

## The weights cover the assets of the training window, and the test prediction uses the same
## assets.
DataFrame(; asset = res.ctx.returns.nx, weight = res.w)

#=
## 6. A fitted split returns new data unchanged

The rows of a fitted holdout belong to the window that it was fitted on. They mean nothing on
another window, so [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) returns
the data unchanged.

You can therefore still fit on the history and predict on new rows. Every other fitted step still
applies its state, such as the training assets, the training fill values and the conversion to
returns. Only the split does nothing.
=#

## We treat the last 41 prices as data that arrived after the fit.
future = PricesResult(; X = X[(end - 40):end])
pred_future = predict(res, future)

DataFrame(; source = ["held-out window (fit_predict)", "fresh data (predict)"],
          observations = [length(pred_test.rd.X), length(pred_future.rd.X)],
          variance = [expected_risk(SCM(), pred_test), expected_risk(SCM(), pred_future)])

#=
## 7. One evaluation protocol per call

A holdout and cross-validation both choose the rows to test on, and you cannot use both at
once. Cross-validation already defines the training window and the test window of every fold.
A split left in the pipeline would cut a second holdout off the training data of every fold,
and it would keep a test window that nothing uses.

The pipeline would lose training data with no warning. `search_cross_validation` therefore
throws an error for a pipeline with a split.
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
Without the split, the same pipeline tunes with no error, and cross-validation chooses the test
rows.
=#

pipe_cv = Pipeline(;
                   steps = (MissingDataFilter(), PriceGapFill(), PricesToReturns(),
                            EmpiricalPrior(),
                            MeanRisk(; r = Variance(),
                                     opt = JuMPOptimiser(; slv = slv,
                                                         pe = EmpiricalPrior()))))
scv_res = search_cross_validation(pipe_cv, gscv, pr)
scv_res.idx

#=
## 8. Choosing between them

|                | Holdout, `TrainTestSplit`                  | Cross-validation                             |
|:---------------|:-------------------------------------------|:---------------------------------------------|
| Fits           | one                                        | one per fold                                 |
| Answers        | *how does the model do on rows it never saw?* | *how does the model do over many windows?* |
| Test rows      | the most recent rows, once                 | every row, in turn                           |
| Use it for     | a last score of a model you no longer change | choosing parameters, and seeing how much the score varies |

Tune with cross-validation, and do not look at the
held-out score until the model is fixed. If you tune against the held-out score, the held-out
rows become part of the fit, and the score is in sample again.

## Summary

  - [`train_test_split`](@ref) cuts data into its first rows for training and its last rows
    for testing. If you give one size, the other window takes the rest. If you give both, the
    rows between the two windows are an embargo.
  - [`TrainTestSplit`](@ref) is the first step of a [`Pipeline`](@ref), so the asset filter,
    the gap fill, the prior and the optimiser after it see only the training window. The
    `Pipeline` constructor throws an error for a split in any other position.
  - [`fit_predict`](@ref) on a pipeline with a split scores on the held-out window. The fitted
    result of the split keeps both windows, `res["split"].train` and `res["split"].test`.
  - A fitted split returns new data unchanged, and a prediction on new rows still works.
  - Cross-validation throws an error for a pipeline with a split. Use one of the two per
    call.
=#
