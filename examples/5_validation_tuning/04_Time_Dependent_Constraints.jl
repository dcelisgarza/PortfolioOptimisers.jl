#=
```@meta
Description = "Time-dependent constraints in PortfolioOptimisers.jl: a TimeDependent schedule or callable in any optimiser field, varied fold by fold."
```

# Time-dependent constraints

The constraints on the earlier pages are static. You set them when you build the optimiser, and
every optimisation uses the same values. Under cross-validation a fold is a separate optimisation
over its own window of time. Some constraints must change with that window, such as a weight cap
that falls over time, a turnover limit measured from the last rebalance, or bounds that tighten
when the training window is volatile.

[`TimeDependent`](@ref) lets a field change from fold to fold, and this page calls it a schedule.
It wraps a vector with one value per fold, a vector schedule, or a function of the fold's
[`TimeDependentContext`](@ref), a callable. You put the schedule in the field it changes, as in
`JuMPOptimiser(; wb = TimeDependent([...]))`, so the field names what changes. A field holds a
static value or a schedule, never both. A field that takes a schedule, such as a field whose type
is [`TD_Option`](@ref), accepts three forms.

 1. Static. You set the field, and every fold uses the same value.
 2. A vector schedule, `TimeDependent([v₁, …, vₙ])`. Entry `i` is the field's whole value for fold
    `i`, with the folds in the order that `split` returns them.
 3. A callable, `TimeDependent(f)`. The loop that runs one optimisation per fold, the fold loop,
    calls `f(ctx)` on every fold to compute the value. `f` is a function, or a struct that subtypes
    [`TimeDependentConstraintCallable`](@ref).

Two rules decide which value a fold gets.

  - Entry `i` belongs to fold `i` of `split(cv, rd)`. The library never reorders the folds. A
    walk-forward returns its folds in calendar order, so there entry `i` applies to the `i`-th
    window in time. When the order of a scheme is not the calendar, key the value to the fold's
    indices, which the context gives you.
  - A schedule has no effect outside a fold loop. A plain `optimise` call has no folds. There the
    field takes its static default, or the value you give in the schedule's `default` keyword.

We run one portfolio problem under four cross-validation schemes, [`IndexWalkForward`](@ref),
[`KFold`](@ref), [`CombinatorialCrossValidation`](@ref) and [`MultipleRandomised`](@ref). Under
each scheme we compare the static optimiser with one schedule or more.
=#
using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=2)) %" : v
    end
end;
#=
## 1. Setting up

We load three years of daily prices and build a minimum-variance [`MeanRisk`](@ref) optimiser with
two Clarabel solvers. The library tries the second solver when the first fails. The field we vary in
most sections is `wb`, the bounds of the weights. We change its upper bound, because the effect
shows in the largest weight of a fold. Sections 2.4, 3.1 and 7.6 put a schedule in `card`, `tn` and
`lcse`.
=#
using CSV, TimeSeries, DataFrames, Clarabel, Statistics, StableRNGs

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 3):end]

## Compute the returns.
rd = prices_to_returns(X)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))];

## Static baseline: an uncapped minimum-variance optimiser.
mr_static = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

## A helper that reads the largest weight of each fold's solution.
max_weights(pred) = [maximum(p.res.w) for p in pred.pred];
#=
## 2. The three forms

### 2.1 Static

A static cap of 20 % is a value of the `wb` field.
=#
mr_capped = MeanRisk(;
                     opt = JuMPOptimiser(; slv = slv,
                                         wb = WeightBounds(; lb = 0.0, ub = 0.2)))
#=
### 2.2 Vector schedule

A vector schedule has one value per fold. The field decides what the value means, so the same
[`Threshold`](@ref) schedule is a long threshold in `lt` and a short threshold in `st`.

This schedule lowers the cap from 35 % on the first fold to 20 % on the last. We size it later, when
we know how many folds the scheme returns. A vector schedule must have one entry per fold. The fold
loop checks the length when it splits the data, before any fold runs.
=#
function deleverage(n, bind = :outermost)
    return TimeDependent([WeightBounds(; lb = 0.0,
                                       ub = 0.35 - 0.15 * (i - 1) / max(n - 1, 1))
                          for i in 1:n], bind)
end;
#=
### 2.3 Callable

A callable computes the value when the fold runs. The fold loop calls it with a
[`TimeDependentContext`](@ref), whose fields are these.

  - `i`, the fold's position in the order of `split`, and `n`, the number of folds.
  - `rd`, the returns data. Under [`MultipleRandomised`](@ref) it has only the path's assets.
  - `train_idx` and `test_idx`, the training and test indices of all the folds.
  - `w_prev`, the previous fold's weights. It is `nothing` unless the fold loop passes the
    previous weights from fold to fold.
  - `path_id`, the fold's path under a scheme with many paths, and `nothing` otherwise.

This callable computes the falling cap of section 2.2 from `i` and `n` alone. We use it to
compare a vector schedule and a callable that state the same rule.
=#
deleverage_fn = TimeDependent(ctx -> WeightBounds(; lb = 0.0,
                                                  ub = 0.35 -
                                                       0.15 * (ctx.i - 1) /
                                                       max(ctx.n - 1, 1)));
#=
This callable changes with the data. It computes the annualised volatility of an equal-weight
portfolio over the fold's training window, and it lowers the cap as that volatility rises. It
indexes `ctx.train_idx` with `ctx.i`. `i` is the fold's position in the scheme's own order, so
`ctx.train_idx[ctx.i]` and `ctx.test_idx[ctx.i]` are this fold's windows under any scheme.
=#
function vol_cap(ctx)
    Xtr = ctx.rd.X[ctx.train_idx[ctx.i], :]
    vol = std(Xtr * fill(1 / size(Xtr, 2), size(Xtr, 2))) * sqrt(252)
    ## 35 % cap in calm regimes, tightening towards 15 % as annualised volatility rises.
    return WeightBounds(; lb = 0.0, ub = clamp(0.35 - vol, 0.15, 0.35))
end
vol_cap_td = TimeDependent(vol_cap);
#=
A callable can also be a struct with a method that takes the context. The struct subtypes
[`TimeDependentConstraintCallable`](@ref), which is a subtype of [`TimeDependentCallable`](@ref). A
struct has two advantages over a function. Its parameters are fields you can inspect. A
[`needs_previous_weights`](@ref) method on its type can declare that it uses the previous weights,
with no [`PreviousWeightsFunction`](@ref) wrapper.

We write the falling cap once more, as a type whose two parameters are the first cap and the last.
=#
struct DeleverageCap <: PortfolioOptimisers.TimeDependentConstraintCallable
    hi::Float64
    lo::Float64
end
function (c::DeleverageCap)(ctx::TimeDependentContext)
    return WeightBounds(; lb = 0.0,
                        ub = c.hi - (c.hi - c.lo) * (ctx.i - 1) / max(ctx.n - 1, 1))
end
deleverage_struct = TimeDependent(DeleverageCap(0.35, 0.2))
#=
### 2.4 When a wrong entry fails

You pass a schedule as a keyword value, and a wrong field name is an ordinary keyword error. The
constructor also builds the optimiser once with each entry of a vector schedule in the field. An
entry that the field does not accept fails when you build the optimiser, not when a fold runs. A
callable has no entries to test. The fold loop checks its value on every fold instead.

We put a [`Threshold`](@ref) in `card`, which takes an integer, and then `0`, which is not positive.
=#
try
    JuMPOptimiser(; slv = slv, card = TimeDependent([Threshold(; val = 0.01)]))
catch err
    err
end
#
try
    JuMPOptimiser(; slv = slv, card = TimeDependent([0]))
catch err
    err
end
#=
### 2.5 No effect outside a fold loop

A plain `optimise` call has no folds. The schedule has no effect, and `wb` takes its static default.
We solve the scheduled optimiser and the uncapped baseline on the full sample and compare their
weights with `isapprox`.
=#
mr_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(4)))
res_sched = optimise(mr_sched, rd)
res_static = optimise(mr_static, rd)
isapprox(res_sched.w, res_static.w)
#=
## 3. Walk-forward

A walk-forward suits a time-dependent constraint. Its folds are consecutive rebalances. Fold `i`
covers a later date range than fold `i - 1`. We train on one year and test on the quarter after
it.
=#
wf = IndexWalkForward(252, 63)
n_wf = n_splits(wf, rd)
#=
The vector schedule needs one entry per fold, so we size it with [`n_splits`](@ref). We backtest
the four optimisers over the same walk-forward with [`cross_val_predict`](@ref) and print the
largest weight of every fold. On every fold the loop puts the fold's value in the field and
solves the optimiser as it would without a schedule.
=#
mr_wf_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(n_wf)))
mr_wf_fn = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage_fn))
mr_wf_vol = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = vol_cap_td))

pred_wf_static = cross_val_predict(mr_static, rd, wf)
pred_wf_sched = cross_val_predict(mr_wf_sched, rd, wf)
pred_wf_fn = cross_val_predict(mr_wf_fn, rd, wf)
pred_wf_vol = cross_val_predict(mr_wf_vol, rd, wf)

pretty_table(DataFrame(:fold => 1:n_wf, :static => max_weights(pred_wf_static),
                       :schedule => max_weights(pred_wf_sched),
                       :callable => max_weights(pred_wf_fn),
                       :vol_callable => max_weights(pred_wf_vol)); formatters = [resfmt])
#=
The static optimiser has no cap. Under the vector schedule the largest weight stays under a cap that
falls from fold to fold. The `callable` column states the same falling cap, so compare it with the
`schedule` column fold by fold. The volatility
callable moves its cap with the volatility of the training window, not with the fold number.

We also backtest the struct and compare its weights with those of the function, fold by fold.
=#
pred_wf_struct = cross_val_predict(MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv,
                                                                wb = deleverage_struct)),
                                   rd, wf)
all(isapprox(a.res.w, b.res.w) for (a, b) in zip(pred_wf_struct.pred, pred_wf_fn.pred))
#=

The composition plot shows the schedule's weights on every fold. On the last folds the cap
binds, and the weight spreads over more assets.
=#
using StatsPlots, GraphRecipes
plot_composition(pred_wf_sched)
#=
### 3.1 Previous weights

The fold loop puts the schedule's value in the field before it gives the optimiser the previous
fold's weights. So a turnover constraint that a schedule puts in a fold gets the previous weights.
A callable can also use `ctx.w_prev`, but a function cannot declare that it does. Wrap it in
[`PreviousWeightsFunction`](@ref) to declare it. The wrapper makes
[`needs_previous_weights`](@ref) return `true`. On a scheme whose folds are a timeline, the fold
loop then runs the folds in sequence. A callable with no wrapper gets `w_prev === nothing`, unless
another part of the optimiser makes the loop run in sequence.

Here no weight can move more than 2 percentage points from its previous value at a rebalance. Fold
1 has no previous weights. The callable returns `nothing` there, and that fold has no turnover
constraint.
=#
tn_budget = TimeDependent(PreviousWeightsFunction(ctx -> if isnothing(ctx.w_prev)
                                                      nothing
                                                  else
                                                      Turnover(; w = ctx.w_prev, val = 0.02)
                                                  end))
mr_wf_tn = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tn = tn_budget))
pred_wf_tn = cross_val_predict(mr_wf_tn, rd, wf)

## One-norm distance between consecutive folds, with and without the budget.
function l1turnover(pred)
    return [sum(abs, pred.pred[i].res.w - pred.pred[i - 1].res.w)
            for i in 2:length(pred.pred)]
end
pretty_table(DataFrame(:rebalance => 2:n_wf, :static => l1turnover(pred_wf_static),
                       :budgeted => l1turnover(pred_wf_tn)); formatters = [resfmt])
#=
The budgeted run moves its weights less than the static run at every rebalance. The run also logs
an informational message. It says that the loop runs in sequence because the optimiser needs the
previous weights, and that a time-dependent constraint alone does not force this. A schedule that
needs no previous weights leaves the folds free to run in parallel.

## 4. KFold

`KFold` splits the data into consecutive blocks of time. Fold `i` tests on block `i` and trains on
the other blocks. A schedule works here with no change. Entry `i` is the constraint while block `i`
is out of sample. The library does not treat these folds as a timeline. No fold gets the previous
fold's weights, and the folds run in parallel.
=#
kfold = KFold(; n = 4)
mr_kf_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(4)))
pred_kf_static = cross_val_predict(mr_static, rd, kfold)
pred_kf_sched = cross_val_predict(mr_kf_sched, rd, kfold)
pred_kf_fn = cross_val_predict(MeanRisk(;
                                        opt = JuMPOptimiser(; slv = slv,
                                                            wb = deleverage_fn)), rd, kfold)

pretty_table(DataFrame(:fold => 1:4, :static => max_weights(pred_kf_static),
                       :schedule => max_weights(pred_kf_sched),
                       :callable => max_weights(pred_kf_fn)); formatters = [resfmt])
#=
A vector schedule of the wrong length fails when the loop splits the data, before any fold is
solved. The
error gives the number of folds the scheme returns.
=#
try
    cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(7))), rd,
                      kfold)
catch err
    err
end
#=
## 5. Combinatorial

Under [`CombinatorialCrossValidation`](@ref) a fold is a split into training and test data. The test
data of a split is a union of several separate blocks of time. `ctx.test_idx[ctx.i]` is a vector of
those blocks. The splits come in a combinatorial order, not a timeline, and the library does not
make one up. Entry `i` belongs to split `i` of `split(ccv, rd)`, which you can inspect. To tie a
constraint to time under this scheme, use a callable that orders the split's windows,
`ctx.train_idx[ctx.i]` and `ctx.test_idx[ctx.i]`, as your problem needs.
=#
ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
n_ccv = n_splits(ccv)
#=
Four blocks with two test blocks per split give six splits. The schedule needs six entries, one per
split in the order of `split(ccv, rd)`. A split covers several blocks of time, so a cap per split is
coarser than a cap per walk-forward fold. We run the schedule and the static optimiser under this
scheme.
=#
mr_cc_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(n_ccv)))
pred_cc_static = cross_val_predict(mr_static, rd, ccv)
pred_cc_sched = cross_val_predict(mr_cc_sched, rd, ccv)

## Largest weight across each path's folds; paths recombine the splits, so the cap shows
## up path by path.
pretty_table(DataFrame(:path => 1:length(pred_cc_sched.pred),
                       :static => [maximum(maximum(p.res.w) for p in path.pred)
                                   for path in pred_cc_static.pred],
                       :schedule => [maximum(maximum(p.res.w) for p in path.pred)
                                     for path in pred_cc_sched.pred]);
             formatters = [resfmt])
#=
The table gives the largest weight on each path. A path joins the test blocks of several splits,
and each block ran under the cap of its own split. The largest weight of a path mixes those caps,
and the table does not show the cap of one split.

## 6. MultipleRandomised

[`MultipleRandomised`](@ref) draws random subsets of the assets and runs a walk-forward on every
subset, which gives one path per subset. Two things change for a schedule.

  - `ctx.i` is the fold's position in its path. After the run the library sorts the predictions
    by test window for the report. The order of the output never changes which
    entry a fold gets.
  - Every path has its own assets. The fold loop restricts the optimiser, schedule entries
    included, to the path's assets before it puts the fold's value in the field. A callable finds
    the path's assets in `ctx.rd.nx`.

A callable that uses the number of assets suits this scheme. This one caps every weight at twice the
equal weight of the path.
=#
mrand = MultipleRandomised(IndexWalkForward(252, 63); subset_size = 15, n_subsets = 3,
                           rng = StableRNG(987654321), seed = 42)
universe_cap = TimeDependent(ctx -> WeightBounds(; lb = 0.0, ub = 2.0 / length(ctx.rd.nx)))
mr_mr_fn = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = universe_cap))
pred_mr_static = cross_val_predict(mr_static, rd, mrand)
pred_mr_fn = cross_val_predict(mr_mr_fn, rd, mrand)

pretty_table(DataFrame(:path => 1:length(pred_mr_fn.pred),
                       :static => [maximum(maximum(p.res.w) for p in path.pred)
                                   for path in pred_mr_static.pred],
                       :callable => [maximum(maximum(p.res.w) for p in path.pred)
                                     for path in pred_mr_fn.pred],
                       :cap => fill(2.0 / 15, length(pred_mr_fn.pred)));
             formatters = [resfmt])
#=
A vector schedule also works under this scheme. Size it to the folds of one path, which is the fold
count of the walk-forward. Every path uses the same schedule.
=#
n_mr = n_splits(IndexWalkForward(252, 63), rd)
mr_mr_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(n_mr)))
pred_mr_sched = cross_val_predict(mr_mr_sched, rd, mrand)
length(pred_mr_sched.pred)
#=
## 7. Nested fold loops: which loop uses the schedule?

A meta-optimiser such as [`Stacking`](@ref) or [`NestedClustered`](@ref) runs a cross-validation of
its own when you give it a `cv`. It uses that cross-validation to estimate the out-of-sample returns
of its inner optimisers. A schedule inside such a meta-optimiser can then be inside two fold loops.
One is the meta-optimiser's inner loop. The other is the outer loop of a backtest, when you pass
the meta-optimiser to [`cross_val_predict`](@ref). The schedule's `bind` decides which
loop uses it.

  - With the default, `:outermost`, the outermost fold loop uses the schedule.
  - With `:nearest`, in section 7.5, the nearest loop that encloses the schedule uses it.

In both cases, size the entries to the loop that uses them.

### 7.1 Outer cross-validation, no inner cross-validation

This is every run on the page so far. The backtest's fold loop puts each fold's value in the
field.

### 7.2 No outer cross-validation, inner cross-validation

A meta-optimiser that you pass to `optimise` has one fold loop, its inner cross-validation. That
loop uses the schedules of the inner optimisers, so you size them to the inner folds. The
meta-optimiser's solves over the full window have no folds, and they use the static defaults, as a
plain `optimise` does.

Here one of the two stacked optimisers lowers its cap across the inner `KFold(4)` folds. Those folds
compute the out-of-sample returns that the outer optimiser combines.
=#
st_inner = Stacking(;
                    opti = [MeanRisk(;
                                     opt = JuMPOptimiser(; slv = slv, wb = deleverage(4))),
                            mr_static], opto = mr_static,
                    cv = OptimisationCrossValidation(; cv = KFold(; n = 4)))
res_st_inner = optimise(st_inner, rd)
maximum(res_st_inner.w)
#=
With a schedule of the wrong length, the same estimator fails at the inner split.
=#
try
    optimise(Stacking(;
                      opti = [MeanRisk(;
                                       opt = JuMPOptimiser(; slv = slv, wb = deleverage(7))),
                              mr_static], opto = mr_static,
                      cv = OptimisationCrossValidation(; cv = KFold(; n = 4))), rd)
catch err
    err
end
#=
### 7.3 Outer and inner cross-validation

A backtest of the same meta-optimiser adds an outer fold loop. The outer loop now puts the outer
fold's value in every field that has a schedule, in the meta-optimiser and in its inner optimisers,
before the meta-optimiser runs. The inner `KFold(4)` then sees only static optimisers. So the
schedule must have one entry per outer fold, although the meta-optimiser has an inner `KFold(4)`.
=#
st_nested = Stacking(;
                     opti = [MeanRisk(;
                                      opt = JuMPOptimiser(; slv = slv,
                                                          wb = deleverage(n_wf))),
                             mr_static], opto = mr_static,
                     cv = OptimisationCrossValidation(; cv = KFold(; n = 4)))
pred_st_nested = cross_val_predict(st_nested, rd, wf)
pretty_table(DataFrame(:fold => 1:n_wf, :stacked => max_weights(pred_st_nested));
             formatters = [resfmt])
#=
The estimator of section 7.2, with its schedule sized to the inner folds, now fails at the outer
split. The estimator is the same, but the outer walk-forward now uses the schedule, and it has
`n_wf` folds.
=#
try
    cross_val_predict(st_inner, rd, wf)
catch err
    err
end
#=
### 7.4 Tuning over schedules

A schedule is a field value, so you can tune it like any other hyperparameter.
[`GridSearchCrossValidation`](@ref) puts each candidate in the field through the estimator's
constructor, which checks it. The search then scores the candidate with its own cross-validation.
Here it picks one of three candidates for `wb`: no cap, the falling cap of section 2.2, and a cap
that falls less.
=#
function gentler(n)
    return TimeDependent([WeightBounds(; lb = 0.0,
                                       ub = 0.45 - 0.1 * (i - 1) / max(n - 1, 1))
                          for i in 1:n])
end;
candidates = ["opt.wb" => [WeightBounds(), deleverage(n_wf), gentler(n_wf)]]
gs = GridSearchCrossValidation(candidates; cv = wf)
gs_res = search_cross_validation(mr_static, gs, rd)
gs_res.idx
#=
`gs_res.idx` is the position of the candidate with the best out-of-sample score, and
`gs_res.val_grid[gs_res.idx]` is a tuple with that candidate in it. The search's fold loop uses
the entries of each vector schedule among the candidates, so each of them must have
`n_splits(gs.cv, rd)` entries.

### 7.5 Bind a schedule to the inner loop with `:nearest`

In section 7.3 the outer loop used the schedule of an inner optimiser. With `bind = :nearest`, the
meta-optimiser's inner cross-validation uses the schedule, also when an outer loop backtests the
meta-optimiser. Use it for an inner optimiser whose cap must follow the inner `KFold(4)` under any
outer backtest. Here the second argument of `deleverage`
passes `bind`. The outer loop skips the schedule and the inner loop uses it, so you size it to the
inner folds under an outer backtest too.
=#
st_bind_near = Stacking(;
                        opti = [MeanRisk(;
                                         opt = JuMPOptimiser(; slv = slv,
                                                             wb = deleverage(4, :nearest))),
                                mr_static], opto = mr_static,
                        cv = OptimisationCrossValidation(; cv = KFold(; n = 4)))
pred_bind_near = cross_val_predict(st_bind_near, rd, wf)
pretty_table(DataFrame(:fold => 1:n_wf, :nearest => max_weights(pred_bind_near));
             formatters = [resfmt])
#=
The schedule of four entries that failed at the outer split in section 7.3 now runs. With
`:nearest`, the inner `KFold(4)` uses it and the outer loop does not. You can mix the two values of
`bind`. The `wb` of a meta-optimiser can be `:outermost`, sized to the backtest, while the schedule
of an inner optimiser is `:nearest`, sized to the inner scheme. Each loop checks the length of its
own schedules when it splits the data.

### 7.6 A schedule of constraint vectors

A field that takes a vector of constraints changes from fold to fold through a vector of vectors.
Entry `i` is the whole constraint vector of fold `i`. Here the linear constraints `lcse` cap the
weight of WMT at 15 % on every fold, and they add a cap of 15 % on JNJ in the second half of the
folds. The uncapped baseline puts its largest weight on WMT on the first fold and on JNJ on the last
two folds.
=#
sets = UniverseSets(; dict = Dict("nx" => rd.nx))
cap_a = LinearConstraintEstimator(; val = "WMT <= 0.15")
cap_b = LinearConstraintEstimator(; val = "JNJ <= 0.15")
lcse_sched = TimeDependent([i <= n_wf ÷ 2 ? [cap_a] : [cap_a, cap_b] for i in 1:n_wf])
mr_lcse = MeanRisk(; opt = JuMPOptimiser(; slv = slv, sets = sets, lcse = lcse_sched))
pred_lcse = cross_val_predict(mr_lcse, rd, wf)

## The weight of one asset on each fold.
asset_weights(pred, name) = [p.res.w[findfirst(==(name), rd.nx)] for p in pred.pred]
pretty_table(DataFrame(:fold => 1:n_wf, :WMT_static => asset_weights(pred_wf_static, "WMT"),
                       :WMT_schedule => asset_weights(pred_lcse, "WMT"),
                       :JNJ_static => asset_weights(pred_wf_static, "JNJ"),
                       :JNJ_schedule => asset_weights(pred_lcse, "JNJ"));
             formatters = [resfmt])
#=
The weight of WMT stays at or under its cap on every fold. In the first half of the folds JNJ has no
cap, and on the first two folds it takes a part of the weight that the cap removes from WMT. In the
second half, the cap of JNJ binds on the folds where the baseline puts more than 15 % on it.

A schedule is always the whole value of a field. An entry of a constraint vector cannot be a
schedule. To change some entries of a constraint vector and keep the
others, build the fold's vector in a callable,
`TimeDependent(ctx -> [dynamic(ctx), a_static_constraint])`. The static parts then stay in one
place.

## 8. Summary

| Form | How you write it | When the library checks it | Parallel folds | Use it for |
|---|---|---|---|---|
| Static | the field itself | when you build the optimiser | yes | constraints that do not change |
| Vector schedule | `field = TimeDependent([v₁, …, vₙ])` | every entry when you build the optimiser, and the length when the fold loop splits the data | yes, unless an entry needs the previous weights on a timeline scheme | plans known in advance: a falling cap, a mandate in phases, regime dates |
| Callable | `field = TimeDependent(f)`, with `f` a function or a [`TimeDependentConstraintCallable`](@ref) struct | its value on every fold, through the constructor of the field's optimiser | yes, unless it declares the previous weights on a timeline scheme, through the [`PreviousWeightsFunction`](@ref) wrapper or the struct's method | values computed from the fold: the volatility regime, the number of assets, the previous weights |

Under every scheme, entry `i` is the value for fold `i` in the order of `split`, and
`ctx.train_idx[ctx.i]` and `ctx.test_idx[ctx.i]` are the windows of that fold. In every form, each
fold solves an ordinary static optimiser. The fold loop builds it through the constructor that
checks the field, so the fold's solve never sees a schedule. A field that takes a vector of
constraints takes a schedule of vectors, as in section 7.6.

The meta-optimisers [`NestedClustered`](@ref), [`Stacking`](@ref) and [`SubsetResampling`](@ref)
take schedules in their own fields, such as `wb`, `fees`, `pe`, `sets` and `wf`, and in their
optimiser fields. They pass the fold's value on to their inner optimisers. The `bind` of a schedule
chooses the fold loop that uses it. With the default, `:outermost`, an outer loop over a
meta-optimiser uses an inner schedule with the outer folds. A meta-optimiser that runs alone uses
it in its inner cross-validation. With `:nearest`, as in section 7.5, the inner cross-validation
uses the schedule under an outer backtest.

One rule decides which inputs take a schedule. An input that states what the fold solves can change
from fold to fold: the prior, the constraints, the risk measures, the objective, the asset sets and
the fallbacks. An input that states how the fold is solved stays static: the solvers, the random
number generators and the cross-validation scheme of a meta-optimiser. The type of each constructor
argument enforces the rule. The optimiser is part of what the fold solves, so a schedule of whole
optimisers also works, passed to [`cross_val_predict`](@ref) or held in a field that takes an
optimiser. [Time-dependent optimisers](06_Time_Dependent_Optimisers.md) covers that case.
=#
