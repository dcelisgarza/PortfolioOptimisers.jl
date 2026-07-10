#=
# Time-dependent constraints

Every constraint we have used so far is *static*: it is fixed when the optimiser is constructed and applies unchanged to every optimisation. Under cross-validation, however, each fold is a separate optimisation over a different slice of time — and sometimes the constraint itself should change with time: a de-leveraging schedule that tightens position caps, a turnover budget relative to the previous rebalance, bounds that react to the volatility regime of the training window.

[`TimeDependent`](@ref) expresses exactly this. It wraps *either* a vector of per-fold values (a **schedule**) *or* a function of the fold's [`TimeDependentContext`](@ref) (a **callable**), and is stored *directly in the optimiser field it varies* — `JuMPOptimiser(; wb = TimeDependent([...]))`. The field's position names what varies, so a field holds either a static value or a schedule, never both. This gives three ways to specify any input typed [`TD_Option`](@ref):

 1. **Static** — set the field itself; the same value applies to every fold.
 2. **Schedule-based** — `TimeDependent([v₁, …, vₙ])`; entry `i` is the complete field value for fold `i` of the consuming scheme's `split` enumeration.
 3. **Callable** — `TimeDependent(f)`; `f(ctx)` computes the value per fold on the fly. `f` can be a bare function or a [`TimeDependentCallable`](@ref) functor struct.

Two rules govern the behaviour:

  - **Enumeration-order indexing, no hidden ranking**: entry `i` maps to fold `i` of `split(cv, rd)` — the machinery never re-orders folds behind your back. Walk-forward and (unshuffled) KFold enumerate chronologically, so there "fold time" is calendar time; for schemes whose enumeration is not a timeline it is *your* job to key entries off the fold's indices, which the context provides.
  - **Inert outside fold loops**: a plain `optimise` call has no folds, so the schedule simply does not participate — the affected fields run at their static defaults.

In this example we run the same portfolio problem under each cross-validation scheme — [`IndexWalkForward`](@ref), [`KFold`](@ref), [`CombinatorialCrossValidation`](@ref) and [`MultipleRandomised`](@ref) — and compare how the three methods behave in each.
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

We use three years of daily data, a [`MeanRisk`](@ref) minimum-variance optimiser, and a short Clarabel fallback chain. The constraint we vary throughout is the per-asset weight cap (the `wb` field), because its effect is easy to read straight off the optimal weights.
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
## 2. The three methods

### 2.1 Static

Nothing new here — a fixed cap of 20 % is just the `wb` field:
=#
mr_capped = MeanRisk(;
                     opt = JuMPOptimiser(; slv = slv,
                                         wb = WeightBounds(; lb = 0.0, ub = 0.2)))
#=
### 2.2 Schedule-based

A schedule is a vector of per-fold values; the field it sits in says what it varies, so there is nothing else to name — the same [`Threshold`](@ref) schedule means "long threshold" in `lt` and "short threshold" in `st`.

The schedule below is a de-leveraging plan: the cap tightens as we walk forward through time. We size it later, once we know how many folds the consuming cross-validation scheme produces — **a schedule must have exactly one entry per fold**, which is validated as soon as the scheme is split, before any fold runs.
=#
function deleverage(n)
    return TimeDependent([WeightBounds(; lb = 0.0,
                                       ub = 0.35 - 0.15 * (i - 1) / max(n - 1, 1))
                          for i in 1:n])
end;
#=
### 2.3 Callable

A callable computes the value when the fold runs. It receives a [`TimeDependentContext`](@ref) carrying the fold's enumeration index `i`, the fold count `n`, the (possibly asset-viewed) returns data `rd`, the scheme's fold index vectors, and — only when previous weights are threaded — `w_prev`.

This one reproduces the same de-leveraging plan from the rank alone, so we can check that schedules and callables are two spellings of the same thing:
=#
deleverage_fn = TimeDependent(ctx -> WeightBounds(; lb = 0.0,
                                                  ub = 0.35 -
                                                       0.15 * (ctx.i - 1) /
                                                       max(ctx.n - 1, 1)));
#=
And this one is genuinely dynamic — it reads the volatility of the fold's training window and tightens the cap in turbulent regimes. Note that it indexes `ctx.train_idx` with `ctx.i`; because `i` is the fold's position in the scheme's own enumeration, `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` are always the fold's *own* windows, under every scheme.
=#
function vol_cap(ctx)
    Xtr = ctx.rd.X[ctx.train_idx[ctx.i], :]
    vol = std(Xtr * fill(1 / size(Xtr, 2), size(Xtr, 2))) * sqrt(252)
    ## 35 % cap in calm regimes, tightening towards 15 % as annualised volatility rises.
    return WeightBounds(; lb = 0.0, ub = clamp(0.35 - vol, 0.15, 0.35))
end
vol_cap_td = TimeDependent(vol_cap);
#=
Callables need not be bare functions. A struct subtyping [`TimeDependentCallable`](@ref) whose functor takes the context does the same job with two advantages: its parameters are inspectable data, and — being a type — a [`needs_previous_weights`](@ref) method can declare a previous-weights requirement directly, without the [`PreviousWeightsFunction`](@ref) wrapper.

Here is the de-leveraging plan once more, as a reusable parameterised type:
=#
struct DeleverageCap <: PortfolioOptimisers.TimeDependentCallable
    hi::Float64
    lo::Float64
end
function (c::DeleverageCap)(ctx::TimeDependentContext)
    return WeightBounds(; lb = 0.0,
                        ub = c.hi - (c.hi - c.lo) * (ctx.i - 1) / max(ctx.n - 1, 1))
end
deleverage_struct = TimeDependent(DeleverageCap(0.35, 0.2))
#=
### 2.4 Validation happens as early as possible

Because the schedule lives in the field itself, a wrong target is an ordinary keyword error — there is no symbol to typo and no way to give a field both a static value and a schedule. What remains is entry validity: every schedule entry is test-substituted through the constructor, so a type-incompatible entry fails immediately:
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
### 2.5 Inert outside fold loops

A fold-less `optimise` has no time axis, so the schedule does not participate — the optimiser behaves exactly like the static baseline (the field runs at its static default):
=#
mr_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(4)))
res_sched = optimise(mr_sched, rd)
res_static = optimise(mr_static, rd)
isapprox(res_sched.w, res_static.w)
#=
## 3. Walk-forward

Walk-forward is the natural home of time-dependent constraints: folds are consecutive rebalances, so "fold time" *is* calendar time. We train on one year and test on the following quarter.
=#
wf = IndexWalkForward(252, 63)
n_wf = n_splits(wf, rd)
#=
The schedule needs one entry per fold, so we size it with [`n_splits`](@ref). All three estimators run through the same [`cross_val_predict`](@ref) call — the fold loop resolves any schedule into an ordinary static optimiser before each solve.
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
The static column is free to concentrate; the schedule column respects the tightening cap fold by fold; the rank-based callable matches the schedule exactly (same rule, different spelling); and the volatility callable moves with the regime instead of the calendar.

The struct form is a third spelling of the same rule — its fold weights are identical to the function form's:
=#
pred_wf_struct = cross_val_predict(MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv,
                                                                wb = deleverage_struct)),
                                   rd, wf)
all(isapprox(a.res.w, b.res.w) for (a, b) in zip(pred_wf_struct.pred, pred_wf_fn.pred))
#=

The composition plot makes the de-leveraging visible — later folds are forced to spread weight across more assets:
=#
using StatsPlots, GraphRecipes
plot_composition(pred_wf_sched)
#=
### 3.1 Previous weights

Schedules and callables are resolved *before* the previous-weights factory pass, so a per-fold turnover constraint swapped in by a schedule still receives the previous fold's weights. A callable can also read the previous weights directly, but because a bare function cannot be inspected, it must declare the requirement by wrapping itself in [`PreviousWeightsFunction`](@ref) — this is what flips [`needs_previous_weights`](@ref) and forces the fold loop to run sequentially (an *undeclared* callable would see `w_prev === nothing`).

Here each asset may move at most 2 percentage points per rebalance relative to its previous weight; fold 1 has no previous weights, so returning `nothing` leaves the turnover constraint off:
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
Note the informational message: it is the previous-weights requirement that forces sequential execution, not time dependence itself. Schedules and undeclared callables keep the fold loop fully parallel because entry `i` is known upfront.

## 4. KFold

KFold's folds are also time-ordered slices (shuffling is rejected for optimisation cross-validation), so schedules carry over unchanged — with one difference in interpretation: fold `i` *tests* on the `i`-th slice while training on the rest, so a schedule reads "the constraint in force while slice `i` is out of sample". Everything stays parallel.
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
A mis-sized schedule fails at `split` time — before a single fold is solved — with the fold count the scheme actually produced:
=#
try
    cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(7))), rd,
                      kfold)
catch err
    err
end
#=
## 5. Combinatorial

Under [`CombinatorialCrossValidation`](@ref) a *fold* is a train/test split, and each split's test set is a union of several disjoint time groups. Splits are enumerated combinatorially, so the enumeration is *not* a timeline — and the machinery deliberately does not invent one: any single "position in time" for a split whose test set is a union of disjoint groups would be an arbitrary hidden policy. Entry `i` simply belongs to split `i` of `split(ccv, rd)`, which you can inspect; to key a constraint to time here, prefer a callable that reads its own windows (`ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]`) and derives whatever ordering your problem calls for.
=#
ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
n_ccv = n_splits(ccv)
#=
With 4 groups choose 2 test groups we get 6 splits, so the schedule needs 6 entries — one per split, in `split(ccv, rd)`'s enumeration order. Because a split spans several time groups, "the constraint in force for this split" is inherently coarser in meaning than under walk-forward; the schedule below is kept to show the mechanics (entry `i` caps split `i`), and a genuinely time-keyed policy is better expressed as a callable deriving its value from its own windows.
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
## 6. MultipleRandomised

[`MultipleRandomised`](@ref) crosses random *asset subsets* with a walk-forward over time, producing one path per subset. Two things happen to a schedule here:

  - `ctx.i` is the fold's position in the path's enumeration; predictions are re-sorted by test window *afterwards, for reporting only* — output order never influences which entry a fold received.
  - Each path sees a different asset universe: the optimiser is *viewed* down to the subset before the schedule is resolved, so schedule entries are sub-selected along with everything else, and callables see the viewed universe through `ctx.rd` (its `nx` are the subset's names).

A universe-aware callable is the natural fit — here the cap adapts to however many assets the path drew, allowing at most twice the equal weight:
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
Schedules work here too — sized to the folds *per path* (the walk-forward fold count), shared across paths:
=#
n_mr = n_splits(IndexWalkForward(252, 63), rd)
mr_mr_sched = MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = deleverage(n_mr)))
pred_mr_sched = cross_val_predict(mr_mr_sched, rd, mrand)
length(pred_mr_sched.pred)
#=
## 7. Summary

| Method | Spelling | Sized/validated | Parallel? | Best for |
|---|---|---|---|---|
| Static | the field itself | at construction | yes | constraints that do not change |
| Schedule | `field = TimeDependent([v₁, …, vₙ])` | each entry test-substituted at construction; length vs fold count at `split` time | yes | known calendars: de-leveraging plans, phased mandates, regime dates fixed in advance |
| Callable | `field = TimeDependent(f)` — a function, or a [`TimeDependentCallable`](@ref) struct that can declare previous-weights needs as a trait | output validated by the host constructor at each fold | yes, unless previous weights are declared ([`PreviousWeightsFunction`](@ref) wrapper or the struct's trait) | values computed from the fold: volatility regimes, universe size, previous weights |

Across schemes, the entry index always means the same thing — fold `i` of the consuming scheme's `split` enumeration, with `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` as the fold's own windows — and whatever the spelling, each fold ends up solving an ordinary static optimiser: the schedule is resolved through the host's validated constructor, so nothing downstream of the fold loop knows time dependence exists. Meta-optimisers ([`NestedClustered`](@ref), [`Stacking`](@ref), [`SubsetResampling`](@ref)) accept schedules in their own `wb`/`fees` and forward the resolution into their inner estimators like any other wrapper, so an outer fold loop over a meta resolves an inner schedule against the outer folds; standalone, the meta's inner cross-validation consumes it instead.
=#
