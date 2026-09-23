#=
```@meta
Description = "The online walk-forward in PortfolioOptimisers.jl worked through, with one estimator fitted on the first training window and updated fold by fold."
```

# The online walk-forward: one estimator, updated fold by fold

A walk-forward refits every fold from its training window. Fold `i` uses every row that fold
`i - 1` used and a few more, fits the moments over all of them, and solves. On an expanding
window the cost of the whole run therefore grows with the square of the number of folds, and
the moments of fold `i` are recomputed from scratch when almost all of their input is the
input of fold `i - 1`.

An online walk-forward, built by [`OnlineIndexWalkForward`](@ref), fits one estimator on the
first training window, adds the *new* rows of each later fold to it, and computes the weights
where a refit would have run. We call the step that adds rows to a fitted estimator an update,
and [`partial_fit!`](@ref) is the function that makes it. The online run gives the weights of the
batch expanding-window walk-forward, fold by fold.

The list below says what each part of a run does in an online walk-forward.

  - A prior updates its moment estimators with the new rows, and it keeps the rows themselves,
    because a prior result holds `X` for the scenario risk measures.
  - An optimiser passes the new rows to its prior alone. To compute weights, it rebuilds a
    returns result from the prior's rows and runs the ordinary batch fit, so every constraint,
    every clustering and every uncertainty set is the one the batch run builds. A set with no
    prior of its own is calibrated on the prior result the optimiser gives it.
  - An optimiser that solves a JuMP problem, such as [`MeanRisk`](@ref), builds a new model each
    time it computes weights. The solve is the cost of the update, and no model is kept between
    folds.
  - The type of the walk-forward declares the update, and the fold loop starts from an estimator
    with no state.
  - A search scores every candidate through the same fold loop, online and batch alike.
  - A `Pipeline` passes the update to its steps. A step that works row by row updates, a step
    that changes the universe takes a view, and a step with no online form throws an error
    unless the pipeline declares a refit.
  - A result lets you continue a run. `Resume(res)` starts the loop again after the folds the
    result holds.

A prior with no update formula, such as [`FactorPrior`](@ref), goes in `Online` instead.
`Online(pe; max_history)` keeps a buffer of rows, and the library refits the prior from that
buffer each time it computes weights. The cap on the buffer gives a rolling window.

!!! tip "When to reach for this"
    Reach for `OnlineIndexWalkForward` in place of an expanding walk-forward whose data and
    estimators the update accepts, because it costs nothing in accuracy. The run gives the weights
    of the batch run, fold by fold. Sections 2 and 3 show inputs the update refuses. Reach for it
    for speed when the prior has a `CoveragePolicy` and the optimiser computes its weights
    cheaply, as a hierarchical or a naive optimiser does, because there the batch fit itself
    runs row by row. Reach for it whenever you will continue a run, because the result of an
    online run resumes over a longer history and a batch result cannot. With a prior that has no
    `CoveragePolicy`, through `MeanRisk`, the update saves no time, and section 9 prints the
    ratio.

This example runs one walk-forward through each of those estimators, and each section prints the
comparison it makes. Section 8 measures the accuracy of the update and its speed on one
covariance, and section 9 measures its speed through the walk-forward.

 1. The panel, a universe that moves inside the window, and the two walk-forwards.
 2. The online run against the batch expanding run, fold by fold.
 3. What the loop does, written by hand.
 4. A prior with no update formula refits from its buffer.
 5. A capped buffer is the rolling window.
 6. A search picks the batch candidate.
 7. A run resumes from its result.
 8. Speed and accuracy of the moment update.
 9. The gain through the loop, measured.
 10. What to take away.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, DataFrames,
      PrettyTables, Clarabel

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. The panel, and the two walk-forwards

The data is a small factor panel drawn from a seeded generator, with eight assets, three hundred
observations and two factors. Five assets list after the first row, and three delist before the
last. The universe therefore moves inside the window, and an online update must follow it. We
write the listing calendar out as a fact about the instruments, and the returns are `NaN`
wherever an asset is not listed. The [`AssetPanel`](@ref) takes the calendar as its active mask.
We set the estimation mask to the active one, because the online update takes the active mask
alone. Section 2 shows the error that a narrower estimation mask throws.

The function `cut` keeps the rows `i` of the returns, of the timestamps and of the panel masks.
Sections 3 and 7 use it to cut windows by hand. The last line prints the first and the last
listed row of each asset.
=#

function synthetic_panel(; T = 300, N = 8, K = 2, seed = 20260912)
    rng = StableRNG(seed)
    F = 0.01 .* randn(rng, T, K)
    B = randn(rng, N, K)
    ivol = 0.008 .+ 0.012 .* rand(rng, N)
    X = F * transpose(B) .+ randn(rng, T, N) .* transpose(ivol)
    listed = [1, 34, 42, 1, 67, 10, 175, 1]
    delisted = [T, T, T, 252, T, 291, T, 264]
    amsk = [listed[i] <= t <= delisted[i] for t in 1:T, i in 1:N]
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2015, 1, 1):Day(1):(Date(2015, 1, 1) + Day(2 * T + 10)))[1:T]
    nx = ["A" * lpad(i, 2, '0') for i in 1:N]
    pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk))
    rd = ReturnsResult(; nx = nx, X = ifelse.(amsk, X, NaN), ts = days, pnl = pnl)
    rdf = ReturnsResult(; nx = nx, X = rd.X, ts = days, nf = ["F1", "F2"], F = F, pnl = pnl)
    return rd, rdf
end

rd, rdf = synthetic_panel()
T, N = size(rd.X)
amsk = Matrix(rd.pnl.amsk)
function cut(rd, i)
    return ReturnsResult(; nx = rd.nx, X = rd.X[i, :], ts = rd.ts[i],
                         pnl = AssetPanel(; amsk = amsk[i, :], emsk = amsk[i, :]))
end

[(rd.nx[j], findfirst(amsk[:, j]), findlast(amsk[:, j])) for j in 1:N]

#=
The two walk-forwards enumerate the same folds. The batch one expands its training window, as an
online run does anyway, because an update adds rows to the moments and never removes them.
`OnlineIndexWalkForward` therefore sets `expand_train = true`, and the two schemes cut identical
windows. Both purge three rows before each test window. A rolling window is set on the
estimator instead, as section 5 shows.
=#

w, t, p = 100, 40, 3
batch = IndexWalkForward(w, t; purged_size = p, expand_train = true)
online = OnlineIndexWalkForward(w, t; purged_size = p)

(; train_idx, test_idx) = split(batch, rd)
(train_idx == split(online, rd).train_idx, train_idx, [first(i):last(i) for i in test_idx])

#=
The seventh asset lists at row 175, inside the third training window, and the fourth delists at
row 252, inside the fifth. A prior with no `CoveragePolicy` never admits a late listing over an
expanding window from row 1. It needs a finite return at every row of the window, and the window
always starts at row 1. The prior below has a [`CoveragePolicy`](@ref) on both moments. Each
asset's mean uses the rows where that asset has a return, each covariance entry uses the rows
where both assets have one, and an asset enters the estimate when it lists. Under the policy the
online update gives the batch estimate exactly, because the batch fit under the policy runs the
same update over the rows. Section 9 returns to that fact.
=#

cvg = CoveragePolicy()
pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                    ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = cvg)))
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                             "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
             check_sol = (; allow_local = false, allow_almost = false))
mr = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv))
hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe))

#=
## 2. The online run against the batch expanding run, fold by fold

We make the same call to [`cross_val_predict`](@ref) twice, once under each scheme. The table
prints the largest
weight difference per fold for `MeanRisk`, in `jump_gap`, and for `HierarchicalRiskParity`, in
`hierarchical_gap`.
=#

b_mr = cross_val_predict(mr, rd, batch)
o_mr = cross_val_predict(mr, rd, online)
b_hrp = cross_val_predict(hrp, rd, batch)
o_hrp = cross_val_predict(hrp, rd, online)

weights(res) = [p.res.w for p in res.pred]
masks(res) = [p.res.imsk for p in res.pred]
count_in(m) = isnothing(m) ? N : count(m)
gap(a, b) = [maximum(abs, x - y) for (x, y) in zip(weights(a), weights(b))]

fold_df = DataFrame(; fold = 1:length(train_idx), window = string.(train_idx),
                    investable = count_in.(masks(o_mr)), jump_gap = gap(o_mr, b_mr),
                    hierarchical_gap = gap(o_hrp, b_hrp),
                    masks_agree = masks(o_mr) .== masks(b_mr))
pretty_table(fold_df; formatters = [numfmt])

#=
The `hierarchical_gap` column is zero on every fold, because under the policy the batch fit and
the online update compute the same numbers. The `jump_gap` column is zero here too. In general it
sits within the tolerance of the solver, because `MeanRisk` solves the same problem from the same
rows.

Both runs follow the universe as it moves. Seven assets are investable in the first two folds,
all eight once the seventh lists, and seven again once the fourth delists. `imsk` is `nothing` on
a fold where every asset was investable. Only the prior ran online. `HierarchicalRiskParity`
clustered and `MeanRisk` solved as they do in batch, over a returns result rebuilt from the
prior's rows.

The online update takes the active mask of the panel with the rows, and no other part of the
panel. A panel whose estimation mask differs from its active mask, or that has time-varying panel
fields, throws an error at the warm-up, the first fit on the first training window. The message
names the field. The batch walk-forward accepts either kind of panel, because each fold refits
over its whole window.
=#

narrow = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts,
                       pnl = AssetPanel(; amsk = amsk, emsk = amsk .& (rd.X .> -0.03)))
try
    cross_val_predict(mr, narrow, online)
catch err
    showerror(stdout, err)
end

#=
## 3. What the loop does, written by hand

The loop calls two functions. `partial_fit!(opt, rd)` adds the rows of a returns result to the
optimiser and solves nothing. `optimise(opt)` with no data computes the weights from the rows
added so far, and solves once. The warm-up uses the first training window, and each later fold
adds the rows its window gained since the one before. We write the loop by hand below, and the
last line prints the largest difference between our weights and the weights of the loop.
=#

est = partial_fit!(mr, cut(rd, train_idx[1]))
hand = [optimise(est).w]
for i in 2:length(train_idx)
    delta = (last(train_idx[i - 1]) + 1):last(train_idx[i])
    global est = partial_fit!(est, cut(rd, delta))
    push!(hand, optimise(est).w)
end

maximum(maximum(abs, a - b) for (a, b) in zip(hand, weights(o_mr)))

#=
The loop keeps the state in the `cache` fields of the estimator, and it starts from an estimator
with no state. An estimator that already has a state throws an error at the warm-up that names
it, so the estimator you give the walk-forward is always a configuration and never a half-fitted
run. A [`TimeDependent`](@ref) schedule on the prior, or on the optimiser itself, throws at the
warm-up for the same reason, because the schedule would replace the estimator that holds the
state. A schedule on any other field works as it does in batch, so the weight bounds below
tighten as the folds advance, online as in batch.
=#

n = length(train_idx)
caps = TimeDependent([WeightBounds(; lb = 0.0, ub = ub) for ub in range(0.4, 0.25, n)])
mr_caps = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv, wb = caps))
gap(cross_val_predict(mr_caps, rd, online), cross_val_predict(mr_caps, rd, batch))

#=
## 4. A prior with no update formula refits from its buffer

A [`FactorPrior`](@ref) regresses the assets on the factors, and the library has no update
formula for a regression. You wrap it in [`Online`](@ref) instead. `Online` keeps a buffer of
rows in the `cache` of the prior. Each update appends the new rows to the buffer, with the factor
rows beside them, because the regression needs both. To compute weights, the library runs the
batch fit of the prior over the buffer. The buffer holds the rows the batch fold uses, so the two
runs fit the same sample. We compare the weights of the two `MeanRisk` runs, one batch and one
online.
=#

fp_batch = MeanRisk(; opt = JuMPOptimiser(; pe = FactorPrior(), slv = slv))
fp_online = MeanRisk(; opt = JuMPOptimiser(; pe = Online(FactorPrior()), slv = slv))

(weights(cross_val_predict(fp_online, rdf, online)) ==
 weights(cross_val_predict(fp_batch, rdf, batch)))

#=
`Online` wraps the estimator in its field, and it does not survive the run. The loop replaces it
at the warm-up with the ordinary estimator and a state, and no `Online` exists after that. An
estimator whose parts mix the two kinds updates each part that has an update formula, and refits
each other part from the rows it keeps. You therefore write the same estimator for an online run
as for a batch run.

## 5. A capped buffer is the rolling window

An online run only ever grows its history, so the estimator declares a rolling window, not the
loop. `Online(pe; max_history = w)` keeps the last `w` rows and drops the oldest as each new one
arrives. A rolling batch scheme of window `w + p` with purge `p` trains over `w` rows, and the
capped online run computes its weights over those same rows. We check the window length and
compare the weights and the masks of the two runs.
=#

rolling = IndexWalkForward(w + p, t; purged_size = p)
stepped = OnlineIndexWalkForward(w + p, t; purged_size = p)
cap(pe) = Online(pe; max_history = w)

hrp_cap = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = cap(pe)))
b_roll = cross_val_predict(hrp, rd, rolling)
o_roll = cross_val_predict(hrp_cap, rd, stepped)

(all(length.(split(rolling, rd).train_idx) .== w), weights(o_roll) == weights(b_roll),
 masks(o_roll) == masks(b_roll))

#=
A cap on a prior that updates its moments does something else. `max_scenarios` on a prior bounds
the rows its result keeps for the scenario risk measures, and it leaves the moments fitted over
every observation. A prior capped that way therefore matches no batch fit. `max_history` on
`Online` limits the whole fit to a window, and it has a batch equal, which is the comparison
above. You can set both, and they nest.

## 6. A search picks the batch candidate

A search scores every candidate through the same fold loop, whatever the fit. The grid tunes the
weight bounds, which bind and so separate the candidates. We compare the two score matrices and
the column each search picked.
=#

r = ConditionalValueatRisk()
grid = ["opt.wb" =>
            [WeightBounds(; lb = 0.0, ub = 1.0), WeightBounds(; lb = 0.0, ub = 0.2),
             WeightBounds(; lb = 0.1, ub = 1.0)]]
gs(cv) = GridSearchCrossValidation(grid; cv = cv, r = r, train_score = true)

s_b = search_cross_validation(hrp, gs(batch), rd)
s_o = search_cross_validation(hrp, gs(online), rd)

(s_o.test_scores == s_b.test_scores, s_o.train_scores == s_b.train_scores,
 s_o.idx == s_b.idx, s_o.val_grid[s_o.idx])

#=
Under an online scheme the folds run in sequence whatever executor the search is given, because
fold `i` starts from the state that fold `i - 1` left. The candidates are what gains from
threads, and the `ex` of `GridSearchCrossValidation` applies to them.

`RandomisedSearchCrossValidation` takes an online scheme too, because it samples a grid from the
search space and runs a `GridSearchCrossValidation` over it.

## 7. A run resumes from its result

The [`MultiPeriodPredictionResult`](@ref) of an online run holds the estimator in `opt`, as it
stood at the end of the last training window. When more rows arrive, [`Resume`](@ref) gives that
result back to the loop with the full history and the same scheme. The loop skips the folds the
result holds, adds the rows between the last training end and the new end to a copy of the
state, and continues from the next fold. The resumed result holds the new folds alone, and `vcat`
stacks the two.

We compare the stack with the run over the full history at once. The loop finds the fold to
resume from by the last timestamp the state holds, so a run that resumes needs timestamps, and
the panel has dates. The first run gets the first 260 rows, and the resumed run gets all 300.
=#

short = cut(rd, 1:260)
res_1 = cross_val_predict(hrp, short, online)
res_2 = cross_val_predict(Resume(res_1), rd, online)
stacked = vcat(res_1, res_2)

(length(res_1.pred), length(res_2.pred), maximum(gap(stacked, o_hrp)),
 stacked.mrd.X == o_hrp.mrd.X)

#=
`Resume` copies every state when it starts, so it never writes to `res_1`, and one result resumes
any number of times. To deploy a resumed state, you update it by hand from the last training end
with the value form. `partial_fit(res.opt, rows)` updates a copy of every state and leaves `res`
resumable. The form with a bang, `partial_fit!`, writes the held timestamps in place, and
`Resume(res)` then refuses the result, so use it on an estimator you keep and will not resume.
The result keeps the state at the end of the last training window. The library does not update
it through the purge and test rows, because such a state matches no fold of any run, and an
update cannot be taken back.

## 8. Speed and accuracy of the moment update

Everything above uses [`partial_fit!`](@ref) on the moments. One row updates a covariance in
`O(N²)`. A refit over a window of `m` rows costs `O(m · N²)`, so its cost grows with the window.
The update is also more accurate, and this section measures both.

The update is a Welford recursion, and Welford is not the textbook one-pass formula
`(Σ xxᵀ − n μμᵀ) / (n − 1)`. On a level series such as the one below, with a mean near a thousand
and a spread near one, the textbook formula subtracts two large numbers and loses most of its
digits. The recursion updates around the running mean and loses none. You meet that case as soon
as you update with a level, such as a price, rather than a change. The batch estimator centres
the rows before it multiplies them, so it avoids that subtraction. We take it as the reference,
and the cell prints the largest entrywise difference between it and each of the two formulas.
=#

rng = StableRNG(987654321)
Z = randn(rng, 50, 4) .+ 1000.0
fold(est, rows) = foldl(partial_fit!, eachrow(rows); init = est)

function textbook(Z)
    return (transpose(Z) * Z - size(Z, 1) * mean(Z; dims = 1)' * mean(Z; dims = 1)) /
           (size(Z, 1) - 1)
end
exact = cov(Covariance(), Z)

(maximum(abs, cov(fold(Covariance(), Z)) - exact), maximum(abs, textbook(Z) - exact))

#=
The update sits at the rounding floor of the exact answer, and the textbook formula sits several
orders of magnitude above it on fifty rows. The gap widens with the level and with the sample.

The update is faster as well. The cost of one update does not grow with the window, and the cost
of a refit does. We print the cost of one refit over the cost of one updated row, timed on a
sample of the panel's size with no gaps.
=#

Xs = randn(rng, T, N) ./ 100
step_cost = let est = fold(Covariance(), Xs[1:(end - 1), :]), row = Xs[end, :]
    minimum(@elapsed(partial_fit!(est, row)) for _ in 1:200)
end
refit_cost = minimum(@elapsed(cov(Covariance(), Xs)) for _ in 1:200)
round(refit_cost / step_cost; digits = 1)

#=
## 9. The gain through the loop, measured

That ratio is for the moments alone, not for the loop. In the fold loop a fold costs the moment
update plus the cost of computing the weights, and that second cost decides whether the update
saves time. Two things decide it, and the table below measures each one on this panel. Each
number is the minimum of three runs after a warm one, so read it as a direction and not as a
benchmark.

The prior comes first. Without a `CoveragePolicy`, the batch covariance is one BLAS product over
the window. The product is cheap, and the bookkeeping of the loop costs what the product saves,
so the rows labelled `plain` show no gain. With a `CoveragePolicy`, the batch fit itself runs the
update row by row, because each entry uses its own rows. There the online loop does that
arithmetic once instead of once per fold, and that is where the gain comes from.

The optimiser comes second. `HierarchicalRiskParity` computes its weights with a clustering over
the moments, so its fold costs the moment fit, and the gain shows. `MeanRisk` computes its
weights with a solve, which the update does not change, so its ratio sits near one on this panel.
=#

function ratio(opt)
    cross_val_predict(opt, rd, batch)
    cross_val_predict(opt, rd, online)
    tb = minimum(@elapsed(cross_val_predict(opt, rd, batch)) for _ in 1:3)
    to = minimum(@elapsed(cross_val_predict(opt, rd, online)) for _ in 1:3)
    return to / tb
end
plain = EmpiricalPrior()
gain_df = DataFrame(; prior = ["plain", "plain", "policy", "policy"],
                    head = ["hierarchical", "JuMP", "hierarchical", "JuMP"],
                    online_over_batch = [ratio(HierarchicalRiskParity(;
                                                                      opt = HierarchicalOptimiser(;
                                                                                                  pe = plain))),
                                         ratio(MeanRisk(;
                                                        opt = JuMPOptimiser(; pe = plain,
                                                                            slv = slv))),
                                         ratio(hrp), ratio(mr)])
pretty_table(gain_df; formatters = [numfmt])

#=
With a prior that has no `CoveragePolicy`, reach for the update for the accuracy of section 8 and
for the resume of section 7, not for the speed.

## 10. What to take away

  - `OnlineIndexWalkForward(w, t)` is the whole declaration. It sets the expanding window, and a
    rolling window is `Online(pe; max_history = w)` on the prior.
  - The online run gives the weights of the batch expanding run, fold by fold. Through
    `HierarchicalRiskParity` the numbers are the same, and through `MeanRisk` they sit within the
    tolerance of the solver. Above the prior, both runs compute the weights with the same batch
    code, over a returns result rebuilt from the prior's rows.
  - A prior with no update formula, such as `FactorPrior`, goes in `Online` and is refitted from
    the rows it keeps, so you write the same estimator for both runs.
  - A search scores through the same loop whatever the fit. A result resumes a run over a longer
    history. A `Pipeline` passes the update to its steps.
  - The update is a Welford recursion, and it keeps its digits on a level series where the
    textbook formula does not. It saves time through the loop where the batch fit itself runs
    row by row, as with a `CoveragePolicy` through `HierarchicalRiskParity`. It saves none with a
    prior that has no `CoveragePolicy`, whose batch fit is one product.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New page for #1034, the fog patch of map #861, written from the closing test
#src   `test/test_24h_online_walk_forward_closing.jl` (#874). Fixture is a StableRNG factor
#src   panel with the same listing calendar as #677's draw, so the same windows carry a
#src   listing (row 175, fold 3) and a delisting (row 252, fold 5).
#src - §9's ratios are wall-clock on the build machine and vary; the prose describes the
#src   direction. #874's table: plain family 1.0–1.6 at every size; policy family through the
#src   hierarchical head 0.55 at 8 × 300, 0.31 at 30 × 1500, 0.20 at 60 × 3000; through the
#src   JuMP head 0.97 at 8 × 300 and 0.46 at 60 × 3000.
#src - §8 is the moment seam's own example that #316 handed to map #861: the speed as the
#src   advertised benefit and the accuracy as the real one.
