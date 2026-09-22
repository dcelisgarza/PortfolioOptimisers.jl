#=
```@meta
Description = "Warm one estimator on the first training window and step it fold by fold, instead of refitting every fold of a walk-forward."
```

# The online walk-forward

A walk-forward refits every fold from its training window, so on an expanding window each fold
re-reads every row the fold before it read. The online walk-forward reads each row once. It warms
one estimator up on the first training window, folds the new rows of each later fold into it, and
reads it out where a refit would have run. The folds are the same folds. Only the fit of each one
changes, and the online run reaches the weights of the batch expanding-window walk-forward, fold
for fold.

You ask for it with one constructor. Write `OnlineIndexWalkForward(w, t)` where you would have
written `IndexWalkForward(w, t)`.

Every layer above the moments takes the step. A prior folds its moments and keeps its rows. An
optimiser hands the rows down to its prior, and when it needs weights it rebuilds the training
data and runs the ordinary batch path, so every constraint, clustering and uncertainty set is the
same one the batch path builds. A search scores every candidate through the same loop. A
[`Pipeline`](@ref) passes the step down to its steps, and a Result lets you resume a run. A member
with no exact recursion goes in [`Online`](@ref)`(est; max_history)`, which refits it from the rows
its buffer keeps, and that cap is the rolling window. This page shows the call, the weights the
two runs share, the wrapper, where the step is faster and where it is not, and the resume.
=#

using PortfolioOptimisers, StableRNGs, Statistics, Dates, Clarabel

#=
## 1. A universe that moves, and the one call

The fixture is a seeded factor panel of eight assets over three hundred observations. Its listing
calendar is a fact about the instruments: three assets list after the first row, and three delist
before the last. The returns are `NaN` where an asset is not listed, and the
[`AssetPanel`](@ref) carries the calendar as its active mask.
=#

function synthetic_panel(; T = 300, N = 8, K = 2, seed = 20260912)
    rng = StableRNG(seed)
    F = 0.01 .* randn(rng, T, K)
    X = F * transpose(randn(rng, N, K)) .+
        randn(rng, T, N) .* transpose(0.008 .+ 0.012 .* rand(rng, N))
    listed = [1, 34, 42, 1, 67, 10, 175, 1]
    delisted = [T, T, T, 252, T, 291, T, 264]
    amsk = [listed[i] <= t <= delisted[i] for t in 1:T, i in 1:N]
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2015, 1, 1):Day(1):(Date(2015, 1, 1) + Day(2 * T + 10)))[1:T]
    return ReturnsResult(; nx = ["A" * lpad(i, 2, '0') for i in 1:N],
                         X = ifelse.(amsk, X, NaN), ts = days,
                         pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
end

rd = synthetic_panel()
N = size(rd.X, 2)

#=
The prior carries a [`CoveragePolicy`](@ref) on both moments, so each cell is fitted on the
observations it has, and an asset enters the universe when it lists. Its moments fold exactly
under that policy. The two schemes below cut the same windows. An online run can only expand,
because a fold cannot take a row back out, so `OnlineIndexWalkForward` sets `expand_train = true`.
=#

cvg = CoveragePolicy()
pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                    ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = cvg)))
hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe))

batch = IndexWalkForward(100, 40; purged_size = 3, expand_train = true)
online = OnlineIndexWalkForward(100, 40; purged_size = 3)

b = cross_val_predict(hrp, rd, batch)
o = cross_val_predict(hrp, rd, online)

weights(res) = [p.res.w for p in res.pred]
masks(res) = [p.res.imsk for p in res.pred]
count_in(m) = isnothing(m) ? N : count(m)

(weights(o) == weights(b), masks(o) == masks(b), count_in.(masks(o)))

#=
Every weight of the online run equals the weight of the batch run, and both runs follow the
universe as it moves. Seven assets are investable, then eight once the seventh asset lists inside
the third window, then seven once the fourth delists inside the fifth. Through a JuMP head the
two runs match to the tolerance of the solver, because the online run solves the batch problem
over the batch data.

The loop calls two functions, and you can call them yourself. [`partial_fit!`](@ref) folds the rows
of a `ReturnsResult` into the optimiser and solves nothing. `optimise(opt)` with no data reads the
folded state out and solves once. Two calls below reach the weights of the second fold.
=#

function cut(i)
    return ReturnsResult(; nx = rd.nx, X = rd.X[i, :], ts = rd.ts[i],
                         pnl = AssetPanel(; amsk = rd.pnl.amsk[i, :],
                                          emsk = rd.pnl.emsk[i, :]))
end
(; train_idx) = split(online, rd)

est = partial_fit!(hrp, cut(train_idx[1]))                     # warm-up: fold 1's window
est = partial_fit!(est, cut((last(train_idx[1]) + 1):last(train_idx[2])))   # fold 2's new rows

optimise(est).w == weights(o)[2]

#=
The loop starts cold. An estimator that already holds a state throws at warm-up, and so does a
[`TimeDependent`](@ref) schedule on the prior or on the optimiser itself, because such a schedule
replaces the value the loop threads the state through. A schedule on any other field works as it
does in batch.

The step folds the active mask of the panel with the rows, and nothing else of the panel. A panel
whose estimation mask is narrower than its active mask throws at warm-up, and so does one that
carries time-varying panel fields. The batch loop takes both panels.

## 2. A member with no exact recursion, and the rolling window

A [`FactorPrior`](@ref) is a regression, and the library ships no incremental fold for one. Wrap
it in [`Online`](@ref). The wrapper puts a buffer of rows in the prior's `cache`, each fold
appends its new rows to the buffer, and the prior is fitted over the buffer in the ordinary way.
The buffer holds the rows the batch fold reads, so the two runs return the same weights.

`max_history` caps that buffer, and the cap is how you ask for a rolling window online. The loop
itself only expands, so the window belongs to the estimator. A batch scheme with window `w + p`
and purge `p` trains over `w` rows, and a buffer capped at `w` fits over those same `w` rows.
=#

rolling = IndexWalkForward(103, 40; purged_size = 3)
stepped = OnlineIndexWalkForward(103, 40; purged_size = 3)
hrp_cap = HierarchicalRiskParity(;
                                 opt = HierarchicalOptimiser(;
                                                             pe = Online(pe;
                                                                         max_history = 100)))

weights(cross_val_predict(hrp_cap, rd, stepped)) ==
weights(cross_val_predict(hrp, rd, rolling))

#=
You write the same estimator online that you write in batch. An estimator folds every member that
can fold, and refits every member that cannot from the rows that member kept.

## 3. Where the online step is faster, and where it is not

Most callers take the online step for speed. It is faster where the batch fit is itself a
recursion over the rows. The batch covariance of the plain family is one BLAS product, and the
online step is no faster at any width the library has measured. The ratio sits at or above one.
The batch arm of the `CoveragePolicy` family folds row by row, so there the loop does that
arithmetic once instead of once per fold. Under a hierarchical head, which ends in a clustering,
the online run is faster, and the margin grows with the panel. The library's own measurements put
it near half of batch at this size, and near a fifth at sixty assets over three thousand
observations. Under a JuMP head the run ends in a solve, which the step does not touch, so the
ratio sits near one until the moment fit grows into the solve. The numbers below are wall-clock
on the build machine, the minimum of three runs after a warm one. Read them as a direction and
not as a benchmark.
=#

function ratio(opt)
    cross_val_predict(opt, rd, batch)
    cross_val_predict(opt, rd, online)
    tb = minimum(@elapsed(cross_val_predict(opt, rd, batch)) for _ in 1:3)
    to = minimum(@elapsed(cross_val_predict(opt, rd, online)) for _ in 1:3)
    return round(to / tb; digits = 2)
end
hrp_plain = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = EmpiricalPrior()))

(; plain = ratio(hrp_plain), policy = ratio(hrp))

#=
The second gain is accuracy. The fold is a Welford recursion, and not the textbook one-pass
formula `(Σ xxᵀ − n μμᵀ) / (n − 1)`. The difference shows as soon as you fold a level instead of
a change, such as prices with a mean near one thousand and unit spread. The textbook formula
subtracts two large numbers and loses digits. The recursion updates around the running mean, and
it returns what the batch estimator returns, which centres the rows first.
=#

Z = randn(StableRNG(987654321), 50, 4) .+ 1000.0
function textbook(Z)
    return (transpose(Z) * Z - size(Z, 1) * mean(Z; dims = 1)' * mean(Z; dims = 1)) /
           (size(Z, 1) - 1)
end
exact = cov(Covariance(), Z)
folded = foldl(partial_fit!, eachrow(Z); init = Covariance())

(maximum(abs, cov(folded) - exact), maximum(abs, textbook(Z) - exact))

#=
Reach for the step on a plain prior for that accuracy and for the resume below, and not for the
clock. Reach for it on a policy prior under a cheap head for the clock as well.

## 4. A run resumes from its Result

The [`MultiPeriodPredictionResult`](@ref) of an online run holds the estimator the loop threaded,
folded through the end of the last training window. When more rows arrive, hand that Result to
[`Resume`](@ref) with the extended history and the same scheme. The loop skips the folds the
Result already holds, folds the new rows into a copy of the state, and continues from the next
fold. The resumed Result holds the new folds alone. `vcat` stacks the two Results, and the stack
equals the run over the whole history at once. The data needs timestamps, because the loop finds
the fold to resume from by the last timestamp the state holds.
=#

res_1 = cross_val_predict(hrp, cut(1:260), online)              # the history at the first run
res_2 = cross_val_predict(Resume(res_1), rd, online)            # the full history

(length(res_1.pred), length(res_2.pred), weights(vcat(res_1, res_2)) == weights(o))

#=
`Resume` copies every state when it starts, so it never writes to `res_1`, and one Result resumes
any number of times.

## Where to go next

  - [The online walk-forward example](../examples/5_validation_tuning/09_Online_Walk_Forward.md)
    runs every comparison above through a JuMP head as well, with the search, the errors, and the
    speed by prior family and by head.
  - [Online portfolio selection](10_Online_Portfolio_Selection.md) covers the one family whose
    last step is its own recursion rather than a batch fit, run through the step on this page.
  - [Validation and tuning](05_Validation_and_Tuning.md) covers the walk-forward and the search
    this page runs.
  - [The point-in-time universe](08_Point_in_Time_Universe.md) covers the panel whose active mask
    the step folds.
  - [Data and priors](01_Data_and_Priors.md) covers the prior that folds here.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New guide page for #1034, the fog patch of map #861; the deep dive is
#src   examples/5_validation_tuning/09_Online_Walk_Forward.jl. Same StableRNG fixture as the
#src   example, and the same listing calendar as #677's draw in test_24h.
#src - §3's ratios are wall-clock and vary; the prose states the direction, from #874's table
#src   (plain 1.0–1.6 at every size; policy through HRP 0.55 / 0.31 / 0.20 at 8 × 300,
#src   30 × 1500, 60 × 3000; through JuMP 0.97 and 0.46).
