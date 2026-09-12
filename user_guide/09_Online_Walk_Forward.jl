#=
# The online walk-forward

A walk-forward refits every fold from its training window, so on an expanding window each
fold re-reads every row the fold before it read. The online step does the reading once. A
walk-forward that declares the **Fold Fit** [`OnlineStep`](@ref) warms one estimator up on
the first training window, folds each later fold's *new* rows into it, and reads it out where
a refit would have run. The folds are the same folds; only the fit of each one changes, and
the run reaches the weights of the batch expanding-window walk-forward fold for fold.

The rule is one keyword:

> `IndexWalkForward(w, t; ff = OnlineStep())`

Every layer above the moments takes the step. A prior folds its moments and carries its rows;
an optimiser forwards the rows to its prior and, at read-out, rebuilds the carrier and runs the
ordinary batch path, so every constraint, clustering and uncertainty set is identical to batch
by construction; a search scores every candidate through the one loop; a [`Pipeline`](@ref)
is a host of the step; and a Result resumes a run. A member with no exact recursion is
wrapped — `Online(est; max_history)` — and refits from the rows its buffer keeps, and that cap
is the rolling window. This page shows the call, the identity, the wrapper, the gain measured
honestly, and the resume.
=#

using PortfolioOptimisers, StableRNGs, Statistics, Dates, Clarabel

#=
## 1. A universe that moves, and the one call

The fixture is a seeded factor panel of eight assets over three hundred observations, with a
listing calendar written out as a fact about the instruments: three assets list after the
first row and three delist before the last. The returns are `NaN` where an asset is not
listed, and the [`AssetPanel`](@ref) carries the calendar as its active mask.
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
observations it has and an asset enters the universe when it lists. Its moments fold exactly
under the policy. The two schemes below cut identical windows: an online run is expanding by
construction — a fold cannot un-fold a row — so `OnlineStep()` derives `expand_train = true`.
=#

cvg = CoveragePolicy()
pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                    ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = cvg)))
hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe))

batch = IndexWalkForward(100, 40; purged_size = 3, expand_train = true)
online = IndexWalkForward(100, 40; purged_size = 3, ff = OnlineStep())

b = cross_val_predict(hrp, rd, batch)
o = cross_val_predict(hrp, rd, online)

weights(res) = [p.res.w for p in res.pred]
masks(res) = [p.res.imsk for p in res.pred]
count_in(m) = isnothing(m) ? N : count(m)

(weights(o) == weights(b), masks(o) == masks(b), count_in.(masks(o)))

#=
The weights agree to the bit, and both routes watch the universe move: seven investable
assets, then eight once the seventh lists inside the third window, then seven once the
fourth delists inside the fifth. Through a JuMP head the same identity holds to the solver's
tolerance, because the read-out solves the batch problem from the batch carrier.

The loop is two verbs, and writing them out reaches its own weights: [`partial_fit!`](@ref)
folds the rows of a carrier into the optimiser and solves nothing, and `optimise(opt)` with
no data reads the state out and solves once.
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
The loop starts cold. An estimator that already carries a state is refused at the door, and
so is a [`TimeDependent`](@ref) schedule on the prior or on the optimiser itself, because a
schedule replaces the value a state is threaded through; a schedule on any other field
composes with no rule.

The step folds the panel's **active mask** with the rows and nothing else. A panel whose
estimation mask is narrower than its active mask, or one that carries time-varying Panel
Fields, is refused at warm-up by name — the batch loop takes both.

## 2. A member with no exact recursion, and the rolling window

A [`FactorPrior`](@ref) is a regression, and a regression has no incremental fold the library
ships. Wrap it in [`Online`](@ref): the wrapper seeds a Sample Buffer in the prior's `cache`,
the fold appends the rows, and the read-out runs the batch verb over them. The buffer holds
exactly the rows the batch fold reads, so the two match to the bit. The same wrapper's
`max_history` is how a **rolling** window is declared online: the loop is expanding, so the
window is the estimator's. A rolling batch scheme of window `w + p` with purge `p` trains over
`w` rows, and a buffer capped at `w` reads out over exactly those rows.
=#

rolling = IndexWalkForward(103, 40; purged_size = 3)
stepped = IndexWalkForward(103, 40; purged_size = 3, ff = OnlineStep())
hrp_cap = HierarchicalRiskParity(;
                                 opt = HierarchicalOptimiser(;
                                                             pe = Online(pe;
                                                                         max_history = 100)))

weights(cross_val_predict(hrp_cap, rd, stepped)) ==
weights(cross_val_predict(hrp, rd, rolling))

#=
An estimator you would write in batch is the estimator you write online: a host folds every
member that folds and refits every member that does not from its own rows.

## 3. The gain, measured honestly

The seam's **advertised** benefit is speed, and through the loop that benefit is where the
batch fit is itself a recursion over the rows. The plain family's batch covariance is one BLAS
product, and its online step is no faster at any width the library has measured — the ratio
sits at or above one. The `CoveragePolicy` family's batch arm folds row by row, so there the
loop does the arithmetic once instead of once per fold; through a hierarchical head, whose
read-out is a clustering, the gain shows, and it grows with the panel (the library's own
measurements put it near half of batch at this size and a fifth at sixty assets over three
thousand observations). Through a JuMP head the read-out is a solve, which the step does not
touch, so the ratio sits near one until the moment fit grows into the solve. The numbers below
are wall-clock on the build machine, the minimum of three runs after a warm one, and are a
direction rather than a benchmark.
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
The seam's **real** benefit is accuracy. The fold is a Welford recursion, not the textbook
one-pass formula `(Σ xxᵀ − n μμᵀ) / (n − 1)`, and the difference shows the moment a caller
folds a level rather than a change — prices, mean a thousand and unit spread. The textbook
formula subtracts two large numbers and loses its digits; the recursion updates around the
running mean and reaches the batch estimator, which centres first and is exact.
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
Reach for the step on a plain prior for that property and for the resume below, not for the
clock; reach for it on a policy prior through a cheap head for the clock as well.

## 4. A run resumes from its Result

An online run's [`MultiPeriodPredictionResult`](@ref) carries the estimator the loop
threaded, folded through the last training end. When more rows arrive, [`Resume`](@ref) hands
that Result back over the **full history extended** and the same scheme: the loop skips the
folds the Result holds, folds the ordinary delta into a copy of the state, and continues from
the next fold. The resumed Result holds the new folds only, `vcat` stacks the two, and the
stack equals the one-shot run. Timestamps are required, because the fold to resume from is
located by the state's last held timestamp.
=#

res_1 = cross_val_predict(hrp, cut(1:260), online)              # the history at the first run
res_2 = cross_val_predict(Resume(res_1), rd, online)            # the full history

(length(res_1.pred), length(res_2.pred), weights(vcat(res_1, res_2)) == weights(o))

#=
`res_1` is never written — `Resume` copies every state at entry — so one Result resumes any
number of times.

## Where to go next

  - [The online walk-forward example](../examples/5_validation_tuning/09_Online_Walk_Forward.md)
    — every identity above through a JuMP head as well, the search, the refusals, and the
    gain measured by prior family and by head.
  - [Validation and tuning](05_Validation_and_Tuning.md) — the walk-forward and the search
    this page runs.
  - [The point-in-time universe](08_Point_in_Time_Universe.md) — the panel whose active mask
    the step folds.
  - [Data and priors](01_Data_and_Priors.md) — the prior that does the folding here.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New guide page for #1034, the fog patch of map #861; the deep dive is
#src   examples/5_validation_tuning/09_Online_Walk_Forward.jl. Same StableRNG fixture as the
#src   example, and the same listing calendar as #677's draw in test_24h.
#src - §3's ratios are wall-clock and vary; the prose states the direction, from #874's table
#src   (plain 1.0–1.6 at every size; policy through HRP 0.55 / 0.31 / 0.20 at 8 × 300,
#src   30 × 1500, 60 × 3000; through JuMP 0.97 and 0.46).
