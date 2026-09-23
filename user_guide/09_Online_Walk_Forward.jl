#=
```@meta
Description = "Fit one estimator on the first training window and update it with each later fold, instead of refitting every fold of a walk-forward."
```

# The online walk-forward

A walk-forward refits each fold from its training window, so on an expanding window each fold
uses again every row that the fold before it used. The online walk-forward adds each row to the
moments once. It fits one estimator on the first training window, adds the new rows of each later
fold to it, and computes the weights where a refit would have run. On this page the online run
gives exactly the weights of the batch expanding-window walk-forward, fold by fold.

You ask for it with one constructor. Write `OnlineIndexWalkForward(w, t)` where you would have
written `IndexWalkForward(w, t)`.

Adding rows to a fitted estimator is an update, and [`partial_fit!`](@ref) makes it. A prior
updates its moments and keeps its rows. An optimiser passes the new rows to its prior. When it
needs weights, it rebuilds the training data and runs the ordinary batch fit, so every
constraint, clustering and uncertainty set is the one the batch run builds. A search scores every
candidate through the same loop. A [`Pipeline`](@ref) passes the update to its steps, and the
result of an online run lets you resume the run. A prior with no update formula goes in
[`Online`](@ref)`(pe; max_history)`, which refits it from the rows its buffer keeps, and the cap
on the buffer gives a rolling window. This page shows the call, the weights the two runs share,
the wrapper, where the update is faster and where it is not, and the resume.
=#

using PortfolioOptimisers, StableRNGs, Statistics, Dates, Clarabel

#=
## 1. A universe that moves, and the one call

The data is a seeded factor panel of eight assets over three hundred observations. Its listing
calendar is a fact about the instruments. Five assets list after the first row, and three delist
before the last. The returns are `NaN` where an asset is not listed, and the
[`AssetPanel`](@ref) takes the calendar as its active mask.
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
The prior has a [`CoveragePolicy`](@ref) on both moments. Each asset's mean uses the rows where
that asset has a return, each covariance entry uses the rows where both assets have one, and an
asset enters the universe when it lists. Under that policy the online update gives the batch
estimate exactly. The two schemes below cut the same windows. An update of the moments adds rows
and never removes them, so `OnlineIndexWalkForward` sets `expand_train = true`, and a rolling window goes
on the estimator, as section 2 shows.
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
the third window, then seven once the fourth delists inside the fifth. The
[online walk-forward example](../examples/5_validation_tuning/09_Online_Walk_Forward.md) runs the
same comparison through [`MeanRisk`](@ref) and prints the largest weight difference per fold.

The loop calls two functions, and you can call them yourself. [`partial_fit!`](@ref) adds the rows
of a `ReturnsResult` to the optimiser and solves nothing. `optimise(opt)` with no data computes
the weights from the rows added so far, and solves once. We update the estimator with the first
training window and then with the new rows of the second fold, and we compare `optimise(est)`
with the weights of the second fold above.
=#

function cut(i)
    return ReturnsResult(; nx = rd.nx, X = rd.X[i, :], ts = rd.ts[i],
                         pnl = AssetPanel(; amsk = rd.pnl.amsk[i, :],
                                          emsk = rd.pnl.emsk[i, :]))
end
(; train_idx) = split(online, rd)

est = partial_fit!(hrp, cut(train_idx[1]))
est = partial_fit!(est, cut((last(train_idx[1]) + 1):last(train_idx[2])))

optimise(est).w == weights(o)[2]

#=
The loop starts from an estimator with no state. An estimator that already holds a state throws
an error at the first fit, and so does a [`TimeDependent`](@ref) schedule on the prior or on the
optimiser itself, because such a schedule would replace the estimator that holds the state. A
schedule on any other field works as it does in batch. A schedule that the update itself uses,
such as the allocation set of an [`OnlinePortfolioSelection`](@ref), is swapped in before the
rows of the fold are added.

The update takes the active mask of the panel with the rows, and no other part of the panel. A
panel whose estimation mask differs from its active mask throws at the first fit, and so does one
that has time-varying panel fields. The batch walk-forward accepts either kind of panel, because
each fold refits over its whole window.

## 2. A prior with no update formula, and the rolling window

A [`FactorPrior`](@ref) is a regression, and the library has no update formula for one. Wrap it
in [`Online`](@ref). The wrapper puts a buffer of rows in the `cache` of the prior, each fold
appends its new rows to the buffer, and the prior is fitted over the buffer in the ordinary way.
The buffer holds the rows the batch fold uses, so the two runs return the same weights. The
wrapper also takes a prior that has an update formula, and the cell below wraps the
`EmpiricalPrior` of section 1.

`max_history` caps that buffer, and the cap is how you ask for a rolling window online. The loop
only expands the training window. You set a rolling window on the estimator, not on the scheme. A
batch scheme with window `w + p` and purge `p` trains over `w` rows, and a buffer capped at `w`
fits over those same `w` rows.
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
The cell prints `true`. The buffer capped at 100 rows gives every weight of the rolling batch
walk-forward. You write the same estimator online that you write in batch. An estimator updates
each part that has an update formula, and refits each other part from the rows it keeps.

## 3. Where the online update is faster, and where it is not

The update is faster where the batch fit itself runs row by row. With the default covariance of
[`EmpiricalPrior`](@ref), the batch fit is one BLAS product, and the online run is no faster.
With a `CoveragePolicy` on both moments, the batch fit runs the same update row by row, so there
the loop does that arithmetic once instead of once per fold. When the optimiser is
`HierarchicalRiskParity`, which ends in a clustering, the online run is then faster. When the
optimiser solves a JuMP problem, such as `MeanRisk`, the run ends in a solve, which the update
does not change.

We print the ratio of online time to batch time through `HierarchicalRiskParity`, without and
with the policy. The numbers are wall-clock on the build machine, the minimum of three runs after
a warm one. Read them as a direction and not as a benchmark.
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
A ratio below one means the online run took less time than the batch run. The update has a second
gain, accuracy. The update is a Welford recursion, and not the textbook one-pass formula
`(Σ xxᵀ − n μμᵀ) / (n − 1)`. The difference shows as soon as you update with a level
instead of a change, such as a series with a mean near one thousand and a spread near one. The
textbook formula subtracts two large numbers and loses digits. The recursion updates around the
running mean, and the batch estimator centres the rows before it multiplies them.
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
The first number is the largest difference between the recursion and the batch covariance, and
the second is the largest difference between the textbook formula and the batch covariance.

With the default covariance of `EmpiricalPrior`, reach for the update for that accuracy and for
the resume below, and not for the speed. With a `CoveragePolicy` and an optimiser that computes
its weights cheaply, such as `HierarchicalRiskParity`, reach for it for the speed as well.

## 4. A run resumes from its result

The [`MultiPeriodPredictionResult`](@ref) of an online run holds the estimator as it stood at the
end of the last training window. When more rows arrive, hand that result to [`Resume`](@ref) with
the extended history and the same scheme. The loop skips the folds the result already holds,
adds the new rows to a copy of the state, and continues from the next fold. The resumed result
holds the new folds alone. `vcat` stacks the two results, and we compare the stack with the run
over the whole history at once. The first run sees the first 260 rows, and the resume sees all 300
rows. The data needs timestamps, because the loop finds the fold to resume from by the last
timestamp the state holds.
=#

res_1 = cross_val_predict(hrp, cut(1:260), online)
res_2 = cross_val_predict(Resume(res_1), rd, online)

(length(res_1.pred), length(res_2.pred), weights(vcat(res_1, res_2)) == weights(o))

#=
`Resume` works on a copy of every state. `res_1` does not change, and you can resume it any number
of times.

## Where to go next

  - [The online walk-forward example](../examples/5_validation_tuning/09_Online_Walk_Forward.md)
    runs every comparison above through `MeanRisk` as well, with the search, the errors, and the
    speed with and without a `CoveragePolicy` through each optimiser.
  - [Online portfolio selection](10_Online_Portfolio_Selection.md) covers the optimisers whose
    weights come from a recursion of their own rather than a batch fit, run through the online
    walk-forward of this page.
  - [Validation and tuning](05_Validation_and_Tuning.md) covers the walk-forward and the search
    this page runs.
  - [The point-in-time universe](08_Point_in_Time_Universe.md) covers the panel whose active mask
    the update takes.
  - [Data and priors](01_Data_and_Priors.md) covers the prior that updates here.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New guide page for #1034, the fog patch of map #861; the deep dive is
#src   examples/5_validation_tuning/09_Online_Walk_Forward.jl. Same StableRNG fixture as the
#src   example, and the same listing calendar as #677's draw in test_24h.
#src - §3's ratios are wall-clock and vary; the prose states the direction, from #874's table
#src   (plain 1.0–1.6 at every size; policy through HRP 0.55 / 0.31 / 0.20 at 8 × 300,
#src   30 × 1500, 60 × 3000; through JuMP 0.97 and 0.46).
