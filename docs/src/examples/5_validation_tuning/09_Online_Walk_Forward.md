The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).
```@meta
EditURL = "../../../../examples/5_validation_tuning/09_Online_Walk_Forward.jl"
```

# The online walk-forward: one estimator, stepped fold by fold

A walk-forward refits every fold from its training window. Fold `i` reads every row fold
`i - 1` read and a few more, fits the moments over all of them, and solves. On an expanding
window the cost of the whole run therefore grows with the square of the number of folds, and
the moments of fold `i` are recomputed from scratch when almost all of their input is the
input of fold `i - 1`.

The library's answer is a **Fold Fit**. A walk-forward that declares
[`OnlineStep`](@ref) warms one estimator up on the first training window, folds each later
fold's *new* rows into it, and reads it out where a refit would have run. The estimator is
threaded from fold to fold, and the run reaches the weights of the batch expanding-window
walk-forward fold for fold. Nothing in the enumeration of folds changes; only the fit of
each one does.

The mechanism is the incremental moment seam, [`partial_fit!`](@ref), and every layer above
the moments takes the step through it:

  - A **prior folds and carries**: its moments come off its members' own folds, and it keeps
    the rows only because a Prior Result carries `X` for the scenario risk measures
    (ADR 0136).
  - An **optimiser forwards to its prior alone**, and a read-out reconstitutes the carrier
    and runs the ordinary batch path, so every constraint, every clustering and every
    uncertainty set is identical to batch by construction (ADR 0137). A set with no prior of
    its own is calibrated on the prior result it is handed (ADR 0138).
  - A **JuMP head builds a fresh model at every read-out**: the solve is the step's cost,
    and no model is kept warm (ADR 0139).
  - The **walk-forward declares the Fold Fit**, and the online arm threads the estimator
    from a cold start (ADR 0140).
  - A **search scores every candidate through the one fold loop**, online and batch alike
    (ADR 0141).
  - A **Pipeline is a host**: its row-local steps fold, its universe steps view, and a step
    with no online form is refused unless the pipeline declares a refit (ADR 0142).
  - A **Result resumes a run**: `Resume(res)` re-enters the loop from the folds it holds
    (ADR 0144).

A member with no exact recursion is wrapped: `Online(est; max_history)` seeds a Sample Buffer
that the read-out refits from, and its cap is the rolling window.

!!! tip "When to reach for this"
    Reach for `ff = OnlineStep()` on any expanding walk-forward, because it costs nothing in
    accuracy: the run equals the batch one fold for fold. Reach for it for the *clock* when
    the prior carries a `CoveragePolicy` and the head's read-out is cheap — a hierarchical or
    a naive optimiser — because that is where the batch fit is itself a recursion over the
    rows. Reach for it whenever a run will be **continued**: a Result of an online run resumes
    over the history extended, and a batch Result cannot. On a plain prior through a JuMP head
    the step buys no time, and the page says so with numbers.

This example runs one walk-forward through every one of those layers and measures each
identity, then measures the gain honestly, because the gain is not where a first reading puts
it.

 1. The panel: a universe that moves inside the window, and the two walk-forwards.
 2. The online run equals the batch expanding run, fold for fold.
 3. What the loop does, written by hand.
 4. A member with no exact recursion refits from its buffer, to the bit.
 5. A capped buffer is the rolling window.
 6. A search picks the batch candidate.
 7. A run resumes from its Result.
 8. The moment seam on its own: the advertised benefit and the real one.
 9. The gain through the loop, measured.
 10. What to take away.

````@example 09_Online_Walk_Forward
using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, DataFrames,
      PrettyTables, Clarabel

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;
nothing #hide
````

## 1. The panel, and the two walk-forwards

The fixture is a small factor panel drawn from a seeded generator: eight assets, three
hundred observations, two factors. Three assets list after the first row and three delist
before the last, so the universe moves inside the window, which is the case the whole seam
exists for. The listing calendar is written out as a fact about the instruments, and the
returns are `NaN` wherever an asset is not listed. The [`AssetPanel`](@ref) carries the
calendar as its active mask, and the estimation mask is set to the active one, because the
online step folds the active mask alone (§2 says why).

````@example 09_Online_Walk_Forward
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
# The rows `i` of the carrier, panel included: the walk-forward cuts its windows this way.
function cut(rd, i)
    return ReturnsResult(; nx = rd.nx, X = rd.X[i, :], ts = rd.ts[i],
                         pnl = AssetPanel(; amsk = amsk[i, :], emsk = amsk[i, :]))
end

[(rd.nx[j], findfirst(amsk[:, j]), findlast(amsk[:, j])) for j in 1:N]
````

The two walk-forwards enumerate the same folds. The batch one expands its training window,
which is what an online run does by construction — a fold cannot un-fold an observation — so
`OnlineStep()` derives `expand_train = true` and the two schemes cut identical windows. Both
purge three rows before each test window.

````@example 09_Online_Walk_Forward
w, t, p = 100, 40, 3
batch = IndexWalkForward(w, t; purged_size = p, expand_train = true)
online = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep())

(; train_idx, test_idx) = split(batch, rd)
(train_idx == split(online, rd).train_idx, train_idx, [first(i):last(i) for i in test_idx])
````

The seventh asset lists at row 175, inside the third training window, and the fourth delists
at row 252, inside the fifth. A plain prior over an expanding window from row 1 never admits
a late listing: its Coverage Universe asks for a finite return at every row of the window,
and the window always starts at row 1. The prior below carries a [`CoveragePolicy`](@ref) on
both moments, so each cell is fitted on the observations it has, and an asset enters the
universe when it lists. Its moments fold exactly under the policy — the available-case
batch arm is itself a recursion over the rows — which is what §9 turns on.

````@example 09_Online_Walk_Forward
cvg = CoveragePolicy()
pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                    ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = cvg)))
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                             "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
             check_sol = (; allow_local = false, allow_almost = false))
mr = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv))
hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pe))
````

## 2. The online run equals the batch expanding run, fold for fold

The identity is the same call twice, once under each scheme. The loop's online arm says so
on its info line; everything else is the ordinary [`cross_val_predict`](@ref).

````@example 09_Online_Walk_Forward
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
````

The hierarchical head agrees to the bit on every fold, because the available-case batch arm
*is* the fold. The JuMP head agrees to the bit here too, and in general to the solver's
tolerance: it solves the same problem from the same carrier, and a solver started from the same
point ends at the same point. Both routes see the universe move: seven investable assets in the first two folds,
all eight once the seventh lists, seven again once the fourth delists. `imsk` is `nothing` on
a fold where every asset was investable. Nothing above the prior did anything online: the
hierarchical head clustered and the JuMP head solved exactly as they do in batch, from a
carrier the read-out rebuilt.

The step folds the **active mask** of the panel with the rows and nothing else. A panel
whose estimation mask differs from its active mask, or that carries time-varying Panel
Fields, is refused at warm-up by name — the buffers hold the rows and the mask, and a field
or a second mask has no row to ride on. The batch loop takes both panels, because a refit
reads the whole window.

````@example 09_Online_Walk_Forward
narrow = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts,
                       pnl = AssetPanel(; amsk = amsk, emsk = amsk .& (rd.X .> -0.03)))
try
    cross_val_predict(mr, narrow, online)
catch err
    showerror(stdout, err)
end
````

## 3. What the loop does, written by hand

The loop is two verbs. `partial_fit!(opt, rd)` folds the rows of a carrier into the
optimiser — it solves nothing — and `optimise(opt)` with no data reads the state out and
solves once. The warm-up is the first training window; each later fold is the rows the
window gained since the last one. Writing it out reaches the loop's own weights.

````@example 09_Online_Walk_Forward
est = partial_fit!(mr, cut(rd, train_idx[1]))                 # warm-up: fold 1's window
hand = [optimise(est).w]
for i in 2:length(train_idx)
    delta = (last(train_idx[i - 1]) + 1):last(train_idx[i])
    global est = partial_fit!(est, cut(rd, delta))            # only the new rows
    push!(hand, optimise(est).w)
end

maximum(maximum(abs, a - b) for (a, b) in zip(hand, weights(o_mr)))
````

The state lives in the estimator's `cache` fields, and the loop **starts cold**: an estimator
that already carries a state at entry is refused by name, so the estimator handed to the
walk-forward is always the configuration and never a half-fitted run. A
[`TimeDependent`](@ref) schedule on the prior, or on the optimiser itself, is refused at
warm-up for the same reason — a schedule replaces the value a state is threaded through. A
schedule on any other field composes with no rule, so the weight bounds below tighten as the
folds advance, online exactly as in batch.

````@example 09_Online_Walk_Forward
n = length(train_idx)
caps = TimeDependent([WeightBounds(; lb = 0.0, ub = ub) for ub in range(0.4, 0.25, n)])
mr_caps = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv, wb = caps))
gap(cross_val_predict(mr_caps, rd, online), cross_val_predict(mr_caps, rd, batch))
````

## 4. A member with no exact recursion refits from its buffer, to the bit

A [`FactorPrior`](@ref) regresses the assets on the factors, and a regression has no
incremental fold the library ships. It takes the step **wrapped**: [`Online`](@ref) seeds a
Sample Buffer in the prior's `cache`, the fold appends the rows — and the factor rows beside
them, because the prior's tree reads them — and the read-out runs the batch verb over the
buffer. The buffer holds exactly the rows the batch fold reads, so the two fit the same
sample and every weight agrees to the bit, the JuMP head's included.

````@example 09_Online_Walk_Forward
fp_batch = MeanRisk(; opt = JuMPOptimiser(; pe = FactorPrior(), slv = slv))
fp_online = MeanRisk(; opt = JuMPOptimiser(; pe = Online(FactorPrior()), slv = slv))

(weights(cross_val_predict(fp_online, rdf, online)) ==
 weights(cross_val_predict(fp_batch, rdf, batch)))
````

`Online` is stored in the field it wraps and it is transient: the loop resolves it at
warm-up into an ordinary estimator carrying a state, and no `Online` exists from that point
on. A host that holds one folding member and one buffered member folds the first and refits
the second from its own rows, so the estimator you write online is the estimator you would
write in batch.

## 5. A capped buffer is the rolling window

An online run is expanding by construction, so a rolling window is the estimator's to
declare, not the loop's: `Online(pe; max_history = w)` keeps the last `w` rows and drops the
oldest as each new one arrives. A rolling batch scheme of window `w + p` with purge `p`
trains over `w` rows, and the capped online run reads out over exactly those rows.

````@example 09_Online_Walk_Forward
rolling = IndexWalkForward(w + p, t; purged_size = p)
stepped = IndexWalkForward(w + p, t; purged_size = p, ff = OnlineStep())
cap(pe) = Online(pe; max_history = w)

hrp_cap = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = cap(pe)))
b_roll = cross_val_predict(hrp, rd, rolling)
o_roll = cross_val_predict(hrp_cap, rd, stepped)

(all(length.(split(rolling, rd).train_idx) .== w), weights(o_roll) == weights(b_roll),
 masks(o_roll) == masks(b_roll))
````

A cap on a folding member is a different thing. `max_scenarios` on a prior bounds the rows
its Result carries for the scenario measures and leaves the moments fitted over every
observation, so a capped fold-and-carry equals no batch fit. `max_history` on `Online`
windows the whole fit and has a batch equal, which is the identity above. Both may be set,
and they nest.

## 6. A search picks the batch candidate

A search scores every candidate through the one fold loop, and does not read the Fold Fit.
The grid tunes the weight bounds, which bind and so separate the candidates; the two
searches score identical matrices and pick the same column.

````@example 09_Online_Walk_Forward
r = ConditionalValueatRisk()
grid = ["opt.wb" =>
            [WeightBounds(; lb = 0.0, ub = 1.0), WeightBounds(; lb = 0.0, ub = 0.2),
             WeightBounds(; lb = 0.1, ub = 1.0)]]
gs(cv) = GridSearchCrossValidation(grid; cv = cv, r = r, train_score = true)

s_b = search_cross_validation(hrp, gs(batch), rd)
s_o = search_cross_validation(hrp, gs(online), rd)

(s_o.test_scores == s_b.test_scores, s_o.train_scores == s_b.train_scores,
 s_o.idx == s_b.idx, s_o.val_grid[s_o.idx])
````

Under `OnlineStep` the fold axis is sequential whatever executor the search is handed,
because fold `i` reads the state fold `i - 1` left; the candidate axis is the one that gains
from threads, and `GridSearchCrossValidation`'s `ex` stays on it.

`RandomisedSearchCrossValidation` is also compatible with `OnlineStep` as it wraps an instance
of `GridSearchCrossValidation` and samples from the search space.

## 7. A run resumes from its Result

An online run's [`MultiPeriodPredictionResult`](@ref) carries the estimator the loop
threaded, in `opt`, folded through the last training end. When more rows arrive,
[`Resume`](@ref) hands that Result back to the loop over the **full history extended** and
the same scheme: the loop skips the folds the Result holds, folds the ordinary delta from
the last training end into a copy of the state, and continues from the next fold. The
resumed Result carries the new folds only, and `vcat` stacks the two.

The oracle is the one-shot run. Timestamps are required — the loop locates the fold to
resume from by the state's last held timestamp — and the fixture carries a calendar.

````@example 09_Online_Walk_Forward
short = cut(rd, 1:260)                                   # the history at the first run
res_1 = cross_val_predict(hrp, short, online)
res_2 = cross_val_predict(Resume(res_1), rd, online)     # the full history, one fold later
stacked = vcat(res_1, res_2)

(length(res_1.pred), length(res_2.pred), maximum(gap(stacked, o_hrp)),
 stacked.mrd.X == o_hrp.mrd.X)
````

`res_1` is never written — `Resume` copies every state at entry — so one Result resumes any
number of times. Deployment of a resumed state is one hand step, `partial_fit!(res.opt,
rows)` from the last training end, and there is no `refit_last`: a state folded through the
purge and test rows equals no fold of any run and cannot be unfolded.

## 8. The moment seam on its own: the advertised benefit and the real one

Everything above stands on [`partial_fit!`](@ref) at the moment layer. Its advertised
benefit is speed: one row folds into a covariance in `O(N²)`, where a refit over the window
costs `O(t · N²)` and grows with every row. Its real benefit is accuracy, and the two are
worth telling apart.

The fold is a Welford recursion, and Welford is not the textbook one-pass formula
`(Σ xxᵀ − n μμᵀ) / (n − 1)`. On returns the two agree. On **prices** — mean a thousand, unit
spread, which is the case a caller lands in the moment they fold a level rather than a
change — the textbook formula subtracts two large numbers and loses most of its digits,
while the recursion updates around the running mean and loses none. The batch estimator
centres first and is exact; the fold reaches it.

````@example 09_Online_Walk_Forward
rng = StableRNG(987654321)
Z = randn(rng, 50, 4) .+ 1000.0                          # a level, not a return
fold(est, rows) = foldl(partial_fit!, eachrow(rows); init = est)

function textbook(Z)
    return (transpose(Z) * Z - size(Z, 1) * mean(Z; dims = 1)' * mean(Z; dims = 1)) /
           (size(Z, 1) - 1)
end
exact = cov(Covariance(), Z)

(maximum(abs, cov(fold(Covariance(), Z)) - exact), maximum(abs, textbook(Z) - exact))
````

The fold sits at the rounding floor of the exact answer; the textbook formula sits several
orders of magnitude above it, on fifty rows. The gap widens with the level and with the
sample, and a fold that reads a whole day of prices would carry it into every covariance
downstream. That is the property the seam is built to keep, and the test that guards it
asserts the gap so that a future simplification to the textbook formula fails loudly.

The speed, at the moment layer, is real: the per-row fold does not grow with the window,
and the refit does.

````@example 09_Online_Walk_Forward
Xs = randn(rng, T, N) ./ 100                             # a clean sample of the fixture's size
step_cost = let est = fold(Covariance(), Xs[1:(end - 1), :]), row = Xs[end, :]
    minimum(@elapsed(partial_fit!(est, row)) for _ in 1:200)
end
refit_cost = minimum(@elapsed(cov(Covariance(), Xs)) for _ in 1:200)
round(refit_cost / step_cost; digits = 1)
````

## 9. The gain through the loop, measured

That ratio is the moment layer's, and it is not the loop's. Through the fold loop a fold's
cost is the moment fit **plus** the read-out, and what the read-out costs decides whether the
step shows. Two things decide it, and each is measured below on this panel — the minimum of
three runs after a warm one, so the numbers are a direction and not a benchmark.

The **prior family** first. The plain family's batch covariance is one BLAS product over the
window, and at any width the library has measured (eight assets to sixty, three hundred
observations to three thousand) its online step is no faster: the product is cheap, and
the loop's bookkeeping costs what the product saves. The policy family's batch arm folds
row by row — the available-case fit *is* a recursion — so there the online loop does the
same arithmetic once instead of once per fold, and the gain is the seam's own.

The **head** second. A hierarchical head's read-out is a clustering over the moments, so its
fold cost is the moment fit and the gain shows; a JuMP head's read-out is a solve, which the
step does not touch (ADR 0139 measured the model build at a few percent of it), so the
ratio sits near one at this width and falls only as the moment fit grows into the solve.

````@example 09_Online_Walk_Forward
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
````

Read the policy rows against the plain ones. The plain family gains nothing through either
head, and that is the honest reading: on a plain prior, reach for the step for the accuracy
of §8 and for the shape of §7, not for the clock. The policy family through the hierarchical
head is the case that pays, and it pays more as the panel widens — the library's own
measurements put it at about half of batch at this size and a fifth at sixty assets over
three thousand observations. Through the JuMP head the solve is the cost at every size, and
the ratio falls only as the moment fit grows into it.

## 10. What to take away

  - `IndexWalkForward(w, t; ff = OnlineStep())` is the whole declaration. The folds are the
    walk-forward's and do not change; the fit of each fold does. The scheme derives the
    expanding window, and a rolling window is `Online(pe; max_history = w)` on the
    estimator.
  - The online run equals the batch expanding run fold for fold: to the bit below a JuMP
    head, to the solver's tolerance through one. Everything above the prior runs the batch
    path from a carrier the read-out rebuilds, so it is identical by construction rather than
    by test.
  - A member with no recursion is wrapped in `Online`, refits from its buffer, and matches
    the batch fit over the same rows exactly. A host folds what folds and refits the rest,
    so the online estimator is the batch estimator.
  - A search does not read the Fold Fit; a Result resumes a run over the history extended;
    a Pipeline is a host of the step on the same terms.
  - The seam's real benefit is accuracy: the fold is a Welford recursion, and it holds its
    digits on a level where the textbook formula does not. Its speed shows through the loop
    where the batch fit is itself a recursion — the `CoveragePolicy` family, through a
    head whose read-out is cheap — and not on a plain prior, whose batch fit is one product.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
