#=
```@meta
Description = "The online walk-forward in PortfolioOptimisers.jl worked through: one estimator warmed on the first fold and stepped through the rest."
```

# The online walk-forward: one estimator, stepped fold by fold

A walk-forward refits every fold from its training window. Fold `i` reads every row fold
`i - 1` read and a few more, fits the moments over all of them, and solves. On an expanding
window the cost of the whole run therefore grows with the square of the number of folds, and
the moments of fold `i` are recomputed from scratch when almost all of their input is the
input of fold `i - 1`.

The library's answer is an online scheme, the walk-forward wrapped in `Online` and built by
[`OnlineIndexWalkForward`](@ref). It warms one estimator up on the first training window, folds
each later fold's *new* rows into it, and reads it out where a refit would have run. The loop
threads the estimator from fold to fold, and the run reaches the weights of the batch
expanding-window walk-forward fold for fold. The enumeration of folds does not change. Only the
fit of each fold does.

The mechanism is [`partial_fit!`](@ref) on the moments, and every layer above the moments takes
the step through it.

  - A prior folds the new rows into its members and keeps the rows themselves, because a prior
    result carries `X` for the scenario risk measures.
  - An optimiser forwards the step to its prior alone. Reading the state out rebuilds a returns
    result and runs the ordinary batch path, so every constraint, every clustering and every
    uncertainty set is the one batch builds. A set with no prior of its own is calibrated on the
    prior result it is handed.
  - A JuMP head builds a fresh model every time it reads the state out. The solve is the step's
    cost, and no model is kept warm.
  - The scheme's type declares the step, and the online arm of the loop threads the estimator
    from a cold start.
  - A search scores every candidate through the one fold loop, online and batch alike.
  - A `Pipeline` hands the online update to the steps it holds. A row-local step folds, a
    universe step takes a view, and a step with no online form fails unless the pipeline
    declares a refit.
  - A result resumes a run. `Resume(res)` re-enters the loop from the folds the result holds.

A member with no exact recursion is wrapped instead. `Online(est; max_history)` seeds a sample
buffer, and the library refits from that buffer when it reads the state out. The cap on the
buffer is the rolling window.

!!! tip "When to reach for this"
    Reach for `OnlineIndexWalkForward` in place of any expanding walk-forward, because it costs
    nothing in accuracy. The run gives the batch run's weights fold for fold. Reach for it for
    the *clock* when the prior carries a `CoveragePolicy` and the head is cheap to read out, a
    hierarchical or a naive optimiser, because that is where the batch fit is itself a recursion
    over the rows. Reach for it whenever a run will be continued, because the result of an online
    run resumes over a longer history and a batch result cannot. On a plain prior through a JuMP
    head the step saves no time, and this page gives the numbers.

This example runs one walk-forward through every one of those layers. Each section prints the
comparison it measures, and the last two sections measure the time the step saves, which is not
where a first reading puts it.

 1. The panel: a universe that moves inside the window, and the two walk-forwards.
 2. The online run against the batch expanding run, fold for fold.
 3. What the loop does, written by hand.
 4. A member with no exact recursion refits from its buffer.
 5. A capped buffer is the rolling window.
 6. A search picks the batch candidate.
 7. A run resumes from its result.
 8. The moment step on its own: the advertised benefit and the real one.
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

The fixture is a small factor panel drawn from a seeded generator: eight assets, three
hundred observations, two factors. Three assets list after the first row and three delist
before the last, so the universe moves inside the window, which is the case the step exists
for. The listing calendar is written out as a fact about the instruments, and the returns are
`NaN` wherever an asset is not listed. The [`AssetPanel`](@ref) carries the calendar as its
active mask, and we set the estimation mask to the active one, because the online step folds
the active mask alone. Section 2 says why.
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
## The rows `i` of the carrier, panel included: the walk-forward cuts its windows this way.
function cut(rd, i)
    return ReturnsResult(; nx = rd.nx, X = rd.X[i, :], ts = rd.ts[i],
                         pnl = AssetPanel(; amsk = amsk[i, :], emsk = amsk[i, :]))
end

[(rd.nx[j], findfirst(amsk[:, j]), findlast(amsk[:, j])) for j in 1:N]

#=
The two walk-forwards enumerate the same folds. The batch one expands its training window,
which is what an online run does anyway, because a fold cannot un-fold an observation.
`OnlineIndexWalkForward` therefore sets `expand_train = true`, and the two schemes cut identical
windows. Both purge three rows before each test window.
=#

w, t, p = 100, 40, 3
batch = IndexWalkForward(w, t; purged_size = p, expand_train = true)
online = OnlineIndexWalkForward(w, t; purged_size = p)

(; train_idx, test_idx) = split(batch, rd)
(train_idx == split(online, rd).train_idx, train_idx, [first(i):last(i) for i in test_idx])

#=
The seventh asset lists at row 175, inside the third training window, and the fourth delists
at row 252, inside the fifth. A plain prior over an expanding window from row 1 never admits a
late listing, because it asks for a finite return at every row of the window and the window
always starts at row 1. The prior below carries a [`CoveragePolicy`](@ref) on both moments, so
each cell is fitted on the observations it has, and an asset enters the estimate when it lists.
Its moments fold exactly under the policy, because the available-case batch fit is itself a
recursion over the rows. Section 9 returns to that fact.
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
## 2. The online run against the batch expanding run, fold for fold

The comparison is the same call twice, once under each scheme. The loop's online arm says so on
its info line, and everything else is the ordinary [`cross_val_predict`](@ref). The table prints
the largest weight difference per fold under each head.
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
The `hierarchical_gap` column is zero on every fold, because the available-case batch fit *is*
the fold. The `jump_gap` column is zero here too, and in general it sits within the solver's
tolerance. The head solves the same problem from the same rows, and a solver started from the
same point ends at the same point. Both routes see the universe move. Seven assets are
investable in the first two folds, all eight once the seventh lists, and seven again once the
fourth delists. `imsk` is `nothing` on a fold where every asset was investable. Nothing above the
prior ran online. The hierarchical head clustered and the JuMP head solved as they do in batch,
over a returns result rebuilt from the state.

The step folds the active mask of the panel with the rows and nothing else. A panel whose
estimation mask differs from its active mask, or that carries time-varying panel fields, fails
at warm-up with a message that names the field. The buffers hold the rows and the mask, and they
hold nothing else. The batch loop takes both panels, because a refit reads the whole window.
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

The loop is two verbs. `partial_fit!(opt, rd)` folds the rows of a returns result into the
optimiser and solves nothing. `optimise(opt)` with no data reads the state out and solves once.
The warm-up is the first training window, and each later fold is the rows the window gained
since the last one. The last line below prints the largest difference between the weights
written by hand and the weights the loop produced.
=#

est = partial_fit!(mr, cut(rd, train_idx[1]))                 # warm-up: fold 1's window
hand = [optimise(est).w]
for i in 2:length(train_idx)
    delta = (last(train_idx[i - 1]) + 1):last(train_idx[i])
    global est = partial_fit!(est, cut(rd, delta))            # only the new rows
    push!(hand, optimise(est).w)
end

maximum(maximum(abs, a - b) for (a, b) in zip(hand, weights(o_mr)))

#=
The state lives in the estimator's `cache` fields, and the loop starts cold. An estimator that
already carries a state at entry fails with a message that names it, so the estimator you hand
to the walk-forward is always the configuration and never a half-fitted run. A
[`TimeDependent`](@ref) schedule on the prior, or on the optimiser itself, fails at warm-up for
the same reason, because a schedule replaces the value the state is threaded through. A schedule
on any other field composes with no rule, so the weight bounds below tighten as the folds
advance, online as in batch.
=#

n = length(train_idx)
caps = TimeDependent([WeightBounds(; lb = 0.0, ub = ub) for ub in range(0.4, 0.25, n)])
mr_caps = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv, wb = caps))
gap(cross_val_predict(mr_caps, rd, online), cross_val_predict(mr_caps, rd, batch))

#=
## 4. A member with no exact recursion refits from its buffer

A [`FactorPrior`](@ref) regresses the assets on the factors, and the library carries no
incremental fold for a regression. It takes the step wrapped instead. [`Online`](@ref) seeds a
sample buffer in the prior's `cache`, and the fold appends the rows, and the factor rows beside
them because the prior's members read them. Reading the state out then runs the batch verb over
the buffer. The buffer holds the rows the batch fold reads, so the two fit the same sample. The
next cell compares the weights of the two runs, the JuMP head's included.
=#

fp_batch = MeanRisk(; opt = JuMPOptimiser(; pe = FactorPrior(), slv = slv))
fp_online = MeanRisk(; opt = JuMPOptimiser(; pe = Online(FactorPrior()), slv = slv))

(weights(cross_val_predict(fp_online, rdf, online)) ==
 weights(cross_val_predict(fp_batch, rdf, batch)))

#=
`Online` is stored in the field it wraps, and it does not survive the run. The loop resolves it
at warm-up into an ordinary estimator carrying a state, and no `Online` exists from that point
on. An estimator that holds one folding member and one buffered member folds the first and
refits the second from its own rows, so the estimator you write online is the estimator you
would write in batch.

## 5. A capped buffer is the rolling window

An online run only ever grows its history, so a rolling window is the estimator's to declare and
not the loop's. `Online(pe; max_history = w)` keeps the last `w` rows and drops the oldest as
each new one arrives. A rolling batch scheme of window `w + p` with purge `p` trains over `w`
rows, and the capped online run reads out over those same rows. The next cell checks the window
length and compares the weights and the masks of the two runs.
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
A cap on a folding member is a different thing. `max_scenarios` on a prior bounds the rows its
result carries for the scenario measures and leaves the moments fitted over every observation,
so a capped fold-and-carry prior matches no batch fit. `max_history` on `Online` windows the
whole fit and has a batch equal, which is the comparison above. You can set both, and they nest.

## 6. A search picks the batch candidate

A search scores every candidate through the one fold loop, whatever the scheme's fit. The grid
tunes the weight bounds, which bind and so separate the candidates. The next cell compares the
two score matrices and the column each search picked.
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
Under an online scheme the fold axis is sequential whatever executor the search is handed,
because fold `i` reads the state fold `i - 1` left. The candidate axis is the one that gains
from threads, and `GridSearchCrossValidation`'s `ex` stays on it.

`RandomisedSearchCrossValidation` takes an online scheme too, because it wraps an instance of
`GridSearchCrossValidation` and samples from the search space.

## 7. A run resumes from its result

An online run's [`MultiPeriodPredictionResult`](@ref) carries the estimator the loop threaded,
in `opt`, folded through the last training end. When more rows arrive, [`Resume`](@ref) hands
that result back to the loop over the full history and the same scheme. The loop skips the folds
the result holds, folds the rows between the last training end and the new end into a copy of
the state, and continues from the next fold. The resumed result carries the new folds alone, and
`vcat` stacks the two.

The run to compare against is the one-shot run. The loop finds the fold to resume from by the
state's last held timestamp, so a run that resumes needs timestamps, and the fixture carries a
calendar.
=#

short = cut(rd, 1:260)                                   # the history at the first run
res_1 = cross_val_predict(hrp, short, online)
res_2 = cross_val_predict(Resume(res_1), rd, online)     # the full history, one fold later
stacked = vcat(res_1, res_2)

(length(res_1.pred), length(res_2.pred), maximum(gap(stacked, o_hrp)),
 stacked.mrd.X == o_hrp.mrd.X)

#=
`Resume` copies every state at entry, so it never writes `res_1` and one result resumes any
number of times. To deploy a resumed state you take one step by hand from the last training end,
in the value form. `partial_fit(res.opt, rows)` folds a copy of every state and leaves `res`
resumable. The bang form `partial_fit!` writes the held timestamps in place, and `Resume(res)`
then refuses the result, which is the rule for an estimator you keep. There is no `refit_last`, because a
state folded through the purge and test rows matches no fold of any run and cannot be unfolded.

## 8. The moment step on its own: the advertised benefit and the real one

Everything above uses [`partial_fit!`](@ref) at the moment layer. Its advertised benefit is
speed: one row folds into a covariance in `O(N²)`, where a refit over the window costs
`O(t · N²)` and grows with every row. Its real benefit is accuracy, and the two are worth
telling apart.

The fold is a Welford recursion, and Welford is not the textbook one-pass formula
`(Σ xxᵀ − n μμᵀ) / (n − 1)`. On returns the two give the same answer. On prices, which carry a
mean near a thousand and a spread near one, the textbook formula subtracts two large numbers and
loses most of its digits, while the recursion updates around the running mean and loses none. A
caller meets that case the moment they fold a level rather than a change. The batch estimator
centres first and is exact, and the fold reaches it.
=#

rng = StableRNG(987654321)
Z = randn(rng, 50, 4) .+ 1000.0                          # a level, not a return
fold(est, rows) = foldl(partial_fit!, eachrow(rows); init = est)

function textbook(Z)
    return (transpose(Z) * Z - size(Z, 1) * mean(Z; dims = 1)' * mean(Z; dims = 1)) /
           (size(Z, 1) - 1)
end
exact = cov(Covariance(), Z)

(maximum(abs, cov(fold(Covariance(), Z)) - exact), maximum(abs, textbook(Z) - exact))

#=
The fold sits at the rounding floor of the exact answer, and the textbook formula sits several
orders of magnitude above it on fifty rows. The gap widens with the level and with the sample,
and a fold that reads a whole day of prices would carry it into every covariance computed from
it. The library's test suite asserts that gap, so a later change to the textbook formula fails.

The speed at the moment layer is real. The per-row fold does not grow with the window, and the
refit does. The next cell prints the cost of one refit over the cost of one folded row.
=#

Xs = randn(rng, T, N) ./ 100                             # a clean sample of the fixture's size
step_cost = let est = fold(Covariance(), Xs[1:(end - 1), :]), row = Xs[end, :]
    minimum(@elapsed(partial_fit!(est, row)) for _ in 1:200)
end
refit_cost = minimum(@elapsed(cov(Covariance(), Xs)) for _ in 1:200)
round(refit_cost / step_cost; digits = 1)

#=
## 9. The gain through the loop, measured

That ratio belongs to the moment layer, not to the loop. Through the fold loop a fold costs the
moment fit plus the cost of reading the state out, and that second cost decides whether the step
shows. Two things decide it, and the cells below measure each one on this panel. Each number is the minimum of
three runs after a warm one, so read it as a direction and not as a benchmark.

Take the prior family first. The plain family's batch covariance is one BLAS product over the
window, and at every width the library has measured, eight assets to sixty and three hundred
observations to three thousand, its online step is no faster. The product is cheap, and the
loop's bookkeeping costs what the product saves. The policy family's batch fit folds row by row,
because the available-case fit *is* a recursion, so there the online loop does the same
arithmetic once instead of once per fold, and that is where the gain comes from.

Take the head second. A hierarchical head reads the state out with a clustering over the
moments, so its fold costs the moment fit and the gain shows. A JuMP head reads the state out
with a solve, which the step does not change, and the model build measures at a few per cent of
the solve. The ratio therefore sits near one at this width, and it falls only as the moment fit
grows into the solve.
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
Read the policy rows against the plain ones. The plain family gains nothing through either head,
so on a plain prior reach for the step for the accuracy of section 8 and for the shape of
section 7, not for the clock. The policy family through the hierarchical head is the case that
pays, and it pays more as the panel widens. The library's own measurements put it at about half
of batch at this size, and a fifth at sixty assets over three thousand observations. Through the
JuMP head the solve is the cost at every size, and the ratio falls only as the moment fit grows
into it.

## 10. What to take away

  - `OnlineIndexWalkForward(w, t)` is the whole declaration. The folds are the walk-forward's
    and do not change, and the fit of each fold is what changes. The scheme derives the
    expanding window, and a rolling window is `Online(pe; max_history = w)` on the estimator.
  - The online run gives the batch expanding run's weights fold for fold. Without a JuMP head
    the numbers are the same, and through one they sit within the solver's tolerance. Everything
    above the prior runs the batch path over a returns result rebuilt from the state, so both
    routes run the same code.
  - A member with no recursion is wrapped in `Online`, refits from its buffer, and fits the same
    sample the batch fold fits. An estimator that holds members folds what folds and refits the
    rest, so the online estimator is the batch estimator.
  - A search scores through the same loop whatever the fit. A result resumes a run over a longer
    history. A `Pipeline` hands the online update to the steps it holds, on the same terms.
  - The step's real benefit is accuracy. The fold is a Welford recursion, and it holds its
    digits on a level where the textbook formula does not. Its speed shows through the loop
    where the batch fit is itself a recursion, the `CoveragePolicy` family through a head that
    is cheap to read out, and not on a plain prior, whose batch fit is one product.
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
