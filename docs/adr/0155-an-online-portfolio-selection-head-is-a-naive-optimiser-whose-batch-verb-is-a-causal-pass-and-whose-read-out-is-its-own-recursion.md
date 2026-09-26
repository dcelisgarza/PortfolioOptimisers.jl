---
status: proposed
---

# An online portfolio selection head is a naive optimiser whose batch verb is a causal pass and whose read-out is its own recursion

## Context

[Map #1148](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1148) brings online
portfolio selection into the library: the family of algorithms that update a portfolio directly
from the last realised price relative, fit no moment, solve no programme of their own — a rule
may hold an optimisation estimator and re-solve it on the rows it selects (ADR 0156) — and carry
a regret guarantee against every price sequence (Cover 1991; Helmbold, Schapire, Singer and Warmuth 1998;
Agarwal, Hazan, Kale and Schapire 2006; Li, Zhao, Hoi and Gopalkrishnan 2012; Li and Hoi 2012;
Huang, Zhou, Li, Hoi and Zhou 2016; Li and Hoi 2014 for the survey). The starting artifact is the
library's own prototype, `research/prototypes/09_online_portfolio_selection.jl`: seven update
rules, each a closure handed to one causal driver, `run_online`, which enforces the one property
that makes the family what it is — the portfolio held during period `t` reads rows `1:t-1` and
nothing later.

[Issue #1151](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1151) is the map's
first decision: what such an estimator *is* in this library. It also takes the online-selection
half of [#312](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/312) on map #304, *are a
frontier and a weight sequence OptimisationResults at all?*

Two research ledgers fixed the ground. The literature ledger
([#1149](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1149)) counts 36 algorithms
whose only varying part is the one-step update and its private carriers — a Gram matrix, a
price-level window, expert log-wealths. The seams ledger
([#1150](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1150)) measured every seam
the family would consume and found the loop keeps and the read-out breaks: `thread_online_folds!`
threads anything `partial_fit!`-able, but `optimise(opt)` under
[ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)
reconstitutes a carrier and runs the batch path, and an unknown optimiser type falls to
`online_readout(opt) = (opt, ReturnsResult())`. The ledger also found that the previous weights the
loop threads reach the per-fold copy through `factory` *after* `partial_fit!` ran on the threaded
estimator, so the recursion's `w_t` cannot come from the loop; that `NaiveOptimisationResult`
already carries everything `predict` reads; and that no online verb dispatches on
`NaiveOptimisationEstimator`, so a subtype inherits no batch read-out by accident.

### The literature's protocol

Every paper's loop has the same shape (the survey's Algorithm 1): at the start of period `t` the
manager commits `b_t` from `x_1 … x_{t-1}`, with `b_1 = 1/N`; the market reveals `x_t`; wealth
compounds by `b_tᵀx_t`; the manager runs the update to form `b_{t+1}`. The update is the last step
of every iteration, so after `n` rows the algorithm holds `b_{n+1}` — Cover writes the universal
portfolio as `b_{t+1} = ∫ b S_t(b) dμ(b) / ∫ S_t(b) dμ(b)` outright. What the papers *report* is
`S_n`, which never uses `b_{n+1}`, and a backtest driver, the prototype's included, stops at
`b_n`. That is a reporting convention, not the algorithm's; a live deployment trades `b_{n+1}`.

## Decision

### The destination, confirmed and amended

The map's destination stands: solver-free optimisers that update from realised price relatives,
take the online step through the seams the library already has, read a Prior wherever an
algorithm's forecast *is* a forecast, and honour the constraint vocabulary through a constrained
update in each algorithm's own geometry — with one amendment. **The family uses an existing seam
where the seam fits and is new functionality where it does not.** Nothing is forced through an
unsuitable seam, and a lesson the family teaches flows back as a generalisation of existing code
when one is worth making. The four out-of-scope lines stand on their merits: the event loop is a
consumer of the two verbs the family already has, and multi-period optimisation contradicts the
one-step definition that carries the regret bound.

### One head, the rule on a slot, under the naive family

`OnlinePortfolioSelection <: NaiveOptimisationEstimator` is the one concrete head. It holds
`fb`, `strict` and `cache` as every naive head does, its constraints as one Allocation Set on `set`
in place of `wb`, `sets` and a Weight Finaliser
([ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)),
its fees on `fees`
([ADR 0160](0160-an-online-update-starts-from-its-own-allocation-and-its-trade-is-measured-from-the-price-adjusted-one.md)),
its Start Allocation on `w0`
([ADR 0162](0162-an-online-selection-head-buffers-returns-and-starts-from-a-given-allocation-or-a-uniform-one-over-the-pinned-universe.md)),
and the update rule on `alg::AbstractOnlinePortfolioSelectionAlgorithm`, an unexported abstract type under
`AbstractAlgorithm`, one concrete rule per algorithm (`ExponentiatedGradient`,
`MovingAverageReversion`, …; the roster is a later ticket's). The rule is the only thing that
varies across the 36 rows of the ledger, so it is the only thing a member writes: a rule struct, a
state for its private carriers, and one update method. Everything shared — the projection, the
block step, the read-out, the `partial_fit!` loop, the refusals, `held_timestamps`, `copy`,
`merge_states`, `port_opt_view` — is written once on the head. A search addresses the rule's
knobs by path, `"alg.window" => 2:10`, through the lenses
[ADR 0141](0141-a-search-scores-every-candidate-through-the-one-fold-loop-online-and-batch-alike.md)
introduced. A meta-learning aggregator over experts is a rule over a vector of rules, not a
second head.

The head is naive because solver-free is the shared property, the naive result already carries
what `predict` reads, and the six thin forwards the naive family provides — the time-dependent
machinery, `needs_previous_weights` from `fb`, the two no-op asserts — are the right ones.
`NaiveOptimisationEstimator`'s docstring gains the clause that a naive head may also update its
weights from the last price relative.

### The batch verb is the Causal Pass

`optimise(opt, rd)` on the head is the **Causal Pass**: from the start weights `w0`, apply the
Online Update to every row of `rd` in order, and return the **Next-Period Allocation** — after
rows `1:T`, the portfolio for period `T+1`. This is one more update than the prototype's driver
takes, because an optimiser's answer is always the portfolio for the period after its data;
parity with the prototype is the library's answer after rows `1:t-1` equalling the prototype's
`W[t, :]`. A batch call on a head that already carries a state runs from `w0` over `rd` and
neither reads nor writes the state, as every host does today; *continue from here* has its own
spelling, `partial_fit!` then `optimise(opt)`.

### The read-out is a Recursion Read-out, and ADR 0137 gains a second kind

`partial_fit!(opt, rd)` folds each row of `rd` through the Online Update into the state.
`optimise(opt)` with no data reads the state's allocation, wraps it in a Result, and runs **no
batch path**. This is the **Recursion Read-out**, and it is a named second kind of read-out
beside the reconstitution
[ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)
fixed, not an exception to it: 0137 is amended to say that an optimiser's read-out is one of the
two, and which one is the family's to declare. The identity 0137 promises, `optimise(opt)` equals
`optimise(opt, rd[1:t])` after `t` folded rows, holds here **exactly** — both arms take the same
sequence of single-row updates — where the reconstitution keeps it to the moment layer's
tolerance.

The state is `OnlinePortfolioSelectionState <: AbstractPartialFitState`, one type on the head's
`cache`: the allocation `w_t`, the rule's private carriers, the Fold Context's pinned asset names
and static Asset Panel, and the timestamps folded. It holds **no shared column buffer** — no
returns, factor or benchmark column — because the read-out rebuilds nothing; the rows any rule
of the tree reads — a reversion's window, a follow-the-leader's prefix — are held once on the
head's state, capped by the rule tree's need (ADR 0156, ADR 0157); the pin-and-check every host
runs at the first step and after it is kept, so a hand-called step on a reordered universe is
refused by name and not folded silently, and the timestamps serve
[ADR 0144](0144-an-online-runs-result-carries-the-threaded-estimator-and-resume-re-enters-the-fold-loop-from-the-folds-it-holds.md)'s
`Resume`. `merge_states` refuses, naming order-dependence: an online update is not a sufficient
statistic for its block. The field list of the state, the one verb a rule writes, the block step,
and what a step does with a `NaN` price relative are ADR 0157's.

### `Online(head)` is refused at the construction door, because it adds no answer

`Online` declares a refit from a buffer, so `Online(head)` would run the Causal Pass over
the buffer at every fold. Both of its settings are a batch walk-forward the library already runs,
measured over the SP500 panel of `examples/` for seven rules of the tree:

| the wrapper | the scheme that already gives it | the allocation |
| --- | --- | --- |
| `Online(head)` | `IndexWalkForward(…; expand_train = true)` | the online arm's, to the last bit |
| `Online(head; max_history = w)` | `IndexWalkForward(w, test_size)` | the recursion restarted from `w0` in each window |

Both columns were run against their hand-taken folds and agreed to `0.0` in every weight, and the
expanding walk-forward cost 12 to 110 times the online arm over 300 rows. So the wrapper is a
slower spelling of two routes that exist, and the refusal takes nothing from a caller.

The door is the **constructor**, not the warm-up. A wrapper reaches the warm-up from an estimator
field alone, and every field that could hold one already refuses a wrapped head by its own type
bound, so a refusal at the warm-up is a sentence no caller ever reads. The constructor is the
earliest point and the only reachable one. `online_state_seed` therefore writes no method for the
head.

### A head does not own a Pipeline's rows

A Pipeline's row owner is its prior step, else its optimisation step, and the read-out
reconstitutes the carrier from the owner before it refits the universe steps over it
(ADR 0142). A head owns no rows in that sense, and two separate things stop it:

1. **It holds no carrier.** Its read-out is the recursion, so it rebuilds nothing, and a rule
   whose tree reads no rows holds no rows at all — `rows_needed` is `0` for most of the tree, and
   the rows buffer is then absent rather than short.
2. **A carrier would not be enough.** The read-out expresses a universe step's selection as a
   **view** of the owner's state over the surviving columns, and promises the view equals the
   batch fit over those columns. That promise holds for a prior, whose estimate over columns is
   the submatrix of the estimate over all of them. It fails for a recursion: the allocation is a
   path, the projection couples the columns, and the wealth factor `⟨w, x⟩` reads every one of
   them. Measured over four of eight columns, the view differs from the run over those columns by
   `1.0` in a weight for `AntiCorrelation`, `1.05e-2` for `GradientProjection` and `2.5e-5` for
   `ExponentiatedGradient`.

So the head is refused by name where the pipeline is entered — at the fold loop's door and at the
first hand-driven fold — and the message names the two routes that give the batch answer: a prior
step before the head, which makes the prior the row owner, or `Online(pipe)`, which refits every
step over the observations folded so far and was measured to reproduce the batch fit exactly. The
asset-subset path is untouched and stays exact, because `MultipleRandomised` takes its view once,
at the warm-up, so the recursion runs on the chosen columns from the first row.

### The Result is `NaiveOptimisationResult`, and a weight sequence is not an OptimisationResult

The head returns a `NaiveOptimisationResult`: one weight vector for the next period, the
Investable Mask, `wb`, `retcode` and `fb`. `pr` is the prior result when the head holds a prior
and `nothing` otherwise, in **both** arms, so the batch–online identity holds on the whole Result
and not on `w` alone. No path and no state ride on the Result. The per-row path over a window is
the product of the walk-forward the library already has —
`OnlineIndexWalkForward(1, 1)` is `T` read-outs at `O(N)`
each, drifted, fee-charged, scored and plotted through the existing machinery — and a path on the
Result would be a second, unscored copy of it whose shape differed between the arms. This rules the
online half of #312: **a weight sequence is a walk-forward's product, never an optimisation
result.** If a later ticket rules that the family takes fees, `NaiveOptimisationResult` grows an
optional `fees` property, which the other naive heads then share.

### What the naive contract gives, and what the head writes

Derived from the rulings above, not decided separately:

- `needs_previous_weights` is inherited: `fb` alone. The state owns `w_t`, so the loop's thread is
  redundant for the update; a turnover-aware read-out that reads `w_prev` through `factory` is the
  start-weights-drift-turnover-fees ticket's, and would add the head's own method.
- `non_investable_universe` is `InverseVolatility`'s: rebuild with `non_investable_sets`.
- `factory(head, w)` is the identity until that same ticket says otherwise.
- `@fprop @vprop cache` on the head, and `port_opt_view` on the state slices `w` by asset and
  renormalises, forwarding to the rule's carriers; a rule whose carriers cannot slice refuses.
- The fallback chain walks in batch exactly as it does for every naive head. At a read-out it is
  never reached, because an update over a finite row cannot fail.
- `show_fields` hides `cache`, so no doctest moves.

## Considered options

1. **One head per algorithm** — `ExponentiatedGradient`, `MovingAverageReversion`, … each a
   full estimator under one abstract family type. Rejected: the six shared fields and the dozen
   shared verbs repeated per algorithm, for a family whose members differ in one function.
2. **A new unexported abstract family type** `<: NonFiniteAllocationOptimisationEstimator` with
   the one head under it, and an own Result type. Chosen first, then withdrawn on reflection:
   a branch for one concrete head, its six forwards rewritten as one-liners, and a naive-shaped
   Result under a new name, buy nothing the naive family does not give.
3. **The head directly under `NonFiniteAllocationOptimisationEstimator`** with no abstract type.
   Rejected for the same six duplicated forwards.
4. **Return `w_T`, the last portfolio held**, mirroring the prototype's matrix. Rejected: the
   answer would be for a period whose return is already in the data, and every consumer would
   apply it one period late. The literature's loop computes `b_{n+1}`.
5. **A batch verb that is not the Causal Pass** — the hindsight best constant rebalanced
   portfolio, say. Rejected: that is `MeanRisk` under `LogarithmicReturn` (map ground truth 5),
   and a batch verb that differs from the online pass breaks the identity the loop relies on.
6. **Amend 0137 with an exception for this family.** Rejected: the next optimiser with its own
   recursion would amend again, and the rule would read as bent rather than extended.
7. **Leave 0137 untouched and state the departure in this ADR alone.** Rejected: 0137's opening
   rule would stay false as written on `main`.
8. **A `ReturnsBufferState` beside the recursion state.** Rejected: it buffers benchmark and
   factor columns the family never reads, unboundedly unless capped, and the state has two halves
   to copy and slice.
9. **No Fold Context at all.** Rejected: a reordered universe would fold silently, and `Resume`
   would have no timestamps.
10. **An own Result carrying the batch pass's path and the terminal state.** Rejected: an
    unscored duplicate of the walk-forward whose shape differs between the arms, and a state on
    a Result invites *continue from here* without `Resume`.
11. **An own Result with the naive fields.** Chosen while the head had its own supertype,
    withdrawn with it.

## Consequences

- ADR 0137 carries an amendment naming the two kinds of read-out. `CONTEXT.md`'s *Fold Context*
  entry, which said nothing above the prior takes a step of its own, gains the sibling sentence.
- `CONTEXT.md` gains *Online Portfolio Selection*, *Online Update*, *Next-Period Allocation*,
  *Causal Pass* and *Recursion Read-out*, and the naive section lists the head.
- The map's remaining decision tickets build on this one: the state's field list and the block
  step of a `test_size > 1` fold; the Prior slot for the thirteen forecast-reading rules; the
  constrained update in each rule's geometry; start weights, drift, turnover and fees; regret and
  the hindsight benchmarks. The build tickets graduate once those land.
- #312's online half is answered here; its critical-line half stays on map #304.
- `test/test_24b_optimiser_partial_fit.jl`'s identity gains its first exact case.
- #1242 closes: the `Online(head)` refusal moves from `online_state_seed`, which no caller could
  reach, to the constructor, and the seed method goes. The head's private API page loses its row
  and the public page gains one.
- #1241 closes: a Pipeline whose row owner is a head is refused at the fold loop's door
  (`assert_online_owner`) and at the first hand-driven fold (`fold_pipeline_owner`), where it
  folded every row and then died at the read-out. `returns_result` is written for no head.
- `test/test_67g_cross_validation_and_tuning.jl` gains the group that locks both refusals and the
  routes they name.
