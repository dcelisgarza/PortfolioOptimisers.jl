---
status: accepted
---

# A search scores every candidate through the one fold loop, online and batch alike

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) closes on one test
whose second half is a search: the online grid search and the online randomised search pick the
candidate the batch searches pick over the same folds. The fold loop's online arm exists
([ADR 0140](0140-a-walk-forward-declares-its-fold-fit-and-the-online-arm-threads-the-estimator-from-a-cold-start.md)):
a walk-forward that declares `ff = OnlineStep()` warms up once, folds each fold's new rows into one
threaded estimator, and reads it out where a refit would have run.
[#871](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/871) decided how the two
searches take that arm.

Four facts shaped the decision.

1. **The scheme composes; the search does not.** `IndexWalkForward(w, t; ff = OnlineStep())` is
   the type the search's `cv` slot already takes, so no new scheme is needed. But the general
   `search_cross_validation` never reaches `fold_loop`. Per candidate it loops over the folds and
   calls `fit_and_score`, which calls `fit_and_predict(opt, rd; train_idx, test_idx)` — an
   independent refit from the fold's training window. The search was an eighth fold loop beside
   the seven that [ADR 0067](0067-the-cross-validation-fold-loop-is-one-seam.md) folded into one,
   and it was outside the seam. Under `OnlineStep` a per-fold refit cannot stand at all: a step
   folds rows into an estimator threaded from the fold before, and no fold has one.
2. **The batch search was already the odd one out.** Because it never built a
   `TimeDependentContext` and read `pws` only to derive the held-weights drift, a walk-forward
   search threaded **no previous weights** between folds and resolved **no schedule** per fold. A
   search over an optimiser with a `Turnover` term and `pws` set scored different folds than
   `fit_and_predict(opt, rd, cv)` over the same scheme. The Pipeline's search mirrored it body for
   body. Nothing documented that divergence, so it is a defect, not a rule.
3. **A step is a fold.** `split(cv, rd)` enumerates the same folds online as in batch (ADR 0140),
   so an online candidate yields `M` predictions, one per fold — the matrix shape the scorer
   already reads. The ticket's "one score per step" was the same thing under another name.
4. **A failed solve is already a non-finite fold.** A failure writes `NaN` weights and the
   fallback chain walks
   ([ADR 0120](0120-a-fold-scores-on-the-investable-mask-a-prior-free-head-and-pre-selection-reduce-to-the-coverage-universe-and-a-failed-candidate-loses-the-search.md)),
   so a fold's score is `NaN` and `finite_candidate_index` never hands the column to the scorer.
   The read-out is pure ([ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)),
   so a failed read-out leaves the threaded state intact and the later steps still run.

The reference's online search is a separate class from its batch search. It clones the
candidate, runs its own online walk-forward, and scores the **concatenated** multi-period
portfolio once per candidate. Under its default `raise_on_failure = True` a solver error fails the
whole candidate and no later step runs; under `False` a failed step is a portfolio of `NaN`
returns, the run continues, the last successful weights are the ones threaded on, and the
aggregate score is `NaN`. Neither regime scores a candidate on the steps that ran.

## Decision

**A search scores every candidate through the one fold loop.** For every contiguous scheme —
`KFold`, the two walk-forwards, and a `MultipleRandomised` over either — `search_cross_validation`
builds candidate `i` from the search's estimator through its lenses, runs
`fit_and_predict(opt_i, rd, gscv.cv; ex = SequentialEx())`, and writes `expected_risk(r, pred[j])`
into row `j` of column `i`, signed as today. Whether the loop refits every fold or steps is the
scheme's Fold Fit, and the search does not read it. The per-fold `fit_and_score` families go, for
the optimiser and for the `Pipeline`. The combinatorial method already had this shape, and keeps
it.

**The batch search therefore threads previous weights and resolves schedules for the first
time.** This moves a released number only where a walk-forward search set `pws`, held a term that
reads the previous weights, or held a `TimeDependent` schedule — every such call scored a
walk-forward it did not declare. A search with none of those scores the same matrix as before.

**Candidates run in parallel, folds in sequence.** `gscv.ex` stays on the candidate axis, as
released; the loop inside each candidate takes `SequentialEx()`. Under `OnlineStep` the fold axis
is sequential whatever executor it is handed, so this is the one axis that gains from threads in
both regimes.

**The search's argument is the configuration alone.** Before any candidate is built, the search
refuses an estimator carrying a partial-fit state, once and by name, through the state walk the
loop's online arm uses (`online_entry_state`); under `OnlineStep` it runs the arm's whole entry
check (`assert_online_entry`), so a schedule on a stateful field is refused once at the door and
not `N` times from worker threads. A cold estimator seeds one state per candidate at that
candidate's own warm-up, because a lens is applied before the warm-up, so no reset exists and none
is needed. A warm estimator is refused, not reset: the same ruling ADR 0140 made for the loop, for
the same cost.

**One row per fold, unchanged.** The score matrix stays `M × N`, rows in `split` order, and the
scorer reads it as it does today. The online searches and the batch searches read the same matrix
to the tolerances ADR 0137 measured and pick the same column, which is the map's closing identity
for the search. The randomised search under the same seed draws the same grid, so it inherits the
identity from the grid search.

**A failed step drops the candidate, and every step still runs.** ADR 0120 is unchanged: the
column holds `NaN` at the failed step, the candidate never reaches the scorer, and the raw matrix
shows which step failed. The later steps run and score, so the online column reads as the batch
column does. This is the reference's `raise_on_failure = False` regime, which is what a
`NaN`-weights fallback chain already is.

**The `Pipeline`'s search takes the same route now**, in batch. Its door keeps the refusal of a
Fold Fit that [#969](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/969) placed
there until [#872](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/872) decides what
a fold with no training window means to a `Pipeline`'s `fit`.

## Considered options

1. **One loop for both searches, batch too** — chosen. One body, ADR 0067's seam extended to
   the last loop outside it, the batch defect closed in passing, and online ≡ batch by construction.
2. **One loop for the online arm only**, a function barrier on `fold_fit(gscv.cv)` with the
   batch arm keeping its per-fold refit — rejected. No released number moves, but two bodies
   remain, the identity holds only for an estimator with no previous-weights term and no schedule,
   and the unthreaded `pws` stays a defect on the batch side.
3. **Step inside the search's own loop** — rejected. Warming up `opt_i`, folding the delta rows
   and reading out through `fit_and_predict(opt_i, rd; test_idx)` matches the batch search exactly,
   because neither threads anything, but it is a second online loop beside `fold_loop`, the shape
   ADR 0140 rejected for the reference's `online_predict`.

On the executor, **candidates sequential and folds parallel** (the combinatorial method's
convention) was rejected because a batch grid loses most of its parallelism and an online search
gains none; **reading the estimator to pick the axis** was rejected because a caller could not
predict which loop their `ex` reaches.

On the reset, **letting the loop refuse per candidate** was rejected because the refusal would
surface after the grid is built, inside `@floop`, up to `N` times; **resetting each candidate
cold**, the reference's `clone`, was rejected because it needs the per-state-type "empty, keep the
cap" verb ADR 0140 costed, and the search would then disagree with the loop on the same input.

On the score, **one aggregate row**, the reference's statistic over the concatenated series, was
rejected because it changes every released walk-forward search's matrix and winner, and a
`HighestMeanScore` over one row is a plain `argmax`; **a switch on the search** was rejected as a
knob the map did not ask for. An aggregate score for a walk-forward search is a scoring feature of
the search, batch and online alike, and is not this map's.

On the failed candidate, **short-circuiting after the first non-finite step**, the reference's
default, was rejected because "failed" and "not run" then read alike in the raw matrix, the online
column diverges from the batch one, and it needs a per-fold hook the loop does not have;
**scoring on the finite steps** was rejected because a candidate that failed once could win, which
ADR 0120 forbids, and neither reference regime does it.

## Consequences

- `search_cross_validation` for the contiguous schemes is one candidate loop over
  `fit_and_predict(opt_i, rd, gscv.cv; ex = SequentialEx())`, for the optimiser and for the
  `Pipeline`. The two `fit_and_score` families are deleted, and #969's
  `assert_batch_fold_fit(gscv.cv, "`search_cross_validation`", "#871")` at the optimiser's door
  goes with them. The `Pipeline`'s door keeps its refusal for #872.
- ADR 0067 is amended: the search is inside the seam.
- A walk-forward search that set `pws`, or held a term that reads the previous weights, or held a
  `TimeDependent` schedule, scores a different matrix than before, because the folds are now the
  walk-forward it declared. Every other released search is unchanged, and the build pins that
  against a hand loop of `fit_and_predict(opt, rd; train_idx, test_idx)`.
- The search refuses a warm estimator whatever the scheme's Fold Fit, which is stricter than the
  batch arms of the loop. No released caller holds a state to hand it.
- Two facts for the build. Under `MultipleRandomised` the loop returns one
  `MultiPeriodPredictionResult` per path, its folds sorted by test start, and the rows of the
  matrix must stay in `split`'s enumeration order. And the online identity holds against the
  materialised carrier at the tolerances ADR 0137 measured — `1e-5` through a solver, `1e-10`
  without, `5e-5` for `RiskBudgeting`.
- Three facts the build ([#1020](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1020))
  met. A `MultipleRandomised` with no `seed` draws from its `rng` afresh at every `split`, and
  the loop splits once per candidate, so the search pins one seed off that `rng` before the
  grid (`pin_draw`) and every candidate scores the same folds, as the released search did by
  splitting once. The `Pipeline`'s search handed its scorer the raw matrix, so a candidate that
  failed a fold could win there, against ADR 0120; it now goes through `finite_candidate_index`
  like the optimiser's. And `expected_risk(r, res)` with the result's own prior paired the
  weights, expanded to the caller's universe, with a prior reduced to the Investable Mask, so a
  train score over a point-in-time window threw; the weights are now viewed at the result's mask
  when no prior is passed.
- One finding outside this decision: `previous_weights` hands the next fold `prev.res.w` or
  `prev.hw.w` verbatim, so after a failed fold the sequential arm threads a `NaN` vector into a
  `Turnover` term, where the reference keeps the last successful weights. That is the fold loop's,
  batch and online alike, and belongs with the previous-weights map that
  [#1004](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1004) seeds.
