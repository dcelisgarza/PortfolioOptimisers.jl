---
status: accepted
---

# A failed fold holds: the loop threads the last threadable fold, a failed fold drifts what it was handed, and `PreviousWeights` is the hold-only fallback

## Context

[#1021](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1021) was found on
[#871](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/871) while reading the
reference's online search: `previous_weights` handed the next fold the previous fold's `res.w`, or
its `hw.w` under a Previous-Weights Source, verbatim, and a failed solve writes a vector of `NaN`
weights ([ADR 0120](0120-a-fold-scores-on-the-investable-mask-a-prior-free-head-and-pre-selection-reduce-to-the-coverage-universe-and-a-failed-candidate-loses-the-search.md)).
The ticket said the tail of the walk-forward then went `NaN`. Measured, it was worse, twice.

1. **Batch, no drift.** The fold after the failure did not go `NaN`; it **threw**.
   `factory(::TurnoverEstimator, w)` asserts finiteness on the threaded weights, so the run died
   at the next fold's `resolve` with `DomainError: all(isfinite, w) must hold`.
2. **Any drift** (`wd` or `pws`). The failed fold never returned. Its own `predict` drifted the
   `NaN` target, `drift_wealth` gave `NaN`, `non_positive_wealth_index` read `!(NaN > 0)` as
   ruined, and `NonPositiveWealthError` came out of `calc_net_returns` before the loop saw a
   result.

One failed fold therefore ended a walk-forward that would have solved every later fold on its own,
for any optimiser carrying a term that reads the previous weights.

The reference has two behaviours, and which one applies is the user's configuration.

- **A failed step.** Its online loop, `_online_predict`, keeps no separate variable and walks
  nothing back: it skips `set_params(previous_weights = …)` on a `FailedPortfolio`, so the sticky
  parameter still holds the last successful weights and the next step reads them. Its batch
  sequential path has no such guard and threads the failed step's `NaN` weights, which is #1021's
  defect in the reference. A failed step's weights and returns are `NaN`, and the multi-period
  series has a hole.
- **`fallback = "previous_weights"`.** An estimator-level fallback the user opts into.
  `_fallback_to_previous_weights_or_raise` sets `weights_ = previous_weights`, so the step
  returns an ordinary `Portfolio` with finite returns, the chain records
  `("previous_weights", "success")`, and the loop threads it as a success. The previous weights are
  used verbatim: shape-validated, a scalar broadcast, a name-keyed mapping zero-filled, no bounds
  and no renormalisation. It raises when there are none.

The reference has no Weight Drift, so it is silent on what a failed fold *held*; its previous
weights are always the target weights.

Two facts about this library shaped the shape of the fix. The loop is stateless — `resolve(i,
prev, …)` is handed a prediction, not a parameter it can leave alone — so the reference's omission
has to be written as a rule about which prediction is handed on. And a `HeldWeightsResult` rebuilt
its weight path from the **target** the reader passed in (`weight_path(pred.hw, pred.res.w)`), so
a failed fold whose drift started anywhere but its own `NaN` target would have rebuilt `NaN` while
its `hw.w` was finite.

The maintainer had ruled on 2026-09-10 that the previous-weights mechanism across a walk-forward
gets a map of its own, seeded by
[#1004](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1004). This decision
takes #1021 and the previous-weights fallback out of that map, on the maintainer's instruction of
2026-09-11; #1004, the frontier's matrix of weights, stays in it.

## Decision

**`prev` is the last threadable fold, not the fold before.** The two sequential arms of
`fold_loop`, `run_folds` and `online_folds`, carry one `prev` and advance it to fold `i` only when
`threads_weights(pws, predictions[i])` holds. What is read decides what is tested: with no source
the target weights are read, and they are finite exactly when every member's return code is an
`OptimisationSuccess`; with a source the held weights are read, and they are finite when the drift
ran. `previous_weights` is unchanged and never yields `NaN`. A failed fold is skipped over by the
target read, so the fold after it reads the last solved fold — the reference's sticky parameter,
written stateless — and both arms, batch and online, do it, where the reference guards its online
loop alone. A failure at fold 1 threads nothing, so fold 2 reads the estimator's own reference
weights, as the reference's constructor value stands.

**A failed fold under a drift holds what it was handed.** `Fold` carries `w_prev`, the weights that
were threaded into its estimator, and `predict` takes it as a keyword through `fit_and_predict`
and the `Pipeline`'s predict. `held_start_weights(retcode, w, w_prev)` names the weights the
fold's drift starts from: its own on a success, `w_prev` on a failure, member by member under a
population, with one `w_prev` vector serving every failed member or one per member. Under a source
`w_prev` is the previous fold's held book, so the record drifts the book the fund carried; under a
Weight Drift alone `w_prev` is the last chosen target, which is what that switch threads by design
(map #746: the two switches are separate), so the failed fold rebalances to the last decision and
drifts. The failed fold's `res.w` and `rd.X` stay `NaN`, no fee is charged, and ADR 0120's rule
that a failed candidate loses the search is untouched: the held record is the one finite thing on
a failed fold. With no `w_prev` — fold 1, or a scheme whose folds are not a timeline — the drift is
skipped, the record is `NaN`, and nothing throws; such a member is not ruined and keeps its own
failure code. `calc_net_returns` under a drift gives a `NaN` series for a non-finite vector rather
than raising, so the drifted arm and the undrifted one agree on what a failed fold's series is.

**`HeldWeightsResult` records its start weights.** The record gains `w0`, the first row of the
path, and `rebuild_weight_path` reads it; the target a reader passes to `weight_path(hw, w)` is
ignored on a record, as the record's `X` already was. A record is self-contained for every fold,
and a rebuild of a failed fold's path is finite when its drift was. The type is not on `main`, so
the field costs no amendment.

**`PreviousWeights` is the hold-only head, and the reference's `previous_weights` fallback.**
`PreviousWeights <: NaiveOptimisationEstimator`, fields `w` and `fb`. `factory(leaf, w::VecNum)`
fills `w` and recurses into `fb` — the same pass that writes the threaded weights into a
`TurnoverEstimator`, reached through the `@fprop fb` tag every optimiser already carries, so the
fallback chain, `needs_previous_weights` and the loop all work unchanged. `_optimise` returns `w`
verbatim on the full universe as a `NaiveOptimisationResult` with `imsk = nothing` and no bounds,
so a hold is never rewritten; with `w = nothing` it returns an `OptimisationFailure` naming the
missing weights, so a chain that reaches it walks on — the reference's raise, in the idiom of a
chain that walks on a failure code. `needs_previous_weights(leaf)` is `true`, so an optimiser
carrying it runs sequentially, as the reference's property forces. Its online step is the
identity. It is also usable as a primary: `PreviousWeights(; w = w)` is a walk-forward that holds
`w`, which the reference cannot express. A weight on an asset that left the panel is still held
and its returns are zeroed as a Held Gap; what a hold does with that weight is
[#956](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/956)'s.

## Considered options

On the carrier: **a `last_ok` local beside `prev`** was folded into the decision as the same
rule with a better name — `prev` *is* the last threadable fold; **`previous_weights` walking back
over `predictions`** was rejected because the scan is `O(k)` after `k` failures and hands the
loops' whole vector down; **the failed fold's result carrying the threaded weights** was rejected
because it changes what a failed fold is, which ADR 0120 reads.

On the advance predicate: **retcode success everywhere** was rejected because under a source a
failed fold's drifted `hw.w` would then be recorded for `weight_path` and threaded to no one;
**`all(isfinite, w)`** was rejected as a value-level test where the library reads the type;
**per-member advance for a population** was rejected because `prev` stops being a fold.

On the failed fold under a drift: **a NaN record and no drift** was rejected because fold 3 would
read fold 1's end-of-window book, stale by every failed window; **keeping the throw** was rejected
because it leaves half of #1021 open. On the drift's start under a Weight Drift alone, **the
previous fold's held weights** were considered and withdrawn: the switch threads the chosen
target by design, and the mechanism already covers it.

On the failed fold's returns: **the realised return of the held book, always** was rejected
because it goes past the reference's loop default, inverts ADR 0120, and makes `res.w` and `rd.X`
disagree on one record; the reference makes a finite held-book return an opt-in fallback, and so
does this decision.

On the record: **a failed fold always storing `U`** was rejected because the rule is invisible in
the type and two folds of one scheme carry different-shaped records; **readers passing `w_prev`
themselves** was rejected because it pushes the loop's job onto every reader.

On the fallback's delivery: **a stateless sentinel plus an `optimise` keyword** was rejected
because it threads a keyword through every entry point's `fit` and is dead outside a fold loop;
**loop-level substitution under a scheme switch** was rejected because it diverges from the
reference's estimator-level fallback and skips the `fb` record. On the leaf's fields, **re-applying
the weight bounds** was rejected because a hold that is rewritten is no longer a hold, and
**reducing to the Coverage Universe** was rejected as pre-empting #956.

## Consequences

- `fold_loop`'s sequential arms take `pws` and advance `prev` through `advance_previous_fold`;
  `threads_weights`, `fold_solved` and `held_weight_members` sit beside `previous_weights`.
- `Fold` gains `w_prev`; `predict(res, rd; …)`, `fit_and_predict` and the `Pipeline`'s predict
  take `w_prev = nothing`; every fold-loop callback passes `fold.w_prev`.
- `held_start_weights` is the seam in `predict`; `held_weights_result`, `rebuild_weight_path` and
  `calc_net_returns` give `NaN` for a non-finite start vector rather than raising, member-wise.
- `HeldWeightsResult(; X, U, w0, w, wd)`: one new required keyword, and
  `assert_held_start_shape` checks `w0` against `w`.
- `PreviousWeights` is exported, with `factory(pw::PreviousWeights, w::VecNum)`,
  `needs_previous_weights`, `_optimise`, the no-fallback `optimise`, and `partial_fit!`.
- A walk-forward with a failed fold and a previous-weights term now runs to the end where it
  threw; the folds after the failure read the last solved fold's weights, or the held book under
  a source. No released number moves for a run with no failed fold.
- The `fb` record on a result was `nothing` after a chain walked, for every fallback: `optimise`
  hands `factory(res, fb)` a vector of `(estimator, result)` pairs, and every result type bound
  `fb` to `Option{<:OptE_Opt}`, so the generic identity factory ran. Found while testing the
  leaf; it was pre-existing, filed as
  [#1024](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1024), and fixed there by
  widening the bound to `Option{<:OptE_Opt_FbChain}` (ADR 0011, amendment of 2026-09-11).
