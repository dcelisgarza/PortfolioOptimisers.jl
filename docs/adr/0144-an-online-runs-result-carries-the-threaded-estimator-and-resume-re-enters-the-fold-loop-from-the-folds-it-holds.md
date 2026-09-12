---
status: accepted
---

# An online run's Result carries the threaded estimator, and `Resume` re-enters the fold loop from the folds it holds

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) makes every layer
above the moments take the online step. The fold loop takes it
([ADR 0140](0140-a-walk-forward-declares-its-fold-fit-and-the-online-arm-threads-the-estimator-from-a-cold-start.md),
built by [#969](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/969)): a walk-forward
with `ff = OnlineStep()` warms up once, folds each fold's new rows into one threaded estimator, and
reads it out where a refit would have run. ADR 0140 ruled two things that made this decision
sharp. The loop **starts cold** — an estimator carrying a state at entry is refused by name,
because folding a warm-up on top of a carried state double-counts rows in silence and the pinned
context, which checks names and shape, does not trip. And the loop **returns no estimator** — the
result is the batch `MultiPeriodPredictionResult`, and the threaded estimator is a local of
`online_folds` (`02_CrossValidation/12_OnlineFoldLoop.jl`) that escapes only as `Fold.est` to a
caller's own callback. So when new rows arrive, the online walk-forward, whose whole point is not
refitting from zero, must itself rerun from zero over the longer history.
[#1018](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1018) decided how a state
leaves a run and re-enters one.

Five facts decided it, measured on `dev` at `2dcfa371a3`.

1. **What the state has folded at loop end.** `online_folds` folds through
   `last(train_idx[end])` — the last *training* end, not the end of the data — and nothing
   records which row that was.
2. **What fold `n + 1` of a longer run reads.** Beside the state: the previous fold's weights and
   held weights (`online_folds` passes `predictions[i - 1]` to `fit_fold`, and a turnover term or
   a Weight Drift reads them), the fold index and count for a schedule (`TimeDependentContext(i,
   n, …)`), and the row the state stopped at, to align the carrier. None of the first three is on
   the estimator; all are on, or derivable from, the `MultiPeriodPredictionResult`.
3. **A capped state cannot locate its own row.** `SampleBufferState` and `ReturnsBufferState`
   hold `n` held rows and no total, so under `Online(pe; max_history = w)` the count says `w`,
   not the row folded through. The buffer's timestamps survive the cap, trimmed from the front:
   `ReturnsBufferState.ts` holds the last `h` folded timestamps.
4. **A Result in the estimator slot already means something else.**
   `cross_val_predict(res::OptimisationResult, rd, cv)` and the three
   `fit_and_predict(res, rd, cv)` methods predict every fold with a *fixed* result and fit nothing
   (`06_Validation.jl:38`, `04_WalkForward.jl:960`).
5. **The reference.** skfolio's `online_predict` and `online_score` clone the estimator before the
   loop (`model_selection/_online/_validation.py:188`, `:389`) and return the portfolio alone, so
   the state is discarded and nothing resumes. Only `OnlineGridSearch` keeps a fitted estimator,
   `best_estimator_`, folded through the end of the last test window by `refit_last=True`
   (`_search.py:979`, `_validation.py:568`); it serves `predict(X_new)` and records no row and
   checks no continuity. In its batch loop the question does not exist: a stateless loop reruns
   from zero and a once-partial fold comes out full, recomputed. A caller of the reference cannot
   continue an evaluation, cannot learn where the state stopped, and folds a gap in silence.

Every state is an immutable `@concrete struct` of numeric and Boolean matrices, name vectors, a
`Set{Int}`, an `AssetPanel`, an integer cap and a timestamp vector — thirteen types, none holding a
JuMP model (ADR 0139 keeps the model on the result), a closure, an RNG or a `Task`. The stdlib
`Serialization` round-trips them as they stand. Nothing in `src/`, `test/` or `docs/` serialises
anything today.

## Decision

**A resume continues the evaluation.** Its oracle is the one-shot run: for an online scheme `cv`,
a history `rd_T` and its extension `rd_{T+k}`, `vcat(old.pred, new.pred)` equals
`cross_val_predict(mr, rd_{T+k}, cv).pred` fold for fold, at ADR 0137's tolerances for the
weights and exactly for the carrier the read-out rebuilds — a capped run included. Deployment of
a resumed state is one hand step, `partial_fit!(res.opt, rows)`; the library offers no
`refit_last`, because a state folded through purge and test rows equals no fold of any run and
cannot be unfolded, while from the last training end the caller reaches the end of the data in
one call.

**`MultiPeriodPredictionResult` carries the threaded estimator**, in `opt`, folded through
`last(train_idx[end])`; `nothing` on a batch run. The field is named for the type it holds, as
`SearchCrossValidationResult.opt` is. The resumed Result carries the **new folds only** and `opt`
again, so the chain continues; `vcat` on two Results stacks them for scoring.

**`Resume(res)` is the entry**, a transient declaration in the estimator slot in the idiom of
`Online` and `TimeDependent`: it resolves at the door to a **copy** of `res.opt`, to
`res.pred[end]` for the previous and held weights, and to the folds to skip, `n_old`, read off
the state — the fold whose training window ends at the state's last held timestamp — and is
gone before any verb below the loop meets it. The count is not `length(res.pred)`: a resumed
Result holds the new folds only, so the chain `Resume(res2)` must skip every fold before it, and
the timestamp names that fold where the count cannot. `res` is never written, because
`partial_fit!` promises nothing about a kept estimator and every state answers `Base.copy`, so
one Result resumes any number of times. The ordinary argument stays refused unchanged, and the
fixed-result reading of the slot is untouched.

**The carrier is the full history extended.** `split(cv, rd)` enumerates every fold as the
one-shot run does; the first `n_old` are skipped; the resumed loop's first act is the
ordinary delta `(last(train_idx[n_old]) + 1):last(train_idx[n_old + 1])`. No warm-up exists
under a resume — the warm-up was fold 1's window, and fold 1 is skipped — so `purged_size <
train_size` is untouched, a `DateWalkForward` anchors its periods on the whole `ts` for free, and
a schedule's `i` and `n` are the combined split's.

**What the re-entry checks**, each an `ArgumentError` by name:

- a batch Result (`res.opt === nothing`), a `PopulationPredictionResult`, a scheme that is not a
  walk-forward, and a scheme with `fold_fit(cv) === nothing`;
- a carrier that adds no fold;
- **timestamps are required.** The carrier's `ts` and the state's `ReturnsBufferState.ts` must
  both be present, and every held timestamp must equal its row of the carrier:
  `state.ts == rd.ts[(r - h + 1):r]` for `r = last(train_idx[n_old])` and `h` held. That is exact
  over the held span at `O(h)`, and it is the one check that pins a prefix — a row dropped or
  inserted before `r` moves `rd.ts[r]`, a changed scheme moves `r`, and a count check catches
  neither under a cap and only the second without one. A `ReturnsResult` binds `ts` to a
  vector of `Dates.AbstractTime`, so an index-only caller attaches a synthetic calendar,
  `ts = Date(1) .+ Day.(0:(T - 1))`, rather than `1:T`.
- **a partial last fold is terminal.** With `reduce_test = true` the old run's last fold is the
  first half of a full window of the longer run — the same training end, the same weights, too
  short a span — so skipping it loses rows and completing it needs a mid-window entry in `predict`
  and the fee clock. The check is `length(test_idx_new[n_old]) == nobs(res.pred[end].rd)`, which
  fails exactly when the window was partial. The message names the workaround: resume the same
  Result **terminally** with `reduce_test = true` for the live view of the leftover rows, and
  resume it again with `reduce_test = false` when rows arrive. The resumed run may itself end
  partial. This is the reference's own reading made explicit: after `refit_last` its state is a
  deployment object, never resumed.
- the pinned context — names, static panel, column presence — on every delta step, through
  `partial_fit!` on the `ReturnsBufferState` as before.

**Serialisation is a fact, not a mechanism.** A `MultiPeriodPredictionResult` of an online run
round-trips through the stdlib `Serialization` as it stands, pinned by one test; the format is
Julia-version-bound, and no other format is provided. No dependency.

## Considered options

**The contract.**

1. **Continue the evaluation** — chosen. The only shape with an oracle, and it subsumes the
   other two: from the last training end the caller reaches the end of the data in one hand
   step, but from the end of the data nobody reaches the last training end.
2. **Only the estimator leaves, no re-entry** — the Result carries `opt` and nothing continues.
   Rejected: nothing pins the fold-for-fold identity, and continuing an evaluation means
   rewriting the schedule, the purge, the previous-weights thread and the held-weights logic by
   hand, which is what the loop exists to own.
3. **The reference's `refit_last`** — the estimator leaves folded through the last test window.
   Rejected: it equals no fold of any run and cannot be unfolded, so the resume is unreachable
   from it, and the flag is a second exit beside the first.

**What re-enters.** The **Result** — chosen; every input of fold `n + 1` is on it. The
**estimator alone** — rejected: a `MeanRisk` with a turnover term reads `nothing` as the previous
weights where the one-shot fold reads fold `n`'s, so the identity breaks for every optimiser
`needs_previous_weights` is true for, and a capped state cannot locate its row.

**The carrier.** The **full history** — chosen. A **tail** — rejected: the Index form can rebuild
`T` from `cv` and the fold count, but a tail that starts one row late is undetectable without a
calendar, and the Date form must re-anchor from timestamps the loop no longer holds. **Both** —
rejected: two alignment rules where one carries the identity, and the tail rule's blindness ships
beside the exact one.

**The entry.** **`Resume(res)`** — chosen. The **bare Result in the slot**, mirroring
`optimise(opt::OptimisationResult, …)` — rejected: the slot would read "predict with these fixed
weights" for one Result type and "continue this run" for another, and a batch
`MultiPeriodPredictionResult` in it has no honest meaning. A **keyword `resume = res`** beside the
configuration — rejected: two sources of the estimator that can disagree, and a keyword threaded
through every door. A **Fold Fit member** carrying the Result — out before the vote: a Fold Fit
carries no data, and a cross-validation scheme is an Estimator, which holds no Result.

**Alignment.** **Timestamps required** — chosen. **Timestamps when present, count otherwise** —
rejected: the count catches a changed scheme and nothing else, and under a cap not that; the
refusal's strength would depend on a keyword the caller may not know matters. **Timestamps and
held values** — rejected: a carry prior holds rows up to `max_scenarios` and an exact-fold moment
state holds none, so the check fires for some estimators and not others, and giving every state
rows to compare is the second copy of `X` that #704 refused.

**The partial last fold.** **Refuse, two resumes** — chosen. **Complete the partial fold from
its stored result** — rejected here: exact on rows, but `pred` then has one more fold than the
one-shot run, so the identity moves from folds to rows, and it needs a continuation entry in
`predict` that starts from the held weights and charges no fixed fee again — prediction-layer
work with a use of its own if a caller ever wants it. **Re-predict and replace** — rejected: the
two Results overlap by a fold the caller must drop, and a naive `vcat` double-counts it.

## Consequences

- `MultiPeriodPredictionResult` gains a field; no `show` doctest renders one, so none moves; no
  released number moves, because a batch run writes `nothing`.
- `fold_loop` gains a resumed arm beside the online one, sharing its per-fold body; `Resume`
  joins `Online` and `TimeDependent` as the third transient declaration, refused at every door
  that is not an online walk-forward. `Base.vcat` on two Results stacks them.
- The two-resume rule makes a Result a value: `Resume` copies the state, so resuming twice gives
  the same answer, and a terminal view never spoils a later continuation.
- A `MultipleRandomised` run (`PopulationPredictionResult`) is refused: whether a longer history
  reproduces each path's asset subset is unmeasured, and the map carries a resumed population as
  fog. A resumed search is fog too. A Pipeline host follows ADR 0142, because the entry walks the
  host generically. A failed last fold threads `NaN` weights as the loop does today, which is
  [#1021](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1021)'s.
- The build is [#1025](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1025).
- `CONTEXT.md` gains **Resume**, and **Fold Fit** says the loop starts cold *or resumes*.
