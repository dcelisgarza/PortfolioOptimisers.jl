---
status: accepted
---

# A walk-forward declares its Fold Fit, and the online arm threads the estimator from a cold start

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) makes every layer
above the moments take the online step, and closes on one test: an online walk-forward with a
warm-up reaches the same weights as the batch expanding-window walk-forward. The optimiser's step
is built — `partial_fit!(opt, rd)` folds a carrier into the prior alone, and `optimise(opt)` reads
the state out through the ordinary batch path
([ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)).
What did not exist was the loop that takes the step fold after fold.
[#870](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/870) decided its shape.

The batch loop is one seam, `fold_loop`
([ADR 0067](0067-the-cross-validation-fold-loop-is-one-seam.md)). Per fold it takes the fold's view
of the estimator and the data, resolves every `TimeDependent` schedule against the fold's context,
threads the previous fold's weights through `factory`, and hands the callback one `Fold`. It has
two arms, chosen from types alone: `run_folds` when the scheme is a timeline and the estimator
needs the previous weights, `parallel_folds` otherwise. Every fold refits from its training window,
so under an expanding walk-forward fold `i` re-reads rows `1 : t_i` that fold `i - 1` already read.

Five facts shaped the decision.

1. **The walk-forwards already carry execution switches.** `wd`, `pws`, `fa`,
   `store_weight_path` and `strict` sit on `IndexWalkForward` and `DateWalkForward` beside the
   fields that make the split, each `nothing` by default meaning the released behaviour, and one
   per-type verb `fold_evaluation(cv)` reads them. "How a fold is fitted" is a switch of that kind.
2. **Both walk-forwards are `@concrete`**, so a field's type is a type parameter of the scheme's
   type. A branch on `isnothing(cv.ff)` is decided by inference and the dead arm is eliminated —
   measured for a tag-typed field on a `@concrete` struct and for a wrapping type, with identical
   return types on both. The type-stability argument does not separate the two shapes.
3. **The window is the estimator's, not the loop's.** A fold cannot un-fold an observation, so an
   online run is expanding by construction. A rolling window computed online is the composition
   [ADR 0136](0136-a-prior-folds-and-carries-the-buffer-is-owned-once-and-a-cap-is-either-a-scenario-cap-or-a-window.md)
   names: `Online(pe; max_history = w)` caps the prior's buffer, and the read-out is a batch fit over
   the window. The loop carries no window, no buffer mode and no `expand_train` of its own.
4. **A schedule and a state pull opposite ways on one slot.** A `TimeDependent` schedule replaces
   its field's value every fold; a state is threaded *through* that value. The comment on #870 that
   [#967](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/967) left measured it: a
   callable schedule returning a stepped estimator hands back the state as it stood when the closure
   was built. The step already refuses a schedule on `pe` and on a head's `opt` (ADR 0137), but its
   messages promise that "a fold loop resolves the schedule before it steps" — a promise that cannot
   be kept for a field that carries a state, because the resolved value never saw the earlier rows.
5. **The reference's online loop is a function, not a scheme.** It takes the walk-forward's own
   keywords, builds an expanding walk-forward inside, clones the estimator so every run starts cold,
   folds the delta rows between folds, threads the previous weights, and returns the same result as
   batch. A caller of the reference cannot hand the online loop to the batch grid search as a
   scheme, cannot run it through a date-based walk-forward as the same object, and cannot resume.

## Decision

**A walk-forward declares its Fold Fit.** `IndexWalkForward` and `DateWalkForward` gain
`ff::Option{<:AbstractFoldFit} = nothing`, a sixth execution switch beside the five, read by
`fold_fit(cv)` beside `fold_evaluation(cv)`. `nothing` refits every fold from its training window,
which is the released behaviour. `OnlineStep()` makes the loop fold the delta rows into one
threaded estimator and read it out. `fold_fit(::Any)` answers `nothing`, so a split result and
every other scheme reach the batch arms; `fold_fit(cv::MultipleRandomised)` forwards to the
walk-forward it wraps, as `fold_evaluation` does. `AbstractFoldFit` is a selector-tag family, open
to a member that a caller's subtype defines.

**`expand_train` derives from the Fold Fit.** The constructor keyword takes `nothing` by default
and resolves to `!isnothing(ff)`, so the struct field stays a `Bool`, a bare call resolves to
`false` as released, and `IndexWalkForward(252, 21; ff = OnlineStep())` is one statement. An
explicit `expand_train = false` beside `OnlineStep()` is refused by name, pointing at
`Online(pe; max_history = train_size - purged_size)` for a rolling window computed online.

**The online arm is the third arm of `fold_loop`, taken first.** It warms up once — resolves every
`Online` through `update_online_estimator`, folds the first training window `train_idx[1]` — and
then per fold folds rows `(last_end + 1) : last(train_idx[i])` into the threaded estimator `est`,
makes the per-fold copy `esti` exactly as the batch arms do (schedules resolved against the fold's
context, previous weights threaded through `factory`), and hands the callback
`Fold(i, n, esti, rdi, nothing, test_idx[i])`. A `Fold` whose `train` is `nothing` says *the
estimator holds its window*, and the per-fold `fit_and_predict(opt, rd; train_idx = nothing,
test_idx)` reads it out through `optimise(opt)`. Every entry point's callback is therefore
unchanged, and the arm emits `cv_online_info()` as the sequential arm emits its own. Under a
multiple-randomised path the asset view is taken once at warm-up, because a path is one asset
subset crossed with the walk-forward's folds, and the sliced estimator is threaded
([ADR 0107](0107-the-update-seam-has-two-verbs-and-a-view-slices-by-asset-and-drops-by-observation.md):
a view slices a state by asset).

**The loop starts cold.** An estimator that carries a state when it enters the online arm is
refused by name, before any solve, by one predicate that walks the tree as
`update_online_estimator` does. The loop's argument is the configuration and nothing else, which is
the reading the batch loop already has — `factory` carries a state and `prior(pe, X)` never reads
it. A resume — a state that leaves a result and re-enters a loop — is its own ticket on the map,
with an explicit entry.

**A schedule reaches stateless fields only.** Under the online arm, a `TimeDependent` on a field
that carries a state — `pe`, or a JuMP or hierarchical head's `opt` — is refused at warm-up, before
the first fold. A schedule on any other field composes with no rule, because the schedule writes
the per-fold copy and the state threads through the unresolved estimator, so the two never touch
one slot. The composition that works is the one #967 pinned: an `Online` inside an estimator that
holds schedules, and a host with a wrapper in one field and a schedule in another.

**The return is unchanged.** The online arm returns a `MultiPeriodPredictionResult`, so the scoring
and the plots read it as they read a batch run. No estimator is returned.

**Two identities are the contract**, fold for fold, to the tolerances ADR 0137 measured for the
weights and exactly for the carrier the read-out rebuilds.

- Expanding: `IndexWalkForward(w, t; purged_size = p, expand_train = true)` equals
  `IndexWalkForward(w, t; purged_size = p, ff = OnlineStep())`. This is the map's closing test.
- Capped: `IndexWalkForward(w + p, t; purged_size = p)`, the rolling scheme, equals
  `IndexWalkForward(w + p, t; purged_size = p, ff = OnlineStep())` with `Online(pe; max_history =
  w)` on the prior. The rolling window is `train_size - purged_size` rows, so the cap is `w` and the
  warm-up `w + p`.

The purge holds in both: fold `i` steps rows up to `test_start_i - p`, and the purged rows are
folded by a later fold, never dropped — exactly the rows the batch expanding fold reads. The fold
count is known before the loop runs, because `split(cv, rd)` enumerates every fold first, online as
in batch, so `assert_time_dependent_fold_count` runs unchanged and a short vector schedule is
refused as today.

## Considered options

1. **A Fold Fit switch on the two walk-forwards** — chosen. The sixth switch in the shape of the
   five; the date form gets the online loop for free; the closing identity differs by one keyword;
   the family is open to a future member that refits every `k` folds; type-level under `@concrete`.
   It is also the design from zero: window (rolling or expanding) and fit (refit or step) are two
   axes, and this keeps them two fields.
2. **`OnlineWalkForward(cv)`, a decorator over either walk-forward** — the alternative to switch to
   if the switch proves the wrong home. One type serves both forms and the two released types are
   untouched; `GridSearchCrossValidation{<:Any, <:OnlineWalkForward}` is a dispatch target for the
   search that mirrors the combinatorial method. Not chosen because it would be the first type in
   the library that wraps a scheme, and it forwards `split`, `n_splits`, `fold_evaluation`,
   `folds_are_time_ordered` and `show` today and one more method for every switch the walk-forwards
   ever gain. Inference is identical to option 1, measured.
3. **`OnlineWalkForward(warmup_size, test_size; …)`, a sibling scheme in the reference's
   vocabulary** — the cleanest call and no dead knob. Rejected because the date form needs an
   `OnlineDateWalkForward` twin duplicating thirteen fields, or is lost — a capability the reference
   has through `freq`.
4. **`online_predict(opt, rd; warmup_size, …)`, the reference's function** — rejected: it cannot
   be the `cv` field of a search or of a `Pipeline`'s `cross_val_predict`, so the search would need
   an `OnlineGridSearch` of its own as the reference has, and a second loop would sit beside
   `fold_loop`, undoing ADR 0067.
5. **Reusing `Online(cv)`** — rejected: the word would carry a second meaning, "run the loop by
   stepping" beside "seed a refit buffer", which is what the glossary's *Avoid* lines exist to
   prevent.
6. **An estimator-declared route with no scheme change** — rejected: a bare `EmpiricalPrior()`
   steps with no wrapper, so a caller could not ask for the online path without one, and the
   scheme's window would go unchecked.

On the entry state, **reset** (the reference's clone) was costed and not chosen: after `Online`
resolves, the cap lives only in the state, so a reset must empty a state and keep its cap — one
verb per state type, fourteen today. **Use** — fold the warm-up on top of what the estimator holds
— was rejected on a worked case: an estimator hand-stepped over rows `1 : 100` and then handed a
warm-up of `252` reads out over rows `1 : 100` twice, and the pinned context, which checks names
and shape alone, does not trip.

On schedules, **carrying the state across a swap** when the incoming value is "the same estimator"
needs a definition of sameness the library does not have, and a genuine switch still lands cold;
**transplanting a `SampleBufferState` alone** helps only a member under `Online`, so one schedule
behaves three ways by which member it lands on. Both are recorded as a possible future extension
on the map, and neither is built.

## Consequences

- `IndexWalkForward` and `DateWalkForward` gain a field and a derived keyword, and their `show`
  doctests gain one line. No released number moves: a bare call resolves `expand_train` to `false`
  and `ff` to `nothing`.
- `fold_loop` gains one arm and one predicate. The `Fold` contract gains one meaning: `train ===
  nothing` says the estimator holds its window. The per-fold `fit_and_predict` gains the read-out
  arm, which is also a public entry for a hand-stepped estimator:
  `fit_and_predict(opt, rd; test_idx)`.
- ADR 0137's two refusal messages are corrected: a schedule on a stateful field is refused, in the
  step and in the loop, and no loop resolves it before stepping.
- The search ([#871](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/871)) scores
  each fold of a contiguous scheme independently today and never reaches `fold_loop`, so it must
  restructure its inner loop for a stepping candidate; it dispatches on `fold_fit(gscv.cv)`
  through a function barrier rather than on a new scheme type. The Pipeline is
  [#872](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/872)'s.
- The build is [#969](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/969). The
  resume, and the schedule-swap extension, are fog on the map until their own tickets.
- `CONTEXT.md` gains **Fold Fit**.
