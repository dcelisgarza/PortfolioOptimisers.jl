---
status: proposed
---

# A scheme takes the online step by wrapping in `Online`, and its function constructor is the only door

## Context

[ADR 0140](0140-a-walk-forward-declares-its-fold-fit-and-the-online-arm-threads-the-estimator-from-a-cold-start.md)
gave `IndexWalkForward` and `DateWalkForward` a sixth execution switch, `ff::Option{<:AbstractFoldFit}`,
whose one member `OnlineStep()` sends `fold_loop` down its online arm. It derived `expand_train`
from that switch, refused an explicit `expand_train = false` beside it, and recorded two
alternatives it did not take: a decorator `OnlineWalkForward(cv)` over either walk-forward, and
`Online(cv)`, rejected because the word would carry a second meaning. The switch shipped in
0.31.0 and is released.
[Issue #1207](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1207) reopens the
choice: `ff = OnlineStep()` reads badly, and the maintainer wants the checks the online loop makes
decided at the type level, at compilation or at instantiation, never on a value at run time.

Six facts measured on `dev` at `ca03684ff5` shaped the decision.

1. **The construction-time checks the ticket asks for already hold.** Both walk-forwards are
   `@concrete`, so the type of `ff` is a type parameter, `OnlineStep` or `Nothing`, and every
   reader asks `isnothing(fold_fit(cv))`. Inference decides the arm. The one mismatch,
   `expand_train = false` beside an `OnlineStep()`, is refused in the inner constructor by
   `assert_fold_fit_expands` (`04_WalkForward.jl:1233`). So the ticket's stated ground does not
   separate the two shapes, and the defect is elsewhere.
2. **The defect is three-fold.** One concept has two spellings: `Online(pe)` and `Online(pipe)`
   declare the online step in the estimator tree, and the scheme spells the same declaration
   `ff = OnlineStep()`. The window knob is not the caller's to state: `expand_train` is derived
   from another knob, and an explicit `false` is refused. And `OnlineStep` is a Bool dressed as a
   type: `AbstractFoldFit` has one member, no method in `src/` or `ext/` dispatches on
   `::OnlineStep`, and the family's openness to "a member that refits every `k` folds" has no
   taker.
3. **Only one check reads a value.** The arm, `Online(pe)` under a batch scheme
   (`online_wrapper_path`, a type-decided walk), a state at entry, a schedule on a stateful field,
   and the `Resume`, search, covariance-forecast and Pipeline doors are all decided by types. The
   `expand_train` mismatch alone reads a `Bool` field, because `@concrete` parametrises on
   `typeof(true) == Bool` and not on the value. A type-level window would need a marker pair or a
   static Bool, neither of which the library carries: `Val` appears only as a method selector,
   and Static.jl is not a dependency.
4. **`Online` is one struct with one supertype.** `Online{T1, T2} <: AbstractEstimator` holds
   `est` and `max_history`. A struct has one supertype, so `Online(cv)` cannot be a
   `WalkForwardEstimator`, and eight bounds name that family: `MultipleRandomised(cv::WalkForwardEstimator)`
   three times, `cv::WFCVER` five times, and the `NonCombOptCV` alias. The library already widens
   an estimator slot to admit a wrapper by a Union alias, `Online_Option{X}`.
5. **`Online(IndexWalkForward(252, 21))` constructs today.** `CrossValidationEstimator <: AbstractEstimator`,
   so the generic `Online(est::AbstractEstimator)` accepts a scheme, and the value then fails at
   the first door with a `MethodError` on the `WFCVER` bound that names the whole type and no
   cause — the shape [#1033](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1033)
   removed from the estimator doors.
6. **A `HindsightSplit` with `prefix = true` is a nested-prefix scheme.** Fold `t` trains on
   rows `1:t` and tests on row `t` (`04_WalkForward.jl`, `split(hs::HindsightSplit, rd)`), so the
   online arm serves it as it serves an expanding walk-forward: warm up on `1:start`, fold row
   `t`, read out, score row `t`. With `prefix = false` a fold trains on row `t` alone, a window
   of one row that a fold cannot un-fold. `prefix` is therefore the split's `expand_train`. The
   gain is narrow: the usual comparators are a JuMP head, whose step is its solve
   ([ADR 0139](0139-the-online-read-out-builds-a-fresh-jump-model-every-step-and-sets-no-primal-start.md)),
   and `BestConstantRebalancedPortfolio`, whose read-out runs the fixed point over every row it
   holds (`09_OnlineOptimisation.jl:577`), so neither runs faster online.

## Decision

### The declaration is `Online`, at every level

A scheme takes the online step by being wrapped in `Online`, the one word the estimator tree
already uses. The wrapper is the existing struct: an **Online Scheme** is an
`Online{<:WalkForwardEstimator, Nothing}`, an `AbstractEstimator` whose `est` is the scheme and
whose `max_history` is `nothing`. On a scheme the wrapper is **not transient**: nothing resolves
it away, and the loop reads it at every fold. `split`, `n_splits`, `fold_evaluation`,
`folds_are_time_ordered` and `show` forward to `est`. The eight walk-forward bounds widen with one
Union alias in the shape of `Online_Option`, so a `MultipleRandomised` wraps an Online Scheme as
it wraps a plain one, `MultipleRandomised(OnlineIndexWalkForward(252, 21); subset_size = 5)`, and
the loop reads `MultipleRandomised{<:Online}` by dispatch where `fold_fit` forwarded. There is no
`OnlineMultipleRandomised`.

### The function constructors are the only door

Three exported functions build an Online Scheme, and nothing else does:

- `OnlineIndexWalkForward(train_size, test_size; purged_size, reduce_test, wd, pws, fa, store_weight_path, strict)`
- `OnlineDateWalkForward(train_size, test_size; period, period_offset, purged_size, adjuster, previous, reduce_test, wd, pws, fa, store_weight_path, strict)`
- `OnlineHindsightSplit(; start, wd, pws, fa, store_weight_path, strict)`

Each takes its scheme's keywords **minus the window knob** — `expand_train` on the two
walk-forwards, `prefix` on the split — builds the scheme with that knob `true`, and wraps it. The
rule is one sentence: **every nested-prefix scheme steps.** The window mismatch cannot be
written, so no value is read anywhere, which is what the ticket asked for. `max_history` is not a
keyword of the constructors, because the window is the estimator's to declare
([ADR 0136](0136-a-prior-folds-and-carries-the-buffer-is-owned-once-and-a-cap-is-either-a-scenario-cap-or-a-window.md)),
and a rolling window computed online stays `Online(pe; max_history = train_size - purged_size)`
on the prior.

`Online(cv)` on any `CrossValidationEstimator` **by hand is refused by name**, by a method on
that type that points at the three constructors. The method closes fact 5 in the same stroke: a
scheme never reaches the generic estimator constructor.

### The window knobs are the caller's, and the tag family goes

`expand_train::Bool = false` on `IndexWalkForward` and `DateWalkForward` is a plain keyword again,
the released default, stated by the caller and derived from nothing. `prefix` on `HindsightSplit`
is unchanged. `ff`, `resolve_expand_train`, `assert_fold_fit_expands`, `fold_fit`,
`AbstractFoldFit` and `OnlineStep` are removed. The `Fold` contract is unchanged: a `Fold` whose
`train` is `nothing` still says the estimator holds its window, and the online arm of `fold_loop`
is unchanged behind its new selector.

### The online arm announces nothing

`cv_online_info`, the message the online arm emits once at entry, is removed. Its sibling
`cv_sequential_info` stands because it reports a fact the caller did not choose: a sequential
run the loop took on its own, from `needs_previous_weights`, where the caller expected a
parallel one. An Online Scheme is chosen by name through its constructor, and a `Resume` is
declared by name in the estimator slot, so neither has anything to announce: `cv_resume_info`
goes with it, and both arms run silent.

### Every refusal is a method on the refused type

The arm, the search door, the covariance-forecast door, the Pipeline door and `Resume`'s door
read the scheme's type. Where a combination is refused — a wrapped estimator under a plain
scheme, `Resume` under a plain scheme, `Online(est)` handed to the covariance-forecast evaluation
under a plain scheme, `Online(pipe)` under a plain scheme — the refusal is a method on the
refused type that throws by name, so dispatch decides it and the message names the cause,
as issue #1033 asked. A `MethodError` from a bound stands where its text is already the
answer: a keyword the constructor does not take.

### A clean break

`ff = OnlineStep()` is removed in one commit, with no deprecation path, released with the next
minor bump, as [ADR 0044](0044-matrix-sources-are-named-not-flagged.md) retired `cle_pr`. The
package is pre-1.0 and `src/` carries no `depwarn`. A deprecation would keep the tag family alive
for a release and make `IndexWalkForward(…; ff = OnlineStep())` return an `Online`, a constructor
that returns another type, which is the shape refused below.

## Considered options

1. **Reuse the `Online` struct** — chosen. One word, one type, `isa(x, Online)` true. Costs the
   eight bound widenings in the library's own idiom and one refusal of `max_history`.
2. **A distinct `OnlineWalkForward{C} <: WalkForwardEstimator` behind an `Online(cv)` method** —
   keeps the bounds, but `show`, every message and the API page name the second type, and a
   constructor that returns a type other than its own breaks the convention that `T(…)` is a
   `T`. The two-spellings defect in a new place.
3. **Both doors: the constructors and a hand `Online(iwf)` with a value check** — keeps one
   `ArgumentError` on `iwf.expand_train` at instantiation, the one value check the ticket wants
   gone, for the sake of wrapping a scheme a caller already holds. Refused: the caller rebuilds
   through the constructor, which is one call.
4. **A type-level window marker, `Expanding()` / `Rolling()`, with the `expand_train` keyword
   kept and `Online(cv)` bound on the marker** — the only shape under which a hand `Online(cv)`
   is refused by a bound. Refused: it adds a two-member tag family for a Bool, the defect
   `OnlineStep` was, and makes `split` dispatch on it.
5. **A deprecation release** — refused above.
6. **`HindsightSplit` left batch-only** — refused: the exception's only reason is "no gain
   measured", which is a weak reason for a type-level exclusion, and the constructor costs one
   function, one identity test and one sentence. The docstring line "a `HindsightSplit`, whose
   fold reads its test row and is refit by construction" goes with it.

## Consequences

- ADR 0140 is on `main` and is amended: the sixth switch and its derived keyword are withdrawn,
  option 5 is taken with the meaning it feared given a definition instead, and option 2's
  forwarding cost is paid on the reused struct. ADRs 0141 to 0144 are history and stand. ADR
  0155, a draft, is rewritten at its one line that spells the old declaration.
- `CONTEXT.md`: **Fold Fit** is struck, **Online Scheme** is written, the *Sample Buffer*,
  *Covariance Forecast Evaluation*, *Online Portfolio Selection*, *Online Update* and *Resume*
  entries lose the old spelling, and the `Online` wrapper gains its third referent: on an
  estimator it seeds a refit buffer, on a Pipeline it declares the workflow's refit, on a scheme
  it makes the loop step.
- The build owes: the three constructors, exported; the Union alias and the eight widenings; the
  five forwarding methods; the by-name refusal of `Online(::CrossValidationEstimator)`; the
  removal of `ff`, `OnlineStep`, `AbstractFoldFit`, `fold_fit`, `resolve_expand_train` and
  `assert_fold_fit_expands` with their private-API entries; the removal of `cv_online_info` and
  `cv_resume_info`, their private-API entries, the three `@test_logs` in `test_24c` and the one
  in `test_24g` that expect them; the dispatch that replaces every
  `isnothing(fold_fit(cv))` read (`01_Base_CrossValidation.jl`, `05_MultipleRandomised.jl`,
  `09_Base_SearchCrossValidation.jl`, `13_CovarianceForecastEvaluation.jl`, `15_Resume.jl`,
  `21_Pipeline/06_OnlinePipeline.jl`, `01_Base/15_Online.jl`); the `Online` docstring's paragraph
  on a scheme; the `expand_train` and `ff` rows of `field_dict`; the two `show` doctests; the 12
  example and user-guide call sites; the 11 test files; and one identity test of
  `OnlineHindsightSplit` against the batch prefix split. The two identities of ADR 0140 are the
  contract still, with `OnlineIndexWalkForward(w, t; purged_size = p)` on the online side.
- `OnlineHindsightSplit` is recorded as parity by rule and no speed gain on the usual
  comparators; a timing assertion on it is not owed.
- No released number moves: a bare `IndexWalkForward(252, 21)` resolves `expand_train` to `false`
  as before, and the batch arms are untouched.
