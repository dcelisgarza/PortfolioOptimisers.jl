---
status: accepted
---

# Time-dependent constraints as in-field wrappers consumed by CV loops

## Context

Cross-validation backtests re-optimise the same estimator over a sequence of time-ordered folds,
but every constraint field is fixed for the whole horizon. The hooks for per-period updates
(`is_time_dependent`, `update_time_dependent_estimator`, `needs_previous_weights` in
[01_Base_Optimisation.jl](../../src/20_Optimisation/01_Base_Optimisation.jl)) existed as
scaffolding with no concrete overload — nothing in the library actually varied a constraint over
time. The naive approach of "put a vector in the field" is unrepresentable: several fields
already accept vectors with field-level meaning (`smtx`/`sgmtx` = multiple subset spaces,
`ple` = multiple phylogeny constraints), so "vector = per-fold" is ambiguous, and any side
channel naming its target with a symbol re-encodes the field list and invites typos.

## Decision

**A `TimeDependent(val)` wrapper stored directly in the optimiser field it varies —
`JuMPOptimiser(; lt = TimeDependent([...]))` — consumed by whichever cross-validation fold loop
processes the estimator, and replaced by the field's static default everywhere else.**

- The field's position names the target: a field holds either a static value or a per-fold
  schedule, never both, so double-specification and target typos are unrepresentable — a wrong
  keyword is an ordinary construction error. `val` is either a vector — entry `i` is the
  complete field value for fold `i`, so a field that itself accepts a *vector of constraints*
  carries a per-fold vector of vectors (`TimeDependent([[c₁ᵃ, c₁ᵇ], [c₂ᵃ], …])`) — or a callable
  `f(ctx::TimeDependentContext)` evaluated per fold. There is no separate "vector of
  `TimeDependent`" facility and none is needed: because the wrapper is recognised only at a
  top-level field, varying individual entries within a constraint vector is done by assembling
  the fold's vector in a callable (`TimeDependent(ctx -> [dynamic(ctx), static])`), which also
  keeps shared static parts in one place. Callables may be structs subtyping
  `TimeDependentCallable <: AbstractAlgorithm` (a functor over the context): being types, they
  can define `needs_previous_weights` to declare previous-weights requirements directly — the
  `PreviousWeightsFunction` wrapper exists only for the uninspectable bare-function case.
- **The widened signatures are the admissibility table**: the constructor signatures using the
  `TD_Option{X} = Union{Nothing, <:TimeDependent, X}` alias (most `JuMPOptimiser` constraint
  fields; `wb`/`fees` on `HierarchicalOptimiser` and on the meta-optimisers) are the single
  source of truth for which inputs may vary over folds. Discovery is a generic `fieldnames`
  scan for `TimeDependent`-valued fields (`time_dependent_fields`) — no hand-maintained list.
  `TimeDependent` is recognised at top-level optimiser fields only, never nested inside another
  input.
- **Enumeration-order indexing, no hidden ranking**: entry `i` corresponds to fold `i` of the
  consuming scheme's `split` enumeration — the machinery never re-ranks folds. `ctx.i` therefore
  always indexes `ctx.train_idx`/`ctx.test_idx`, so a callable can identify its own fold's
  windows under every scheme. Walk-forward and (unshuffled) KFold enumerate chronologically;
  for schemes whose enumeration is not a timeline (combinatorial splits, randomised paths) it
  is the *user's* responsibility to key entries off the fold's indices. One shared vector
  serves all paths; its length must equal folds-per-path, validated immediately after
  `split(cv, rd)` before any fold runs.
- **Swap-in is a constructor rebuild**: the fold loop rebuilds the host through its validated
  keyword constructor with each `TimeDependent`-valued field replaced by its per-fold value, so
  every existing invariant re-runs each fold (including on function outputs) and the per-fold
  optimiser is an ordinary static estimator. At construction time every vector entry is
  test-substituted through the same constructor (with the other time-dependent fields at their
  static defaults), surfacing type and cross-field errors immediately; eager validation
  coupling a time-dependent field to static fields is deferred to the per-fold rebuild.
- **Parallel-safe**: a `TimeDependent` field alone never forces sequential fold execution;
  sequentiality remains tied to `needs_previous_weights`, which inspects vector-entry contents
  recursively. A function form that needs previous weights declares it by wrapping the callable
  in `PreviousWeightsFunction(f)` (contributes `true` to the trait); `ctx.w_prev` is populated
  only in sequential runs. The fold loop applies the time-dependent swap *first* and
  `factory(·, prev.w)` *second* (order swapped from the original scaffolding), so per-fold
  constraint values also receive previous weights.
- `TimeDependentContext` carries `i`, `n` (folds in path), the possibly asset-viewed
  `ReturnsResult`, `train_idx`, `test_idx`, `w_prev::Option`, `path_id::Option`;
  `update_time_dependent_estimator(opt, ctx)` replaces the ad-hoc
  `(opt, i, rd, train_idx, test_idx)` argument list. Views need no per-field machinery:
  `TimeDependent` maps `port_opt_view` over its stored values exactly as host fields already do,
  and function forms see the viewed universe through `ctx.rd`.
- **Inert outside fold loops, resolved in `_optimise`**: a time-dependent constraint is defined
  *only* over folds, so each estimator's `_optimise` first passes itself through
  `reset_time_dependent_estimator`, which rebuilds hosts with every `TimeDependent`-valued
  field at its static default (`time_dependent_field_defaults`: `wb → WeightBounds()`,
  `bgt → 1.0`, otherwise `nothing`) and recurses through wrapper optimisers. Placing the seam
  in `_optimise` covers the fallback chain, the `fb = Nothing` fast paths, and meta-optimisers'
  fold-less full-window solves uniformly; per-fold estimators produced by the swap contain no
  `TimeDependent` values and pass through at the cost of one field scan. Meta-optimisers reset
  *only their own fields*: their inner schedules must survive the seam to be consumed by the
  meta's inner cross-validation leg, while the meta's fold-less full-window inner solves reset
  themselves at their own `_optimise`.
- **Meta-optimisers compose, never distribute**: NestedClustered, Stacking, and
  SubsetResampling accept `TimeDependent` in their *own* `wb`/`fees` (the post-combination
  inputs) and forward `is_time_dependent` / `update_time_dependent_estimator` / the fold-count
  assertion into their inner and outer estimators like every other wrapper — an outer fold loop
  over a meta resolves both the meta's own schedules and the inner ones against the *outer*
  folds (swap-in produces static estimators, so inner legs see ordinary per-fold optimisers). A
  meta-level schedule is never pushed down into inner estimators: sharing one schedule across
  hosts is passing the same `TimeDependent` object into each host's own field, and a host without
  the field has no keyword to put it in. Because swap-in and reset build new structs, the
  original estimator is never mutated and one `TimeDependent` object may be safely shared between
  hosts (e.g. a primary and its fallback).
- **`bind` chooses the consuming fold loop**: `TimeDependent(val, bind::Symbol = :outermost)`.
  The default `:outermost` is the composition rule above — the outermost fold loop consumes the
  schedule, so an inner estimator's schedule under an outer backtest is resolved against the
  *outer* folds. `:nearest` binds the schedule to the *nearest enclosing* fold loop instead:
  inside a meta-optimiser's inner estimators that is the meta's own cross-validation leg, which
  consumes the schedule even when the meta is backtested under an outer loop (sizing it to the
  inner folds). The mechanism is a positional `all_binds::Bool = true` flag threaded through the
  scan/update/assert recursion: it is a property of the *recursion position*, not of the
  schedule, and cannot be read off `bind` — the same `:nearest` field is *visited* by both the
  outer loop (recursing through the meta) and the meta's inner loop, and only their position
  distinguishes which should consume it. Every ordinary (outermost/standalone) fold loop calls
  with `true` (it is both outermost and nearest, so it takes everything remaining, `:nearest`
  included); a meta forces `false` only into the estimators its own inner cross-validation owns
  (`Stacking`/`NestedClustered` `opti`), leaving their `:nearest` schedules for that inner loop.
  `needs_previous_weights` is deliberately left scope-blind (it never consults `bind`): declaring
  sequentiality conservatively is always safe, and threading the flag through the trait would buy
  nothing.
- **Pipelines: swap, then inject — with the swap outside `fit` entirely.** The `Pipeline` fold
  loop (`cross_val_predict(pipe, data::Prices_RR, cv)`) splits the *input* at its own level
  (contiguous windows only: combinatorial and asset-resampling schemes are rejected — a fitted
  workflow cannot replay non-contiguous windows and cannot take an asset view) and resolves
  time-dependent steps per fold *before* `fit` runs on the training window, so injection
  (`inject_context`/`maybe_inject_step`) only ever sees a plain, already-resolved optimiser and
  `fit`/`run_step` never learn about folds. A schedule may *be* the optimisation step: the
  statically-typed forms (a vector of optimisers/results, a declared
  `TimeDependentOptimiserCallable`) classify as optimisation steps directly
  (`pipe_writes = :opt`, `pipe_reads = (:returns,)`), while a bare `ctx -> optimiser` enters via
  `PipelineStep(; est = td, writes = :opt)` with its output checked at the swap. A mixed
  schedule's result entries run predict-only folds — `run_step` writes the result to the `opt`
  slot as-is — and reuse the non-injectable pattern: computed `prior`/`phylogeny` slots pass by,
  populated `uncertainty`/`constraints` slots fail closed. `TimeDependentContext.rd` widens from
  `ReturnsResult` to `Prices_RR`: the pipeline loop passes the raw input data, so pipeline-level
  callables see the fold's data *before* any preprocessing step has transformed it. Schedules of
  non-optimiser families stay un-steppable (rejected at `PipelineStep` construction) — a
  per-fold prior/constraint is spelled as a `TimeDependent` *field* of the optimisation step,
  keeping one spelling per thing. `search_cross_validation` resolves candidate schedules against
  its *tuning* folds (tuning fold `j` runs entry `j`, fold counts asserted per candidate, lenses
  need no schedule semantics), and the fold-less `fit` resolves schedule steps to their explicit
  `default` or throws a `TimeDependentDefaultError` pointing at `cross_val_predict`. When the
  pipeline `needs_previous_weights`, the fold loop runs sequentially and threads the previous
  fold's weights both into `ctx.w_prev` and — post-swap, so freshly swapped-in optimisers receive
  them — into the optimisation steps via `factory`.

## Considered options

- **A `tdc` side-channel field holding `TimeDependent(val, field::Symbol)` wrappers with the
  target named by symbol.** Rejected: the symbol re-encodes the field list (typos need
  did-you-mean machinery, targets need uniqueness and "sole source" checks against constructor
  defaults), sharing across hosts with different field sets needs per-entry leniency flags, and
  the wrapper needs a target-inference trait for ambiguous value types (`Threshold` → `:lt` or
  `:st`). Storing the wrapper in the field it varies makes all of that unrepresentable at the
  cost of widening the constructor signatures — which is the admissibility declaration anyway.
- **A `needs_prev` flag on the wrapper.** Rejected: previous-weight need is already discoverable
  from constraint contents via the recursive trait; a flag would duplicate that channel for
  vector entries. `PreviousWeightsFunction` covers the one uninspectable case (callables) as
  data rather than as a boolean field.
- **Machinery-imposed chronological ranking of folds.** An earlier draft ranked folds by their
  test window's position in time (`invperm(sortperm(first.(test_idx)))`) so entry `i` always
  meant "the `i`-th test window in time". Rejected: the ranking is a hidden policy the user
  never stated, and under combinatorial schemes any single "position" of a split (whose test
  set is a union of disjoint groups) is arbitrary. Ranking by the *training* windows instead is
  worse — under KFold the training set is the test slice's complement, so train-based ordering
  is exactly the reverse of test-based ordering, inverting meaning across schemes. Note the
  test-position rank is *not* a data leak (fold boundaries are scheme metadata fixed before any
  returns are observed, and `sort_predictions!` only orders outputs for reporting); it was
  dropped for its implicitness, not for leakage. The enumeration-order contract puts the keying
  decision where the knowledge is — with the user, who has the fold's indices in the context.
- **Direct field replacement (`@set`) plus a hand-maintained symbol→type admissibility table.**
  Rejected: the table re-encodes constructor invariants and drifts; the constructor rebuild
  makes the existing validation the single source of truth at negligible per-fold cost.
- **A per-host targetable-fields trait instead of the generic field scan.** Rejected: it is the
  hand-maintained list again; the scan is O(fieldcount) once per `split`/`_optimise` and reads
  its truth from the same signatures the user reads.
- **Fail-closed guard (`optimise` errors on time-dependent fields) with an explicit strip step
  for meta-optimisers.** Rejected in favour of reset-to-default semantics: the guard would
  break Stacking/NCO, whose full-window solves of inner estimators are fold-less by design, and
  a `1.0`-expecting consumer cannot shrug off a schedule, so the defaults substitution has to
  exist anyway — one seam in `_optimise` serves every path.
- **Per-path schedules.** Rejected: paths are an artifact of the CV scheme, not something a
  user can author constraints against.

## Consequences

- `update_time_dependent_estimator`'s documented signature changes to `(opt, ctx)`, and the
  WalkForward / MultipleRandomised loops change behaviour: the update applies from fold 1 (the
  `prev !== nothing` guard no longer gates it) and runs before the previous-weights factory.
  The KFold/non-sequential path gains the swap-in hook it never had.
- `is_time_dependent` becomes derived (any field holds a `TimeDependent`, recursing through
  wrapper optimisers) and no longer implies sequential execution on its own.
- A time-dependent estimator passed to a fold-less `optimise` runs at the affected fields'
  static defaults with no message — deliberate (the constraint is defined *only* over folds),
  but it must be stated prominently in the `TimeDependent` docstring since a reader may expect
  an error.
- Fold-count mismatches surface at `split` time, not at construction — the fold count is not
  knowable earlier.
- Constructor validation that eagerly couples several fields (e.g. `scard`/`smtx`/`slt`/`sst`)
  is skipped when a participant is time-dependent and re-runs on every per-fold rebuild and on
  every construction-time entry substitution instead.
- `cross_val_predict` gains a `Pipeline` method — the first whole-workflow fold loop — and
  `fit(pipe, data)` gains a reset seam at its top, mirroring `_optimise`. The
  `TimeDependentContext` inner constructor accepts `Prices_RR` where it required a
  `ReturnsResult`; optimiser fold loops still always pass returns-level data.
