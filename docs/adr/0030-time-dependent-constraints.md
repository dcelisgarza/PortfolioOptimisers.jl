---
status: accepted
---

# Time-dependent inputs as in-field wrappers — and schedules of optimisers — consumed by CV loops

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

The design started with constraint fields (`wb`/`fees`/`lt`/… on `JuMPOptimiser`) and was then
extended across the optimisation layer: every problem-definition field of every
`NonFiniteAllocationOptimisationEstimator` now takes a schedule, including the optimiser-valued
fields (`fb`, the metas' `opti`/`opto`/`opt`) — and a schedule may be handed to a fold loop as
the optimiser *itself*. This ADR is amended in place to state that one current design.

## Decision

**A `TimeDependent(val; default = x)` wrapper consumed by whichever cross-validation fold loop
processes it, held in either of two positions: stored directly in the optimiser field it varies —
`JuMPOptimiser(; lt = TimeDependent([...]))` — or handed to `optimise`/`cross_val_predict` as the
optimiser itself, a per-fold schedule of estimators and/or precomputed results. Everywhere
fold-less, the wrapper resolves to the field's static default, or to its own `default`, or —
where the position is required and neither exists — throws a structured
`TimeDependentDefaultError`.**

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
  `PreviousWeightsFunction` wrapper exists only for the uninspectable bare-function case. A
  callable returning the fold's *optimiser* may declare that statically by subtyping
  `TimeDependentOptimiserCallable <: TimeDependentCallable`, making the schedule admissible
  wherever the optimiser-typed bounds require it; a bare `ctx -> optimiser` is admitted as
  `Base.Callable` and its output checked when the fold loop swaps it in. Schedules never nest:
  neither `val`, nor a vector entry, nor `default` may be a `TimeDependent` (rejected at
  construction) — an estimator swapped in by a schedule may carry schedules *in its own fields*,
  which is recursion, not nesting.
- **Problem definition varies, execution control does not — and the widened signatures encode
  it.** A field is admissible for a schedule iff it is *problem definition* (what is being
  solved: `pe`, `ret`, `sca`, `sets`, `cle`, `wf`, `r`, `obj`, `wi`, constraint fields, `fb`,
  the metas' `opti`/`opto`/`opt`, `scale`, `subset_size`, `n_subsets`, …), never *execution
  control* (how it is solved: `slv`, `ex`, `rng`, `seed`, `sc`, `so`, `brt`, `cle_pr`,
  `strict`, `sq`, and `NestedClustered.cv`, which *is* the inner fold loop). The criterion is
  applied per constructor and the widened signatures are the admissibility table — the single
  source of truth for which inputs may vary over folds, spelled with a small family of aliases:
  `TD_Option{X} = Union{Nothing, <:TimeDependent, X}` for optional fields,
  `TD{X} = Union{<:TimeDependent, X}` for required non-optimiser fields (so widening never
  smuggles `nothing` into a field that never admitted it), and the statically-parametric
  optimiser bounds `TD_OptE_Opt` / `TDO_OptE_Opt` / `TD_VecOptE_Opt`
  (e.g. `TimeDependent{<:AbstractVector{<:OptE_Opt}}`) for optimiser-valued positions.
  Discovery is a generic scan of the host's fields for `TimeDependent`-valued ones
  (`time_dependent_fields`) — no hand-maintained list. The scan is narrowed to the fields whose
  *type* admits a schedule (`time_dependent_candidate_fields`, a generated function over
  `fieldtype`), which is still derived from the widened signatures and so keeps them the single
  source of truth, but folds the scan of a static host to an empty tuple at compile time instead
  of walking all of its fields per `split`/`_optimise`. Within an estimator, `TimeDependent` is
  recognised at top-level fields only, never nested inside another input; it is additionally
  recognised as the estimator handed to a fold loop itself (next bullet).
- **A schedule can *be* the optimiser.** The fold-loop entry points widen to
  `OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}` (and
  `OptE_Opt_TD` where precomputed results are also legal), so
  `cross_val_predict(TimeDependent([opt₁, …, optₙ]), rd, cv)` runs entry `i` on fold `i`:
  `update_time_dependent_estimator` on a schedule resolves entry `i`, checks it, then **recurses
  into it with the same fold context**, so the swapped-in estimator's own schedules resolve
  against the same folds. Schedules may be **mixed**: estimator and precomputed-result entries
  sit side by side, and fold `i` optimises or predicts depending on what entry `i` is — the
  single-fold `fit_and_predict` already dispatches estimator-vs-result, so this costs nothing.
  The one exclusion is asset-subsampling CV (`MultipleRandomised`): a solved result has no
  sub-portfolio semantics — `port_opt_view` on a result throws a structured `ArgumentError`
  (with a `Colon` pass-through) — so result entries are rejected there with a clear message.
  `assert_time_dependent_fold_count` sizes the schedule to the consuming loop **and recurses
  into its entries** — but not into the `default`, which only ever runs fold-lessly. The same
  bound types the optimiser-valued *fields* (`fb` everywhere; `Stacking.opti`,
  `NestedClustered.opti`, `SubsetResampling.opt`, the metas' `opto`), where optional positions
  additionally admit `nothing` entries (`TDO_OptE_Opt`) so a scheduled fallback can switch off
  per fold.
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
- **Inert outside fold loops, resolved in `_optimise` — reset, `default`, or throw**: a
  time-dependent input is defined *only* over folds, so each estimator's `_optimise` first
  passes itself through `reset_time_dependent_estimator`, which rebuilds hosts with every
  `TimeDependent`-valued field at its fold-less value and recurses through wrapper optimisers.
  That value is, in order: the schedule's own `default` if it carries one
  (`TimeDependent(val; default = x)`); else the field's static default, declared once per host
  in a `*_td_defaults` NamedTuple (e.g. `jump_optimiser_td_defaults`: `wb → WeightBounds()`,
  `bgt → 1.0`, otherwise `nothing`) shared by `time_dependent_field_defaults` and the
  constructor's entry-substitution pass; else — for a *required* field with no static default,
  marked `NoDefault()` in that table, i.e. the optimiser-valued positions — a structured
  `TimeDependentDefaultError`. Fields with a static default therefore keep the original silent
  reset, and only the positions where "no value" is unrepresentable fail closed. A defaultless
  `fb` schedule is the deliberate edge: `nothing` is a legal fallback value, so fold-less it
  means *no fallback*, not an error, and the reset runs before the fallback chain is walked. A
  schedule handed to a fold-less `optimise` *as* the optimiser resolves to its `default` the
  same way or throws pointing at `cross_val_predict`. Placing the seam in `_optimise` covers
  the fallback chain, the `fb = Nothing` fast paths, and meta-optimisers' fold-less full-window
  solves uniformly; per-fold estimators produced by the swap may still carry schedules of their
  own (post-swap recursion resolves them against the same fold context before the solve, so
  what reaches `_optimise` passes through at the cost of one field scan). Meta-optimisers reset
  *only their own fields*: their inner schedules must survive the seam to be consumed by the
  meta's inner cross-validation leg, while the meta's fold-less full-window inner solves reset
  themselves at their own `_optimise`. The reset honours the same per-field `:nearest`
  entitlement as the scan (next bullets): `reset_time_dependent_fields` must not wipe a
  `:nearest` optimiser schedule the host's own inner CV has yet to consume.
- **Meta-optimisers compose, never distribute**: NestedClustered, Stacking, and
  SubsetResampling accept `TimeDependent` in their *own* problem-definition fields (`wb`/`fees`
  and, per the criterion, `pe`/`sets`/`wf`/`scale`/`cle`/the optimiser positions — the generic
  field scan, entry reset and TD-aware views cover a newly widened field with no new machinery)
  and forward `is_time_dependent` / `update_time_dependent_estimator` / the fold-count
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
  Entitlement is refined **per field**, not per host: each host declares in an
  `inner_fold_fields` trait which of its *own* fields it hands across its inner fold loop, and
  a pass at `all_binds = true` still skips a `:nearest` schedule in those fields
  (`entitled(opt, f, all_binds)`) — a host opening a nearer fold loop and handing a field across
  it means the loop *reaching* the host is not nearest for that field. The reset seam honours
  the same rule, or it would wipe every `:nearest` optimiser schedule before the inner CV saw
  it. `needs_previous_weights` is deliberately left scope-blind (it never consults `bind`):
  declaring sequentiality conservatively is always safe, and threading the flag through the
  trait would buy nothing.
- **`:nearest` on an optimiser-valued position is legal iff an inner fold loop actually consumes
  the value there.** The two metas split by where they *enter* their inner CV:
  `NestedClustered` enters it per *cluster* (`cross_val_predict(opti, …; cols = cl)`), so a
  schedule on the **field** is consumed by the inner folds and the column identity — the
  cluster — survives; `Stacking` enters it per *candidate* (`cross_val_predict(opti[k], …)`),
  so only an **element** of `opti` may be `:nearest` (a field-level schedule would vary the
  candidate count and `scale` per fold, and the proxy's columns would stop denoting candidates
  — field-level `Stacking.opti` schedules are `TD_VecOptE_Opt`, `:outermost` only). Where
  `:nearest` is legal it requires an explicit `default` and `cv !== nothing`, both checked at
  construction, because `opti` has a second, fold-*less* consumer (the full-sample `wi` fit).
  Everywhere else — every `opto`, `SubsetResampling.opt`, and every fallback `fb` on all eleven
  concrete hosts — no inner *fold* loop owns the position, so a `:nearest` optimiser schedule
  is rejected at construction (`assert_no_nearest_bind_optimiser_schedule`).
- **Pipelines: swap, then inject — with the swap outside `fit` entirely.** The `Pipeline` fold
  loop (`cross_val_predict(pipe, data::Prices_RR, cv)`) splits the *input* at its own level and
  resolves time-dependent steps per fold *before* `fit` runs on the training window. Combinatorial
  and asset-resampling schemes are supported for a **returns-level** pipeline — each fits on the
  split's (possibly non-contiguous) training rows and predicts its test groups / asset-subset
  paths, exactly as the plain-optimiser loops do — and rejected only for a **price-starting**
  pipeline, by the *rolling-window rule*: a rolling, order-dependent price transform (a
  `PricesToReturns`, or any windowed preprocessing) needs contiguous input rows, which the
  recombined groups / resampled paths do not guarantee. Injection
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
- **A new supertype for schedules of optimisers.** Rejected: `TimeDependent` stays
  `<: AbstractEstimator` and the optimiser positions widen to parametric bounds
  (`TimeDependent{<:AbstractVector{<:OptE_Opt}}` and friends), which give static admissibility
  checking without a parallel type tree to keep in sync — one wrapper, two positions.
- **Banning callables in optimiser position.** Rejected: a callable is how per-fold optimisers
  are *derived* rather than enumerated (regime detection over `ctx.rd`). Instead, callables
  declare themselves — `TimeDependentOptimiserCallable` makes the output kind static — and the
  bare `ctx -> optimiser` escape hatch is checked at the fold swap, trading a construction-time
  error for expressiveness.
- **Entry 1 as the fold-less default.** Rejected: fold 1's optimiser is a statement about fold
  1, not about fold-less use, and silently promoting it hides a semantic decision. The explicit
  `default` keyword (or a structured throw when it is missing on a required position) keeps the
  fold-less value something the user actually said.
- **Homogeneous-only schedules (estimators xor results).** Rejected: the single-fold
  `fit_and_predict` already dispatches estimator-vs-result per entry, so mixing costs no
  machinery and covers real use (splicing an externally solved period into a backtest). The one
  place homogeneity is enforced — no result entries under asset-subsampling CV — falls out of
  results having no sub-portfolio semantics, not out of the schedule.

## Consequences

- `update_time_dependent_estimator`'s documented signature changes to `(opt, ctx)`, and the
  WalkForward / MultipleRandomised loops change behaviour: the update applies from fold 1 (the
  `prev !== nothing` guard no longer gates it) and runs before the previous-weights factory.
  The KFold/non-sequential path gains the swap-in hook it never had.
- `is_time_dependent` becomes derived (any field holds a `TimeDependent`, recursing through
  wrapper optimisers) and no longer implies sequential execution on its own.
- A time-dependent estimator passed to a fold-less `optimise` runs at the affected fields'
  static defaults (or the schedules' own `default`s) with no message — deliberate (the schedule
  is defined *only* over folds), but it must be stated prominently in the `TimeDependent`
  docstring since a reader may expect an error. Only a *required* position holding a defaultless
  schedule throws (`TimeDependentDefaultError`), and the message points at `cross_val_predict`.
- The four fold-loop entry points (`cross_val_predict` and the `fit_and_predict` methods)
  widen from `OptE_Opt`-style signatures to `OptE_TD`/`OptE_Opt_TD`, so a schedule is a legal
  optimiser argument everywhere a fold loop begins — but nowhere else.
- `port_opt_view` on an `OptimisationResult` throws a structured `ArgumentError` (with a
  `Colon` pass-through) instead of silently passing the result through: a solved result has no
  sub-portfolio semantics, which is also what excludes result entries from `MultipleRandomised`
  schedules.
- Fold-count mismatches surface at `split` time, not at construction — the fold count is not
  knowable earlier.
- Constructor validation that eagerly couples several fields (e.g. `scard`/`smtx`/`slt`/`sst`)
  is skipped when a participant is time-dependent and re-runs on every per-fold rebuild and on
  every construction-time entry substitution instead.
- `cross_val_predict` gains a `Pipeline` method — the first whole-workflow fold loop — and
  `fit(pipe, data)` gains a reset seam at its top, mirroring `_optimise`. The
  `TimeDependentContext` inner constructor accepts `Prices_RR` where it required a
  `ReturnsResult`; optimiser fold loops still always pass returns-level data.
