---
status: accepted
---

# Time-dependent constraints as a fold-indexed wrapper consumed by CV loops

## Context

Cross-validation backtests re-optimise the same estimator over a sequence of time-ordered folds,
but every constraint field is fixed for the whole horizon. The hooks for per-period updates
(`is_time_dependent`, `update_time_dependent_estimator`, `needs_previous_weights` in
[01_Base_Optimisation.jl](../../src/20_Optimisation/01_Base_Optimisation.jl)) existed as
scaffolding with no concrete overload — nothing in the library actually varied a constraint over
time. Two representation problems block the naive approach of "put a vector in the field":

- Several fields already accept vectors with field-level meaning (`smtx`/`sgmtx` = multiple subset
  spaces, `ple` = multiple phylogeny constraints), so "vector = per-fold" is ambiguous.
- Several fields share a value type (`lt` and `st` are both `Option{<:BtE_Bt}`), so the target
  slot cannot always be inferred from the value alone.

## Decision

**A `TimeDependent(val, field::Symbol)` wrapper stored in a `tdc` field on `JuMPOptimiser` and
`HierarchicalOptimiser`, consumed by whichever cross-validation fold loop processes the
estimator, and inert everywhere else.**

- `val` is either a vector — entry `i` is the complete field value for fold `i`, so vector-valued
  fields nest one level — or a callable `f(ctx::TimeDependentContext)` evaluated per fold.
  `field` defaults via a trait on the stored value type; types that admit several targets
  (`Threshold` → `:lt`/`:st`) have no trait method and require an explicit symbol, as do bare
  function forms. Callables may instead be structs subtyping
  `TimeDependentCallable <: AbstractAlgorithm` (a functor over the context): being types, they
  can define `default_time_dependent_target` to make the target inferable and
  `needs_previous_weights` to declare previous-weights requirements directly — the
  `PreviousWeightsFunction` wrapper exists only for the uninspectable bare-function case.
  `tdc` accepts one wrapper or a vector of them (duplicate targets error).
- **Sole source**: the targeted base field must be left at its constructor default (`nothing`
  for most, `WeightBounds()` for `wb`, `1.0` for `bgt`); the constructor errors if both are set.
  Entry `i` applies to fold `i` *including fold 1*.
- **Enumeration-order indexing, no hidden ranking**: entry `i` corresponds to fold `i` of the
  consuming scheme's `split` enumeration — the machinery never re-ranks folds. `ctx.i` therefore
  always indexes `ctx.train_idx`/`ctx.test_idx`, so a callable can identify its own fold's
  windows under every scheme. Walk-forward and (unshuffled) KFold enumerate chronologically;
  for schemes whose enumeration is not a timeline (combinatorial splits, randomised paths) it
  is the *user's* responsibility to key entries off the fold's indices. One shared vector
  serves all paths; its length must equal folds-per-path, validated immediately after
  `split(cv, rd)` before any fold runs.
- **Swap-in is a constructor rebuild**: the fold loop rebuilds the host through its validated
  keyword constructor with the fold's values substituted and `tdc = nothing`, so every existing
  invariant re-runs each fold (including on function outputs) and the per-fold optimiser is an
  ordinary static estimator. At construction time every vector entry is test-substituted through
  the same constructor, surfacing symbol/type/cross-field errors immediately.
- **Parallel-safe**: `tdc` alone never forces sequential fold execution; sequentiality remains
  tied to `needs_previous_weights`, which inspects vector-entry contents recursively. A function
  form that needs previous weights declares it by wrapping the callable in
  `PreviousWeightsFunction(f)` (contributes `true` to the trait); `ctx.w_prev` is populated only
  in sequential runs. The fold loop applies the time-dependent swap *first* and
  `factory(·, prev.w)` *second* (order swapped from the original scaffolding), so per-fold
  constraint values also receive previous weights.
- `TimeDependentContext` carries `i`, `n` (folds in path), the possibly asset-viewed
  `ReturnsResult`, `train_idx`, `test_idx`, `w_prev::Option`, `path_id::Option`;
  `update_time_dependent_estimator(opt, ctx)` replaces the ad-hoc
  `(opt, i, rd, train_idx, test_idx)` argument list. Views need no per-field machinery:
  `TimeDependent` maps `port_opt_view` over its stored values exactly as host fields already do,
  and function forms see the viewed universe through `ctx.rd`.
- **Inert outside fold loops**: a plain (fold-less) `optimise` ignores `tdc` — the targeted
  fields are at their defaults, so the solve is well-defined without it. Meta-optimisers
  (NestedClustered, Stacking, SubsetResampling) forward `is_time_dependent` /
  `update_time_dependent_estimator` / the fold-count assertion into their inner and outer
  estimators like every other wrapper, so an outer fold loop over a meta resolves the inner
  `tdc` against the *outer* folds; used standalone, the meta's inner CV leg consumes it against
  the inner folds through the ordinary `fit_and_predict` path (outermost fold loop wins —
  swap-in clears `tdc`, so inner legs then see static per-fold estimators). Fold-less
  full-window solves run at the defaults, and because swap-in builds new structs the original
  estimator is never mutated. The entries must be sized for whichever CV consumes them.

## Considered options

- **`needs_prev` flag on the wrapper.** Rejected: previous-weight need is already discoverable
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
- **Fail-closed guard (`optimise` errors on `tdc`) with an explicit strip step for
  meta-optimisers.** Rejected in favour of inert semantics: the guard would break Stacking/NCO,
  whose full-window solves of inner estimators are fold-less by design, and the strip step is
  special-casing the sole-source rule already makes unnecessary — outside folds the constraint
  legitimately reverts to the field's default.
- **Per-path `tdc` vectors.** Rejected: paths are an artifact of the CV scheme, not something a
  user can author constraints against.

## Consequences

- `update_time_dependent_estimator`'s documented signature changes to `(opt, ctx)`, and the
  WalkForward / MultipleRandomised loops change behaviour: the update applies from fold 1 (the
  `prev !== nothing` guard no longer gates it) and runs before the previous-weights factory.
  The KFold/non-sequential path gains the swap-in hook it never had.
- `is_time_dependent` becomes derived (`tdc !== nothing`, recursing through wrapper optimisers)
  and no longer implies sequential execution on its own.
- A `tdc`-bearing estimator passed to a fold-less `optimise` runs at the targeted fields'
  defaults with no message — deliberate (the constraint is defined *only* over folds), but it
  must be stated prominently in the `tdc` docstring since a reader may expect an error.
- Fold-count mismatches surface at `split` time, not at construction — the fold count is not
  knowable earlier.
