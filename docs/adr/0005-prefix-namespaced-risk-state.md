---
status: accepted
supersedes: 0004 §5 (snapshot/restore as the interface)
---

# Namespace nested risk state by key prefix, not snapshot/restore swap

## Context

ADR [0004 §5](0004-typed-jump-model-state.md) decided that the worst hazard in the
JuMP model state — `RiskTrackingRiskMeasure` rebuilding every Category-A key against
a different returns vector — would be solved by modelling Category-A keys as a
registered, enumerable set with a scoped swap (`snapshot_state!` + `restore_state!` /
`with_swapped_state`). It called this the highest-leverage part of the work.

We tried that and **reverted it.** The registry/build-then-swap mechanism still
required the unregister/restore dance to be correct for every key (and every *new*
key), in two places — it relocated the fragility into a registry rather than removing
it, and it was not re-entrant (a tracking measure nested inside another would clobber
the saved snapshot).

## Decision

Build the nested risk expressions under a **namespacing key prefix** from the start,
so they never collide with the outer model's keys and nothing has to be saved,
unregistered, or restored.

A single helper, `preg!(model, prefix, name, val)` in
[08_Base_JuMPOptimisation.jl](../../src/17_Optimisation/05_JuMP/01_Base_JuMPOptimisation.jl),
registers `val` under `Symbol(prefix, name)`. Shared-infrastructure builders and the
read accessors take a `prefix::Symbol = Symbol("")` keyword: the default empty prefix
reproduces the original bare key (so the change is behaviour-preserving everywhere
the default is used), and a nested build (e.g. risk tracking) passes a distinct
non-empty prefix such as `Symbol(:tr_iv_, i, :_)`. The tracking measure stores its
difference weights at `Symbol(prefix, :w)` and threads `prefix` down the whole
risk-build spine; the inner build reads and writes only prefixed keys.

## Consequences

- The ~390-line save→build→restore block and the `:w`↔`:oldw` swap in
  [19_TrackingRiskMeasureConstraints.jl](../../src/17_Optimisation/05_JuMP/09_RiskMeasureConstraints/16_TrackingRiskMeasureConstraints.jl)
  are **deleted**, not relocated. This is the payoff.
- Nested builds are re-entrant: distinct prefixes never alias, so a tracking measure
  can itself contain a tracking measure without special handling.
- A new Category-A key joins automatically — there is no swap block to extend — as
  long as its builder threads `prefix` and registers via `preg!`. That threading
  *is* the new obligation: the discipline moved from "remember to extend the swap
  block" to "remember to pass `prefix`," enforced by the seam-lock test (0004 §6.5),
  not the type system.
- The rule for what to prefix is an **invariant: prefix a key iff it is per-build
  risk state.** That is two kinds of key: (a) **weight-dependent** expressions — a
  nested build (risk tracking) shifts the *weights* via the benchmark difference, so
  any key that is a function of `:w` differs between the inner and outer builds; and
  (b) **build-scoped presence flags** — boolean markers (`:variance_flag`,
  `:rc_variance`) that are not weight-dependent but gate per-build formulation
  decisions, so the inner build must not see the outer build's flags (and vice
  versa). A key that is a pure function of the prior `pr` is identical in the inner
  and outer builds and is correctly shared **bare**: `:fees`, the FRC keys
  (`:frc_W`/`:frc_M`/`:frc_M_PSD`), and the prior-derived caches `:G`, `:Gkt`, `:GV`,
  `:vals_Akt`/`:vecs_Akt` (Cholesky/eigendecompositions of `pr`). Prefixing those
  would break sharing, not protect it.
- The presence flags need care at the **read** site, not just the write. `:rc_variance`
  is read *in-spine* by the `sdp_variance_flag!` predicate to choose the SDP-vs-SOC
  variance formulation, so `prefix` threads through that predicate (and the
  `set_risk!`/`set_variance_risk!`/`rc_variance_constraints!` chain) — its `haskey`
  becomes `haskey(model, Symbol(prefix, :rc_variance))`. `:variance_flag`'s only
  readers are the *outer-level* phylogeny builders (they add a `p·tr(W)` penalty only
  when no variance is present); those stay bare, and prefixing the inner *write* is
  what stops a nested variance from leaking presence to the outer model — the job the
  old save/restore did, now structural.
- That invariant also exposes a latent gap in the old swap, which is direct evidence
  for the fragility argument above: the swap rebuilt the SDP `:W`/`:M`/`:M_PSD` under
  tracking weights but **never `:L2W`** (`= L2·vec(W)`, a function of that `W`). The
  single-measure golden tests never hit it (with no outer Kurtosis, `:L2W` is built
  fresh against the tracking `W`), but a Kurtosis-nested-in-Kurtosis tracking would
  reuse a stale `:L2W`. The prefix approach fixes this for free — `:L2W` is prefixed
  alongside `:W`, byte-identical on the golden tests and correct under re-entrancy —
  precisely because "thread `prefix`" forces the question "is this weight-dependent?"
  that the swap list silently got wrong.
- The Step 5 seam-lock (0004 §6.5) is therefore scoped to **literal** `model[:`
  only, not all `model[`. Prefix-threaded Category-A keys become computed
  `model[Symbol(prefix, name)]`, and Category-B scratch (0004 §1) is computed too, so
  both are exempt; the lock catches bare-literal blackboard leaks, which is the actual
  hazard. Phase 2 gives the cross-file Category-A infra keys (`:X`, `:net_X`, `:Xap1`,
  `:ddap1`, `:dd`, SDP) named prefixed `get_*`/`has_*` accessors; per-measure
  singletons stay on the prefix-computed form.

## Amendment (2026-09-24)

The invariant said that a weight-dependent key and a presence flag are per-build risk state.
That holds for a build on shifted weights. It does not hold for a build that registers the
weights of its enclosing build without a change. `DependentVariableTracking` is such a build:
its prefix holds the head's own `w`. Under the old rule its inner measures built a second
lifted matrix `tr_dv_iW` with its own PSD cone for the same `w` (#1305). Two defects followed:

1. The head's semidefinite phylogeny rows `A ⊙ W = 0` were on the bare `W`, so they did not
   constrain the inner variance `tr(Σ tr_dv_iW)`.
2. The inner variance marked `tr_dv_ivariance_flag`, so a head whose only variance was a
   dependent tracking variance kept the phylogeny's `p · tr(W)` penalty. A head with a plain
   variance dropped it.

**Decision.** The lifted matrix `W` (with `M` and `M_PSD`) and the marks `variance_flag` and
`rc_variance` belong to the weights, not to the build. A build that registers weights it did
not make also registers `w_owner`, the namespace that owns them. `weights_prefix(model,
prefix)` returns that owner, or `prefix` when there is none, and `set_sdp_constraints!`, the
variance builders and the phylogeny builder read it.

- `DependentVariableTracking` registers `w_owner = weights_prefix(model, prefix)` under its
  tracking prefix. Its inner measures read the enclosing build's `W` and marks, and keep
  their own rows and scratch under the prefix.
- The owner is resolved when the weights are registered, so a chain collapses: a dependent
  build inside an `IndependentVariableTracking` build owns the independent build's shifted
  weights, not the bare ones.
- `IndependentVariableTracking` registers no owner. Its weights are `w - wb·k`, so its `W`
  and its marks stay under its own prefix, as this ADR decided.
- A programme Allocation Set in a leader's model registers `w_owner = Symbol("")` under
  `:aset_` (#1303, ADR 0159). This replaces the boolean mark `w_shared` that #1303 added, which
  could not name an owner other than the bare namespace.
- `UncertaintySetVariance` marks `variance_flag` on the owner, as `Variance` does, because it
  builds on the owner's `W`.

**Consequence.** A dependent tracking variance now removes the phylogeny's `p · tr(W)` penalty,
as the head's own variance does. The `[A, Tracking(A)]` cases of
`test/test_27_prefix_registration.jl` assert that a `DependentVariableTracking` build registers
none of these keys under its prefix, and a new testset covers the phylogeny, the penalty and
the nested chain.
