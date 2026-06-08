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
[08_Base_JuMPOptimisation.jl](../../src/20_Optimisation/08_Base_JuMPOptimisation.jl),
registers `val` under `Symbol(prefix, name)`. Shared-infrastructure builders and the
read accessors take a `prefix::Symbol = Symbol("")` keyword: the default empty prefix
reproduces the original bare key (so the change is behaviour-preserving everywhere
the default is used), and a nested build (e.g. risk tracking) passes a distinct
non-empty prefix such as `Symbol(:tr_iv_, i, :_)`. The tracking measure stores its
difference weights at `Symbol(prefix, :w)` and threads `prefix` down the whole
risk-build spine; the inner build reads and writes only prefixed keys.

## Consequences

- The ~390-line save→build→restore block and the `:w`↔`:oldw` swap in
  [18_TrackingRiskMeasureConstraints.jl](../../src/20_Optimisation/19_RiskMeasureConstraints/18_TrackingRiskMeasureConstraints.jl)
  are **deleted**, not relocated. This is the payoff.
- Nested builds are re-entrant: distinct prefixes never alias, so a tracking measure
  can itself contain a tracking measure without special handling.
- A new Category-A key joins automatically — there is no swap block to extend — as
  long as its builder threads `prefix` and registers via `preg!`. That threading
  *is* the new obligation: the discipline moved from "remember to extend the swap
  block" to "remember to pass `prefix`," enforced by the seam-lock test (0004 §6.5),
  not the type system.
- The rule for what to prefix is an **invariant: prefix a key iff it is
  weight-dependent.** A nested build (risk tracking) shifts the *weights* via the
  benchmark difference, not the prior moments, so a key that is a pure function of
  the prior `pr` is identical in the inner and outer builds and is correctly shared
  **bare**. The deliberately-bare keys are therefore `:fees`, the FRC keys
  (`:frc_W`/`:frc_M`/`:frc_M_PSD`), and the prior-derived caches `:G`, `:Gkt`, `:GV`,
  `:vals_Akt`/`:vecs_Akt` (Cholesky/eigendecompositions of `pr`). Prefixing these
  would break sharing, not protect it.
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
