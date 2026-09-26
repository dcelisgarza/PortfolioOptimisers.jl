---
status: accepted
---

# A walk-forward refuses a frontier sweep that reads the previous weights

## Context

Issue #1004. A walk-forward threads the previous fold's weights into every term that reads them
(a turnover, a tracking error, a fee, a custom constraint or a custom objective) through
`factory`, when `needs_previous_weights` holds of the estimator. A frontier sweep does not give
one portfolio. It gives a population, one portfolio per sweep point, so the previous fold's `w`
is a vector of vectors.

No `factory` method takes a population. `factory(opt::JuMPOptimiser, w::AbstractVector)` does
match it, but the term types it forwards to do not, so each falls through to a pass-through and
keeps the `w` the caller stated. Measured on `dev` at `7d1ed02d8d` with a `Turnover` of
`val = 0.003` on `w = 1/N` and a `Frontier(; N = 5)` over `IndexWalkForward(127, 171)`: every
point of fold 2 sat exactly `0.003` from `1/N` in its largest asset and `0.006` from the fold-1
point of the same index. The loop charged the turnover against the first stated `w` on every
fold, and no error or warning said so.

A method that takes the population does not settle it, because no one previous portfolio exists.
`resolve_frontier_bounds` derives each span per fit, from the corner solves over that fold's own
data. So point `k` at fold `t` and point `k` at fold `t + 1` sit at different return levels on
different frontiers. A multi-term frontier sweeps a Cartesian product of spans, which has no index
correspondence across folds at all. A frontier sweep describes a set of choices, and a term that
reads the previous weights describes the trade from the one portfolio that was held. The two
ideas contradict each other.

## Decision

1. **The fold loop refuses the pair by name.** `fold_loop` passes the previous weights through
   `one_previous_portfolio(est, w_prev)` before `factory`. On one portfolio it gives the weights
   back unchanged. On a population it throws an `ArgumentError` that names the estimator, the
   number of portfolios, the reason, and the two ways out: optimise one portfolio per fold, or fix
   the previous weights of the term.
2. **The refusal is at the fold, not at construction.** A frontier sweep with a turnover on a
   stated `w` is a valid problem in one `optimise` call, and a fixed turnover in a walk-forward
   reads no previous weights. Only the threading is contradictory, and only the fold loop threads.
3. **The refusal covers every arm that threads and both Previous-Weights Sources.** The sequential
   arm and the online arm both resolve a fold through the same per-fold copy, and the held weights
   of a population are a population too. The parallel arm threads nothing, so a frontier
   walk-forward with no such term runs as before.
4. **The check reads the shape of the weights.** Only a frontier sweep gives a population, and
   the meta-optimisers carry that shape from their inner sweep, so the shape answers for every
   estimator without a new trait. The cost is that the refusal comes after the first fold's
   solve, because the first fold threads nothing.

## Alternatives, left open

The maintainer ruled that the two ideas contradict each other today, and that the question can be
opened again. These are the candidates issue #1004 names, not refused on their merits:

- **Charge point-wise on a pinned span.** Resolve the span once, at the first fold, and hold it
  for the run, so point `k` means the same return level throughout. This changes what a frontier
  walk-forward means.
- **Charge against one selected portfolio.** The caller names the frontier point that carries the
  previous weights, and the other points sweep unconstrained.
- **Correspond by nearest neighbour** in the objective space rather than by index.

Each has consequences for the online step, for the plotting family, and for what
`held_weights_drift` reports. A change to any of them amends this ADR and replaces the refusal.
