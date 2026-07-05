---
status: accepted
---

# Nested optimisers propagate structured failure diagnostics, not a message string

## Context

[ADR 0011](0011-deep-optimisation-result.md) established that an `OptimisationResult` carries a
`retcode` which, on failure, is an `OptimisationFailure`. For a single JuMP solve that is enough.
But the **composite** optimisers each drive several sub-solves and then reconcile them:

- `NearOptimalCentering` solves `w_min`, `w_opt`, `w_max`, then the centred `noc_opt` problem;
- the meta-optimisers (`Stacking` / `Base_MetaOptimisation`) solve many inner optimisers, an outer
  optimiser, and a weight-bounds finalisation;
- `SubsetResampling` solves one optimiser per resample, then finalises weight bounds.

When any sub-problem failed, these collapsed the whole thing into
`OptimisationFailure(; res = msg)` — a single human-readable string. The individual sub-problems'
own `OptimisationFailure` return codes (each carrying its solver trial diagnostics) were discarded.
A caller inspecting the composite result could see *that* it failed but not *which* sub-problem, nor
reach the solver-level detail already computed underneath — defeating the point of ADR 0011's
inspectable return code for exactly the cases with the most moving parts.

## Decision

**When a composite optimiser fails, its `OptimisationFailure.res` is a named tuple that names each
sub-problem and carries that sub-problem's own return code**, alongside the summary message.

- `NearOptimalCentering`:
  `res = (; msg, w_min, w_opt, w_max, noc_opt)` — the four sub-problem return codes.
- Meta-optimisation (`Base_MetaOptimisation`):
  `res = (; msg, opti, opto, wb)` — the vector of inner return codes, the outer return code, and the
  weight-bounds-finalisation return code.
- `SubsetResampling` (`subset_resampling_retcode`):
  `res = (; msg, opti, wb)` — the per-resample return codes and the finalisation return code;
  returns the finalisation `retcode` *unchanged* when every resample succeeded, and wraps a fresh
  `OptimisationFailure` only when something actually failed.

Because each element is itself an `OptimisationFailure` (or a success code), the structure is
recursive: a caller can walk from the composite failure down to the individual solver trials.

## Considered options

- **Keep the flat message string.** Rejected: it throws away already-computed structured
  diagnostics precisely where nesting makes "which part failed?" hardest to answer by eye.
- **Throw on the first sub-problem failure.** Rejected: a composite optimiser wants to attempt all
  sub-problems and report a *complete* picture (e.g. which resamples failed), which an early throw
  precludes; and it breaks the ADR-0011 contract that failure is a return code, not an exception.
- **A bespoke failure struct per composite.** Rejected: named tuples give the same field access and
  printing with zero new types, and keep the shape self-documenting at the call site.

## Consequences

- The composite return code is now inspectable to arbitrary depth — `res.opti[k]`, `res.noc_opt`,
  etc. — matching ADR 0011's intent for the layered optimisers.
- **Mildly breaking for anyone pattern-matching `res` as a `String`**; `res` is now a named tuple
  whose `msg` field holds the former string.
- `NearOptimalCentering`'s reference output (`MeanRiskDT` fixture) and its regression tests were
  updated to assert on the structured failure shape.
- New composite optimisers should follow suit: aggregate children's return codes into a
  `(; msg, <named sub-problems>)` tuple rather than stringifying.
