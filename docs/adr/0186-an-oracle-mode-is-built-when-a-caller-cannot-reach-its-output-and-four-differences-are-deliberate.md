---
status: accepted
---

# An oracle mode is built when a caller cannot reach its output, and four differences are deliberate

## Context

[Map #1375](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1375) measures the
cross-sectional factor prior and everything around it against an oracle: an independent
implementation of the same models. The oracle has modes that the library does not port.
[Issue #1392](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1392) listed sixteen of
them and decided each one.

A mode of the oracle can be missing from the library in three ways:

1. The library has no route to the mode's output.
2. The library reaches the same output by its own idiom: a composition, a view, a rebuild through a
   checked constructor, or a keyword of a different estimator.
3. The mode contradicts a rule that the library already records.

## Decision

**A mode is built when a caller cannot reach its output today.** A mode that a library route
reaches gets that route documented, and the build of a convenience surface is decided mode by mode.
A mode that contradicts a recorded library rule is a deliberate difference. The parity target is
the output, not the shape of the oracle's call.

### The deliberate differences

| Mode of the oracle | The library | Why |
| --- | --- | --- |
| A nearest-correlation repair by clipping the correlation eigenvalues at `1e-13`, with one retry at `1e-12` | `Posdef(Newton)` in `f_mp` and `mp`, the library default | Newton gives the nearest correlation matrix in the Frobenius norm. The oracle's clip is a cheaper approximation of it. |
| An inactive-cell policy stored on each field: `NaN`, zero, or the value left as it is | Fields stay finite, and every consumer reads the masks. A read of a field takes the policy as an argument, so the caller chooses it for each read. | ADR 0102 rules that a numeric field's values stay finite. Every computed output keeps the oracle's capability: the descriptors write `NaN` on an inactive cell, and the weights are built over the estimation universe. The one view the oracle gives, a field with its inactive cells blanked, becomes a read, not a stored state. |
| A warning when the regime half-life exceeds 138 observations | The `regime_decay` docstring states the threshold `2^(-1/138)` and its effect | The rule of #1282: the docstring states the condition, and the measure does not change its behaviour. |
| An integer `cv`, which means K-fold with that many folds | `cv = KFold(; n)` | No field or verb of the library takes an integer as a short form for an estimator. |

The oracle's history cap on its prior truncates the stored histories and the scenarios, and leaves
the moments folded over every observation. The library reaches the scenario cap through
`pe = EmpiricalPrior(; max_scenarios)`, because the cross-sectional prior builds its scenarios from
the last rows of the factor scenarios. That is a route, not a difference. ADR 0136 already
separates a scenario cap from the window of `Online(…; max_history)`.

### The square root of a covariance is a named policy

The library took the square root of a covariance by three unnamed policies: a bare Cholesky that
throws, `covariance_factor` (a Cholesky, else an exact eigen square root of a positive semidefinite
matrix), and `safe_regime_cholesky` (a Cholesky with a ridge that grows tenfold over three tries,
which is the oracle's policy). An algorithm type now names the last two, and a field on
`CrossSectionalFactorPrior` and `FactorPrior` selects one. The field defaults to `nothing`, which
keeps the bare Cholesky that throws. With the default repair on, all three agree, because the
matrix reaches the Cholesky positive definite. They differ only when a caller turns the repair off
or the repair fails. Issue #1410 decides each other bare Cholesky in the library.

### Built

The build tickets are children of map #1375: #1396 to #1408, and #1411. #1409 decides the route of
the prior onto the online seam. #1392 holds the table of all sixteen modes.

### A lost mask is caught by a test, not by a `NaN`

The oracle's default policy makes a consumer that forgets the mask give `NaN`, which a test sees at
once. With finite fields the same defect gives a plausible answer. #1398 is a real case: a windowed
wrapper dropped the active mask of an Asset Panel. So a census test (#1411) writes an extreme value
into every inactive cell and asserts that the answer of every consumer of an Asset Panel does not
change. It gives the safety of the oracle's `NaN` with no cost at run time.

## Consequences

- A measure ticket of map #1375 that finds a mode the oracle has and the library lacks applies the
  rule above before it asks the maintainer.
- A parity test does not compare the nearest-correlation repair with the oracle's clip. #1383
  records that row as a deliberate difference.
- The premise of #929, that the oracle persists nothing, was wrong. #1399 builds the round trip
  that parity needs.
