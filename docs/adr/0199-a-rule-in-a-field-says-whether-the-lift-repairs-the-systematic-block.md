---
status: accepted
---

# A rule in a field says whether the lift repairs the systematic block

## Context

`cross_sectional_lift` builds the asset covariance of a `CrossSectionalFactorPrior` over the
investable assets. It projects the factor covariance through the latest Factor Exposures, which
gives the systematic block `si = Li F Li'`, and runs every step of `mp` on it. Then it adds the
idiosyncratic block `D` and repairs the sum under `mp.pdm`.

The first step of the default `mp` is the `:pdm` step, a Newton repair to the nearest correlation
matrix. `si` has rank at most `K`, the number of factors, so `LinearAlgebra.isposdef` never
accepts it, and the Newton repair runs at each fit. On the benchmark of
[#1562](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1562) (500 assets, rank 7) it
cost 0.057 s of each read-out of the carry fold, longer than the whole step of the oracle. It moved
`si` by 9.3e-11 relative to its largest entry, because `si` is positive semidefinite to round-off.
The oracle repairs only the factor covariance, never the asset covariance.

[#1570](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1570) decided that the repair
becomes a rule in a field, and that the default does not repair the systematic block. The
maintainer offered a `Bool` or a singleton type. The build takes a singleton type, the form the
maintainer chose for the rules of #1450 and #1451.

## Decision

**`CrossSectionalFactorPrior` takes a field `srep`, the Systematic Repair rule.** Its family is
the `public` abstract type `AbstractSystematicRepair`, with two exported members:

| Member | What the lift does to the systematic block before it adds the idiosyncratic block |
| --- | --- |
| `NoSystematicRepair()`, the default | Runs every step of `mp` except the `:pdm` step, in the order of `mp.order`, then takes the symmetric part `(si + si') / 2`. |
| `SystematicRepair()` | Runs every step of `mp`, the `:pdm` step included. This is the lift before this ADR. |

Under each member the lift then adds `D` and repairs the sum under `mp.pdm`, as before.

**Naming rule.** Each member names what the lift does to the systematic block before it adds the
idiosyncratic block. `GLOSSARY.md` states the rule under *Systematic Repair*.

**The verb.** Each member implements `systematic_processing!(srep, mp, sigma, X; kwargs...)`, which
the `# Interfaces` section of `AbstractSystematicRepair` states. `NoSystematicRepair` reads the
steps of `mp` one by one, so it takes a `MatrixProcessing` alone.

**The symmetric part.** `support_product` gives `si` symmetric only to round-off, and
`isposdef` refuses a matrix that is not exactly symmetric. The Newton repair of `si` made it exactly
symmetric, so the repair of the sum took the path with no repair. With no `:pdm` step on `si`, the
repair of the sum ran the Newton step on every read-out, and the default saved nothing. So
`NoSystematicRepair` takes the symmetric part of the block after its steps.

**One rule for the batch fit and the step.** The lift has one caller, `cross_sectional_assemble`,
which the batch fit and the read-out of the carry fold both run. So the carry fold equals the batch
fit under each member.

## Consequences

- Every stored output of the prior that reads `sigma` moves by round-off under the default, and
  every stored case still passes. The comparisons of `sigma` against the oracle measured a scaled
  error of 3.0e-13 to 7.3e-13 before, and measure 2.1e-16 to 1.2e-14 now. The cell-by-cell error
  fell from up to 1.3e-10 to at most 6.2e-12. So most of the error that the tests explained as a
  cancellation in `L F L'` was the repair of the systematic block. Each test states the new value
  and the old one at its line.
- `SystematicRepair()` gives the outputs from before this ADR, bit for bit.
- The read-out of the carry fold on the benchmark of #1562 does no Newton repair of a block that is
  positive semidefinite to round-off.
- The carry method of `partial_fit!` dispatches on the type parameter of `cache`, which the new
  field moves from place 25 to place 26.
