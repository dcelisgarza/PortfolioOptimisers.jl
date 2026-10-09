---
status: accepted
---

# An Unseen Member has a return of zero by default

## Context

A constrained Factor Family of a `CrossSectionalFactorPrior` carries a benchmark-weighted
zero-sum condition, and the Factor Family Basis imposes it by dropping one member. The exposures
lag the returns, so the condition of an observation reads the benchmark weights of the lagged
observation. An asset that loads on a member there can leave the regression sample of the
observation, for example when it delists.

[#1606](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1606) found such a row. On
`parity_panel(; T = 300, N = 60, seed = 1601)`, the industry Utilities holds one asset, which
delists after data row 225. At data row 226 no asset of the sample loads on Utilities, but the
condition still weights it. The member then fixes the condition by itself, the condition no longer
identifies the other members, and the reduced design is rank-deficient. The pseudo-inverse answer
depends on the dropped member: 7.6e-4 on Utilities between the automatic member and Energy, and
2.5e-5 on the market and the other industries. Every other row agreed to 6.7e-17.

The oracle gives the same answers on that panel, to 2.3e-16 on every row, under both members. Its
batch solve fails on the singular row, and it takes the pseudo-inverse of every row. So the
oracle's answer at such a row depends on the dropped member too.

The data state nothing about the return of the member at such a row, so any value is a choice. A
value of zero matches two cases that the prior already answers. An Empty Factor has a return of
zero. At the rows after 226, the lagged observation gives Utilities no benchmark weight, so its
column of the reduced design is zero, and the fit already gives it a return of zero there. A
minimum-norm answer on the raw axis would not depend on the dropped member either. But it depends
on the scale of the factors, and it gives the member a return that no data state.

## Decision

**`CrossSectionalFactorPrior` takes a field `unseen`, the Unseen Member rule.** Its family is the
`public` abstract type `AbstractUnseenMemberRule`, with two exported members:

| Member | The return of an Unseen Member, and the zero-sum condition of its observation |
| --- | --- |
| `ZeroUnseenMember()`, the default | Zero. The condition holds over the members that the sample sees. |
| `SolvedUnseenMember()` | The member stays in the condition, and the solve algorithm of `cre` gives its return. This is the fit before this ADR, and the oracle's. |

An Unseen Member is a member that no asset of positive regression weight loads on at one
observation. `GLOSSARY.md` states the term.

**The verb.** Each member implements `unseen_member_design(rule, fcb, B, Zl, W)`, which the
`# Interfaces` section of `AbstractUnseenMemberRule` states. The pairs of positive weight in `W` are
the sample of each observation. It returns the design `Z` that the regression solves and one matrix
`P_t` per changed observation, and the fit stores `P_t h_t`, where `h_t` are the coefficients of
`t`. Under `ZeroUnseenMember`, `P_t` is the identity with a zero
column at each retained Unseen Member. When the dropped member is unseen, the condition moves onto
the retained member that the sample sees with the largest ratio. The design times the coefficients
equals the reduced exposures times the factor returns, so the residuals are those of the stored
factor returns. `P_t` is idempotent, so the changed design times the stored factor returns gives
the same fitted values.

**One rule for the batch fit and the carry fold.** The batch fit, the refit of the carry fold and
its step call the verb on the same rows, so the carry fold equals the batch fit under each member.

**The consumers read the design that the fit regressed on
([#1609](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1609)).** The
`CrossSectionalFactorModel` block stores the rule in a field `unseen`. A block that a caller builds
without a rule takes `SolvedUnseenMember()`, which changes no design. The regression diagnostics and
the standard errors of `factor_attribution` call the verb on the stored exposures, the basis and
the regression weights `rw`, so they rebuild `Z` and `P_t` from the block alone:

- The VIF and the condition number read `Z`. An Unseen Member has a zero column there, and so has
  the member that carries the condition when the dropped member is unseen, so their VIF is `NaN`.
- The t-statistics read the diagonal of `P_t G_t^+ P_t'`, where `G_t` is the Gram matrix of `Z`,
  because the stored factor returns are `P_t h_t`. An Unseen Member has a standard error of zero,
  so its t-statistic is `NaN`.
- The sandwich of the attribution is taken on `Z` and mapped to `P_t V_t P_t'`. An Unseen Member has
  a zero row and a zero column.

The bare-array method of `factor_attribution` takes the rule as the keyword `unseen`, with the
default `SolvedUnseenMember()`. The rule types are in `01_CrossSectionalRegression.jl`, which loads
before the block.

## Consequences

- Under the default, the factor returns of every row do not depend on the dropped member. On the
  panel of #1606 they agree to 6.7e-17 on every row, row 226 included. Row 226 is the only row
  that changes, and the residuals of its sample move by 3.5e-17 at most.
- The residual of an asset outside the sample reads the return of each Unseen Member that it loads
  on, so it reads zero. Before, it read a value that depended on the dropped member.
- Every stored case of the oracle states `SolvedUnseenMember()`, through `grid_config` of
  `test/parity_grid.jl`, and gives the outputs from before this ADR. `test_12x` checks that the
  default differs from the solved rule only at the rows where the solved rule depends on the
  dropped member.
- On the small grid panel, whose only Utilities asset delists, the default moves the factor
  returns of one row by up to 4.6% of the largest factor return, and through them the factor
  moments and the asset moments.
- The carry method of `partial_fit!` dispatches on the type parameter of `cache`, which the new
  field moves from place 28 to place 29.
- Complexity: the inner constructor takes one argument per field, so its `arg` rises from 28 to
  29, and `unseen_member_design` takes five arguments, so the `arg` of its file rises from 3 to 5.
  The baseline records both. The inner constructor builds its type parameters from one tuple of
  the fields, so the new field adds no line to it.
- Complexity of #1609: the field `unseen` of the block raises the `arg` of its file from 22 to 23,
  the keyword `P` of `cs_regression_t_stats` raises the `arg` of the diagnostics from 5 to 6, and
  the keyword `unseen` of the bare-array attribution raises the `arg` of its file from 11 to 12.
  The baseline records the three. The inner constructor of the block builds its type parameters
  from one tuple of the fields, as the prior does, so the block file keeps its size.
- On the panel of #1606, the diagnostics and the standard errors of row 226 read the design that
  the fit regressed on, and they do not depend on the dropped member. Before #1609 the old design
  gave `NaN` t-statistics and an `Inf` VIF to the market, Banks and Utilities at that row, and its
  sandwich took the pseudo-inverse of the rank-deficient Gram matrix.
- Under `SolvedUnseenMember()`, the fold of a move of the dropped member
  ([#1605](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1605)) still solves each
  row with an Unseen Member again
  ([#1613](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1613)). Under the default
  it solves no row again.
