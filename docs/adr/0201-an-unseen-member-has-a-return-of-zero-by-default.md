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

**The verb.** Each member implements `unseen_member_design(rule, fcb, B, Zl, X, W)`, which the
`# Interfaces` section of `AbstractUnseenMemberRule` states. It returns the design `Z` that the
regression solves and one matrix `P_t` per changed observation, and the fit stores `P_t h_t`, where
`h_t` are the coefficients of `t`. Under `ZeroUnseenMember`, `P_t` is the identity with a zero
column at each retained Unseen Member. When the dropped member is unseen, the condition moves onto
the retained member that the sample sees with the largest ratio. The design times the coefficients
equals the reduced exposures times the factor returns, so the residuals are those of the stored
factor returns.

**One rule for the batch fit and the carry fold.** The batch fit, the refit of the carry fold and
its step call the verb on the same rows, so the carry fold equals the batch fit under each member.

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
  29, and `unseen_member_design` takes six arguments, so the `arg` of its file rises from 3 to 6.
  The baseline records both. The inner constructor builds its type parameters from one tuple of
  the fields, so the new field adds no line to it.
- The diagnostics and the standard errors that rebuild the design of an observation still read the
  reduced exposures without `P_t`. The follow-up issue that #1606 names holds that work.
- Under `SolvedUnseenMember()`, the fold of a move of the dropped member
  ([#1605](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1605)) still solves each
  row with an Unseen Member again. Under the default it solves no row again.
