---
status: accepted
---

# Every scalariser slot defaults to `nothing`, and `nothing` gives the sum

## Context

A scalariser slot reduces a vector of risk measures, or a vector of search scorers, to one value.
Issue #2069 found that the slots of the library did not agree on `nothing`.

1. **Most slots refused `nothing`.** `JuMPOptimiser.sca`, `HierarchicalRiskParity.sca`, the two
   slots `scai` and `scao` of `HierarchicalEqualRiskContribution`, `RiskLoss.sca`, the `sca` of
   `MeanReturnRiskRatio` and of `ExpectedReturnRiskRatio`, `sca1` and `sca2` of
   `NonOptimisationRiskRatio`, the results that copy these fields, and the `sca` keyword of
   `expected_risk` and its siblings took `Scalariser`, with the default `SumScalariser()`.
2. **Two slots took `nothing` and refused it with a vector.** `NearestQuantilePrediction.sca` and
   `FoldRiskScorer.sca` took `Option{<:Sca_VecToScaM}`. `nothing` was inert for one measure, and
   the constructor refused it with a vector, because `nothing` kept one score for each measure.
3. **The two slots of a search gave `nothing` a third meaning.** ADR 0232 and ADR 0233 let
   `r_sca = nothing` and `s_sca = nothing` keep one value for each member, and ADR 0234 gave one
   ranking for each value.

So one value, `nothing`, was refused, refused in part, or a request for a different result shape,
by the slot that held it. ADR 0222 states that `nothing` spells a trivial default, and that the
method on `Nothing` holds the behaviour.

## Decision

**Every scalariser slot is optional and defaults to `nothing`. With a vector, `nothing` gives the
sum of `SumScalariser`. With one member, every slot is inert, as before.**

| Item | Rule |
| --- | --- |
| The bound | Each slot takes `Option` of its family: `Option{<:Scalariser}`, `Option{<:NonHierarchicalScalariser}` for `JuMPOptimiser`, `TD_Option{<:…}` for a slot that takes a `TimeDependent` schedule, and `Option{<:Sca_VecToScaM}` for a score. |
| The default | `nothing`, on every field and every keyword. A Result copies `nothing` from its estimator, so `expected_risk(res.r, res; sca = res.sca)` still gives the figure that the optimisation reports. |
| The behaviour | The method on `Nothing` holds the sum. `scalarise(f, ::Option{<:SumScalariser}, itr)`, `scalarise_risk_expression!(model, ::Option{<:SumScalariser})`, `scalariser_gradient_weights(::Option{<:SumScalariser}, vals)` and the sum arm of `herc_intra_cluster_weights` take both. A slot that reaches none of these forwards `sca` to one of them. |
| The refusals | `NearestQuantilePrediction`, `sort_by_measure`, `quantile_by_measure` and `FoldRiskScorer` admit `nothing` with a vector. `assert_one_score_per_path` and `assert_one_score_per_asset` are removed, and `assert_path_reduction` checks only the keywords. |
| The search | `r_sca` and `s_sca` follow the same rule. The form that keeps one value for each member (ADR 0232, ADR 0233, ADR 0234) takes its own singleton type, a rule with content of its own, in #2001. `nothing` no longer spells it. |
| One member | Inert. No number of a call with one measure, or one search scorer, moves. |
| The cost | `nothing` dispatches to the method of the sum at compile time, so a call costs what a call with `SumScalariser()` cost. |

## Considered options

- **Keep `nothing` as the per-value form on the two slots of a search.** It is the multi-metric
  form of sklearn. Rejected by the maintainer (#2069): one value must mean one thing on every slot,
  and a per-value result is a rule with content, which takes a type of its own under ADR 0222.
- **Keep `SumScalariser()` as the default and refuse `nothing`.** Rejected: two slots already
  admitted `nothing`, and a slot that refuses the trivial default contradicts ADR 0222.
- **A forward method `scalarise(f, ::Nothing, itr) = scalarise(f, SumScalariser(), itr)` at each
  seam.** Rejected: it doubles each method of the sum. A bound `Option{<:SumScalariser}` on the one
  method gives the same dispatch.

## Consequences

- **A breaking change.** The default of each field is `nothing` in place of `SumScalariser()`. A
  caller that reads the field, such as `opt.sca isa SumScalariser`, now reads `nothing`. The value
  of every call is the same. `NearestQuantilePrediction`, `sort_by_measure`, `quantile_by_measure`
  and `FoldRiskScorer` with a vector and `sca = nothing` now give the sum, where they threw an
  `IsNothingError`. The migration guide names both.
- **ADR 0231, ADR 0232, ADR 0233 and ADR 0234 change in place**, because none of them reached
  `main`. Each `nothing` that named the per-value form now names the singleton of #2001, and the
  rows that refused `nothing` with a vector are removed.
- **The open tickets of map #1980.** #1998 and #2000 build `r_sca` and `s_sca` with the bound
  `Option{<:Sca_VecToScaM}` and the default `nothing`. #2001 builds the per-value singleton, and
  #2002 reuses its `K` matrices.
