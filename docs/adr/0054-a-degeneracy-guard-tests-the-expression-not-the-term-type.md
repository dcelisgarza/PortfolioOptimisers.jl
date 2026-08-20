---
status: accepted
---

# A degeneracy guard tests the expression, not the term type

## Context

The package holds two sentinel types, `NoReturn` and `NoRisk`, and two inclusion flags,
`settings.rte` and `settings.rke`. A sentinel term builds an identically-zero expression. An
unset flag keeps a real term's expression out of the model-global sum. Both routes end in the
same place: a `:ret` or a `:risk` expression that is identically zero.

Every guard in the package tested the **type**. `noreturn_flag` and `norisk_flag` were type
tests, and every assertion keyed on one of them. The flags were guarded almost nowhere.

That asymmetry produced four defects.

1. **`MaximumReturn` accepted an all-`rte = false` configuration.** The objective was
   identically zero, so the solver returned an arbitrary feasible portfolio and reported
   success. The same configuration by the sentinel type was refused.
2. **`FactorRiskContribution` had no return-side guard at all.** It carries its own `obj`
   field, but only `MeanRisk`'s constructor called
   `assert_no_return_objective_compatibility`. So `FactorRiskContribution` was refused under
   `MaximumRatio`, which checks at model build, and accepted under `MaximumReturn`, which
   checked at `MeanRisk`'s constructor. It was refused unevenly against itself.
3. **`MaximumElementReturn(i)` was unguarded.** It maximises `ret_i` directly. A `NoReturn` at
   index `i` gives an identically-zero objective, whatever the other terms do.
4. **The risk axis had no `rke` check anywhere.** A sweep of `src/` found `rke` read only by
   the JuMP constraint builders and by `RiskMeasureTools`. `rke` is the exact twin of `rte`,
   and it was inherited unguarded from PR #21.

## Decision

**A degeneracy guard tests the state of the expression, not the type of the term. The
expression is degenerate when it is identically zero, by whichever route it got there.**

Two predicates carry the rule.

```julia
zero_return_expression_flag(r) = all(x -> isa(x, NoReturn) || !x.settings.rte, r)
zero_risk_expression_flag(r)   = norisk_flag(r) || all(x -> !x.settings.rke, r)
```

### The return axis fuses; the risk axis composes

The return predicate is **one fused test**, not a disjunction of two. Composition gives the
wrong answer on a mixed vector:

```julia
[NoReturn(), ArithmeticReturn(; settings = JuMPReturnsSettings(; rte = false))]
```

Every term is out of `:ret`, yet `all(isa NoReturn) || all(!rte)` is `false`. Only
`all(isa NoReturn || !rte)` sees it.

The risk axis composes because its two halves carry **different quantifiers**. `norisk_flag`
is `any` on the type, which is what ships and which keeps a `NoRisk` beside a real measure
refused on its own terms. The state half is `all`, because one included measure leaves
`:risk` non-zero. An `any` on the state would refuse a **constraint-only** measure:

```julia
[Variance(), Variance(; settings = RiskMeasureSettings(; rke = false, ub = u))]
```

That measure binds a `ub` without entering the objective. It is the risk-side twin of the
constraint-only return term, and it stays expressible.

`noreturn_flag` is **deleted**. `zero_return_expression_flag` subsumes it exactly, and both of
its callers take the fused form, so it loses every caller.

### The guards have two homes, and the split is principled

| Guard | Question it asks | Home |
| --- | --- | --- |
| `assert_return_term_required` | does this *formulation* need a return term? | `NearOptimalCentering`'s constructor |
| `assert_no_return_objective_compatibility` | does this *objective* need a non-zero `:ret`? | `set_return_constraints!` |

The formulation question is answerable from the estimator alone. The objective question is
not: the term lives on `JuMPOptimiser` and the objective on the optimiser, and the only site
that sees every objective-carrying optimiser is the shared model-build seam. Moving the
objective guard there covers `FactorRiskContribution` with no new call site, and no future
objective-carrying estimator can miss it.

The objective guard dispatches on the objective. `MaximumReturn` and `MaximumRatio` refuse a
zero `:ret`. `MaximumElementReturn(i)` refuses a `NoReturn` at index `i` and **ignores `rte`
entirely**, because it reads `ret_i` directly rather than the sum, so the flag cannot make its
objective zero. Every other objective takes a no-op fallback: a zero `:ret` is legitimate
under `MinimumRisk` and `MaximumUtility`.

The `MaximumRatio` empty-numerator check folds out of `set_max_ratio_return_constraints!` and
into this one hook.

The risk-side guards stay at the constructors. Moving them to model build would strip the
clustering optimisers of them, and it is not necessary: every risk-taking estimator except
`MeanRisk` already demands a real risk measure, so only `MeanRisk` can present a zero `:risk`
to an objective.

### `rke` is inert for the clustering optimisers

`assert_risk_measure_required` takes a `flag` keyword. The JuMP optimisers —
`RiskBudgeting`, `FactorRiskContribution` and `NearOptimalCentering` — pass
`zero_risk_expression_flag`. `HierarchicalRiskParity` and `HierarchicalEqualRiskContribution`
keep the default `norisk_flag`.

They keep it because they never reach `set_risk_and_scalarise!`, so `rke` never touches them.
Widening their predicate would refuse a configuration that solves correctly today and that the
flag never affected:

```julia
HierarchicalRiskParity(; r = Variance(; settings = RiskMeasureSettings(; rke = false)))
```

`RelaxedRiskBudgeting` has no risk measure, so it needs no guard.

## Consequences

- `MeanRisk`'s existing `NoReturn` errors move from **construction time to `optimise` time**.
  A caller who built a refused estimator and never solved it now sees no error until they do.
- The risk-axis change touches behaviour shipped since PR #21. An all-`rke = false`
  configuration that previously solved to a degenerate answer now throws a named error.
- `TimeDependent` needs no special handling at the model-build seam, because a schedule is
  already resolved by the time the seam runs. The constructor-level guards keep their schedule
  skip and are reached through `assert_time_dependent_substitution`, which re-runs the host's
  own constructor on each resolved entry.

## Related

- ADR 0052, which owns the shape of the return expression as a weighted sum of terms.
- ADR 0053, which owns the meaning of `settings.scale` in that sum.
