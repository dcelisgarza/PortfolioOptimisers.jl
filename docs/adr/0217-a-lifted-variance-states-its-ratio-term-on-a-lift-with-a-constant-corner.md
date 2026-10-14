---
status: accepted
---

# A lifted variance states its ratio term on a lift with a constant corner

## Context

`MaximumRatio` solves the ratio in the homogenised weights `y = k w`, and a risk expression of
degree `d` in `(y, k)` gives the maximiser of `(ret - rf) / R(w)^(1/d)`. A variance that the model
states on the lifted matrix of `set_sdp_constraints!`, `[W y; y' k] ⪰ 0`, is linear in `W`, so it
has degree one. Four terms do this: the semidefinite `Variance` (risk-contribution rows, a
`SemiDefinitePhylogeny`, every variance of `FactorRiskContribution`) and the box, the ellipsoid and
the covariance norm ball of `UncertaintySetVariance`. Under `MaximumRatio` they maximised
`(ret - rf) / σ²(w)`, not the Sharpe ratio, while the SOC variance and the compact set, of degree
two, gave the Sharpe ratio. So the type of a set changed the ratio, and a set of radius zero did not
give the nominal answer (#1676). Issue #1321 had ruled that the degree belongs to the formulation,
because no convex form of degree two was known.

A form of degree two exists. A second lift with a constant corner, `[Wc y; y' c] ⪰ 0`, gives
`tr(Σ Wc) ≥ y' Σ y / c = k² σ²(w) / c`, which is a linear matrix inequality and has degree two at
every constant `c`. The oracle moved its own lifted worst-case variance to this form at v1.8.5, with
`c = 1`. The bound `risk ≤ ub · k` needs degree one, so it keeps the lift with the corner `k`.
Measured on #1676: the corner `1` conditions the solve badly when a stated `ohf` moves `k` far from
one, and a corner near the expected `k` does not.

## Decision

1. A lifted variance under `MaximumRatio` states its objective term on a second lift with a constant
   corner, `set_sdp_objective_lift!`, when the term is a positive term of the objective (`rke` and
   `scale > 0`). The bound keeps the lift with the corner `k`, and the model builds that lift only
   when the measure states a bound. Under every other objective `k` is `1`, and the model builds one
   lift.
2. `Variance` and `UncertaintySetVariance` take a field `lift::SemiDefiniteLift`. The default
   `ConstantCornerLift()` is the decision above. `ScaleCornerLift()` keeps the degree-one term on the
   lift with the corner `k`, the objective `(ret - rf) / σ²(w)`. The maintainer ruled that the old
   ratio stays as an option behind a field of the measure.
3. Both measures take a field `corner::Option{<:Number}`. A number is the constant corner. `nothing`,
   the default, derives it: the head registers `sdp_corner` with `set_sdp_lift_corner!`, `1` when
   `ohf` is the default, and `ohf / clamp(mean(abs, mu), 1e-3, 1e3)` of the prior's `mu` when the
   caller states `ohf`. The maintainer ruled that the corner is a field of the measure, derived when
   it is `nothing`.
4. The objective term is `tr(Σ Wc)`, which is `y' Σ y / c` at rank one. It is not multiplied by `c`:
   a constant factor does not move the maximiser, and the product puts the term below the tolerance
   of the solver when a small `ohf` makes `k` small.
5. The lift belongs to the weights, as `W` does, so all the measures on one weight vector share it.
   A second measure that resolves to another corner is refused, `assert_one_lift_corner`.
6. The risk-contribution rows of a variance read the lift of its objective term. A semidefinite
   phylogeny reads the lift of the objective when the objective minimises the risk and the lift
   exists, and then adds no penalty. Otherwise it keeps the rule of `mark_objective_variance!`.

## Consequences

- A set of radius zero gives the tangency weights of `Variance()`. Measured on the fixture of
  `test_18j`: 1.0e-3 at most, against 0.15 under `ScaleCornerLift`.
- The answers of four stored columns move: column 10 of the norm-ball fixture of `test_10d`, now the
  oracle at v1.8.5 (the old column is kept as column 11, the `ScaleCornerLift` case), and columns
  17, 18, 23 and 24 of `MeanRisk1.csv.gz`.
- Beside a risk term of another degree, the weight of the term changes with `c`, as the weight of a
  term of degree two changes with `ohf`. A ratio of mixed degrees already depends on `ohf`.
- A model with a lifted objective term and a bound holds two lifted matrices.
- Risk-contribution rows under `MaximumRatio` now relax as they do under `MinimumRisk`. On the
  fixture of `test_18l` the rows hold on `Wc`, but its second eigenvalue is 0.81 of the first, so
  the returned weights miss them (largest share 0.29 against a bound of 0.2). The lift with the
  corner `k` was tight on that fixture. A caller who needs the rows on the weights can state
  `lift = ScaleCornerLift()` and check the shares, as the `Variance` docstring says.

## Alternatives considered

- **The corner `1`, as the oracle.** It matches the default `ohf`, where the derived corner is `1`
  too. Rejected for a stated `ohf`: on the fixture of #1676 it stopped `0.017` from the tangency
  weights at `ohf = 1` and `0.065` at `ohf = 1e-6`.
- **The term `c · tr(Σ Wc)`, equal to `y' Σ y` at rank one.** Rejected: at `ohf = 1e-6` the term is
  of the order of `1e-12`, and the solve stopped `0.16` from the tangency weights while it reported
  success.
- **Remove the degree-one ratio.** Rejected by the maintainer.

## Related

- Issue #1676, issue #1321.
- `ConstantCornerLift`, `ScaleCornerLift`, `set_sdp_objective_lift!`, `set_sdp_lift_corner!`.
- The `## The degree of the risk` subsection of `MaximumRatio`.
