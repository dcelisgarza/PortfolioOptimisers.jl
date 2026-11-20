---
status: accepted
---

# A cost budget is a ceiling, and its floor bounds the invested weights

## Context

`BudgetCosts` and `BudgetMarketImpact` charge the cost of the trade against the net budget. The
trade from the reference `w0` splits into two parts, `w - k w0 = wp - wn` with `wp, wn >= 0`, and
the charge is `vp' wp + vn' wn`, or the same coefficients on the power-cone variables `wip` and
`win`. A number `bgt` wrote the equality `sum(w) + charge == k bgt`, and a `BudgetRange` wrote
`sum(w) + charge` between `k lb` and `k ub`.

The sources write the same rows. Equation 9.15 and Listing 9.8 of `cajas2025` write the equality.
Equation 6.2 of `mosek2023c` writes it too, and Section 6.1 of that work states that the budget
row keeps one of the two parts at zero at every optimum.

That statement is false. Nothing makes `wp` and `wn` complementary, so the charge is exact only at
a solution where `min(wp_i, wn_i) = 0` for each asset. A programme that gains from cash buys and
sells one asset together, charges the cost of both, and holds the charge as cash. Issue #2128
measured it under `MaximumReturn`, where a cap on the trade bound and the other assets had a low
return: the row charged `6.3e-4` for no trade. With every mean negative and caps of `0.05`, the
equality gave `w = [0.15, 0.25, 0.25, 0.197, 0.15]`. It bought two losing assets to the cap,
because the false trades could hide only `0.005` of cash. Under `MinimumRisk` the docstring of
`BudgetCosts` already warned that the row spent `0.0486` of the budget on trades that cost `0.0014`.
Section 6.3 of `mosek2023c` names the same defect for market impact: the solver treats the charge as
a riskless asset and throws money away.

The defect is in the form of the row, not in a coefficient. The cost `C` of a trade is convex. So
the set `sum(w) + C(w - k w0) <= k ub` is convex, and the split describes it exactly: for any
feasible `(w, wp, wn)`, the split `(max(Δw, 0), max(-Δw, 0))` is feasible too, because it charges
less and stays within the caps. The set `sum(w) + C(w - k w0) >= k lb`, and so the equality, is not
convex, and no convex row on the split describes it. Section 4.7 of `boyd2017` states this of the
self-financing condition, Equation 2.6 of that work, and relaxes it to an inequality.

## Decision

**The budget row of a cost budget is a ceiling, and the floor of a `BudgetRange` bounds the
invested weights alone. `BudgetCosts(; xbgt = true)` pins the budget exactly with one binary for
each asset.**

| Item | Rule |
| --- | --- |
| A number `bgt` | One row, `cost_bgt`: `sum(w) + charge <= k bgt`. |
| A `BudgetRange` | Its `ub` writes the ceiling `cost_bgt_ub`. Its `lb` writes `cost_bgt_lb`: `sum(w) >= k lb`, with no charge. |
| Cash | The part of the budget that the weights and their costs do not use. It is not a weight. |
| The charge | Exact in the weights at every solution of a convex programme. Where the ceiling does not bind, `cost_bgt_expr` can exceed the cost of the trade, so a caller computes the cost from the weights. |
| `BudgetCosts(; xbgt = true)` | The exact form, on request. One binary `z_i` for each asset keeps `wp_i <= up_i k z_i` and `wn_i <= un_i k (1 - z_i)`, so the split is complementary and a floor that holds the cost is exact. A number `bgt` then pins `sum(w) + cost`, and a `BudgetRange` floor holds the cost. `set_cost_budget_mip_constraints!` writes the rows in the MIP step of the block builder, which holds the big-M `ss`. When `k` is a variable, the product `k z_i` is a gate variable, as `declare_mip_indicators!` gates its bits. The default is `false`. |

A ceiling is the risk-free security that Section 6.3 of `mosek2023c` adds, at a zero rate. The
flag follows the `xbgt` of `JuMPOptimiser`, which pins the long and short parts with sign bits on
request and bounds them by default.

`BudgetMarketImpact` has no `xbgt`. The binaries make the split complementary, but the cone
variables `wip` and `win` can still exceed `wp^(1/beta)` and `wn^(1/beta)`, and an upper bound on a
convex power is not convex, so no mixed-integer row makes its floor exact.

## Considered options

- **Keep the rows, and state the relaxation in the docstrings.** Rejected: the result is wrong,
  and a docstring does not correct it. The hidden cash also depends on the caps `up` and `un`, so
  the rows state a budget that no caller writes.
- **Make the exact form with binaries the default.** Rejected: every cost budget then needs a
  mixed-integer solver, which costs the convex path its solver class. The maintainer asked for it
  as an option (#2128), so it is the flag `xbgt`.
- **A singleton type in a field in place of a `Bool`.** Rejected: the choice has two members, and
  the `xbgt` of `JuMPOptimiser` and the `fixed` of the same estimator spell the same kind of choice
  as a `Bool`.
- **Penalise the charge in the objective**, the first remedy of Section 6.3 of `mosek2023c`.
  Rejected: the penalty needs a calibrated weight, and it changes the optimum that it protects.
- **Keep the floor on the weights and their costs.** Rejected: it is the non-convex half of the
  equality, so it has the same defect. A floor on the invested weights is convex and exact, and it
  bounds the cash as a caller needs.

## Consequences

- **A breaking change.** A number `bgt` no longer pins `sum(w) + cost`. A programme that gains
  from cash now holds it in the open: `MinimumRisk` under a number sells the book to cash, where it
  spent the budget on false trades. A `BudgetRange` floor now bounds `sum(w)`, and not
  `sum(w) + cost`. A programme that preferred a larger invested budget, such as `MaximumReturn` on
  assets with a positive mean, gives the same weights as before.
- **`MaximumRatio` with `rf = 0`** finds the ratio, but not the share of cash, because cash does
  not change that ratio. A positive `rf`, or a floor, fixes it. The docstring of `BudgetCosts`
  states the measured range.
- **`BudgetCosts` has a field `xbgt` after `fixed`**, so its positional constructor takes eight
  arguments. `xbgt = true` gives the exact equality of v0.33 without its false charge, with a
  mixed-integer solver.
- The migration guide names the break, and `GLOSSARY.md` defines the Cost Budget.
