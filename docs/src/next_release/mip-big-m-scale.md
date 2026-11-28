```@meta
Description = "The default big-M constant of a mixed-integer model reads the bound that the model holds on each gated term and the scale of the variable k, in place of 1000 times the largest weight bound."
```

# [The default big-M constant](@id migration-next-mip-big-m-scale)

- **The default big-M constant of a mixed-integer model, `ss = nothing`.** The previous release read `1000` times the largest weight bound `b`. This release reads the bound that the model holds on each term that the constant releases, with no factor of its own.
  - **Every weight bound finite, and a constant `k`**, as under `MinimumRisk`: the constant is `b`, the exact bound of each weight. A sub-group cardinality or threshold gates the sum of each group, so it takes its own constant, the largest group sum of `max.(abs.(lb), abs.(ub))`.
  - **Every weight bound finite, and a variable `k`**, under `MaximumRatio` or `RiskBudgeting`: the constant is the ceiling of `k` times `max(1, b)`. The ceiling is `10 N / g` for a `RiskBudgeting` of `N` assets, and `10 ohf / mean(max.(min(g, 1) .* mu .- rf, 0))` for a `MaximumRatio`, where `g` is the smaller of `abs(bgt)` and the gross exposure that the weight bounds and the budgets admit. `mu` is the vector of the return terms.
  - **An infinite bound** holds nothing, so the constant stays `1000 b`, or `1000` with no finite bound. Under a variable `k` it is at least the ceiling times `max(1, b)`.

  The rows of these models keep `k` at or below the constant. A small bound gave a constant below `k`, and a large bound under a free `k` gave a constant so far above the gated terms that the solver failed. A `MixedIntegerRiskBudgeting` with bounds of `±10`, `bgt = nothing` and `card = 5` was `INFEASIBLE` with Pajarito at `10^4`. Each new constant releases its rows, so the optimum does not change, but the solver can take another path to the same book, or return another book of equal objective. A stated `ss` does not change. Write `ss = 1000 * maximum(abs, bounds)` for the constant of the previous release.
- **A `BudgetCosts` or a `BudgetMarketImpact` in `bgt`** bounds the sum of the weights by the bound of its own `bgt`. The previous release read no bound from it. The big-M constant that an XatRisk measure derives is smaller for long-only weights, and for long-short weights with a short budget. The default `kmin` of a `RiskBudgeting` is `1e-4 / abs(bgt)`, and it was zero.
