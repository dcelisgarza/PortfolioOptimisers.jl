The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/03_Constraints_and_Costs.jl"
```

# Constraints and costs

Real mandates are not unconstrained. You cap concentration, hold a sector band, limit how much
you trade at each rebalance, and pay transaction costs. In `PortfolioOptimisers.jl` these are
**keywords on the [`JuMPOptimiser`](@ref)** — the optimiser carries the constraints and costs,
the estimator carries the objective. This page shows the common ones with one minimal call
each; for the full treatment see the
[constraints & costs examples](../examples/4_constraints_costs/01_Budget_Constraints.md).

We fix one empirical prior and a minimum-risk objective so each keyword's effect is visible
against the same baseline.

````@example 03_Constraints_and_Costs
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_base = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))
````

## 1. Weight bounds

[`WeightBounds`](@ref) sets the per-asset lower and upper bound through the `wb` keyword. The
default is `lb = 0, ub = 1` (long-only, fully invested). Capping `ub` forces diversification —
no single name can exceed the bound.

````@example 03_Constraints_and_Costs
res_cap = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                wb = WeightBounds(; lb = 0.0, ub = 0.10))))
````

The budget itself is the `bgt` keyword (default `1.0`); `BudgetRange` and a separate short
budget `sbgt` let you build long/short and leveraged mandates — see
[Budget Constraints](../examples/4_constraints_costs/01_Budget_Constraints.md).

## 2. Linear and group constraints

Group and linear constraints are written as plain strings over an [`AssetSets`](@ref) and passed
through `lcse` as a [`LinearConstraintEstimator`](@ref) — the same syntax used for views. Name a
group, then bound it. Here we require the tech group to hold at least 15% (a floor the
unconstrained minimum-risk portfolio would not give it).

````@example 03_Constraints_and_Costs
sets = AssetSets(; dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"]))
res_grp = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                lcse = LinearConstraintEstimator(;
                                                                                 val = ["tech >= 0.15"]))))
````

The same `lcse` handles absolute and relative bounds (`"AAPL <= 0.1"`, `"MSFT >= AMD"`). For
constraints built from the asset *hierarchy* — phylogeny and centrality — see
[Phylogeny & Centrality](../examples/4_constraints_costs/03_Phylogeny_Centrality.md).

## 3. Turnover

Costs enter the same way. [`Turnover`](@ref) (`tn`) limits how far the new weights may drift
from a reference portfolio `w` — your current holdings — so a rebalance stays cheap. Here we
anchor at the current minimum-risk portfolio and re-solve under a turnover budget.

````@example 03_Constraints_and_Costs
res_tn = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               tn = Turnover(; w = res_base.w, val = 0.02))))
````

A tighter `val` keeps the result closer to the reference holdings; a looser one frees the
optimiser to move toward the unconstrained solution.

## 4. Fees

[`Fees`](@ref) (`fees`) charges proportional (and optionally fixed) transaction costs on long
and short positions, which the objective then trades off against return. The minimal form sets a
per-unit long fee.

````@example 03_Constraints_and_Costs
res_fee = optimise(MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                fees = Fees(; l = 0.001))))
````

Soft alternatives to hard turnover/position limits — L1/L2 weight regularisation and an
effective-number-of-assets floor (`l1`, `l2`, `nea`) — are covered in
[Regularisation](../examples/4_constraints_costs/06_Regularisation.md); benchmark
[`Tracking`](@ref) (the `tr` keyword) in
[Turnover & Tracking](../examples/4_constraints_costs/04_Turnover_and_Tracking.md).

## 5. Comparing the effect

Same prior, same objective — only the constraint or cost changes the allocation.

````@example 03_Constraints_and_Costs
results = [res_base, res_cap, res_grp, res_tn, res_fee]
labels = ["Base", "Cap 10%", "Tech ≥ 15%", "Turnover", "Fees"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights under each constraint / cost")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
