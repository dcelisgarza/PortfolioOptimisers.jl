#=
```@meta
Description = "Post-processing in PortfolioOptimisers.jl: turn continuous weights into whole shares under a cash budget, and plot the result."
```

# Post-processing

An optimiser returns continuous weights, which are fractions of the capital. To trade them, you
need whole shares, and to show them to others, you need a report. Post-processing does both. It
turns the weights into whole numbers of shares under a cash budget, and it plots the result. For
more, see the [post-processing examples](../examples/6_post_processing/01_Finite_Allocation.md).
=#

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

res = optimise(MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
## 1. Finite allocation

Finite allocation turns the weights into whole numbers of shares that you can buy with a cash
budget at the last prices. [`GreedyAllocation`](@ref) needs no solver. It goes through the assets
from the largest target weight down, and buys as many whole shares as the target weight pays for.
It then spends the cash that is left on single shares of the assets furthest below their target.
=#

prices = vec(values(X)[end, :])
cash = 100_000.0
alloc = optimise(GreedyAllocation(),
                 FiniteAllocationInput(; w = res.w, prices = prices, cash = cash))

#=
The result holds the whole shares in `shares`, the cost of each asset in `cost`, the weights after
the rounding in `w`, and the cash that is left in `cash`. In the table, the realised weights are
close to the target weights, and the title shows the cash that is left.
=#

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target weight" => res.w,
                       "Shares" => round.(Int, alloc.shares), "Realised weight" => alloc.w);
             formatters = [resfmt],
             title = "Discrete allocation of \$$(round(Int, cash)) — invested \$$(round(Int, invested)), cash left \$$(round(alloc.cash, digits = 2))")

#=
[`DiscreteAllocation`](@ref) solves a mixed-integer program to find an exact allocation, and it
needs a [`Solver`](@ref) that handles integer variables. See the
[finite allocation example](../examples/6_post_processing/01_Finite_Allocation.md).

## 2. Reporting

[`plot_stacked_bar_composition`](@ref) plots the weights, [`plot_measures`](@ref) plots the risk
against the return of a set of portfolios, [`plot_risk_contribution`](@ref) plots the risk that
each asset contributes, and [`plot_prior`](@ref) plots the moments of the prior. We plot the
weights of the minimum-risk portfolio.
=#

plot_stacked_bar_composition([res], rd; xticks = (1:1, ["Min risk"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow guide page: GreedyAllocation (solver-free) as the blessed finite-allocation path,
#src   DiscreteAllocation (MIP) as the exact alternative. Verified on kaimon (f102cae9):
#src   100k budget → $99,986 invested, $14.22 leftover, realised weights within 0.06% of target.
#src - optimise(GreedyAllocation(), FiniteAllocationInput(; w, prices, cash)) — prices = vec(values(X)[end,:]) (latest row).
#src   Result fields: shares / cost / w (realised) / cash (leftover).
#src - Reporting section points to the plot_* family; 6_post_processing/02_Plotting_and_Reporting
#src   example NOT YET AUTHORED — cross-link resolves once that group lands.
