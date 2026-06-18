#=
# Post-processing

An optimiser returns *continuous* weights — fractions of capital. To trade them you need whole
shares, and to communicate them you need a report. Post-processing covers both: turning weights
into an integer share count under a cash budget, and visualising the result. For the full
treatment see the
[post-processing examples](../examples/6_post_processing/01_Finite_Allocation.md).
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

Finite allocation converts the continuous weights into integer share counts you can actually
buy, given the latest prices and a cash budget. [`GreedyAllocation`](@ref) is the solver-free
option: it rounds to whole shares and spends the leftover cash on the largest underweights. The
call takes the weights, the price vector, and the available cash.
=#

prices = vec(values(X)[end, :])
cash = 100_000.0
alloc = optimise(GreedyAllocation(), res.w, prices, cash)

#=
The result carries the integer `shares`, the per-asset `cost`, the *realised* weights `w` (after
rounding), and the leftover `cash`. The realised weights track the target closely, and only a
few dollars are left uninvested.
=#

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target weight" => res.w,
                       "Shares" => round.(Int, alloc.shares), "Realised weight" => alloc.w);
             formatters = [resfmt],
             title = "Discrete allocation of \$$(round(Int, cash)) — invested \$$(round(Int, invested)), cash left \$$(round(alloc.cash, digits = 2))")

#=
For an *exact* (rather than greedy) allocation, [`DiscreteAllocation`](@ref) solves a
mixed-integer program — pass it a MIP-capable [`Solver`](@ref). It is more precise but needs a
MIP solver; the greedy method needs none. See
[Finite Allocation](../examples/6_post_processing/01_Finite_Allocation.md).

## 2. Reporting

The plotting functions used throughout this guide are the reporting toolkit:
[`plot_stacked_bar_composition`](@ref) for weights, [`plot_measures`](@ref) for
risk/return scatters and frontiers, [`plot_risk_contribution`](@ref) for where the risk sits,
and [`plot_prior`](@ref) for the input moments. Here is the realised portfolio's composition.
=#

plot_stacked_bar_composition([res], rd; xticks = (1:1, ["Min risk"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow guide page: GreedyAllocation (solver-free) as the blessed finite-allocation path,
#src   DiscreteAllocation (MIP) as the exact alternative. Verified on kaimon (f102cae9):
#src   100k budget → $99,986 invested, $14.22 leftover, realised weights within 0.06% of target.
#src - optimise(GreedyAllocation(), w, prices, cash) — prices = vec(values(X)[end,:]) (latest row).
#src   Result fields: shares / cost / w (realised) / cash (leftover).
#src - Reporting section points to the plot_* family; 6_post_processing/02_Plotting_and_Reporting
#src   example NOT YET AUTHORED — cross-link resolves once that group lands.
