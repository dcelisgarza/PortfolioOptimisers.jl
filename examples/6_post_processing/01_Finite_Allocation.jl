#=
```@meta
Description = "Finite allocation in PortfolioOptimisers.jl: turn continuous weights into whole shares under a cash budget with greedy or discrete allocation."
```

# Finite allocation

An optimiser returns continuous weights, which are fractions of capital. To trade them you buy
whole shares at market prices with a fixed amount of cash. The rounding moves the portfolio you
hold away from the target weights, and the smaller the account, the larger the move.
`PortfolioOptimisers.jl` has two finite-allocation optimisers, and you call both as
`optimise(allocator, FiniteAllocationInput(; w, prices, cash))`.

  - [`GreedyAllocation`](@ref) rounds each weight to whole shares, or to lots, and then spends the
    cash left over on the assets furthest below their target. It needs no solver.
  - [`DiscreteAllocation`](@ref) solves a mixed-integer programme. It chooses the share counts
    that minimise the sum of the absolute differences, in money, between the target and the bought
    positions, plus the cash left over. It needs a solver for mixed-integer programmes.

!!! tip "When to reach for this"
    Reach for a finite allocation as the last step before you trade, because you buy shares, not
    fractions of capital. Use [`GreedyAllocation`](@ref) when you want an answer at once, and for a
    large portfolio where the mixed-integer programme is slow. Use [`DiscreteAllocation`](@ref)
    when the account is small enough that the rounding matters and you want the share counts that
    the mixed-integer programme finds. The sum of the absolute differences between the weights you
    hold and the target weights is the rounding error, and each table below prints it.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. A target portfolio and prices

A finite allocation needs the target weights and a price per share. We optimise a minimum-risk
portfolio for the weights and take the latest prices from the price table.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res = optimise(MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pr, slv = slv)))
prices = vec(values(X)[end, :])

#=
## 2. Greedy allocation

[`GreedyAllocation`](@ref) needs no solver. Its result holds the whole number of `shares` per asset,
the `cost` per asset, the weights `w` of the shares bought, and the `cash` left over. We allocate a
budget of \$100,000. The function `rounding_error` sums the absolute differences between the weights
bought and the target weights, and the title of the table prints it with the cash left over.
=#

cash = 100_000.0
greedy = optimise(GreedyAllocation(),
                  FiniteAllocationInput(; w = res.w, prices = prices, cash = cash))

rounding_error(alloc) = sum(abs, alloc.w .- res.w)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => res.w,
                       "Shares" => round.(Int, greedy.shares), "Realised" => greedy.w);
             formatters = [resfmt],
             title = "Greedy allocation of \$$(round(Int, cash)) — leftover cash \$$(round(greedy.cash, digits = 2)), rounding error $(round(rounding_error(greedy), digits = 4))")

#=
## 3. Exact allocation with a mixed-integer solver

[`DiscreteAllocation`](@ref) optimises all the share counts together, where the greedy method
takes one asset at a time. It needs a mixed-integer solver, and we use
[HiGHS](https://github.com/jump-dev/HiGHS.jl). The table compares the cash left over and the
rounding error of the two methods.
=#

mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
discrete = optimise(DiscreteAllocation(; slv = mip_slv),
                    FiniteAllocationInput(; w = res.w, prices = prices, cash = cash))

pretty_table(DataFrame("Method" => ["Greedy", "Discrete (MIP)"],
                       "Leftover cash" => [greedy.cash, discrete.cash],
                       "Rounding error" =>
                           [rounding_error(greedy), rounding_error(discrete)]);
             formatters = [resfmt], title = "Greedy vs exact allocation")

#=
## 4. Lot sizes

Many instruments trade in lots, not single shares. `GreedyAllocation(; unit = u)` rounds to
multiples of `u` shares. A larger lot gives a larger rounding error and can leave
more cash unspent. The table compares single shares with lots of ten.
=#

greedy_lots = optimise(GreedyAllocation(; unit = 10),
                       FiniteAllocationInput(; w = res.w, prices = prices, cash = cash))

pretty_table(DataFrame("Allocation" => ["Single shares", "Lots of 10"],
                       "Leftover cash" => [greedy.cash, greedy_lots.cash],
                       "Rounding error" =>
                           [rounding_error(greedy), rounding_error(greedy_lots)]);
             formatters = [resfmt], title = "Lot size coarsens the allocation")

#=
## 5. The budget sets the rounding error

A rounding error too small to matter on a large account is large on a small one. We allocate the
same target with budgets of \$100,000, \$25,000 and \$5,000, and the table prints the rounding error
of each. On a small account, the choice of allocation method and of lot size changes the portfolio
you hold the most.
=#

budgets = [100_000.0, 25_000.0, 5_000.0]
budget_allocs = [optimise(GreedyAllocation(),
                          FiniteAllocationInput(; w = res.w, prices = prices, cash = c))
                 for c in budgets]

pretty_table(DataFrame("Budget" => budgets,
                       "Leftover cash" => [a.cash for a in budget_allocs],
                       "Rounding error" => [rounding_error(a) for a in budget_allocs]);
             formatters = [resfmt], title = "Rounding error by budget")

#=
[`FiniteAllocationInput`](@ref) also takes a `fees` keyword, a [`Fees`](@ref), which needs a
`horizon` in periods too. With fees, both allocators choose share counts whose cost and fee fit in
the cash. [Fees and net returns](../4_constraints_costs/06_Fees_and_Net_Returns.md) covers
the fees.

## 6. Target and allocated weights

The plot puts the target weights and the weights of the two allocations side by side.
=#

plot_stacked_bar_composition([res, greedy, discrete], rd;
                             xticks = (1:3, ["Target", "Greedy", "Discrete"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive (6_post_processing). Verified on kaimon (f102cae9) with HiGHS as MIP solver:
#src   - GreedyAllocation 100k: invested $99,986, leftover $14.22, drift 0.0033.
#src   - DiscreteAllocation(HiGHS) 100k: identical here ($99,986, drift 0.0033) — greedy already
#src     optimal on this book; framed honestly (MIP's edge is on tight budgets).
#src   - Budget sweep monotone: drift 100k=0.0033, 5k=0.0947 (~30x). Clean "small accounts suffer"
#src     story.
#src   - GreedyAllocation(unit=10) 100k: leftover $34.64, drift 0.0407 (rerun 2026-09-23, #1226).
#src     The old NEGATIVE leftover (-$268.44) is gone: both greedy passes now test the cost plus
#src     the fee against the cash left, so a lot never overdraws the budget.
#src   - Budget sweep, discrete vs greedy drift: 25k 0.0147 vs 0.0174, 5k 0.0849 vs 0.0947; at 5k
#src     the discrete allocation leaves more cash ($15.81 vs $5.37). DiscreteAllocation has no lots.
#src - optimise(GreedyAllocation()|DiscreteAllocation(; slv=mip), FiniteAllocationInput(; w, prices, cash)); prices =
#src   vec(values(X)[end,:]). Result fields shares/cost/w/cash.
