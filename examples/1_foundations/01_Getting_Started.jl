#=
```@meta
Description = "Getting started with PortfolioOptimisers.jl: a classic Markowitz mean-variance optimisation with MeanRisk, from prices to weights."
```

# Getting started: a simple `MeanRisk` optimisation

This page runs the classic Markowitz optimisation with `PortfolioOptimisers`, from a table of prices to a portfolio you can buy with a fixed amount of cash.
=#

using PortfolioOptimisers

#=
We use PrettyTables to format the tables the page prints.
=#

using PrettyTables

## Format for pretty tables.
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
mipresfmt = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Load the data

We load the S&P 500 prices from a compressed `.csv` file and keep the last 253 observations.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

#=
The optimiser works on returns, so we compute them from the prices. The [`ReturnsResult`](@ref) that comes back stores the asset names in `nx`, the asset returns in `X` and the timestamps in `ts`. Its other fields store data that this page does not use.
=#

rd = prices_to_returns(X)

#=
## 2. MeanRisk optimisation

### 2.1 Creating a solver instance

Every optimisation needs statistics of the returns, such as their mean and covariance, which the library calls the prior. You can compute the prior before the optimisation and pass it in, or let the optimisation compute it. Computing it first saves time when several optimisations share one prior. Here there is only one optimisation, so we let it compute its own.

The [`MeanRisk`](@ref) estimator states a mean-risk optimisation problem. The library builds that problem as a `JuMP` model, so `MeanRisk` needs a solver that `JuMP` can call. We use `Clarabel`.
=#

using Clarabel

#=
A [`Solver`](@ref) takes the solver to use, an optional name that appears in the logs, optional solver settings, and optional keyword arguments for [`JuMP.assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#assert_is_solved_and_feasible).

A hard problem can fail with one solver or one group of settings and succeed with another. For that case you can pass a vector of `Solver` objects, and the optimisation tries each in turn until one succeeds or all fail. The Markowitz problem is easy to solve, so we use one solver.
=#

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=

### 2.2 Defining the optimisation estimator

`PortfolioOptimisers` builds an optimisation out of smaller estimators. `MeanRisk` takes its solver, its constraints and its prior estimator from a [`JuMPOptimiser`](@ref), in its `opt` field.

We create a `MeanRisk` estimator. The printed output lists many more fields of `JuMPOptimiser` and `MeanRisk` than this page uses.
=#

mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

#=
### 2.3 Performing the optimisation

The [`optimise`](@ref) function runs every optimisation in `PortfolioOptimisers`. It returns a result with a return code, the solution, and the statistics the optimisation used, whether you computed them first or the optimisation did.

The `retcode` field of the result is an `OptimisationSuccess`, which means that the solver found a solution.
=#

res = optimise(mr, rd)

#=
An optimisation result has a `w` property that returns the weights, `sol.w`. The optimisation keeps the assets in the order of the returns, so the table below puts each asset name beside its weight.
=#

pretty_table(DataFrame(:assets => rd.nx, :weights => res.w); formatters = [resfmt])

#=
## 3. Visualising the portfolio

We plot the composition of the portfolio, its cumulative returns, the distribution of its returns and its drawdowns.
=#

using StatsPlots, GraphRecipes
# The portfolio weights as a bar chart.
plot_composition(res, rd)
# The cumulative returns of the portfolio over the sample.
plot_portfolio_cumulative_returns(res, rd)
# A histogram of the portfolio returns, with markers for the value at risk and the conditional value at risk.
plot_histogram(res, rd)
# The drawdowns, each the loss from the highest value the portfolio reached before it.
plot_drawdowns(res, rd)

#=
## 4. Finite allocation

The weights are fractions of capital, but an investor buys whole shares with a fixed amount of cash. A finite allocation turns the weights, the latest prices and the cash into a number of shares per asset. We use [`DiscreteAllocation`](@ref), which solves a mixed-integer programme for the best whole-share portfolio. The library also has a greedy method that can allocate fractional shares, and [Finite allocation](../6_post_processing/01_Finite_Allocation.md) compares the two.

A mixed-integer programme needs a solver that supports it, and we use `HiGHS`.
=#

using HiGHS

mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
da = DiscreteAllocation(; slv = mip_slv)

#=
The allocation needs three inputs. The weights come from the optimisation, the latest prices are the last row of the price table `X`, and we invest `4206.9` USD.

A [`FiniteAllocationInput`](@ref) takes the three. It also takes a time horizon and fees, which this page does not use.
=#

mip_res = optimise(da,
                   FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                         cash = 4206.9))

#=
This result has different fields from the optimisation result. The discrete allocation solves the long and the short positions apart and then combines them, and the fields that start with `l_` or `s_` belong to the two halves.

The table below puts the shares, their cost, the optimised weights and the weights of the allocation side by side.
=#

pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
                       :opt_weights => res.w, :mip_weights => mip_res.w);
             formatters = [mipresfmt])

#=
The weights of the allocation differ from the optimised weights, because a whole number of shares at a fixed cash amount can only come close to them. The `cash` property of the result is the cash left over. We print it beside the starting cash less the sum of the costs. When the input has fees, the allocation also subtracts the fees from the cash left over.
=#

println("cash left over: $(round(mip_res.cash; digits = 2)), starting cash less the costs: $(round(4206.9 - sum(mip_res.cost); digits = 2))")

#=
The cost of each asset is its number of shares times its price, and the next cell compares the two.
=#

println("cost of shares ≈ cost of portfolio: $(all(isapprox.(mip_res.shares .* vec(values(X[end])), mip_res.cost)))")

#=
The last plot puts the optimised weights and the weights of the allocation side by side. The two differ where the allocation rounds a weight to whole shares.
=#

plot_stacked_bar_composition([res, mip_res], rd)
