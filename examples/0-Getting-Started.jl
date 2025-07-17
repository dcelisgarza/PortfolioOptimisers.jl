#=
# Example 1: Simple `MeanRisk` optimisation

Here we show a simple example of how to use `PortfolioOptimisers.jl`. We will perform the classic Markowitz optimisation.
=#
using PortfolioOptimisers

#=
PrettyTables is used to format the example output.
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

Import the S&P500 data from a compressed `.csv` file. We will only use the last 253 observations.
=#
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = tsfmt)

#=
Compute the returns from the prices. The `ReturnsResult` struct contains the asset names in `nx`, asset returns in `X`, and timestamps in `ts`. The other fields are used in other applications which we will not be showcasing here.
=#
rd = prices_to_returns(X)

#=
## 2. MeanRisk optimisation

### 2.1 Creating a solver instance

All optimisations require some prior statistics to be computed. This can be done before calling the optimisation function, or inside it. For certain optimisations, precomputing the prior is more efficient. Here it makes no difference other than simplifying the example.

The `MeanRisk` estimator defines a mean-risk optimisation problem. It is a `JuMPOptimisationEstimator`, which means it requires a `JuMP`-compatible optimiser. For this we will use `Clarabel`.
=#
using Clarabel

#=
We now define a solver object, which can contain an optional user-defined name for logging purposes, the solver we wish to use, optional solver settings and kwargs for [`JuMP.assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#assert_is_solved_and_feasible). 

We can also provide a vector of `Solver` objects, which will be iterated over until one succeeds or all fail. This is useful for trying different solvers and/or settings combinations in a single call as it is often unclear what will work best for a given problem.
=#
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=

### 2.2 Defining the optimisation estimator

Our first hint of the composable design philosophy of `PortfolioOptimisers.jl` will come in the form of `JuMPOptimiser`. This is the structure used for all `JuMPOptimisationEstimator`s.

As you can see, `JuMPOptimiser` contains myriad properties that we are not going to be showcasing here.
=#

opt = JuMPOptimiser(; slv = slv)

#=
We then create the `MeanRisk` estimator, which takes a `JuMPOptimiser` as part of its construction. This also contains a few different properties.
=#

mr = MeanRisk(; opt = opt)

#=
### 2.3 Performing the optimisation

The `optimise!` function is used to perform all optimisations in `PortfolioOptimisers.jl`. Each one returns an `AbstractResult` object containing the optimisation results, which include a return code, a solution object, and as well as any relevant statistics which were either precomputed and provided via the optimisation estimator instances, or were computed during the function call.

From the return code `retcode`, we can see that our optimisation was successful.
=#

res = optimise!(mr, rd)

#=
Lets view the solution results as a pretty table. For convenience, we have ensured all `AbstractResult` have a property called `w`, which directly accesses `sol.w`. The optimisations don't shuffle the asset order so we can simply view the asset names and weights side by side.
=#

pretty_table(DataFrame(:assets => rd.nx, :weights => res.w); formatters = resfmt)

#=
### 3. Finite allocation

We have the optimal solution, but most people don't have access to effectively unlimited funds. So we can perform a finite allocation given a finite cash amount, the optimised weights, and current prices. For this we will use a discrete allocation method which uses mixed-integer programming to find the best allocation. We have another finite allocation method which uses a greedy algorithm, we will reserve it for later.

This time we need a solver capable of solving mixed-integer programming problems, for this we use `HiGHS`.
=#

using HiGHS

mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
da = DiscreteAllocation(; slv = mip_slv)

#=
Aside from the finite allocation estimator, we need the vector of weights assigned to each asset, a vector with their latest prices, and cash at our disposal. Luckily, the weights come from the optimisation result, the latest prices are the last entry of our original time array `X`, and lets say we have 4206.9 USD to invest.

The function can optionally take extra positional arguments to account for a variety of fees, but we will not use them here.
=#

mip_res = optimise!(da, res.w, vec(values(X[end])), 4206.9)

#=
This time, the result contains various interesting pieces of information. The reason various fields are prefixed by `l_`or `s_` is because the discrete allocation method splits the assets into long and short positions, which are recombined in the final result.

We can see the results in a pretty table.
=#

pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
                       :opt_weights => res.w, :mip_weights => mip_res.w);
             formatters = mipresfmt)

#=
We can see that the mip weights do not exactly match the optimal weights, but that is because we only have finite resources. Note that the sum of the costs minus the initial cash is equal to the `cash` property of the result. This changes when we introduce fees, which will be shown in a future example.
=#

isapprox(mip_res.cash, 4206.9 - sum(mip_res.cost))