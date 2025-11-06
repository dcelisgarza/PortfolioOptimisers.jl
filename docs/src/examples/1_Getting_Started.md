The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/1_Getting_Started.jl"
```

# Example 1: Simple `MeanRisk` optimisation

Here we show a simple example of how to use `PortfolioOptimisers`. We will perform the classic Markowitz optimisation.

````@example 1_Getting_Started
using PortfolioOptimisers
````

PrettyTables is used to format the example output.

````@example 1_Getting_Started
using PrettyTables

# Format for pretty tables.
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
nothing #hide
````

## 1. Load the data

Import the S&P500 data from a compressed `.csv` file. We will only use the last 253 observations.

````@example 1_Getting_Started
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = tsfmt)
````

First we must compute the returns from the prices. The `ReturnsResult` struct stores the asset names in `nx`, asset returns in `X`, and timestamps in `ts`. The other fields are used in other applications which we will not be showcasing here.

````@example 1_Getting_Started
rd = prices_to_returns(X)
````

## 2. MeanRisk optimisation

### 2.1 Creating a solver instance

All optimisations require some prior statistics to be computed. This can either be done before the optimisation function, or within it. For certain optimisations, precomputing the prior is more efficient, but it makes no difference here so we'll do it within the optimisation.

The `MeanRisk` estimator defines a mean-risk optimisation problem. It is a `JuMPOptimisationEstimator`, which means it requires a `JuMP`-compatible optimiser, which in this case will be `Clarabel`.

````@example 1_Getting_Started
using Clarabel
````

We have to define a `Solver` object, which contains the optimiser we wish to use, an optional name for logging purposes, optional solver settings, and optional kwargs for [`JuMP.assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#assert_is_solved_and_feasible).

Given the vast range of optimisation options and types, it is often useful to try different solver and settings combinations. To this aim, it is also possible to provide a vector of `Solver` objects, which is iterated over until one succeeds or all fail. The classic Markowitz optimisation is rather simple, so we will use a single solver instance.

````@example 1_Getting_Started
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
````

### 2.2 Defining the optimisation estimator

`PortfolioOptimisers` is designed to heavily leverage composition. The first hint of this design ethos in the examples comes in the form of `JuMPOptimiser`, which is the structure defining the optimiser parameters used in all `JuMPOptimisationEstimator`s.

Let's create a `MeanRisk` estimator. As you can see from the output, `JuMPOptimiser` and `MeanRisk` contain myriad properties that we will not showcase in this example.

````@example 1_Getting_Started
mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
````

### 2.3 Performing the optimisation

The `optimise` function is used to perform all optimisations in `PortfolioOptimisers`. Each method returns an `AbstractResult` object containing the optimisation results, which include a return code, a solution object, and relevant statistics (precomputed or otherwise) used in the optimisation.

The field `retcode` informs us that our optimisation was successful because it contains an `OptimisationSuccess` return code.

````@example 1_Getting_Started
res = optimise(mr, rd)
````

Let's view the solution results as a pretty table. For convenience, we have ensured all `AbstractResult` have a property called `w`, which directly accesses `sol.w`. The optimisations don't shuffle the asset order, so we can simply view the asset names and weights side by side.

````@example 1_Getting_Started
pretty_table(DataFrame(:assets => rd.nx, :weights => res.w); formatters = resfmt)
````

## 3. Finite allocation

We have the optimal solution, but most people don't have access to effectively unlimited funds. Given the optimised weights, current prices and a finite cash amount, it is possible to perform a finite allocation. We will use a discrete allocation method which uses mixed-integer programming to find the best allocation. We have another finite allocation method which uses a greedy algorithm that can deal with fractional shares, but we will reserve it for a later example.

For the discrete allocation, we need a solver capable of handling mixed-integer programming problems, we will use `HiGHS`.

````@example 1_Getting_Started
using HiGHS

mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
da = DiscreteAllocation(; slv = mip_slv)
````

Luckily, we have the optimal weights, the latest prices are the last entry of our original time array `X`, and let's say we have `4206.9` USD to invest.

The function can optionally take extra positional arguments to account for a variety of fees, but we will not use them here.

````@example 1_Getting_Started
mip_res = optimise(da, res.w, vec(values(X[end])), 4206.9)
````

The result of this optimisation contains different pieces of information to the previous one. The reason various fields are prefixed by `l_`or `s_` is because the discrete allocation method splits the assets into long and short positions, which are recombined in the final result.

Let's see the results in another pretty table.

````@example 1_Getting_Started
pretty_table(DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
                       :opt_weights => res.w, :mip_weights => mip_res.w);
             formatters = mipresfmt)
````

We can see that the mip weights do not exactly match the optimal ones, but that is because we only have finite resources. Note that the sum of the costs minus the initial cash is equal to the `cash` property of the result. This changes when we introduce fees, which will be shown in a future example.

````@example 1_Getting_Started
println("used cash ≈ available cash: $(isapprox(mip_res.cash, 4206.9 - sum(mip_res.cost)))")
````

We can also see that the cost of each asset is equal to the number of shares times its price.

````@example 1_Getting_Started
println("cost of shares ≈ cost of portfolio: $(all(isapprox.(mip_res.shares .* vec(values(X[end])), mip_res.cost)))")
````

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
