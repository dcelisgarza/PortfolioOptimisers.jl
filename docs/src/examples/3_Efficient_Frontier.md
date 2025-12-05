The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).
```@meta
EditURL = "../../../examples/3_Efficient_Frontier.jl"
```

# Example 3: Efficient frontier

In this example we will show how to compute efficient frontiers using the `MeanRisk` and `NearOptimalCentering` estimators.

````@example 3_Efficient_Frontier
using PortfolioOptimisers, PrettyTables
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
nothing #hide
````

## 1. ReturnsResult data

We will use the same data as the previous example.

````@example 3_Efficient_Frontier
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Efficient frontier

We have two mutually exclusive ways to compute the efficient frontier. We can do so from the perspective of minimising the risk with a return lower bound, or maximising the return with a risk upper bound. It is possible to provide explicit bounds, or a `Frontier` object which automatically computes the bounds based on the problem and constraints. All four combinations have their use cases. In this example we will only show the use of `Frontier` as a lower bound on the portfolio return.

Since we will be performing various optimisations, we will provide a vector of solver settings because we don't know if a single set of settings will work in all cases.

````@example 3_Efficient_Frontier
using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
              check_sol = (; allow_local = true, allow_almost = true))]
````

This time we will use the `ConditionalValueatRisk` measure and we will once again precompute prior.

````@example 3_Efficient_Frontier
r = ConditionalValueatRisk()
pr = prior(EmpiricalPrior(), rd)
````

Let's create the efficient frontier by setting returns lower bounds and minimising the risk. We will compute a 30-point frontier.

````@example 3_Efficient_Frontier
opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ArithmeticReturn(; lb = Frontier(; N = 30)))
````

We can now use `opt` to create the `MeanRisk` estimator. In order to get the entire frontier, we need to minimise the risk (which is the default value).

````@example 3_Efficient_Frontier
mr = MeanRisk(; opt = opt, r = r)
res1 = optimise(mr)
````

Note that `retcode` and `sol` are now vectors. This is because there is one per point in the frontier. Since we didn't get any warnings that any optimisations failed we can proceed without checking the return codes. Regardless, let's check that all optimisations succeeded.

````@example 3_Efficient_Frontier
all(x -> isa(x, OptimisationSuccess), res1.retcode)
````

We can view how the weights evolve along the frontier.

````@example 3_Efficient_Frontier
pretty_table(DataFrame([rd.nx hcat(res1.w...)], Symbol.([:assets; 1:30]));
             formatters = [resfmt])
````

## 3. Visualising the efficient frontier

Perhaps it is time to introduce some visualisations, which are implemented as a package extesion. For this we need to import the `Plots` and `GraphRecipes` packages.

````@example 3_Efficient_Frontier
using StatsPlots, GraphRecipes

plot_stacked_area_composition(res1.w, rd.nx)
````

The efficient frontier is just a special case of a pareto front, we have a function that can plot pareto fronts and surfaces. We have to provide the weights and the prior. There are optional keyword parameters for the risk measure for the X-axis, Y-axis, Z-axis, and colourbar. Here we will use the Conditional Value at Risk as the X-axis, the arithmetic return, and the risk-return ratio as the colourbar.

````@example 3_Efficient_Frontier
# Risk-free rate of 4.2/100/252
plot_measures(res1.w, res1.pr; x = r, y = ReturnRiskMeasure(; rt = res1.ret),
              c = RatioRiskMeasure(; rt = res1.ret, rk = r, rf = 4.2 / 100 / 252),
              title = "Efficient Frontier", xlabel = "CVaR", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)
````

The `plot_measures` function can plot all sorts of pareto fronts. We can even use the ratio of two risk measures as the colourbar.

````@example 3_Efficient_Frontier
plot_measures(res1.w, res1.pr; x = r, y = ConditionalDrawdownatRisk(),
              c = RiskRatioRiskMeasure(; r1 = ConditionalDrawdownatRisk(), r2 = r),
              title = "Pareto Front", xlabel = "CVaR", ylabel = "CDaR",
              colorbar_title = "\nCDaR/CVaR Ratio", right_margin = 6Plots.mm)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

