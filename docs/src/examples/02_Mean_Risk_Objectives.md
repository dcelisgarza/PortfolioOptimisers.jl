The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/02_Mean_Risk_Objectives.jl"
```

# Example 2: `MeanRisk` objectives

In this example we will show the different objective functions available in `MeanRisk`, and compare them to a benchmark.

````@example 02_Mean_Risk_Objectives
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

````@example 02_Mean_Risk_Objectives
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
````

## 2. MeanRisk objectives

Here we will show the different objective functions available in `MeanRisk`. We will also use the semi-standard deviation risk measure.

````@example 02_Mean_Risk_Objectives
using Clarabel
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
````

Here we encounter another consequence of the design philosophy of `PortfolioOptimisers`. An entire class of risk measures can be categorised and consistently implemented as `LowOrderMoment` risk measures with different internal algorithms. This corresponds to the semi-standard deviation.

````@example 02_Mean_Risk_Objectives
r = LowOrderMoment(; alg = SecondMoment(; alg1 = Semi(), alg2 = SOCRiskExpr()))
````

Since we will perform various optimisations on the same data, there's no need to redo work. Let's precompute the prior statistics using the `EmpiricalPrior` to avoid recomputing them every time we call the optimisation.

````@example 02_Mean_Risk_Objectives
pr = prior(EmpiricalPrior(), rd)
````

We can provide the prior result to `JuMPOptimiser`.

````@example 02_Mean_Risk_Objectives
opt = JuMPOptimiser(; pe = pr, slv = slv)
````

Here we define the estimators for different objective functions.

````@example 02_Mean_Risk_Objectives
# Minimum risk
mr1 = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)
# Maximum utility with risk aversion parameter 2
mr2 = MeanRisk(; r = r, obj = MaximumUtility(), opt = opt)
# Risk-free rate of 4.2/100/252
rf = 4.2 / 100 / 252
mr3 = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
# Maximum return
mr4 = MeanRisk(; r = r, obj = MaximumReturn(), opt = opt)
````

Let's perform the optimisations, but since we've precomputed the prior statistics, we do not need to provide the returns data. We will also produce a benchmark using the `InverseVolatility` estimator.

````@example 02_Mean_Risk_Objectives
res1 = optimise(mr1)
res2 = optimise(mr2)
res3 = optimise(mr3)
res4 = optimise(mr4)
res0 = optimise(InverseVolatility(; pe = pr))
````

Let's view the results as pretty tables.

````@example 02_Mean_Risk_Objectives
pretty_table(DataFrame(; :assets => rd.nx, :benchmark => res0.w, :MinimumRisk => res1.w,
                       :MaximumUtility => res2.w, :MaximumRatio => res3.w,
                       :MaximumReturn => res4.w); formatters = [resfmt])
````

In order to confirm that the objective functions do what they say on the tin, we can compute the risk, return and risk return ration. There are individual functions for each `expected_risk`, `expected_return`, `expected_ratio`, but we also have `expected_risk_ret_ratio` that returns all three at once (`risk`, `return`, `risk-return ratio`) which is what we will use here.

Due to the fact that we provide different expected portfolio return measures, any function that computes the expected portfolio return also needs to know which return type to compute. We will be consistent with the returns we used in the optimisation.

````@example 02_Mean_Risk_Objectives
rk1, rt1, rr1 = expected_risk_ret_ratio(r, res1.ret, res1.w, res1.pr; rf = rf);
rk2, rt2, rr2 = expected_risk_ret_ratio(r, res2.ret, res2.w, res2.pr; rf = rf);
rk3, rt3, rr3 = expected_risk_ret_ratio(r, res3.ret, res3.w, res3.pr; rf = rf);
rk4, rt4, rr4 = expected_risk_ret_ratio(r, res4.ret, res4.w, res4.pr; rf = rf);
rk0, rt0, rr0 = expected_risk_ret_ratio(r, ArithmeticReturn(), res0.w, res0.pr; rf = rf);
nothing #hide
````

Let's make sure the results are what we expect.

````@example 02_Mean_Risk_Objectives
pretty_table(DataFrame(;
                       :obj => [:MinimumRisk, :MaximumUtility, :MaximumRatio,
                                :MaximumReturn, :Benchmark],
                       :rk => [rk1, rk2, rk3, rk4, rk0], :rt => [rt1, rt2, rt3, rt4, rt0],
                       :rr => [rr1, rr2, rr3, rr4, rr0]); formatters = [resfmt])
````

We can see that indeed, the minimum risk produces the portfolio with minimum risk, the maximum ratio produces the portfolio with the maximum risk-return ratio, and the maximum return portfolio produces the portfolio with the maximum return.

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
