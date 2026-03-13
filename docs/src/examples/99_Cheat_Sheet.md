The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/99_Cheat_Sheet.jl"
```

# Cheat Sheet

This is a collection of quickfire tutorials to help you get started with `PortfolioOptimisers.jl` without delving into the examples and/or documentation.

## 1. Downloading data

There are both open and close source providers, in Julia we have [`YFinance.jl`](https://github.com/eohne/YFinance.jl) and [`MarketData.jl`](https://github.com/JuliaQuant/MarketData.jl).

## 2. Computing returns

Usually data is obtained from a provider and the returns have to be computed. `PortfolioOptimisers.jl` has a [`prices_to_returns`](@ref) to do so from price data. It can handle asset, factor, and benchmark returns, as well as implied volatilities, and volatility premiums. It performs appropriate data validation checks to ensure the timestamps match and the data is clean. It can also preprocess missing data and collapse to lower frequencies.

Here we show a quick example of a heterogenous dataset which will only return the data with matching timestamps.

````@example 99_Cheat_Sheet
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz"));
                                 timestamp = :Date)[(end - 7 * 252):(end - 252 * 2)],
                       TimeArray(CSV.File(joinpath(@__DIR__, "Factors.csv.gz"));
                                 timestamp = :Date)[(end - 6 * 252):(end - 252)];
                       B = TimeArray(CSV.File(joinpath(@__DIR__, "SP500_idx.csv.gz"));
                                     timestamp = :Date)[(end - 5 * 252):end])
````

## 3. Basic Optimisations

There are many optimisers available in `PortfolioOptimisers.jl`. Here we will showcase their basic usage. There are three types of non finite allocation optimisers.

### 3.1 JuMP optimisers

The JuMP-based optimisers use traditional mathematical optimisation. As such, they are the most flexible when it comes to constraints, and for those which accept them, objective functions. Most risk measures are also compatible with these, aside from a few exclusively compatible with clustering optimisations, as well as other risk measures which are incompatible with any optimisation. All JuMP-based optimisers require the user to provide a JuMP-compatible solver, which supports the type of constraints being used.

If using open-source solvers we recommend [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.jl) when not using MIP constraints. When using MIP constraints, [`Pajarito`](https://github.com/jump-dev/Pajarito.jl) with [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.jl) as the continuous solver, and [`HiGHS`](https://github.com/jump-dev/HiGHS.jl) as the MIP solver.

Users can provide a vector of solvers which will be iterated over until one solves the problem satisfactorily, or all fail. Other than optimisation-specific constraints, general constraints are applied at the level of the [`JuMPOptimiser`]-(@ref), whether the problem is feasable or not depends on the specific constraints and the provided solver's support for constraint types/ability to solve the problem.

````@example 99_Cheat_Sheet
using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = "verbose" => false),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
       Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                              "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                              "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                              "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                              "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4))];
nothing #hide
````

#### 3.1.1 Mean Risk

[`MeanRisk`]-(@ref) is the traditional portfolio optimisation problem. It seeks to minimise the risk with respect to a target return, or maximise the return with respect to a target risk. It supports four objective functions which use the relationship between risk and returns in different ways.

````@example 99_Cheat_Sheet
# Minimum risk (default)
mr1 = MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; slv = slv))
# Maximum utility
mr2 = MeanRisk(; obj = MaximumUtility(), opt = JuMPOptimiser(; slv = slv))
# Maximum risk adjusted return ratio
mr3 = MeanRisk(; obj = MaximumRatio(), opt = JuMPOptimiser(; slv = slv))
# Maximum return
mr4 = MeanRisk(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv = slv))
# Optimise all objective functions at once
ress = optimise.([mr1, mr2, mr3, mr4], rd);
nothing #hide
````

`PortfolioOptimisers.jl` provides users with the ability to use multiple risk measures per optimisation, which means that some risk measure have to keep certain internal statistics to be able to compute the risk. We provide factory functions that create the risk measures with the appropraite internal statistics.

The [`MeanRisk`]-(@ref) optimiser defaults to the variance so we will use that to compute the risk statistics. All optimisations use the same prior estimator, as well as portfolio return estimator so we will use only the first

````@example 99_Cheat_Sheet
pr = ress[1].pr
r = factory(Variance(), pr)
ret = mr1.opt.ret
rk_rt_ratio = [expected_risk_ret_ratio(r, ret, res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display results
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
````

#### 3.1.2 Factor risk contribution

This is a more advanced estimator, it requires some more set up. It allows users to provide objective functions, but also define risk contributions per factor to the variance risk measure. The minimum risk optimisaion will follow the risk contribution constraints the closest, and with enough data and assets can be quite exact up to the user provided convergence settings for the provided solvers.

It is compatible with other risk measures, but only the variance can take risk contribution constraints, without them or when using other risk measures it is largely the same as the [`MeanRisk`]-(@ref) estimator.

First we need to provide a instance of [`AssetSets`](@ref) which defines sets of assets or factors and their relationships, which lets `PortfolioOptimisers.jl` create linear constraints according to how users define them via [`LinearConstraintEstimator`](@ref). This way it's possible to define groups of assets/factors and how they relate to each other. We will showcase them later. For now, we need these to define the relationship between factors for their risk contribution.

The [`AssetSets`](@ref) has a `key` property which defines the default search key in `dict`, `dict` must contain a key matching `key` whose value is taken to tbe the names of the assets/factors around which the sets are defined.

In this case we use this to define the set of factors with key "nf" (default "nx"). We use these to define the factor risk contribution constraints, and we have to assign the linear constraint estimator for the risk contribution to the [`Variance`](@ref). We'll define the constraints such that each factor contributes between 30% and 12% of the total variance risk.

````@example 99_Cheat_Sheet
sets = AssetSets(; key = "nf", dict = Dict("nf" => rd.nf))
lcs = LinearConstraintEstimator(;
                                val = [["$f <= 0.3" for f in rd.nf];
                                       ["$f >= 0.12" for f in rd.nf]])
r = Variance(; rc = lcs)
````

We can optimise for the different objective functions.

````@example 99_Cheat_Sheet
# Minimum risk (default)
frc1 = FactorRiskContribution(; r = r, obj = MinimumRisk(), sets = sets,
                              opt = JuMPOptimiser(; slv = slv))
# Maximum utility, `l` controls the risk aversion.
frc2 = FactorRiskContribution(; r = r, obj = MaximumUtility(; l = 8), sets = sets,
                              opt = JuMPOptimiser(; slv = slv))
# Maximum risk adjusted return ratio, rf is the risk free rate.
frc3 = FactorRiskContribution(; r = r, obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                              sets = sets, opt = JuMPOptimiser(; slv = slv))
# Maximum return
frc4 = FactorRiskContribution(; r = r, obj = MaximumReturn(), sets = sets,
                              opt = JuMPOptimiser(; slv = slv))
# Optimise all objective functions at once
ress = optimise.([frc1, frc2, frc3, frc4], rd);
nothing #hide
````

We can display the results and factor risk contributions [`factor_risk_contribution`]-(@ref), but we have to normalise them using their sum. Again we use the factory function to set the appropriate internal parameters. The last entry in the risk contribution is the regression intercept.

````@example 99_Cheat_Sheet
r = factory(r, pr)
rkcs = [factor_risk_contribution(r, res.w, pr.X; rd = rd) for res in ress]
rkcs = rkcs ./ sum.(rkcs)

pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:factors => [rd.nf; "Intercept"]),
                  DataFrame(reduce(hcat, rkcs),
                            ["RC MinRisk", "RC Max Util", "RC Max Ratio", "RC Max Ret"]));
             formatters = [resfmt])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
