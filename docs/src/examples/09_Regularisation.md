The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/09_Regularisation.jl"
```

# Example 9: Regularisation

This example shows one of the simplest ways to improve the robustness of portfolios, regularisation penalties.

````@example 09_Regularisation
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
mmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
summary_row = (data, j) -> begin
    if j == 1
        return "N/A"
    else
        return number_effective_assets(data[:, j])
    end
end
````

## 1. Setting up

We will use the same data as the previous example.

````@example 09_Regularisation
using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.70),
              check_sol = (; allow_local = true, allow_almost = true))];
nothing #hide
````

## 2. Regularised portfolios

The optimal regularisation penalty value depends on the data, the investor preferences, and type of regularisation. The specific choice of penalty value is so volatile that it can only be estimated via grid search cross-validation or similar techniques, but the "optimal" (to some definition of optimal) value will also change over time as the market conditions change. Therefore, we will simply show how to set up and solve a regularised portfolio optimisation problem, without attempting to find the optimal penalty value.

We will use the same small penalty for all regularisations to illustrate how they differ.

- L1 regularisation (also known as Lasso regularisation) adds a penalty proportional to the sum of the absolute values of the portfolio weights. This encourages sparsity in the portfolio, leading to fewer assets being selected.
- L2 regularisation (also known as Ridge regularisation) adds a penalty proportional to the sum of the squares of the portfolio weights. This discourages large weights and promotes diversification.
- Lp regularisation via [`LpRegularisation`]-(@ref) adds a penalty proportional to the p-norm of the portfolio weights, where `p > 1` is a positive real number.
- L-Inf regularisation adds a penalty proportional to the maximum absolute value of the portfolio weights. This limits the influence of any single asset in the portfolio.

### 2.1 Efficient frontier

````@example 09_Regularisation
opts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, ret = ArithmeticReturn(; lb = Frontier(; N = 50))),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)), bgt = 1,
                      l1 = 4e-4),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)), bgt = 1,
                      l2 = 4e-4),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)), bgt = 1,
                      lp = LpRegularisation(; p = 5, val = 4e-4)),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)), bgt = 1,
                      linf = 4e-4)]
nocs = [MeanRisk(; opt = opt) for opt in opts]
ress = optimise.(nocs)
````

Let's plot the efficient frontiers.

````@example 09_Regularisation
using StatsPlots, GraphRecipes

r = Variance()
````

No regularisation portfolio weights.

````@example 09_Regularisation
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "No regularisation", legend = :outerright))
````

No regularisation frontier.

````@example 09_Regularisation
plot_measures(ress[1].w, pr; x = r, y = ExpectedReturn(; rt = ress[1].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "No regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

L1 regularisation portfolio weights. As expected, the portfolio is sparsified, with fewer assets with non-zero weight.

````@example 09_Regularisation
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L1 regularisation", legend = :outerright))
````

L1 regularisation frontier. The sparsification makes the pareto front non-smooth.

````@example 09_Regularisation
plot_measures(ress[2].w, pr; x = r, y = ExpectedReturn(; rt = ress[2].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L1 regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

L2 regularisation portfolio weights. Even values of p-norms smooth out the weights, leading to more diversified portfolios. The higher the value, the more highly penalised larger deviations from the mean weight become. This is similar to how moments of even order behave.

````@example 09_Regularisation
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L2 regularisation", legend = :outerright))
````

L2 regularisation frontier.

````@example 09_Regularisation
plot_measures(ress[3].w, pr; x = r, y = ExpectedReturn(; rt = ress[3].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L2 regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Lp regularisation portfolio weights. The higher the value of p, the closer the behaviour is to L-Inf regularisation, where the maximum absolute weight is penalised. This leads to portfolios where all weights are more similar in magnitude, but does not smear the negative weights into positive values like the L2 norm.

````@example 09_Regularisation
plot_stacked_area_composition(ress[4].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Lp (p = 5) regularisation",
                                        legend = :outerright))
````

Lp regularisation frontier.

````@example 09_Regularisation
plot_measures(ress[4].w, pr; x = r, y = ExpectedReturn(; rt = ress[4].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "Lp (p = 5) regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

L-Inf regularisation portfolio weights.

````@example 09_Regularisation
plot_stacked_area_composition(ress[5].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L-Inf regularisation",
                                        legend = :outerright))
````

L-Inf regularisation frontier.

````@example 09_Regularisation
plot_measures(ress[5].w, pr; x = r, y = ExpectedReturn(; rt = ress[5].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L-Inf regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

### 2.2 Minimum risk portfolios

Lets view only the minimum risk portfolios for each regularisation to get more insight into what regularisation does.

````@example 09_Regularisation
opts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1),# no regularisation
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l1 = 4e-4),# L1 regularisation
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l2 = 4e-4),# L2 regularisation
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, lp = LpRegularisation(; p = 5, val = 4e-4)),# Lp regularisation with p = 5
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, linf = 4e-4)]# L-Inf regularisation
nocs = [MeanRisk(; opt = opt) for opt in opts]

ress = optimise.(nocs)
pretty_table(DataFrame(:Assets => rd.nx, :No_Reg => ress[1].w, :L1 => ress[2].w,
                       :L2 => ress[3].w, :L5 => ress[4].w, :LInf => ress[5].w);
             formatters = [resfmt], summary_rows = [summary_row],
             summary_row_labels = ["# Eff. Assets"])
````

The effect of each regularisation depends on the relative values of the objective function with respect to the value of the relevant norm of the optimised portfolio weights multiplied by the penalty.

Generally, regularised portfolios tend to have more effective assets than unregularised ones. The number of effective assets is different to the sparsity in that it measures the concentration of weights as `1/(w ⋅ w)`, rather than counting the number of non-zero (or near zero) weights. Usually, the larger the number of effective assets, the more diversified the portfolio. Sparsity is a non-smooth measure, while the number of effective assets is smooth, so a portfolio can have higher sparsity and still have a larger number of effective assets.

It is possible to combine multiple regularisation penalties in the same optimisation problem by simultaneously specifying multiple regularisation keywords in the `JuMPOptimiser`. This can be useful to combine the benefits of different regularisations, such as sparsity and diversification, but can make the optimisation more difficult to solve and interpret.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
