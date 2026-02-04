The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/09_Regularisation.jl"
```

# Example 9: Regularisation

This example shows one of the simplest ways to imporve the robustness of portfolios, regularisation penalties.

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

The optimal regularisation penalty value depends on the data, the investor preferences, and type of regularisation. The specific choice of penalty value is so volatile that it can only be estimated via cross-validation or similar techniques, but the "optimal" (to some definition of optimal) value will also change over time as the market conditions change. Therefore, we will simply show how to set up and solve a regularised portfolio optimisation problem, without attempting to find the optimal penalty value. We will use the same small penalty for all regularisations to illustrate how they differ.

- L1 regularisation (also known as Lasso regularisation) adds a penalty proportional to the sum of the absolute values of the portfolio weights. This encourages sparsity in the portfolio, leading to fewer assets being selected.
- L2 regularisation (also known as Ridge regularisation) adds a penalty proportional to the sum of the squares of the portfolio weights. This discourages large weights and promotes diversification.
- Lp regularisation via [`LpRegularisation`]-(@ref) adds a penalty proportional to the p-norm of the portfolio weights, where `p > 1` is a positive real number.
- L-Inf regularisation adds a penalty proportional to the maximum absolute value of the portfolio weights. This limits the influence of any single asset in the portfolio.

````@example 09_Regularisation
opts = [JuMPOptimiser(; pr = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1),#
        JuMPOptimiser(; pr = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l1 = 1e-5),#
        JuMPOptimiser(; pr = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l2 = 1e-5),#
        JuMPOptimiser(; pr = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, lp = LpRegularisation(; p = 3, val = 1e-5)),#
        JuMPOptimiser(; pr = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, linf = 1e-5)]
nocs = [MeanRisk(; opt = opt) for opt in opts]

ress = optimise.(nocs)
pretty_table(DataFrame(:Assets => rd.nx, :No_Reg => ress[1].w, :L1 => ress[2].w,
                       :L2 => ress[3].w, :Lp => ress[4].w, :LInf => ress[5].w);
             formatters = [resfmt], summary_rows = [summary_row],
             summary_row_labels = ["# Eff. Assets"])
````

The effect of eac regularisation depends on the relative values between the objective function and the value of the relevant norm of the optimised portfolio weights multiplied by the penalty value. So there is little point in exploring the entire efficient frontier without finding an appropriate penalty value first. However, we can see that for this risk measure and data set, the regularised portfolios tend to have more effective assets. The number of effective assets is different to the sparsity in that it measures the concentration of weights as `1/(w ⋅ w)`, rather than counting the number of non-zero weights. Usually, the larger the number of effective assets, the more diversified the portfolio is despite it possibly having more zero weights.

The regularisations can be combined in the same optimisation by simply specifying multiple penalty types and values. The library takes care of the sign of the penalty depending on the sense of the optimisation.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
