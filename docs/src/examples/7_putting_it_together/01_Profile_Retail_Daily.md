The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/7_putting_it_together/01_Profile_Retail_Daily.jl"
```

# Profile: retail, daily

The earlier examples each isolate one piece of the pipeline. The *putting-it-together* profiles
run the whole pipeline end to end for a concrete investor, so you can see how the choices
compose. This first profile is a **retail investor rebalancing daily** with a small account: the
constraints are compute, trading cost, and capital, not sophistication.

The reasoning, following the [strategy decision framework](../../user_guide/06_Choosing_a_Strategy.md):

- **Compute is cheap but frequent** — rebalancing every day rules out heavy optimisations; a
    single convex solve is right.
- **Trading is the enemy** — daily turnover compounds costs, so we cap turnover and charge fees
    explicitly, letting the optimiser trade only when it is worth it.
- **The account is small** — discretisation matters, so finite allocation is not an afterthought.
- **Robustness over edge** — a tight weight cap buys diversification and stability.

!!! tip "When to reach for this"
    This is the template for any cost- and capital-constrained, high-frequency book: keep the
    optimisation light, control turnover and fees at the optimiser, and finish with a finite
    allocation sized to the real account.

````@example 01_Profile_Retail_Daily
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. Data and current book

We use the S&P 500 slice, and assume the investor currently holds an equal-weight book — the
reference point turnover is measured against.

````@example 01_Profile_Retail_Daily
X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

prices = vec(values(X)[end, :])
current_book = fill(1 / N, N)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
````

## 2. The optimisation

One light convex solve: minimum risk, a 15% per-name cap for diversification, a daily turnover
budget against the current book, and explicit fees so trades must justify their cost.

````@example 01_Profile_Retail_Daily
retail = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               wb = WeightBounds(; lb = 0.0, ub = 0.15),
                                               tn = Turnover(; w = current_book,
                                                             val = 0.05),
                                               fees = Fees(; l = 0.001))))

pretty_table(DataFrame("Asset" => rd.nx, "Current" => current_book, "Target" => retail.w);
             formatters = [resfmt],
             title = "Retail daily target — capped, low-turnover, net of fees")
````

The cap and turnover budget keep the book diversified and close to where it started, so the daily
rebalance is small and cheap.

## 3. Finite allocation

The account is \$10,000. [`GreedyAllocation`](@ref) converts the target into whole shares — no MIP
solver, instant, which suits a daily cadence.

````@example 01_Profile_Retail_Daily
alloc = optimise(GreedyAllocation(), retail.w, prices, 10_000.0)

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => retail.w,
                       "Shares" => round.(Int, alloc.shares), "Realised" => alloc.w);
             formatters = [resfmt],
             title = "\$10,000 allocated — invested \```math(round(Int, invested)), cash left \```(round(alloc.cash, digits = 2))")
````

## 4. The book

````@example 01_Profile_Retail_Daily
plot_stacked_bar_composition([retail], rd; xticks = (1:1, ["Retail daily"]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
