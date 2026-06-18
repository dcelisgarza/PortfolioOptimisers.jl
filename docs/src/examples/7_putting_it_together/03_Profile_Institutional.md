The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/7_putting_it_together/03_Profile_Institutional.jl"
```

# Profile: institutional

The third profile is an **institutional mandate** — a large, benchmarked book hemmed in by rules.
Where the [retail profile](01_Profile_Retail_Daily.md) optimised for cost and the
[desk profile](02_Profile_Desk_Monthly.md) optimised for a view, this one optimises *within
constraints*: concentration limits, sector caps, and a tracking-error budget against a benchmark.
The size makes exact execution worthwhile.

The reasoning, following the [strategy decision framework](../../user_guide/06_Choosing_a_Strategy.md):

- **The mandate is the boss** — per-name caps, sector limits, and a benchmark tracking-error
    ceiling are hard requirements, exactly what the [`JuMPOptimiser`](@ref) constraint keywords
    express.
- **Benchmarked** — this is enhanced indexing: minimise risk but stay within a tracking-error
    budget of the benchmark (see [Turnover and Tracking](../4_constraints_costs/04_Turnover_and_Tracking.md)).
- **Large and precise** — a big book justifies an exact [`DiscreteAllocation`](@ref).

!!! tip "When to reach for this"
    This is the template for a constrained, benchmarked institutional book: stack the mandate's
    rules as `JuMPOptimiser` keywords, bound tracking error to the benchmark, and allocate exactly.
    The constraints, not the prior, are doing most of the work.

````@example 03_Profile_Institutional
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. Data, benchmark, and groups

The benchmark is an equal-weight book; the sectors are named so the mandate's sector caps can
reference them.

````@example 03_Profile_Institutional
X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

prices = vec(values(X)[end, :])
benchmark = fill(1 / N, N)

sets = AssetSets(;
                 dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                             "energy" => ["CVX", "XOM", "RRC"],
                             "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"]))

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
````

## 2. The constrained optimisation

Minimum risk, subject to the full mandate: a 10% per-name cap, an energy sector limit, and a
tracking-error budget against the benchmark. Every rule is a keyword on the [`JuMPOptimiser`](@ref).

````@example 03_Profile_Institutional
institutional = optimise(MeanRisk(; obj = MinimumRisk(),
                                  opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                      wb = WeightBounds(; lb = 0.0,
                                                                        ub = 0.10),
                                                      lcse = LinearConstraintEstimator(;
                                                                                       val = ["energy <= 0.2"]),
                                                      tr = TrackingError(;
                                                                         tr = WeightsTracking(;
                                                                                              w = benchmark),
                                                                         err = 0.005))))

sector_weight(w, g) = sum(w[i] for i in eachindex(w) if rd.nx[i] in sets.dict[g])
pretty_table(DataFrame("Sector" => ["tech", "energy", "healthcare"],
                       "Benchmark" => [sector_weight(benchmark, g)
                                       for g in ["tech", "energy", "healthcare"]],
                       "Mandate book" => [sector_weight(institutional.w, g)
                                          for g in ["tech", "energy", "healthcare"]]);
             formatters = [resfmt],
             title = "Institutional book — capped, energy-limited, benchmark-tracking")
````

The book respects every rule: no name above 10%, energy under its cap, and the whole portfolio
stays within the tracking-error budget of the benchmark — diversified by construction rather than
by a single objective.

## 3. Exact finite allocation

On a \$10,000,000 book, [`DiscreteAllocation`](@ref) with a MIP solver
([HiGHS](https://github.com/jump-dev/HiGHS.jl)) turns the target into whole shares with negligible
residual cash.

````@example 03_Profile_Institutional
mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
alloc = optimise(DiscreteAllocation(; slv = mip_slv), institutional.w, prices, 10_000_000.0)

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => institutional.w,
                       "Shares" => round.(Int, alloc.shares), "Realised" => alloc.w);
             formatters = [resfmt],
             title = "\$10,000,000 allocated — invested \```math(round(Int, invested)), cash left \```(round(alloc.cash, digits = 2))")
````

## 4. The book

````@example 03_Profile_Institutional
plot_stacked_bar_composition([institutional], rd; xticks = (1:1, ["Institutional"]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
