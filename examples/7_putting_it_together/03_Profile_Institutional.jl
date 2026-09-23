#=
```@meta
Description = "An end-to-end profile in PortfolioOptimisers.jl: an institutional mandate under concentration limits, sector caps and a tracking-error budget."
```

# Profile: institutional

The third profile runs a large benchmarked book under a written mandate. The
[retail profile](01_Profile_Retail_Daily.md) optimised against cost and the
[desk profile](02_Profile_Desk_Monthly.md) optimised on a view. This one optimises inside a set of
rules: a cap on each name, a cap on the energy sector, and a limit on how far the book may drift from
its benchmark. The prior is the plain empirical one.

Of the limits in the [strategy decision framework](../../user_guide/07_Choosing_a_Strategy.md),
three shape this mandate's choices.

  - The mandate's caps are hard limits, not preferences. Each one is a keyword on the
    [`JuMPOptimiser`](@ref).
  - The book is measured against a benchmark, so we bound the tracking error while we minimise
    risk. [Turnover and Tracking](../4_constraints_costs/05_Turnover_and_Tracking.md) covers that
    bound.
  - The mandate invests \$10,000,000, and [`DiscreteAllocation`](@ref) turns it into whole shares
    with a mixed-integer solve.

!!! tip "When to reach for this"
    Reach for this profile when a mandate, rather than a forecast, decides what the book may hold.
    Write each rule as a `JuMPOptimiser` keyword, bound the tracking error against the benchmark,
    and allocate with a mixed-integer solve.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data, benchmark, and groups

The benchmark is an equal-weight book. We also name the sectors, because the mandate's sector cap
is written against a sector name.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

prices = vec(values(X)[end, :])
benchmark = fill(1 / N, N)

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "energy" => ["CVX", "XOM", "RRC"],
                                "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"]))

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=
## 2. The constrained optimisation

We minimise risk under the whole mandate. `wb` caps each name at 10%, `lcse` caps energy at 20%,
and `tr` bounds the tracking error against the benchmark at 0.005. Each rule is one
keyword on the [`JuMPOptimiser`](@ref).
=#

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
             title = "Sector weights of the benchmark and the mandate book")

#=
The table gives the benchmark weight and the mandate weight for each of the three named sectors.
Compare the energy row with the 20% the mandate sets. The per-name cap shows in the weight table
of the next section. The tracking error does not show in either table.

## 3. Exact finite allocation

The mandate invests \$10,000,000. We round the target to whole shares with
[`DiscreteAllocation`](@ref), which solves a mixed-integer problem in
[HiGHS](https://github.com/jump-dev/HiGHS.jl). The title of the table prints the cash left over,
the cost of rounding to whole shares.
=#

mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
alloc = optimise(DiscreteAllocation(; slv = mip_slv),
                 FiniteAllocationInput(; w = institutional.w, prices = prices,
                                       cash = 10_000_000.0))

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => institutional.w,
                       "Shares" => round.(Int, alloc.shares), "Realised" => alloc.w);
             formatters = [resfmt],
             title = "\$10,000,000 to invest, \$$(round(Int, invested)) invested, \$$(round(alloc.cash, digits = 2)) left in cash")

#=
## 4. The book
=#

plot_stacked_bar_composition([institutional], rd; xticks = (1:1, ["Institutional"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New end-to-end profile; closes 7_putting_it_together and the ADR 0014 content. Verified on
#src   kaimon (f102cae9): EmpiricalPrior → MeanRisk(MinimumRisk, wb ub=0.10, lcse energy<=0.2,
#src   TrackingError err=0.005 vs equal-weight) → DiscreteAllocation(HiGHS) $10M. Result maxw 10%
#src   (cap binds), 16 names, leftover $7.58.
#src - Composes constraints + tracking (4_constraints_costs) + MIP finite allocation
#src   (6_post_processing). Three profiles now contrast cleanly: retail = cost control, desk =
#src   view + frontier, institutional = constraints + benchmark.
