#=
```@meta
Description = "An end-to-end profile in PortfolioOptimisers.jl: a professional desk rebalancing monthly on a view and the full risk-return trade-off."
```

# [Profile: desk, monthly](@id example-profile-desk-monthly)

The second profile rebalances the book of a professional desk once a month. Every limit that
binds in the [retail profile](@ref example-profile-retail-daily) loosens here. A monthly decision can pay
for a long computation, and a month of return covers more trading cost than a day of it does. What
this desk has instead is a house view, and the time to look at the whole risk-return trade-off
before it picks a book.

Three of the limits in the
[strategy decision framework](@ref user-guide-choosing-a-strategy) change for this desk.

  - A monthly rebalance leaves time for a slower computation, so we trace the whole efficient
    frontier with fifteen solves and then take its tangency point.
  - The desk has a view, so we state it as a constraint on the mean and fit an
    [`EntropyPoolingPrior`](@ref) rather than take the sample mean as given.
  - A monthly decision can wait for a mixed-integer solve, so we buy whole shares with
    [`DiscreteAllocation`](@ref).

!!! tip "When to reach for this"
    Reach for this profile when you trade rarely and can spend the time on the model instead. Put
    that time into the prior and the frontier, choose a point on it, and allocate with a
    mixed-integer solve. Turnover and fee control matter less when you trade once a month.
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
## 1. Data and the house view

The desk expects healthcare to outperform energy. We state that as a view on the mean, and
[`EntropyPoolingPrior`](@ref) turns it into a new probability for each historical scenario. A
scenario that supports the view gets more probability, and the rest get less.
[Entropy Pooling](@ref example-entropy-pooling) covers the method that finds those
probabilities.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
prices = vec(values(X)[end, :])

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "energy" => ["CVX", "XOM", "RRC"],
                                "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"]))
view_prior = EntropyPoolingPrior(; sets = sets,
                                 mu_views = LinearConstraintEstimator(;
                                                                      val = ["healthcare >= energy"]))
pr = prior(view_prior, rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

#=
## 2. The efficient frontier

We solve for minimum risk fifteen times, once at each of fifteen lower bounds on the return. The
fifteen books trace the efficient frontier of the prior that carries the view. The plot shows what
each step up in return costs in risk.
=#

frontier = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 ret = ArithmeticReturn(;
                                                                        settings = JuMPReturnsSettings(;
                                                                                                       lb = Frontier(;
                                                                                                                     N = 15))))))

plot_efficient_frontier(frontier.w, pr; rt = frontier.ret)

#=
## 3. Choosing the book

The frontier shows the trade-off, and the desk still has to pick one point on it. We solve once
more with [`MaximumRatio`](@ref), which maximises the ratio of return above the risk-free rate to
risk. That book is the tangency point of the frontier.
=#

desk = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                         opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame("Asset" => rd.nx, "Tangency weight" => desk.w);
             formatters = [resfmt],
             title = "Desk monthly tangency weights on the prior with the view")

#=
## 4. Exact finite allocation

The desk has \$500,000 to invest, so rounding to whole shares moves each weight by very little.
[`DiscreteAllocation`](@ref) solves a mixed-integer problem with
[HiGHS](https://github.com/jump-dev/HiGHS.jl). It minimises the money by which each position
misses its target, plus the cash left over. The table title gives the amount invested and the cash
left.
=#

mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
alloc = optimise(DiscreteAllocation(; slv = mip_slv),
                 FiniteAllocationInput(; w = desk.w, prices = prices, cash = 500_000.0))

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => desk.w,
                       "Shares" => round.(Int, alloc.shares), "Realised" => alloc.w);
             formatters = [resfmt],
             title = "\$500,000 to invest, \$$(round(Int, invested)) invested, \$$(round(alloc.cash, digits = 2)) left in cash")

#=
## 5. The book
=#

plot_stacked_bar_composition([desk], rd; xticks = (1:1, ["Desk monthly"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New end-to-end profile. Verified on kaimon (f102cae9): EntropyPoolingPrior(view
#src   "healthcare >= energy") → MeanRisk frontier (15 pts, all OptimisationSuccess) + MaximumRatio
#src   tangency → DiscreteAllocation(HiGHS) $500k (leftover ~$9). Tangency concentrates (maxw 77%,
#src   2 names) — return-seeking on this slice; honest, the frontier plot is the real showcase.
#src - Composes view priors (2_moments_priors), frontier (3_optimisers), and MIP finite allocation
#src   (6_post_processing). Contrast with retail profile: rich prior + frontier vs cost control.
