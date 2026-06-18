The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/7_putting_it_together/02_Profile_Desk_Monthly.jl"
```

# Profile: desk, monthly

The second profile is a **professional desk rebalancing monthly**. The trade-offs invert the
[retail profile](01_Profile_Retail_Daily.md): rebalancing infrequently means each decision can
afford real compute and real analysis, and turnover matters far less. The edge here comes from a
*view* and from exploring the whole risk/return trade-off rather than from cost control.

The reasoning, following the [strategy decision framework](../../user_guide/06_Choosing_a_Strategy.md):

- **Compute is abundant, decisions are rare** — a monthly cadence justifies a richer prior and a
    full frontier sweep.
- **The desk has a view** — it encodes a house thesis with an [`EntropyPoolingPrior`](@ref)
    rather than taking the sample moments at face value.
- **Explore, then choose** — instead of one objective, it traces the efficient frontier and
    selects the risk-adjusted (tangency) book.
- **Budget is substantial** — an exact [`DiscreteAllocation`](@ref) is affordable.

!!! tip "When to reach for this"
    This is the template for a research-driven, lower-frequency book: invest the compute in a
    better prior and a frontier sweep, pick a point deliberately, and allocate exactly. Turnover
    and fee control matter less when you trade rarely.

````@example 02_Profile_Desk_Monthly
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS, StatsPlots,
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

## 1. Data and the house view

The desk's thesis: healthcare will outperform energy. It encodes that as an entropy-pooling view,
reweighting the empirical scenarios so the prior reflects the conviction (see
[Entropy Pooling](../2_moments_priors/06_Entropy_Pooling.md)).

````@example 02_Profile_Desk_Monthly
X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
prices = vec(values(X)[end, :])

sets = AssetSets(; dict = Dict("nx" => rd.nx,
                               "energy" => ["CVX", "XOM", "RRC"],
                               "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"]))
view_prior = EntropyPoolingPrior(; sets = sets,
                                 mu_views = LinearConstraintEstimator(; val = ["healthcare >= energy"]))
pr = prior(view_prior, rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252
````

## 2. The efficient frontier

With compute to spare, the desk traces the whole frontier on the view-tilted prior — minimum-risk
books across a sweep of return targets — rather than committing to a single objective up front.

````@example 02_Profile_Desk_Monthly
frontier = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 ret = ArithmeticReturn(; lb = Frontier(; N = 15)))))

plot_efficient_frontier(frontier.w, pr; rt = frontier.ret)
````

## 3. Choosing the book

From the frontier, the desk takes the risk-adjusted optimum — the [`MaximumRatio`](@ref)
(tangency) portfolio on the same view-tilted prior.

````@example 02_Profile_Desk_Monthly
desk = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame("Asset" => rd.nx, "Tangency weight" => desk.w); formatters = [resfmt],
             title = "Desk monthly — risk-adjusted optimum on the view prior")
````

## 4. Exact finite allocation

On a \$500,000 book the rounding is small but the desk wants the provably-best whole-share book, so
it uses [`DiscreteAllocation`](@ref) with a MIP solver ([HiGHS](https://github.com/jump-dev/HiGHS.jl)).

````@example 02_Profile_Desk_Monthly
mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
alloc = optimise(DiscreteAllocation(; slv = mip_slv), desk.w, prices, 500_000.0)

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => desk.w, "Shares" => round.(Int, alloc.shares),
                       "Realised" => alloc.w); formatters = [resfmt],
             title = "\$500,000 allocated — invested \```math(round(Int, invested)), cash left \```(round(alloc.cash, digits = 2))")
````

## 5. The book

````@example 02_Profile_Desk_Monthly
plot_stacked_bar_composition([desk], rd; xticks = (1:1, ["Desk monthly"]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
