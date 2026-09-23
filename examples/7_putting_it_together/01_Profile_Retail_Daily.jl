#=
```@meta
Description = "An end-to-end profile in PortfolioOptimisers.jl: a retail investor rebalancing daily under compute, trading-cost and capital limits."
```

# Profile: retail, daily

Each earlier example covers one part of the pipeline. A profile page runs the whole pipeline once,
for one investor, so you can see how the choices fit together. This first profile rebalances a
small retail account every day. Compute, trading cost and the size of the account set the limits
here, and none of them rewards a more elaborate model.

Of the limits in the [strategy decision framework](../../user_guide/07_Choosing_a_Strategy.md),
three shape this investor's choices.

  - The rebalance is daily, so you pay for the optimisation every trading day. One convex solve is
    enough.
  - Trading cost compounds when you trade every day. We cap how far each weight can move from the
    current book, and we give the optimiser a fee on each long position.
  - The account is small, so one whole share is a large part of a position. The finite allocation
    at the end moves the weights you hold away from the weights you solved for.

!!! tip "When to reach for this"
    Reach for this profile when trading cost and account size bind harder than the model does. Keep
    the optimisation to one convex solve, bound the turnover inside the optimiser, and size the
    last step to the cash you have. A fee changes the weights only under an objective or a
    constraint that uses the portfolio return.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data and current book

We read the S&P 500 slice and take the investor's current book to be equal weight. The turnover
budget in the next section measures every target weight against that book.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

prices = vec(values(X)[end, :])
current_book = fill(1 / N, N)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=
## 2. The optimisation

One convex solve takes all the choices. The objective is minimum risk. The weight bounds cap each
name at 15%. The turnover budget keeps each target weight within 0.05 of the weight the investor
has today. The fee is proportional to each long position.

Two of these choices do not change this book. With an equal-weight book of 20 names and a budget
of 0.05, no weight can pass 10%, so the turnover budget, not the 15% cap, sets the largest
position. The fee enters only the portfolio return. A minimum-risk objective with the default
variance does not use that return, so the fee does not change the weights.
=#

retail = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               wb = WeightBounds(; lb = 0.0, ub = 0.15),
                                               tn = Turnover(; w = current_book,
                                                             val = 0.05),
                                               fees = Fees(; l = 0.001))))

pretty_table(DataFrame("Asset" => rd.nx, "Current" => current_book, "Target" => retail.w);
             formatters = [resfmt],
             title = "Retail daily target — capped, low-turnover, net of fees")

#=
Compare the two weight columns. No target is further than 0.05 from its current weight, and the
largest target is the 10% that the turnover budget permits.

## 3. Finite allocation

The investor has \$10,000. [`GreedyAllocation`](@ref) turns the target weights into whole shares.
It runs no mixed-integer solve, so it is cheap enough to run every day.
=#

alloc = optimise(GreedyAllocation(),
                 FiniteAllocationInput(; w = retail.w, prices = prices, cash = 10_000.0))

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => retail.w,
                       "Shares" => round.(Int, alloc.shares), "Realised" => alloc.w);
             formatters = [resfmt],
             title = "\$10,000 allocated — invested \$$(round(Int, invested)), cash left \$$(round(alloc.cash, digits = 2))")

#=
## 4. The book
=#

plot_stacked_bar_composition([retail], rd; xticks = (1:1, ["Retail daily"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New end-to-end profile (7_putting_it_together). Verified on kaimon (f102cae9): full pipeline
#src   EmpiricalPrior → MeanRisk(MinimumRisk, wb ub=0.15, tn val=0.05 vs equal-weight, Fees l=0.001)
#src   → GreedyAllocation $10k. Result maxw 10% (cap binds), 14 names, leftover $2.99.
#src - Composes blocks verified in 4_constraints_costs (wb/turnover/fees) and 6_post_processing
#src   (GreedyAllocation). No new API; the value is the integrated narrative.
