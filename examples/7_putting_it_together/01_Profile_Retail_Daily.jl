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
four shape this investor's choices.

  - The rebalance is daily, so you pay for the optimisation every trading day. One convex solve is
    enough.
  - Trading cost compounds when you trade every day. We cap how far each weight can move from the
    current book, and we charge a fee on every unit of weight traded. A trade then happens only
    where its expected gain is larger than its fee.
  - A cap of 8% per name limits how much of the book one name can take.
  - The account is small, so one whole share is a large part of a position. The finite allocation
    at the end moves the weights you hold away from the weights you solved for.

!!! tip "When to reach for this"
    Reach for this profile when trading cost and account size bind harder than the model does. Keep
    the optimisation to one convex solve, put the turnover bound and the trading fee inside the
    optimiser, and size the last step to the cash you have. The fee is a deduction from the
    portfolio return, so it changes the weights only under an objective that uses that return.
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

One convex solve takes all the choices. The objective is [`MaximumUtility`](@ref), the expected
return net of the fee, less two times the variance. The weight bounds cap each name at 8%. The
turnover budget keeps each target weight within 0.05 of the weight the investor has today. The
fee is 0.1% of each unit of weight traded, and the solve deducts it from the expected return.

A minimum-risk objective would ignore the fee, as the tip says. A fee on each long position would
not work either, because a fully-invested long-only book pays the same total fee whatever its
weights are.
=#

retail = optimise(MeanRisk(; obj = MaximumUtility(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               wb = WeightBounds(; lb = 0.0, ub = 0.08),
                                               tn = Turnover(; w = current_book,
                                                             val = 0.05),
                                               fees = Fees(;
                                                           tn = Turnover(; w = current_book,
                                                                         val = 0.001)))))

pretty_table(DataFrame("Asset" => rd.nx, "Current" => current_book, "Target" => retail.w);
             formatters = [resfmt], title = "Retail daily target against the current book")

#=
Compare the two weight columns. No target is further than 0.05 from its current weight or above
the 8% cap, and most names keep their current weight.

We solve the same problem two more times, once without the fee and once without the cap, to show
what each of them changes.
=#

no_fee = optimise(MeanRisk(; obj = MaximumUtility(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               wb = WeightBounds(; lb = 0.0, ub = 0.08),
                                               tn = Turnover(; w = current_book,
                                                             val = 0.05))))
no_cap = optimise(MeanRisk(; obj = MaximumUtility(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               tn = Turnover(; w = current_book,
                                                             val = 0.05),
                                               fees = Fees(;
                                                           tn = Turnover(; w = current_book,
                                                                         val = 0.001)))))

books = [retail, no_fee, no_cap]
pretty_table(DataFrame("Book" => ["Retail", "Without the fee", "Without the cap"],
                       "Turnover" => [sum(abs, b.w - current_book) for b in books],
                       "Largest weight" => [maximum(b.w) for b in books],
                       "Names traded" =>
                           [count(>(1e-4), abs.(b.w - current_book)) for b in books]);
             formatters = [resfmt], title = "What the fee and the cap change")

#=
Without the fee, the book trades more than twice as much and in more names. Without the cap, the
largest weight goes past 8%.

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
#src   EmpiricalPrior → MeanRisk → GreedyAllocation $10k.
#src - #1286 (2026-09-23): the first version used MinimumRisk, ub=0.15 and Fees(l=0.001). Neither
#src   the fee nor the cap changed the book: a MinimumRisk variance objective ignores the return the
#src   fee is deducted from, a long fee costs a fully-invested long-only book `l` whatever `w` is,
#src   and turnover 0.05 on 1/20 caps every weight at 0.10. Now MaximumUtility, ub=0.08 and a
#src   turnover fee of 0.001. Bare run: retail turnover 0.30, maxw 0.08 (cap binds), 8 of 20
#src   names traded; without the fee 0.72 and 20 traded; without the cap maxw 0.10. All three
#src   OptimisationSuccess. GreedyAllocation $10k: 17 names, leftover $10.00.
#src - Composes blocks verified in 4_constraints_costs (wb/turnover/fees) and 6_post_processing
#src   (GreedyAllocation). No new API; the value is the integrated narrative.
