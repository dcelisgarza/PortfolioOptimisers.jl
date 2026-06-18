#=
# Finite allocation

An optimiser returns *continuous* weights — fractions of capital. To actually trade them you need
whole shares at real prices under a finite cash budget, and the rounding is not free: it pulls the
realised portfolio away from the target, and the smaller the account the more it hurts.
`PortfolioOptimisers.jl` provides two finite-allocation optimisers — a fast solver-free heuristic
and an exact mixed-integer one — both called through the same `optimise(allocator, w, prices, cash)`
interface.

  - [`GreedyAllocation`](@ref) — a two-pass heuristic: round to whole (or lot-sized) shares, then
    spend the leftover cash on the largest underweights. No solver needed.
  - [`DiscreteAllocation`](@ref) — solves a mixed-integer program for the *optimal* whole-share
    book. Needs a MIP solver.

!!! tip "When to reach for this"
    Reach for finite allocation as the last step before trading, always — continuous weights are
    not executable. Use [`GreedyAllocation`](@ref) when you want an instant, dependable answer
    (and for very large books where the MIP is slow); use [`DiscreteAllocation`](@ref) when the
    account is small enough that the rounding genuinely matters and you want the provably best
    integer book. Watch the realised-vs-target drift — it is your discretisation error.
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
## 1. A target portfolio and prices

We optimise a minimum-risk book, then read the latest prices off the price series — finite
allocation needs both the target weights and a price per share.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res = optimise(MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pr, slv = slv)))
prices = vec(values(X)[end, :])

#=
## 2. Greedy allocation

[`GreedyAllocation`](@ref) needs no solver. The result carries the integer `shares`, the per-asset
`cost`, the *realised* weights `w`, and the leftover `cash`. With a \$100,000 budget the realised
weights track the target tightly and only a few dollars go uninvested.
=#

cash = 100_000.0
greedy = optimise(GreedyAllocation(), res.w, prices, cash)

drift(alloc) = sum(abs, alloc.w .- res.w)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => res.w,
                       "Shares" => round.(Int, greedy.shares), "Realised" => greedy.w);
             formatters = [resfmt],
             title = "Greedy allocation of \$$(round(Int, cash)) — leftover cash \$$(round(greedy.cash, digits = 2)), drift $(round(drift(greedy), digits = 4))")

#=
## 3. Exact allocation with a MIP solver

[`DiscreteAllocation`](@ref) solves for the *optimal* whole-share book instead of a greedy one.
It needs a mixed-integer solver — here [HiGHS](https://github.com/jump-dev/HiGHS.jl). On a book
this size the greedy heuristic is already at (or very near) the optimum, so the two agree; the
value of the MIP shows up on tighter budgets and lot constraints where the greedy pass can leave
gains on the table.
=#

mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
discrete = optimise(DiscreteAllocation(; slv = mip_slv), res.w, prices, cash)

pretty_table(DataFrame("Method" => ["Greedy", "Discrete (MIP)"],
                       "Leftover cash" => [greedy.cash, discrete.cash],
                       "Drift from target" => [drift(greedy), drift(discrete)]);
             formatters = [resfmt], title = "Greedy vs exact allocation")

#=
## 4. Lot sizes

Many instruments trade in lots, not single shares. `GreedyAllocation(; unit = u)` rounds to
multiples of `u` shares. Coarser lots mean a coarser allocation — the drift grows, and a large lot
can even overshoot the budget (the leftover cash goes negative), which is the signal that the lot
size is too big for the account.
=#

greedy_lots = optimise(GreedyAllocation(; unit = 10), res.w, prices, cash)

pretty_table(DataFrame("Allocation" => ["Single shares", "Lots of 10"],
                       "Leftover cash" => [greedy.cash, greedy_lots.cash],
                       "Drift from target" => [drift(greedy), drift(greedy_lots)]);
             formatters = [resfmt], title = "Lot size coarsens the allocation")

#=
## 5. Budget size is the discretisation error

The same rounding that is negligible on a large account dominates a small one. Allocating the
identical target into \$100,000 versus \$5,000 shows the drift growing by an order of magnitude —
on a small account, the *choice* of finite-allocation method (and lot size) matters most.
=#

budgets = [100_000.0, 25_000.0, 5_000.0]
budget_allocs = [optimise(GreedyAllocation(), res.w, prices, c) for c in budgets]

pretty_table(DataFrame("Budget" => budgets,
                       "Leftover cash" => [a.cash for a in budget_allocs],
                       "Drift from target" => [drift(a) for a in budget_allocs]);
             formatters = [resfmt],
             title = "Smaller budgets suffer larger discretisation error")

#=
Both allocators also accept a [`Fees`](@ref) argument, so the share counts can be chosen net of
transaction costs (see [Fees and Net Returns](../4_constraints_costs/05_Fees_and_Net_Returns.md)).

## 6. Target vs realised
=#

plot_stacked_bar_composition([res, greedy, discrete], rd;
                             xticks = (1:3, ["Target", "Greedy", "Discrete"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive (6_post_processing). Verified on kaimon (f102cae9) with HiGHS as MIP solver:
#src   - GreedyAllocation 100k: invested $99,986, leftover $14.22, drift 0.0033.
#src   - DiscreteAllocation(HiGHS) 100k: identical here ($99,986, drift 0.0033) — greedy already
#src     optimal on this book; framed honestly (MIP's edge is on tight budgets / lots).
#src   - Budget sweep monotone: drift 100k=0.0033, 5k=0.0947 (~30x). Clean "small accounts suffer"
#src     story.
#src   - FINDING: GreedyAllocation(unit=10) → leftover cash NEGATIVE (-$268.44), drift 0.034. Large
#src     lots overshoot the budget; documented in §4 as the signal the lot is too big. A guard /
#src     warning when residual cash goes negative would help.
#src - optimise(GreedyAllocation()|DiscreteAllocation(; slv=mip), w, prices, cash); prices =
#src   vec(values(X)[end,:]). Result fields shares/cost/w/cash.
