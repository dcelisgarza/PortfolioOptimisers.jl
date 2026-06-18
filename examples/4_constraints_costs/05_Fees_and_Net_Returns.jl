#=
# Fees and net returns

Every trade and every position costs something, and a portfolio that looks good *gross* can be
mediocre *net*. `PortfolioOptimisers.jl` models costs with [`Fees`](@ref) — proportional and
fixed charges on long and short positions, plus a turnover-based component — and lets you both
**optimise net of fees** (the optimiser trades expected return against cost) and **evaluate** a
portfolio's net return after the fact with [`calc_net_returns`](@ref).

This closes the constraints-and-costs group: where [Turnover and Tracking](04_Turnover_and_Tracking.md)
charged for *moving*, this page charges for *holding*.

!!! tip "When to reach for this"
    Reach for fees whenever costs are material to the decision, not just the reporting: high-fee
    instruments, asymmetric long/short costs, or a strategy whose gross edge is thin enough that
    fees decide whether it is worth running. Use the per-asset or per-group form to make the
    optimiser *prefer cheap exposure*, and `calc_net_returns` to check that a candidate book still
    earns its keep after costs.
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
## 1. ReturnsResult data
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

res_base = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
## 2. Fees erase returns

The starkest way to see why fees matter: take the unconstrained maximum-ratio book and look at
its gross return against its return *net* of a proportional fee. [`calc_fees`](@ref) computes the
cost of holding a weight vector, and [`calc_net_returns`](@ref) deducts it from the gross
returns. A flat 0.2% long fee on this book costs about as much per day as the strategy earns —
the net edge nearly vanishes.
=#

fee = Fees(; l = 0.002)
gross_daily = sum(rd.X * res_base.w) / size(rd.X, 1)
net_daily = sum(calc_net_returns(res_base.w, rd.X, fee)) / size(rd.X, 1)
fee_cost = calc_fees(res_base.w, fee)

pretty_table(DataFrame("Quantity" => ["Gross daily return", "Fee cost", "Net daily return"],
                       "Value" => [gross_daily, fee_cost, net_daily]);
             formatters = [resfmt],
             title = "A 0.2% long fee against this book's daily edge")

#=
## 3. Optimising net of fees

Passing `fees` to the [`JuMPOptimiser`](@ref) makes the optimiser maximise the ratio *after*
costs, so it only takes exposure whose return justifies its fee. A uniform [`Fees`](@ref) charges
the same rate everywhere (`l` for long, `s` for short positions).
=#

res_fee = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                fees = Fees(; l = 0.002))))

#=
## 4. Differential fees steer the allocation

Costs are rarely uniform — some instruments are expensive to hold. A [`FeesEstimator`](@ref)
(the estimator counterpart of [`Fees`](@ref)) accepts per-asset or per-group rates over an
[`AssetSets`](@ref), exactly like the linear constraints. Charging a punitive fee on the sector
the baseline loved makes the optimiser walk away from it.
=#

sets = AssetSets(;
                 dict = Dict("nx" => rd.nx,
                             "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"]))
res_diff = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                             opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                 fees = FeesEstimator(;
                                                                      l = ["healthcare" =>
                                                                               0.02]))))

function healthcare_weight(w)
    return sum(w[i] for i in eachindex(w) if rd.nx[i] in sets.dict["healthcare"])
end
pretty_table(DataFrame("Portfolio" => ["Baseline", "Healthcare fee 2%"],
                       "Healthcare weight" =>
                           [healthcare_weight(res_base.w), healthcare_weight(res_diff.w)]);
             formatters = [resfmt],
             title = "A targeted fee steers exposure away from a sector")

#=
## 5. Fixed fees need a MIP solver

Proportional fees (`l`, `s`) are convex and solve with Clarabel. **Fixed** fees (`fl`, `fs`) — a
flat charge incurred only *if* a position is opened — are semi-continuous (zero or the fee), the
same structure as thresholds and cardinality. They require a mixed-integer-capable solver
([Pajarito](https://github.com/jump-dev/Pajarito.jl) + [HiGHS](https://github.com/jump-dev/HiGHS.jl));
handed to a continuous-only solver the problem fails to solve rather than returning a wrong
answer. See [Budget Constraints](01_Budget_Constraints.md) for the MIP setup. Fees also carry a
turnover component (the `tn` field), tying the holding cost here to the trading cost in
[Turnover and Tracking](04_Turnover_and_Tracking.md).

## 6. Comparing the books

Gross-optimal, net-optimal, and fee-steered — same prior and objective, different cost
assumptions.
=#

results = [res_base, res_fee, res_diff]
labels = ["Gross", "Net (0.2%)", "Healthcare fee"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights under each fee assumption")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive; closes the 4_constraints_costs group. Verified on kaimon (f102cae9):
#src   - calc_net_returns / calc_fees: base gross 19.98 bps/day, Fees(l=0.002) costs 20.0 bps →
#src     net -0.02 bps. Clean "fees erase the edge" story.
#src   - Differential fee via FeesEstimator(l=["healthcare"=>0.02]) over AssetSets: base
#src     healthcare 66%→0%. (NOTE: Fees the RESULT type takes only Number/Vector; group/pair
#src     rates need FeesEstimator, the estimator — easy to trip over, the error is a TypeError on
#src     the `l` kwarg. Documented in §4.)
#src   - Proportional l/s convex (Clarabel ok). FIXED fees fl/fs are MIP: Fees(fl=0.001) on
#src     Clarabel "Failed to solve", returns NaN weights. Documented §5 with Pajarito/HiGHS pointer.
#src - Fees.tn ties holding cost to turnover (04) — noted, not re-demoed.
