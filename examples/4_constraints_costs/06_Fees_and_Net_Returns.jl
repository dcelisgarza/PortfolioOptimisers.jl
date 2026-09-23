#=
```@meta
Description = "Fees and net returns in PortfolioOptimisers.jl: proportional, fixed and turnover fees, optimised net of cost and evaluated with calc_net_returns."
```

# Fees and net returns

A portfolio with a good return before costs can have a poor return after them. A [`Fees`](@ref)
contains the costs: proportional and fixed charges on long and short positions, and a charge on
turnover. You can optimise net of fees, so that the optimiser weighs the expected return against
the cost. You can also compute the returns of a portfolio after costs with
[`calc_net_returns`](@ref).

[Turnover and tracking](05_Turnover_and_Tracking.md) bounds how far the weights move. This page
charges for the positions, and section 5 names the charge on trades.

!!! tip "When to reach for this"
    Reach for fees when costs change the decision: instruments with high fees, different costs
    on the long and the short side, or a strategy with a return before costs so small that the
    fees can remove it. Use the per-asset or per-group form so the optimiser prefers the cheaper
    assets, and `calc_net_returns` to find the return of a candidate book after costs.
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
## 1. Data

We load one year of S&P 500 prices, fit an empirical prior, and optimise a maximum-ratio
portfolio with no fees as the baseline.
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
## 2. What a fee costs this portfolio

We compare the mean daily return of the baseline before and after a long fee of 0.2% per period.
[`calc_fees`](@ref) returns the cost of holding a weight vector as a pair: the charge on every
observation, and, with the default `fa`, a one-time charge on the first observation only.
[`calc_net_returns`](@ref) deducts both from the returns. `l`, `s` and `tn` are rates per
period, so a fee made from them alone has a one-time charge of zero. `fl` and `fs` are the
one-time charges. Compare the fee cost of every day with the gross daily return in the table.
=#

fee = Fees(; l = 0.002)
T_obs = size(rd.X, 1)
gross_daily = sum(rd.X * res_base.w) / T_obs
net_daily = sum(calc_net_returns(res_base.w, rd.X, fee)) / T_obs
fee_cost, fee_one_off = calc_fees(res_base.w, T_obs, fee)

pretty_table(DataFrame("Quantity" => ["Gross daily return", "Fee cost, every day",
                                      "Fee cost, one time", "Net daily return"],
                       "Value" => [gross_daily, fee_cost, fee_one_off, net_daily]);
             formatters = [resfmt],
             title = "Daily return of the baseline before and after a 0.2% long fee")

#=
## 3. Optimising net of fees

With `fees` on the [`JuMPOptimiser`](@ref), the optimiser maximises the ratio after costs, so
the fee of each position counts against its expected return. A [`Fees`](@ref) with a number in
`l` charges the same rate on every long position, and `s` does the same for short positions.
=#

res_fee = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                fees = Fees(; l = 0.002))))

#=
## 4. A fee on one sector

A [`FeesEstimator`](@ref) takes rates per asset or per group, by name through a
[`UniverseSets`](@ref), as the linear constraints do. A [`Fees`](@ref) takes only a number or a
vector, so a rate by name needs the estimator. We charge a long fee of 2% on healthcare, the
sector with the largest weight in the baseline, and print the weight of healthcare in the two
portfolios.
=#

sets = UniverseSets(;
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
             title = "Healthcare weight with and without a 2% healthcare fee")

#=
## 5. Fixed fees need a mixed-integer solver

The proportional fees, `l` and `s`, are convex, and Clarabel solves them. The fixed fees, `fl`
and `fs`, are a flat charge on each position the portfolio holds. Each one is either zero or the
whole fee, which is the same structure as a threshold or a cardinality constraint. They need a
mixed-integer solver, as section 6 of [Linear and group
constraints](02_Linear_Group_Constraints.md) explains. A `Fees` also has a `tn` field, a charge
on the turnover from a reference weight vector, which puts a price on the change that [Turnover
and tracking](05_Turnover_and_Tracking.md) limits.

## 6. Comparing the portfolios

We print and plot the three portfolios: the baseline, the portfolio optimised with a 0.2% long
fee, and the portfolio with a 2% fee on healthcare.
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
#src   - Differential fee via FeesEstimator(l=["healthcare"=>0.02]) over UniverseSets: base
#src     healthcare 66%→0%. (NOTE: Fees the RESULT type takes only Number/Vector; group/pair
#src     rates need FeesEstimator, the estimator — easy to trip over, the error is a TypeError on
#src     the `l` kwarg. Documented in §4.)
#src   - Proportional l/s convex (Clarabel ok). FIXED fees fl/fs are MIP: Fees(fl=0.001) on
#src     Clarabel "Failed to solve", returns NaN weights. Documented §5 with Pajarito/HiGHS pointer.
#src - Fees.tn ties holding cost to turnover (04) — noted, not re-demoed.
