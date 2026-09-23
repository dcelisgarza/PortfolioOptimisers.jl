#=
```@meta
Description = "Near optimal centering in PortfolioOptimisers.jl: give up a little of the objective for the analytic centre of the near-optimal region, which moves less."
```

# Near optimal centering

A plain optimiser returns the one point that maximises or minimises its objective, such as
the maximum-ratio portfolio or the minimum-variance portfolio. That point often sits in a
corner of the feasible set. It puts most of the money on a few assets, and a small change to
the inputs moves it a long way. [`NearOptimalCentering`](@ref), or NOC, gives up a little of
the objective to move less. It returns the analytic centre of the near-optimal region, which
holds the portfolios whose objective is close to the best one. `bins` sets the size of that
region by splitting the efficient frontier into that many steps.

The efficient-frontier page and the Pareto-surface page both use NOC to trace a curve. This
page asks a different question, which is how far the centred portfolio sits from the extreme
point of the same objective.

!!! tip "When to reach for this"
    Reach for NOC when the [`MeanRisk`](@ref) objective is the one you want and its corner
    solution is not. You get a portfolio with a high risk-adjusted return that does not rest on
    two assets, and it moves less when the prior changes. Use
    [`UnconstrainedNearOptimalCentering`](@ref) when weight bounds and budgets are the only
    constraints the centred portfolio must meet, and [`ConstrainedNearOptimalCentering`](@ref)
    when it must also meet the linear, cardinality, turnover and other constraints you set,
    which is a harder solve. If the extreme point is what you want, use [`MeanRisk`](@ref).
=#

using PortfolioOptimisers, PrettyTables

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data

We use one year of daily prices for twenty S&P 500 stocks.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. Prior and solvers

NOC splits the frontier into bins and then centres inside a region, so it solves a harder
problem than a plain [`MeanRisk`](@ref). One solver setting can fail to converge on it. We
pass a vector of solvers whose `max_step_fraction` falls from one to the next, and the
optimiser tries each in turn until one returns a solution.

We also pass [`StandardDeviation`](@ref), which is NOC's default risk measure and states risk
as a second-order cone. A plain [`Variance`](@ref) states it as a quadratic expression. NOC
warns on a quadratic risk expression and then fails to solve on this data, so reach for
[`StandardDeviation`](@ref) here.
=#

using Clarabel

slv = [Solver(; name = Symbol("clarabel$i"), solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => f),
              check_sol = (; allow_local = true, allow_almost = true))
       for (i, f) in enumerate((0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7))]

pr = prior(EmpiricalPrior(), rd)
rf = 4.2 / 100 / 252
opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
## 3. The reference: a `MeanRisk` corner solution

We first solve for the largest risk-adjusted return. This is the extreme point that NOC
centres around. `rf` is the daily risk-free rate that the ratio subtracts from the expected
return, 4.2% a year divided by 252 trading days.
=#

res_mr = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; rf = rf),
                           opt = opt))

#=
## 4. Unconstrained near optimal centering

The next cell runs the same objective through NOC.
[`UnconstrainedNearOptimalCentering`](@ref) keeps the weight bounds and the budgets on the
centred portfolio and leaves the other constraints of the problem off it, which keeps the solve
cheap.
=#

res_noc_u = optimise(NearOptimalCentering(; r = StandardDeviation(),
                                          obj = MaximumRatio(; rf = rf), opt = opt,
                                          alg = UnconstrainedNearOptimalCentering()))

#=
## 5. Constrained near optimal centering

[`ConstrainedNearOptimalCentering`](@ref) asks the centred portfolio to meet every constraint
of the problem, the linear, cardinality, turnover and other constraints you set on the optimiser
as well as the bounds and budgets. It is the harder solve of the two, which is why section 2
passes seven solvers. This page sets no constraint beyond the default bounds and budget, so here
both variants centre under the same bounds.
=#

res_noc_c = optimise(NearOptimalCentering(; r = StandardDeviation(),
                                          obj = MaximumRatio(; rf = rf), opt = opt,
                                          alg = ConstrainedNearOptimalCentering()))

#=
## 6. Comparing the allocations

Read the three weight columns against each other.
=#

pretty_table(DataFrame(; :assets => rd.nx, Symbol("MaxRatio (extreme)") => res_mr.w,
                       Symbol("NOC unconstrained") => res_noc_u.w,
                       Symbol("NOC constrained") => res_noc_c.w); formatters = [resfmt])

#=
We print two numbers per portfolio, the largest single weight and the number of
assets whose weight is above 0.01%.
=#

summarise(w) = (round(maximum(w) * 100; digits = 2), count(>(1e-4), w))
pretty_table(DataFrame(;
                       :portfolio =>
                           ["MaxRatio (extreme)", "NOC unconstrained", "NOC constrained"],
                       Symbol("max weight %") =>
                           [summarise(res_mr.w)[1], summarise(res_noc_u.w)[1],
                            summarise(res_noc_c.w)[1]],
                       Symbol("assets held") =>
                           [summarise(res_mr.w)[2], summarise(res_noc_u.w)[2],
                            summarise(res_noc_c.w)[2]]))

#=
The extreme maximum-ratio portfolio holds far fewer assets than either NOC portfolio, and its
largest weight is larger. The centre of the near-optimal region lies away from the corner that
the extreme point occupies.

## 7. Visualising the compositions

The plot stacks the three allocations, with the extreme portfolio first.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_mr, res_noc_u, res_noc_c], rd)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - On this slice the contrast lands cleanly: MaximumRatio holds ~2 assets (max ~66%) while
#src   both NOC variants hold ~20 (max ~42%), demonstrating the neighbourhood-centring effect.
#src - FRAGILITY (record-only per hybrid policy → issue #125): NOC fails to converge with a
#src   *single* Clarabel solver here ("Failed to solve optimisation problem"), but succeeds
#src   with the 7-solver decreasing-max_step_fraction fallback vector. The harder NOC solve is
#src   noticeably more solver-sensitive than plain MeanRisk (which solves single-solver). Worth
#src   a note in the NearOptimalCentering docstring recommending a solver fallback vector.
#src - ERGO (record-only → issue #125): NOC with a plain quadratic `Variance()` emits
#src   "Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work"
#src   and then fails, whereas the SOC `StandardDeviation()` (NOC's default `r`) works. The
#src   warning is good, but pairing it with a pointer to StandardDeviation/the SDP variance
#src   formulation would close the loop for users who reach for Variance by habit.
