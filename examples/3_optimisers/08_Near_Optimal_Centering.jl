#=
# Near optimal centering

A classic optimiser returns the single point that *exactly* extremises its objective — the
maximum-ratio portfolio, the minimum-variance portfolio, and so on. That point often sits on a
knife edge: a corner solution that loads heavily on a handful of assets and shifts a lot when
the inputs wobble. [`NearOptimalCentering`](@ref) (NOC) trades a sliver of optimality for
stability. Instead of the extreme point, it returns the portfolio at the **analytic centre of
the near-optimal neighbourhood** — the region of solutions that are *almost* as good as the
optimum. The neighbourhood is parametrised by binning the efficient frontier (`bins`).

We met NOC briefly as a frontier/surface engine in the efficient-frontier and Pareto-surface
examples; here we focus on the behaviour that makes it its own optimiser: how its centred
solution differs from the extreme point of the same objective.

!!! tip "When to reach for this"
    Reach for NOC when you like a [`MeanRisk`](@ref) objective but distrust its corner
    solutions — when you want the spirit of "maximum risk-adjusted return" without betting the
    book on two assets, or a more stable allocation that survives small changes in the prior.
    Use [`UnconstrainedNearOptimalCentering`](@ref) for the plain centred portfolio, and
    [`ConstrainedNearOptimalCentering`](@ref) when the centred solution must also satisfy the
    problem's external constraints (at the cost of a harder solve). If you genuinely want the
    extreme point, use [`MeanRisk`](@ref) directly.
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
## 1. ReturnsResult data

We use the same S&P 500 slice as the other optimiser examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. Prior and solvers

NOC solves a harder problem than a plain `MeanRisk` (it bins the frontier and centres within a
neighbourhood), so a single solver configuration can fail to converge. We therefore pass a
*vector* of solvers with decreasing `max_step_fraction`; the optimiser falls back through them
until one succeeds. We also use the SOC-based [`StandardDeviation`](@ref) risk measure — see
the findings note on why a plain quadratic [`Variance`](@ref) is not a good fit for NOC.
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

First the plain maximum risk-adjusted return portfolio. This is the extreme point NOC will
centre around.
=#

res_mr = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; rf = rf),
                           opt = opt))

#=
## 4. Unconstrained near optimal centering

Now the same objective through NOC. [`UnconstrainedNearOptimalCentering`](@ref) does not impose
the problem's external constraints on the centred solution, which keeps the solve tractable.
=#

res_noc_u = optimise(NearOptimalCentering(; r = StandardDeviation(),
                                          obj = MaximumRatio(; rf = rf), opt = opt,
                                          alg = UnconstrainedNearOptimalCentering()))

#=
## 5. Constrained near optimal centering

[`ConstrainedNearOptimalCentering`](@ref) additionally requires the centred portfolio to
satisfy the external constraints. It is a harder solve (hence the solver fallback vector), but
keeps the result feasible with respect to whatever bounds or budgets you have imposed.
=#

res_noc_c = optimise(NearOptimalCentering(; r = StandardDeviation(),
                                          obj = MaximumRatio(; rf = rf), opt = opt,
                                          alg = ConstrainedNearOptimalCentering()))

#=
## 6. Comparing the allocations

The contrast is the whole point. The extreme maximum-ratio portfolio piles into a couple of
assets; both NOC variants spread the same objective across many more names by sitting at the
centre of the near-optimal region rather than at its corner.
=#

pretty_table(DataFrame(; :assets => rd.nx, Symbol("MaxRatio (extreme)") => res_mr.w,
                       Symbol("NOC unconstrained") => res_noc_u.w,
                       Symbol("NOC constrained") => res_noc_c.w); formatters = [resfmt])

#=
A quick numeric summary of the diversification difference: the largest single weight and the
number of materially-held assets.
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
## 7. Visualising the compositions

The stacked-bar composition shows the corner solution collapsing onto a few assets while NOC
fans the allocation out.
=#

# Composition: extreme MaxRatio vs the two NOC variants.
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
