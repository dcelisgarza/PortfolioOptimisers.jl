#=
```@meta
Description = "Trace the efficient frontier in PortfolioOptimisers.jl: the whole risk-return curve from one MeanRisk estimator, and how to read it."
```

# Efficient frontier

A single [`MeanRisk`](@ref) optimisation returns *one* portfolio. The efficient frontier is the
whole curve of portfolios that earn the most return for each level of risk. They are also the
portfolios that take the least risk for each level of return. You compute the whole trade-off first,
and then choose a point on it.

This page adds three things to the [`MeanRisk` objectives](01_MeanRisk_Objectives.md) page. It
computes the frontier from both directions, once by minimising risk above a return floor and once
by maximising return below a risk ceiling, and plots the two on one chart. It introduces
[`Frontier`](@ref), which finds the bounds of the frontier for you. It then compares the
[`MeanRisk`](@ref) frontier with the centred frontier of [`NearOptimalCentering`](@ref), which gives
up a little of the objective for a portfolio that holds more assets at each point.

!!! tip "When to reach for this"
    Reach for an efficient frontier when you do not want to fix one point of risk and return
    in advance. You see the whole curve and choose a portfolio on it, or pass its points to a
    rule that selects one. Read the [`MeanRisk`](@ref) objectives page first, since each point
    of a frontier is one of those optimisations. For more than two criteria, see the
    [Pareto surface](03_Pareto_Surface.md) example.
=#

using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data

We use the same data as the previous example.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X)

#=
## 2. Two directions, four combinations

You compute a frontier in one of two directions.

  - Minimise risk with a *lower bound on the return*, and raise the bound from point to point.
  - Maximise return with an *upper bound on the risk*, and raise the bound from point to point.

Each bound can be a `range` of numbers that you choose, or a [`Frontier`](@ref), which finds the
lowest and the highest feasible value and spaces the points between them. Two directions and two
kinds of bound give four combinations, and this page uses `Frontier` in both directions. The two
directions give the *same* curve, so choose the one whose quantity is easier to set in your
problem.

We use the conditional value at risk (CVaR), [`ConditionalValueatRisk`](@ref), as the risk measure
on the whole page, and compute the prior once and share it across every optimisation. One solver
setting can fail to converge at some points of a frontier, so we pass a vector of two solvers. When
the first fails, the optimiser tries the second.
=#

using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
              check_sol = (; allow_local = true, allow_almost = true))]

r = ConditionalValueatRisk()
pr = prior(EmpiricalPrior(), rd)
rf = 4.2 / 100 / 252

#=
### Direction A: minimise risk above a return floor

We set no objective, so the optimisation takes the default, `MinimumRisk`, and minimises the CVaR. A
[`Frontier`](@ref) of 30 points sets the lower bound on the return. The bound is on the *return*, so
it goes in the settings of [`ArithmeticReturn`](@ref), as `lb`.
=#

optA = JuMPOptimiser(; pe = pr, slv = slv,
                     ret = ArithmeticReturn(;
                                            settings = JuMPReturnsSettings(;
                                                                           lb = Frontier(;
                                                                                         N = 30))))
resA = optimise(MeanRisk(; opt = optA, r = r))

#=
`retcode` and `sol` now have one entry per point of the frontier. We check that every point
solved.
=#

all(x -> isa(x, OptimisationSuccess), resA.retcode)

#=
The table has one column of weights per point. From point 1 to point 30 the portfolio moves from
many assets at the low-risk end to few assets at the high-return end.
=#

pretty_table(DataFrame([rd.nx hcat(resA.w...)], Symbol.([:assets; 1:30]));
             formatters = [resfmt])

#=
### Direction B: maximise return below a risk ceiling

We maximise the return, and a [`Frontier`](@ref) of 30 points sets an *upper bound on the CVaR*.
The bound is on the *risk*, so it goes in the settings of the risk measure,
[`RiskMeasureSettings`](@ref), as `ub`. The prior and the solvers stay the same, and the last line
checks that every point solved.
=#

optB = JuMPOptimiser(; pe = pr, slv = slv)
resB = optimise(MeanRisk(; opt = optB, obj = MaximumReturn(),
                         r = ConditionalValueatRisk(;
                                                    settings = RiskMeasureSettings(;
                                                                                   ub = Frontier(;
                                                                                                 N = 30)))))
all(x -> isa(x, OptimisationSuccess), resB.retcode)

#=
We compute the CVaR and the arithmetic return of each point of both frontiers, and plot the two
on one chart. The points of both directions lie on one curve. They sit at different places along
it, because each `Frontier` spaces its points over a different quantity.
=#

rcvar = factory(ConditionalValueatRisk(), pr)
xs_A = [expected_risk(rcvar, w, pr.X) for w in resA.w]
ys_A = [expected_return(ArithmeticReturn(), w, pr) for w in resA.w]
xs_B = [expected_risk(rcvar, w, pr.X) for w in resB.w]
ys_B = [expected_return(ArithmeticReturn(), w, pr) for w in resB.w]

using StatsPlots, GraphRecipes

plot(xs_A, ys_A; seriestype = :scatter, marker = (:circle, 5),
     label = "Min risk | return floor", xlabel = "CVaR", ylabel = "Arithmetic return",
     title = "Same frontier from both directions")
plot!(xs_B, ys_B; seriestype = :scatter, marker = (:cross, 7),
      label = "Max return | risk ceiling")

#=
## 3. The `MeanRisk` frontier vs the `NearOptimalCentering` frontier

Each point of the frontier above lies at the *edge* of the set of feasible portfolios, because it
is the exact optimum of its problem. [`NearOptimalCentering`](@ref),
NOC for short, computes a centred frontier instead. At each point it returns the portfolio at the
analytic centre of the set of near-optimal portfolios, not the one at its edge. That portfolio
gives up a little of the objective, holds more assets, and moves less when the prior changes. The
[NOC page](15_Near_Optimal_Centering.md) shows how it finds the centre. Here we use it only to
compute a frontier.

NOC solves a harder problem than `MeanRisk`, so one solver setting can fail to converge. We give it
seven solvers, whose `max_step_fraction` falls from 0.99 to 0.7.
=#

slv_noc = [Solver(; name = Symbol("clarabel$i"), solver = Clarabel.Optimizer,
                  settings = Dict("verbose" => false, "max_step_fraction" => f),
                  check_sol = (; allow_local = true, allow_almost = true))
           for (i, f) in enumerate((0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7))]

#=
We build *both* frontiers with the same return floor, a `Frontier` of 15 points, so that only the
optimiser differs.
=#

ret15 = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = 15)))
resM = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv_noc, ret = ret15),
                         r = r))
resN = optimise(NearOptimalCentering(;
                                     opt = JuMPOptimiser(; pe = pr, slv = slv_noc,
                                                         ret = ret15), r = r))

#=
NOC solves many `MeanRisk` problems and returns one `retcode` for all of them. We check that it is
a success.
=#

isa(resN.retcode, OptimisationSuccess)

#=
We print the largest single weight at each point of both frontiers. A lower number means that the
portfolio spreads its weight over more assets. The NOC column is lower at most points. The two
columns come close at the high-return end, where both frontiers approach the one portfolio that
maximises the return.
=#

maxw(ws) = [round(maximum(w) * 100; digits = 1) for w in ws]
pretty_table(DataFrame("point" => 1:15, "MeanRisk max w %" => maxw(resM.w),
                       "NOC max w %" => maxw(resN.w));
             title = "Largest single weight along each frontier")

#=
We plot both frontiers with the CVaR on the x-axis and the return on the y-axis. The NOC frontier
lies *inside* the `MeanRisk` frontier, to its right, because for a given return it takes a little
more CVaR. That extra CVaR is the cost of a portfolio at the centre of the near-optimal set instead
of at its edge.
=#

xs_M = [expected_risk(rcvar, w, pr.X) for w in resM.w]
ys_M = [expected_return(ArithmeticReturn(), w, pr) for w in resM.w]
xs_N = [expected_risk(rcvar, w, pr.X) for w in resN.w]
ys_N = [expected_return(ArithmeticReturn(), w, pr) for w in resN.w]

plot(xs_M, ys_M; seriestype = :scatter, marker = (:circle, 5), label = "MeanRisk (extreme)",
     xlabel = "CVaR", ylabel = "Arithmetic return", title = "Extreme vs centred frontier")
plot!(xs_N, ys_N; seriestype = :scatter, marker = (:diamond, 6), label = "NOC (centred)")

#=
The stacked areas show the weights at each point of the `MeanRisk` frontier.
=#

plot_stacked_area_composition(resM.w, rd.nx)

#=
The `MeanRisk` frontier holds a few assets at its low-risk end and one at its high-return end.
Compare the number of coloured bands at each end with the same plot for the NOC frontier below.
=#

plot_stacked_area_composition(resN.w, rd.nx)

#=
## 4. Visualising the frontier

The efficient frontier is a special case of a Pareto front, and [`plot_measures`](@ref) plots one
on any pair of axes. Keywords set the measure on the x-axis, the y-axis, the z-axis and the colour
bar. Here the CVaR is on the x-axis, the arithmetic return on the y-axis, and the colour shows the
risk-return ratio.
=#

plot_measures(resA.w, resA.pr; x = r, y = ExpectedReturn(; rt = resA.ret),
              c = ExpectedReturnRiskRatio(; rt = resA.ret, rk = r, rf = rf),
              title = "Efficient Frontier", xlabel = "CVaR", ylabel = "Arithmetic Return",
              colorbar_title = "\nReturn/Risk Ratio", right_margin = 6Plots.mm)

#=
`plot_measures` takes *any* pair of measures, so you can view the same 30 portfolios on other
axes. Here the y-axis is the conditional drawdown at risk (CDaR),
[`ConditionalDrawdownatRisk`](@ref), which is the CVaR of the drawdowns. The colour is the ratio
of the CDaR to the CVaR. These
portfolios do not minimise the CDaR, so this plot is not a front of the CVaR against the CDaR.
=#

plot_measures(resA.w, resA.pr; x = r, y = ConditionalDrawdownatRisk(),
              c = RiskRatio(; r1 = ConditionalDrawdownatRisk(), r2 = r),
              title = "CDaR of the CVaR frontier", xlabel = "CVaR", ylabel = "CDaR",
              colorbar_title = "\nCDaR/CVaR Ratio", right_margin = 6Plots.mm)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Deepened (ADR 0014, examples-are-deep-dives): the page now delivers what its intro promised
#src   — both frontier directions AND the NearOptimalCentering frontier. Verified on kaimon
#src   (session 56c8906d):
#src   - Direction A (min CVaR | return-lb Frontier, N=30) and Direction B (max return | CVaR-ub
#src     Frontier via RiskMeasureSettings, N=30) both solve all 30 points (OptimisationSuccess)
#src     and trace the same curve.
#src   - NOC frontier (N=15, 7-solver fallback) solves; summary retcode OptimisationSuccess.
#src     Apples-to-apples vs MeanRisk (same N=15 return-lb sweep): NOC avg max-weight 53.7% vs
#src     59.5%, peak 93.6% vs 100% — NOC fans out except at the return-max corner where both pin.
#src     NOC points sit inside the MeanRisk frontier (higher CVaR for a given return), as expected.
#src - API confirmed: risk-side frontier bound = RiskMeasureSettings(; ub = Frontier(; N)); NOC
#src   accepts ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(...))) and returns a vector frontier.
#src - No doc/ergonomics/bug findings beyond the harmless `Plots.mm not public` warning (pre-existing).
#src   Group rollup: issue #125.
