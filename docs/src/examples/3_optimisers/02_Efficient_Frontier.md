The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/3_optimisers/02_Efficient_Frontier.jl"
```

# Efficient frontier

A single [`MeanRisk`](@ref) objective returns *one* portfolio. The **efficient frontier** is the
whole curve: the set of portfolios that earn the most return for each level of risk (equivalently,
that take the least risk for each level of return). Instead of committing to one risk/return point
up front, you trace the entire trade-off and choose by eye, hand it to a stakeholder, or feed the
sweep to a downstream selection rule.

This example does three things the [`MeanRisk` objectives](01_MeanRisk_Objectives.md) page does
not. First, it shows the frontier traced from **both** directions — minimising risk at a return
floor, and maximising return at a risk ceiling — and that they recover the same curve. Second, it
introduces the [`Frontier`](@ref) helper, which computes the sweep bounds automatically. Third, it
contrasts the extreme [`MeanRisk`](@ref) frontier with the **centred** frontier produced by
[`NearOptimalCentering`](@ref), which trades a sliver of optimality for a more diversified, stable
allocation at every point.

!!! tip "When to reach for this"
    Reach for an efficient frontier when you do not want to commit to a single risk/return
    point up front — you want to *see the whole trade-off curve* and choose a portfolio by
    eye, hand it to a stakeholder, or feed the sweep to a downstream selection rule. It is
    the natural next step once you understand the [`MeanRisk`](@ref) objectives: instead of
    one objective value, you sweep the risk/return frontier. For more than two competing
    criteria, see the [Pareto surface](03_Pareto_Surface.md) example.

````@example 02_Efficient_Frontier
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
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
nothing #hide
````

## 1. ReturnsResult data

We use the same S&P 500 slice as the other optimiser examples.

````@example 02_Efficient_Frontier
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Two directions, four combinations

There are two mutually exclusive ways to trace a frontier:

- **Minimise risk** subject to a *lower bound on return* — sweep the return floor upward.
- **Maximise return** subject to an *upper bound on risk* — sweep the risk ceiling upward.

Each bound can be supplied **explicitly** (a `range` of numbers you pick) or as a
[`Frontier`](@ref) object, which inspects the problem, finds the feasible extremes, and lays out
the sweep for you. That is the four combinations — two directions × explicit/automatic bounds — and
they all have their uses. The two directions trace the *same* curve; the choice is about which
quantity is more natural to pin in your problem.

We will use the [`ConditionalValueatRisk`](@ref) measure throughout, and precompute the prior once
so every optimisation reuses it. Since we run many optimisations and cannot assume a single solver
configuration converges at every point, we pass a *vector* of solver settings and let the optimiser
fall back through them.

````@example 02_Efficient_Frontier
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
````

### Direction A — minimise risk along a return floor

We minimise CVaR (the default objective) while a [`Frontier`](@ref) sweeps the return lower bound,
giving a 30-point frontier. The lower bound lives on the *return* side, so it is set through
[`ArithmeticReturn`](@ref)'s `lb`.

````@example 02_Efficient_Frontier
optA = JuMPOptimiser(; pe = pr, slv = slv,
                     ret = ArithmeticReturn(; lb = Frontier(; N = 30)))
resA = optimise(MeanRisk(; opt = optA, r = r))
````

`retcode` and `sol` are now *vectors* — one entry per frontier point. We had no warnings about
failed optimisations, but let's confirm every point solved.

````@example 02_Efficient_Frontier
all(x -> isa(x, OptimisationSuccess), resA.retcode)
````

The weights evolve smoothly from the low-risk end (diversified) toward the high-return end
(concentrated) as we walk up the frontier.

````@example 02_Efficient_Frontier
pretty_table(DataFrame([rd.nx hcat(resA.w...)], Symbol.([:assets; 1:30]));
             formatters = [resfmt])
````

### Direction B — maximise return under a risk ceiling

The dual route: maximise return while a [`Frontier`](@ref) sweeps an *upper bound on CVaR*. The
bound now lives on the *risk* side, so it is attached to the risk measure through its
[`RiskMeasureSettings`](@ref). Everything else is identical.

````@example 02_Efficient_Frontier
optB = JuMPOptimiser(; pe = pr, slv = slv)
resB = optimise(MeanRisk(; opt = optB, obj = MaximumReturn(),
                         r = ConditionalValueatRisk(;
                                                    settings = RiskMeasureSettings(;
                                                                                   ub = Frontier(;
                                                                                                 N = 30)))))
all(x -> isa(x, OptimisationSuccess), resB.retcode)
````

The two directions trace the same trade-off curve. Computing the CVaR and the arithmetic return of
each point lets us overlay them: the risk-floor sweep and the return-ceiling sweep land on top of
one another (up to where the automatic bounds place their points).

````@example 02_Efficient_Frontier
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
````

## 3. The `MeanRisk` frontier vs the `NearOptimalCentering` frontier

The frontier above is built from *extreme* points — each one exactly extremises the objective, and
the high-return end loads heavily on a couple of names. [`NearOptimalCentering`](@ref) (NOC) traces
a **centred** frontier instead: at each point it returns the portfolio at the analytic centre of
the near-optimal neighbourhood rather than the corner. The result sits just inside the extreme
frontier — slightly less optimal, noticeably more diversified and more stable to changes in the
prior. (NOC's neighbourhood-centring behaviour is dissected on its own
[page](15_Near_Optimal_Centering.md); here we only use it as a frontier engine.)

NOC solves a harder problem than plain `MeanRisk`, so a single solver configuration can fail to
converge. We give it a richer fallback vector with decreasing `max_step_fraction`.

````@example 02_Efficient_Frontier
slv_noc = [Solver(; name = Symbol("clarabel$i"), solver = Clarabel.Optimizer,
                  settings = Dict("verbose" => false, "max_step_fraction" => f),
                  check_sol = (; allow_local = true, allow_almost = true))
           for (i, f) in enumerate((0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7))]
````

For an apples-to-apples comparison we build *both* frontiers over the same 15-point return-floor
sweep — only the optimiser changes.

````@example 02_Efficient_Frontier
ret15 = ArithmeticReturn(; lb = Frontier(; N = 15))
resM = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv_noc, ret = ret15),
                         r = r))
resN = optimise(NearOptimalCentering(;
                                     opt = JuMPOptimiser(; pe = pr, slv = slv_noc,
                                                         ret = ret15), r = r))
````

NOC summarises its many internal `MeanRisk` solves into a single `retcode`; let's confirm success.

````@example 02_Efficient_Frontier
isa(resN.retcode, OptimisationSuccess)
````

The diversification difference is the point. We tabulate, at each frontier point, the largest
single weight: NOC consistently holds a lower maximum weight — it fans the allocation out — except
at the high-return end, where both frontiers are forced into the same return-maximising corner.

````@example 02_Efficient_Frontier
maxw(ws) = [round(maximum(w) * 100; digits = 1) for w in ws]
pretty_table(DataFrame("point" => 1:15, "MeanRisk max w %" => maxw(resM.w),
                       "NOC max w %" => maxw(resN.w));
             title = "Largest single weight along each frontier")
````

Plotted on the risk/return plane, the NOC frontier sits *inside* (to the upper-left of) the
`MeanRisk` frontier: for a given return it accepts a little more CVaR, the price of sitting at the
centre of the near-optimal region rather than at its edge.

````@example 02_Efficient_Frontier
xs_M = [expected_risk(rcvar, w, pr.X) for w in resM.w]
ys_M = [expected_return(ArithmeticReturn(), w, pr) for w in resM.w]
xs_N = [expected_risk(rcvar, w, pr.X) for w in resN.w]
ys_N = [expected_return(ArithmeticReturn(), w, pr) for w in resN.w]

plot(xs_M, ys_M; seriestype = :scatter, marker = (:circle, 5), label = "MeanRisk (extreme)",
     xlabel = "CVaR", ylabel = "Arithmetic return", title = "Extreme vs centred frontier")
plot!(xs_N, ys_N; seriestype = :scatter, marker = (:diamond, 6), label = "NOC (centred)")
````

The composition along each frontier makes the same story visual: the `MeanRisk` frontier collapses
onto a handful of names as it climbs, while the NOC frontier keeps more assets in play for longer.

````@example 02_Efficient_Frontier
plot_stacked_area_composition(resM.w, rd.nx)
````

The same sweep under NOC — visibly more names carried up the frontier.

````@example 02_Efficient_Frontier
plot_stacked_area_composition(resN.w, rd.nx)
````

## 4. Visualising the frontier

The efficient frontier is a special case of a Pareto front, and [`plot_measures`](@ref) draws it on
any pair of risk/return axes. There are optional keyword parameters for the risk measure on the
X-axis, Y-axis, Z-axis, and colourbar. Here we put CVaR on the X-axis, the arithmetic return on the
Y-axis, and colour by the risk-return ratio.

````@example 02_Efficient_Frontier
plot_measures(resA.w, resA.pr; x = r, y = ExpectedReturn(; rt = resA.ret),
              c = ExpectedReturnRiskRatio(; rt = resA.ret, rk = r, rf = rf),
              title = "Efficient Frontier", xlabel = "CVaR", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)
````

Because `plot_measures` works on *any* pair of measures, the same call plots arbitrary Pareto
fronts — we can even use the ratio of two risk measures as the colourbar.

````@example 02_Efficient_Frontier
plot_measures(resA.w, resA.pr; x = r, y = ConditionalDrawdownatRisk(),
              c = RiskRatio(; r1 = ConditionalDrawdownatRisk(), r2 = r),
              title = "Pareto Front", xlabel = "CVaR", ylabel = "CDaR",
              colorbar_title = "\nCDaR/CVaR Ratio", right_margin = 6Plots.mm)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
