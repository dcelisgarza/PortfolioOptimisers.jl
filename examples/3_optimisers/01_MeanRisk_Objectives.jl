#=
# `MeanRisk` objectives

[`MeanRisk`](@ref) is the workhorse optimiser: it casts portfolio selection as an explicit
trade-off between expected return and risk, and an **objective function** decides *which* point on
that trade-off you get. The same estimator, prior and risk measure can produce four very different
portfolios depending on the objective:

  - [`MinimumRisk`](@ref) — ignore return, take the least-risk portfolio.
  - [`MaximumReturn`](@ref) — ignore risk, take the highest-return portfolio (a corner solution).
  - [`MaximumRatio`](@ref) — maximise the risk-adjusted ratio (return over risk, net of the
    risk-free rate) — the tangency portfolio.
  - [`MaximumUtility`](@ref) — maximise `return − l · risk`, where the risk-aversion `l` dials
    continuously between the return-seeking and risk-averse ends.

This page runs all four against a common benchmark, confirms each does what it claims, and shows how
`MaximumUtility`'s risk-aversion parameter sweeps between the extremes.

!!! tip "When to reach for this"
    [`MeanRisk`](@ref) is the workhorse optimiser: reach for it whenever you want to express
    a portfolio as an explicit *trade-off between expected return and risk* and let a single
    objective pick the point — minimise risk, maximise return, maximise the risk-adjusted
    ratio, or maximise a risk-averse utility. If you instead want to *allocate risk itself*
    rather than trade it against return, see [`RiskBudgeting`](@ref); if you want the whole
    trade-off curve rather than one point, see the [efficient-frontier](02_Efficient_Frontier.md)
    example.
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
## 1. ReturnsResult data

We use the same S&P 500 slice as the other optimiser examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X)

#=
## 2. The four objectives

We will hold the risk measure fixed and vary only the objective. For the risk measure we reach for
the **semi–standard deviation** — and here we meet a consequence of the package's design
philosophy: an entire class of risk measures is expressed as a single
[`LowOrderMoment`](@ref) parametrised by an internal algorithm. Semi–standard deviation is the
second lower partial moment (`Semi()`) rendered as a second-order cone expression
([`SOCRiskExpr`](@ref)).
=#

using Clarabel
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

r = LowOrderMoment(; alg = SecondMoment(; alg1 = Semi(), alg2 = SOCRiskExpr()))

#=
Since every optimisation runs on the same data, we precompute the prior statistics once with
[`EmpiricalPrior`](@ref) and pass the result to [`JuMPOptimiser`](@ref), so they are not recomputed
on every call.
=#

pr = prior(EmpiricalPrior(), rd)
opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
Now the four objectives. Only the `obj` field changes — same prior, same risk measure, same
optimiser.
=#

## Minimum risk
mr1 = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)
## Maximum utility (default risk aversion l = 2)
mr2 = MeanRisk(; r = r, obj = MaximumUtility(), opt = opt)
## Maximum risk-adjusted ratio, risk-free rate of 4.2/100/252
rf = 4.2 / 100 / 252
mr3 = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
## Maximum return
mr4 = MeanRisk(; r = r, obj = MaximumReturn(), opt = opt)

#=
We optimise each. Because the prior is precomputed, we do not pass the returns data. For a
reference point we also compute an [`InverseVolatility`](@ref) benchmark — a naive, solver-free
allocation that ignores both objective and expected returns.
=#

res1 = optimise(mr1)
res2 = optimise(mr2)
res3 = optimise(mr3)
res4 = optimise(mr4)
res0 = optimise(InverseVolatility(; pe = pr))

#=
The weights side by side. Reading left to right, the benchmark spreads evenly, minimum-risk hugs
the low-volatility names, and maximum-return collapses onto the single highest-return asset.
=#

pretty_table(DataFrame(; :assets => rd.nx, :benchmark => res0.w, :MinimumRisk => res1.w,
                       :MaximumUtility => res2.w, :MaximumRatio => res3.w,
                       :MaximumReturn => res4.w); formatters = [resfmt])

#=
## 3. Risk aversion: tuning `MaximumUtility`

[`MinimumRisk`](@ref) and [`MaximumReturn`](@ref) are the two extremes of the trade-off.
[`MaximumUtility`](@ref) interpolates between them: it maximises `return − l · risk`, so the
risk-aversion `l` is the dial. As `l → 0` utility chases return (toward the maximum-return corner);
as `l` grows large the risk term dominates (toward the minimum-risk portfolio). The default is
`l = 2`.

We sweep a range of `l` and read off the realised risk and return of each portfolio.
=#

lambdas = [1, 2, 8, 32, 128]
util = [optimise(MeanRisk(; r = r, obj = MaximumUtility(; l = l), opt = opt))
        for l in lambdas]

sweep = map(zip(lambdas, util)) do (l, res)
    rk, rt, rr = expected_risk_ret_ratio(r, res.ret, res.w, res.pr; rf = rf)
    return (l, rk, rt, rr)
end
pretty_table(DataFrame(; Symbol("risk aversion l") => [s[1] for s in sweep],
                       :risk => [s[2] for s in sweep], :return => [s[3] for s in sweep],
                       :ratio => [s[4] for s in sweep]); formatters = [resfmt],
             title = "MaximumUtility: higher l ⇒ lower risk and lower return")

#=
Both risk and return fall monotonically as `l` rises — the portfolio slides down the frontier from
the return-seeking end toward the minimum-risk end. Plotting the realised (risk, return) of each
step traces that path explicitly.
=#

using StatsPlots, GraphRecipes

plot([s[2] for s in sweep], [s[3] for s in sweep]; seriestype = :path,
     marker = (:circle, 5), xlabel = "Semi-deviation risk", ylabel = "Arithmetic return",
     title = "MaximumUtility risk-aversion path",
     label = "l = " * join(string.(lambdas), ", "))

#=
## 4. Confirming the objectives

To check each objective did what it says on the tin, we compute the risk, return and risk-return
ratio of every portfolio. There are individual functions ([`expected_risk`](@ref),
[`expected_return`](@ref), [`expected_ratio`](@ref)), but [`expected_risk_ret_ratio`](@ref) returns
all three at once, which is what we use here.

Any function that computes the expected portfolio return needs to know *which* return type to use;
we stay consistent with the return measure used in each optimisation.
=#

rk1, rt1, rr1 = expected_risk_ret_ratio(r, res1.ret, res1.w, res1.pr; rf = rf);
rk2, rt2, rr2 = expected_risk_ret_ratio(r, res2.ret, res2.w, res2.pr; rf = rf);
rk3, rt3, rr3 = expected_risk_ret_ratio(r, res3.ret, res3.w, res3.pr; rf = rf);
rk4, rt4, rr4 = expected_risk_ret_ratio(r, res4.ret, res4.w, res4.pr; rf = rf);
rk0, rt0, rr0 = expected_risk_ret_ratio(r, ArithmeticReturn(), res0.w, res0.pr; rf = rf);

#=
The table confirms it: `MinimumRisk` posts the lowest risk, `MaximumRatio` the highest ratio, and
`MaximumReturn` the highest return — each column extremised on its own objective.
=#

pretty_table(DataFrame(;
                       :obj =>
                           [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn,
                            :Benchmark], :rk => [rk1, rk2, rk3, rk4, rk0],
                       :rt => [rt1, rt2, rt3, rt4, rt0], :rr => [rr1, rr2, rr3, rr4, rr0]);
             formatters = [resfmt])

#=
## 5. Visualising the objectives

The stacked-bar composition contrasts the five allocations at a glance — note how the
return-driven objectives concentrate while the benchmark and minimum-risk books spread out.
=#

plot_stacked_bar_composition([res0, res1, res2, res3, res4], rd)

#=
The return histogram for the minimum-risk portfolio shows the distribution of daily returns and the
VaR / CVaR tail-risk markers.
=#

plot_histogram(res1, rd)

#=
Drawdown time series for the minimum-risk portfolio.
=#

plot_drawdowns(res1, rd)

#=
Per-asset semi-standard-deviation risk contribution for the minimum-risk portfolio.
=#

plot_risk_contribution(r, res1, rd)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Deepened (ADR 0014, examples-are-deep-dives): fixed broken section numbering (was 1,2,4),
#src   enriched the per-objective prose, and added §3 — a MaximumUtility risk-aversion (l) sweep.
#src   Verified on kaimon (session 56c8906d): l = [1,2,8,32,128] gives monotone (risk, return) of
#src   (0.673%,0.119%) → (0.651%,0.075%) — l interpolates from the return-seeking corner toward the
#src   minimum-risk portfolio (risk at l=128 ≈ the MinimumRisk risk). MaximumUtility field is `l`
#src   (default 2). The risk-aversion path plot constructs.
#src - Original sweep still clean: all four objectives + InverseVolatility benchmark solve, and the
#src   expected_risk_ret_ratio table confirms each objective extremises its own criterion. No doc,
#src   ergonomics, plotting, or bug findings. Group rollup: issue #125.
