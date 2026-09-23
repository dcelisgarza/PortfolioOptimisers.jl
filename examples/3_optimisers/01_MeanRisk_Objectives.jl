#=
```@meta
Description = "The MeanRisk objectives in PortfolioOptimisers.jl: minimum risk, maximum utility, maximum ratio and maximum return on one prior and risk measure."
```

# `MeanRisk` objectives

[`MeanRisk`](@ref) trades expected return against risk, and its objective, the `obj` field,
picks the point on that trade-off that you get. With one prior and one risk measure, the four
objectives give four different portfolios.

  - [`MinimumRisk`](@ref) ignores return and takes the portfolio with the least risk.
  - [`MaximumReturn`](@ref) ignores risk and takes the portfolio with the highest return, which
    is a corner solution.
  - [`MaximumRatio`](@ref) maximises the ratio of return, net of the risk-free rate, to risk.
    This is the tangency portfolio.
  - [`MaximumUtility`](@ref) maximises `return − l · risk`. The risk aversion `l` moves the
    portfolio between the high-return end and the low-risk end.

This page runs all four next to a benchmark. It shows how the risk aversion of `MaximumUtility`
changes the portfolio, and then prints the risk, return and ratio of each portfolio.

!!! tip "When to reach for this"
    Reach for [`MeanRisk`](@ref) when you want a portfolio that trades expected return against
    risk, with one objective to pick the point. The objective can minimise risk, maximise
    return, maximise the risk-adjusted ratio, or maximise a utility that penalises risk. If you
    want to allocate the risk itself instead of trading it against return, see
    [`RiskBudgeting`](@ref). If you want the whole curve of the trade-off instead of one point,
    see the [efficient-frontier](02_Efficient_Frontier.md) example.
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

We load the last 253 days of the S&P 500 prices that the other optimiser examples use, and
compute their returns.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X)

#=
## 2. The four objectives

We hold the risk measure fixed and change only the objective. The risk measure is the
semi-standard deviation, and the library has no type of that name. We build it from
[`LowOrderMoment`](@ref), which covers a whole class of risk measures. `SecondMoment` selects the
second moment, `SemiMoment()` keeps only the returns below the mean, and
[`SOCRiskExpr`](@ref) takes the square root as a second-order cone expression.
=#

using Clarabel
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

r = LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment(), alg2 = SOCRiskExpr()))

#=
Every optimisation below runs on the same data, so we compute the prior once with
[`EmpiricalPrior`](@ref) and pass the result to [`JuMPOptimiser`](@ref). No call computes it again.
=#

pr = prior(EmpiricalPrior(), rd)
opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
!!! note "Precomputed result vs estimator"
    `pe = pr` passes the *result* of `prior(...)`. The optimiser reuses those statistics on
    every `optimise` call, so we do not pass the returns data again. You can instead pass the
    estimator, `pe = EmpiricalPrior()`, and call `optimise(model, rd)`. The optimiser then
    computes the prior from the data you give it. Most examples here precompute the prior
    because they solve many times on one fixed window. Use the estimator form when the data
    changes between calls.

    - Cross-validation fits the optimiser again on each training fold, so it needs the estimator form. A result fitted on the whole sample has already seen the test data, and the library does not accept one.
    - The meta-optimisers [`Stacking`](@ref) and [`NestedClustered`](@ref) give their *outer* optimiser synthetic returns that they build from the inner solves. The assets of the outer optimiser are the inner portfolios, so a prior computed on the original assets does not apply, and that field takes an estimator.

    The [meta-optimisers](13_Meta_Optimisers.md) and
    [subset resampling and cross-validation](14_Subset_Resampling_and_Cross_Validation.md)
    examples show each case.

We build one `MeanRisk` per objective. Only the `obj` field differs between them.
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
We optimise each one. As a reference we also compute an [`InverseVolatility`](@ref) benchmark. It
needs no solver, and it uses neither an objective nor the expected returns.
=#

res1 = optimise(mr1)
res2 = optimise(mr2)
res3 = optimise(mr3)
res4 = optimise(mr4)
res0 = optimise(InverseVolatility(; pe = pr))

#=
The table prints the weights side by side. The benchmark gives each asset a weight in
proportion to the inverse of its volatility, so every asset gets one. The minimum-risk portfolio
holds mostly assets with low volatility, and the maximum-return portfolio puts all its weight in
the asset with the highest expected return.
=#

pretty_table(DataFrame(; :assets => rd.nx, :benchmark => res0.w, :MinimumRisk => res1.w,
                       :MaximumUtility => res2.w, :MaximumRatio => res3.w,
                       :MaximumReturn => res4.w); formatters = [resfmt])

#=
## 3. Risk aversion: tuning `MaximumUtility`

[`MinimumRisk`](@ref) and [`MaximumReturn`](@ref) are the two ends of the trade-off, and
[`MaximumUtility`](@ref) moves between them. It maximises `return − l · risk`, so the risk
aversion `l` sets the point. As `l → 0` the utility favours return, and the portfolio moves toward
the maximum-return corner. As `l` grows, the risk term dominates, and the portfolio moves toward
the minimum-risk portfolio. The default is `l = 2`.

We optimise for five values of `l` and print the risk, return and ratio of each portfolio.
Section 4 explains the call that computes them.
=#

lambdas = [1, 2, 8, 32, 128]
util = [optimise(MeanRisk(; r = r, obj = MaximumUtility(; l = l), opt = opt))
        for l in lambdas]

sweep = map(zip(lambdas, util)) do (l, res)
    rk, rt, rr = expected_risk_ret_ratio(res.r, res.ret, res.w, res.pr; sca = res.sca,
                                         rf = rf)
    return (l, rk, rt, rr)
end
pretty_table(DataFrame(; Symbol("risk aversion l") => [s[1] for s in sweep],
                       :risk => [s[2] for s in sweep], :return => [s[3] for s in sweep],
                       :ratio => [s[4] for s in sweep]); formatters = [resfmt],
             title = "MaximumUtility: higher l ⇒ lower risk and lower return")

#=
Risk and return both fall as `l` rises, so the portfolio moves down the efficient frontier from
the high-return end toward the minimum-risk end. We plot the risk and return of each
portfolio to show that path.
=#

using StatsPlots, GraphRecipes

plot([s[2] for s in sweep], [s[3] for s in sweep]; seriestype = :path,
     marker = (:circle, 5), xlabel = "SemiMoment-deviation risk",
     ylabel = "Arithmetic return", title = "MaximumUtility risk-aversion path",
     label = "l = " * join(string.(lambdas), ", "))

#=
## 4. The risk, return and ratio of each portfolio

We compute the risk, the return and the risk-return ratio of every portfolio.
[`expected_risk`](@ref), [`expected_return`](@ref) and [`expected_ratio`](@ref) compute one each,
and [`expected_risk_ret_ratio`](@ref) returns all three at once, so we use it here.

A function that computes the expected return of a portfolio needs to know *which* return measure
to use. We use the one that each optimisation used.

The result of an optimisation holds what the call needs. `res.r` is the risk measure of the
optimisation and `res.ret` its return measure. The result stores both as the optimisation used
them. If you gave an estimator, the result holds the fitted measure, and if you left a field unset,
the result holds the value the optimiser took from the prior. `res.sca` is the scalariser. So the
call takes all its arguments from the result:

```julia
expected_risk_ret_ratio(res.r, res.ret, res.w, res.pr; sca = res.sca, rf = rf)
```

Prefer this form. You can name the measure by hand, but the number then matches the optimisation
only if you name the *same* measure and the same scalariser, and pass the same `fees` and `rf`.
The benchmark is the exception. [`InverseVolatility`](@ref) uses no risk measure and no return
measure, so its result has no `r` or `ret`, and we name both ourselves.
=#

rk1, rt1, rr1 = expected_risk_ret_ratio(res1.r, res1.ret, res1.w, res1.pr; sca = res1.sca,
                                        rf = rf);
rk2, rt2, rr2 = expected_risk_ret_ratio(res2.r, res2.ret, res2.w, res2.pr; sca = res2.sca,
                                        rf = rf);
rk3, rt3, rr3 = expected_risk_ret_ratio(res3.r, res3.ret, res3.w, res3.pr; sca = res3.sca,
                                        rf = rf);
rk4, rt4, rr4 = expected_risk_ret_ratio(res4.r, res4.ret, res4.w, res4.pr; sca = res4.sca,
                                        rf = rf);
rk0, rt0, rr0 = expected_risk_ret_ratio(r, ArithmeticReturn(), res0.w, res0.pr; rf = rf);

#=
In the table, `rk` is the risk, `rt` the return and `rr` the risk-return ratio. Read each
objective off the column it optimises. `MinimumRisk` has the lowest `rk`, `MaximumRatio` the
highest `rr`, and `MaximumReturn` the highest `rt`.
=#

pretty_table(DataFrame(;
                       :obj =>
                           [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn,
                            :Benchmark], :rk => [rk1, rk2, rk3, rk4, rk0],
                       :rt => [rt1, rt2, rt3, rt4, rt0], :rr => [rr1, rr2, rr3, rr4, rr0]);
             formatters = [resfmt])

#=
## 5. Visualising the objectives

The stacked bars show the weights of the five portfolios. The objectives that maximise return or
ratio put their weight in few assets, and the benchmark and the minimum-risk portfolio spread it
over many.
=#

plot_stacked_bar_composition([res0, res1, res2, res3, res4], rd)

#=
The histogram shows the daily returns of the minimum-risk portfolio, with lines at its mean, its
VaR, its CVaR and several other risk levels.
=#

plot_histogram(res1, rd)

#=
The drawdown plot shows how far the minimum-risk portfolio stands below its last peak on each
day.
=#

plot_drawdowns(res1, rd)

#=
The last plot shows how much each asset adds to the semi-standard deviation of the minimum-risk
portfolio.
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
