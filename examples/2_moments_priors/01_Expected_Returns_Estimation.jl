#=
```@meta
Description = "Expected returns estimation in PortfolioOptimisers.jl: shrinkage estimators that pull the noisy sample mean toward a structured target."
```

# Expected returns estimation

The sample mean is the noisiest input of a portfolio optimisation. One year of daily data gives
a poor estimate of the average return of each asset. An optimiser that seeks return, such as
maximum return, the maximum risk-adjusted ratio or maximum utility, turns that noise into
extreme weights. A shrinkage estimator pulls the sample mean toward a structured target. It
accepts a little bias for a large fall in variance.

A prior takes its expected returns estimator in the `me` field. The default is
[`SimpleExpectedReturns`](@ref), the plain sample mean. [`ShrunkExpectedReturns`](@ref) wraps
it with one of three algorithms, [`BayesStein`](@ref), [`BodnarOkhrinParolya`](@ref) or
[`JamesStein`](@ref). Each algorithm shrinks toward one of three targets,
[`GrandMean`](@ref), [`VolatilityWeighted`](@ref) or [`MeanSquaredError`](@ref).

!!! tip "When to reach for this"
    Reach for a shrunk expected returns estimator when your objective reads the mean, as
    [`MaximumReturn`](@ref), [`MaximumRatio`](@ref) and [`MaximumUtility`](@ref) do, and
    above all when the window is short next to the number of assets. [`MinimumRisk`](@ref)
    and risk budgeting ignore the mean, so if you only run those, the default estimator is
    enough.
=#

using PortfolioOptimisers, PrettyTables

mmtfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=4)) %" : v
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

We use the same S&P 500 slice as the other examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. Expected-returns estimators

We build one prior per estimator and change only the `me` field, so every prior carries the
same covariance. Any difference in the optimisation later comes from the expected returns
alone. We compare the sample mean with Bayes-Stein, Bodnar-Okhrin-Parolya and James-Stein
shrinkage toward different targets.
=#

mes = ["Vanilla" => SimpleExpectedReturns(),
       "BS(GM)" => ShrunkExpectedReturns(; alg = BayesStein(; tgt = GrandMean())),
       "BS(VW)" => ShrunkExpectedReturns(; alg = BayesStein(; tgt = VolatilityWeighted())),
       "BOP(MSE)" =>
           ShrunkExpectedReturns(; alg = BodnarOkhrinParolya(; tgt = MeanSquaredError())),
       "JS(GM)" => ShrunkExpectedReturns(; alg = JamesStein(; tgt = GrandMean()))]

prs = [k => prior(EmpiricalPrior(; me = me), rd) for (k, me) in mes]

#=
Shrinkage toward the grand mean, in `BS(GM)` and `JS(GM)`, keeps the average over the assets
and narrows the spread around it. The volatility-weighted target and the mean squared error
target move the average as well.
=#

pretty_table(DataFrame(["Assets" => rd.nx; [k => p.mu for (k, p) in prs]]);
             formatters = [mmtfmt], title = "Expected returns by estimator")

#=
## 3. Visualising the shrinkage

[`plot_mu`](@ref) draws the expected return of each asset as a bar. The bars of the shrunk
estimator spread less than the bars of the sample mean.
=#

using StatsPlots, GraphRecipes
# The sample mean.
plot_mu(prs[1].second, rd.nx)
# Bayes-Stein shrinkage toward the volatility-weighted target.
plot_mu(prs[3].second, rd.nx)

#=
## 4. Expected returns in a maximum-ratio portfolio

The expected returns change a portfolio only when the objective reads them. We maximise the
risk-adjusted ratio with each prior in turn, and compare the weights. The covariance is the
same in every prior, so only the mean differs. The sample mean concentrates the weights in a
few assets, and shrinkage spreads them over more.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

ress = [k => optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                               opt = JuMPOptimiser(; pe = p, slv = slv))) for (k, p) in prs]

pretty_table(DataFrame(["Assets" => rd.nx; [k => r.w for (k, r) in ress]]);
             formatters = [resfmt], title = "Maximum-ratio weights by mu estimator")

#=
Each bar of the composition plot is the maximum-ratio portfolio of one estimator, and the
expected returns estimator is the only input that changes between the bars.
=#

plot_stacked_bar_composition([r for (_, r) in ress], rd;
                             xticks = (1:length(ress), [k for (k, _) in ress]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end (split from ex08, focused on the `me` field only; covariance held
#src   fixed so the optimisation differences isolate expected returns). GrandMean shrinkage
#src   preserves the cross-sectional average while VW/MSE targets shift it, as described.
#src - API note (record-only → #126): `plot_mu` takes a *single* prior + asset-name vector,
#src   not a vector of priors — so estimator comparisons need one call per prior (or the
#src   weights composition). A vector-of-priors method (like the weight plots have) would make
#src   side-by-side mu comparisons one call. `plot_stacked_bar_composition` likewise has no
#src   `names` kwarg; labelling bars requires passing `xticks` through to groupedbar.
