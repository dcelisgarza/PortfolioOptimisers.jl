#=
```@meta
Description = "BrownianDistanceVariance and VarianceSkewKurtosis in PortfolioOptimisers.jl: a dispersion measure built from distances, and a measure of the higher moments."
```

# `BrownianDistanceVariance` and `VarianceSkewKurtosis`

Some risk measures respond to parts of the return distribution that the variance and the
conditional value at risk do not measure. This page shows two of them.

  - [`BrownianDistanceVariance`](@ref) measures the dispersion of the portfolio returns with
    their distance variance, the Brownian distance covariance of the return series with
    itself. It is built from the distance between the returns of every pair of days, not
    from their deviations about the mean, so it captures non-linear structure that the
    variance misses.
  - [`VarianceSkewKurtosis`](@ref) combines the variance, the skewness and the kurtosis in one
    objective. Its model is a semidefinite relaxation with one large positive semidefinite
    matrix, so a first-order solver such as SCS suits it.

!!! tip "When to reach for this"
    Reach for `BrownianDistanceVariance` when you want a measure of dispersion that captures
    non-linear structure in the return series. Reach for `VarianceSkewKurtosis` when the
    third and fourth moments of the portfolio return matter, for example when the assets
    have fat tails or skewed payoffs that mean-variance optimisation ignores.

!!! warning "Solvers and the size of the sample"
    - `BrownianDistanceVariance` builds a `T × T` distance matrix inside the model, so the
    number of extra variables grows with the square of the number of observations. This
    example uses 50 observations to keep the model small.
    - `VarianceSkewKurtosis` builds a large semidefinite problem. This example solves it
    with SCS, and you can use another solver that handles large semidefinite problems.
    It also needs a [`HighOrderPriorEstimator`](@ref), which computes the coskewness and the
    cokurtosis that the risk measure uses. This example uses the same 50 observations.
=#

using PortfolioOptimisers, PrettyTables, DataFrames

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

using CSV, TimeSeries, Clarabel, SCS

#=
## 1. The data: 50 observations

Both measures use the last 50 daily returns. The number of returns `T` sets the size of the
model of `BrownianDistanceVariance`.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 50):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
              check_sol = (; allow_local = true, allow_almost = true))]

opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
## 2. Brownian distance variance

### 2.1 `alg2`: the absolute distances

By default [`BrownianDistanceVariance`](@ref) writes the absolute distances with a norm-one
cone, `NormOneConeBrownianDistanceVariance`. The model adds one variable for each entry of the
distance matrix and bounds it with that cone.
=#

res_bdvar = optimise(MeanRisk(; r = BrownianDistanceVariance(), opt = opt))

#=
The other value of `alg2`, `IneqBrownianDistanceVariance`, writes the absolute distances with
linear inequalities instead of a cone. On a large problem it can be faster, because it avoids a
large cone.
=#

res_bdvar_ineq = optimise(MeanRisk(;
                                   r = BrownianDistanceVariance(;
                                                                alg2 = IneqBrownianDistanceVariance()),
                                   opt = opt))

#=
### 2.2 `alg1`: the sum of squares

A second choice, `alg1`, sets how the model writes the sum of the squared distances. The
default, `QuadRiskExpr`, writes it as a quadratic expression. `RSOCRiskExpr` writes it with a
rotated second-order cone, which can be faster on a dense problem. We print the weights of the
three formulations side by side.
=#

res_bdvar_rsoc = optimise(MeanRisk(; r = BrownianDistanceVariance(; alg1 = RSOCRiskExpr()),
                                   opt = opt))

pretty_table(DataFrame(; :assets => rd.nx, :BDVar_default => res_bdvar.w,
                       :BDVar_ineq => res_bdvar_ineq.w, :BDVar_rsoc => res_bdvar_rsoc.w);
             formatters = [resfmt])

#=
The three columns show the same portfolio. The formulations differ in how they build the model,
not in what the model optimises.

We also minimise the variance, and plot the two portfolios. The Brownian distance variance
depends on the distances between the returns of pairs of days, and the variance on the squared
deviations about the mean, so the two portfolios can differ.
=#

res_var = optimise(MeanRisk(; r = Variance(), opt = opt))

using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_var, res_bdvar], rd)

#=
## 3. VarianceSkewKurtosis

### 3.1 High-order prior and SCS solver

We solve `VarianceSkewKurtosis` with SCS, and compute its prior with
`HighOrderPriorEstimator` on the same 50 observations as section 1.
=#

pr_ho = prior(HighOrderPriorEstimator(), rd)

scs_slv = Solver(; name = :scs, solver = SCS.Optimizer, settings = "verbose" => false,
                 check_sol = (; allow_local = true, allow_almost = true))
opt_ho = JuMPOptimiser(; pe = pr_ho, slv = scs_slv)

#=
### 3.2 Default composite

[`VarianceSkewKurtosis`](@ref) combines three measures.

  - [`Variance`](@ref) penalises dispersion.
  - [`Skewness`](@ref) enters with a negative sign, so a larger skewness lowers the risk, and
    returns skewed to the left raise it.
  - [`Kurtosis`](@ref) penalises fat tails.

Each measure has a scale of 1 by default. A larger scale gives that moment more weight in the
objective. We optimise with the default scales and print the return code.
=#

res_vsk = optimise(MeanRisk(; r = VarianceSkewKurtosis(), opt = opt_ho), rd)
println("VarianceSkewKurtosis retcode: $(res_vsk.retcode)")

#=
### 3.3 Custom component scales

To set how much each higher moment counts next to the variance, pass your own `Skewness` and
`Kurtosis` with a scale in their settings. The skewness takes `MaxRiskMeasureSettings`, and the
kurtosis takes [`RiskMeasureSettings`](@ref). We give both a scale of 2.
=#

r_vsk_heavy = VarianceSkewKurtosis(;
                                   sk = Skewness(;
                                                 settings = MaxRiskMeasureSettings(;
                                                                                   scale = 2.0)),
                                   kt = Kurtosis(;
                                                 settings = RiskMeasureSettings(;
                                                                                scale = 2.0)))
res_vsk_heavy = optimise(MeanRisk(; r = r_vsk_heavy, opt = opt_ho), rd)

#=
We print the two portfolios side by side. The larger scales double the skewness and kurtosis
terms of the objective. On these 50 daily returns both terms are at least 500 times smaller
than the variance term, so the two columns differ by less than 0.1 of a percentage point.
=#

pretty_table(DataFrame(; :assets => rd.nx, :VarianceSkewKurtosis => res_vsk.w,
                       :VSK_heavy_tail => res_vsk_heavy.w); formatters = [resfmt])

#=
### 3.4 Comparison with plain variance (SCS)

We also minimise the variance with SCS on the same prior, so that only the risk measure differs,
and plot the three portfolios.
=#

res_var_scs = optimise(MeanRisk(; r = Variance(), opt = opt_ho))

plot_stacked_bar_composition([res_var_scs, res_vsk, res_vsk_heavy], rd)

#=
## Summary

  - We minimised [`BrownianDistanceVariance`](@ref) with Clarabel on 50 returns. Keep
    `T` small, because the size of the model grows with the square of `T`.
  - We minimised [`VarianceSkewKurtosis`](@ref) with SCS, because the model is a
    large semidefinite problem. Pair it with [`HighOrderPriorEstimator`](@ref), whose
    coskewness and cokurtosis grow with the number of assets.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env) on the 50-obs slice. All three
#src   `BrownianDistanceVariance` formulations (NormOneCone default, `IneqBrownianDistanceVariance`,
#src   `RSOCRiskExpr`) return the same portfolio, confirming the section-2 prose. Both
#src   `VarianceSkewKurtosis` solves (default scales and heavy skew/kurtosis) return
#src   `OptimisationSuccess` with SCS.
#src - The 50-obs slice and the SCS requirement are load-bearing, not stylistic: VSK needs large PSD
#src   cones (Clarabel cannot solve them directly), and BDVar's O(T²) distance matrix makes T the binding size. Both
#src   are flagged in the opening `!!! warning` so a reader who swaps in Clarabel or a 252-obs
#src   slice knows why it breaks. No solver warnings or plotting deprecations observed.
#src - COSMETIC: the `RSOCRiskExpr` column prints `-0.0 %` for zero weights (signed-zero from the
#src   rotated-cone formulation). Harmless in the table; not worth a fix.
