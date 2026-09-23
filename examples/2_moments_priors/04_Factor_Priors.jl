#=
```@meta
Description = "Factor priors in PortfolioOptimisers.jl: asset moments implied by a regression onto common risk factors, to cut estimation error."
```

# Factor priors

A factor model writes the return of each asset as a function of a few common risk factors. It
has fewer parameters to estimate than the moments of the assets have, so its estimates carry
less error. This page builds priors from factor models and compares the optimisations they
give.

!!! tip "When to reach for this"
    Reach for a factor prior when you have a set of common risk factors, of style, macro or
    statistical kind, and want the asset moments that a regression onto them implies, rather
    than moments estimated freely. The model writes the ``N`` assets through a few factors,
    so it cuts the number of parameters and the estimation error. This helps most when the
    number of assets is large next to the length of the history. If you have no meaningful
    factors, or a long history next to the number of assets, a denoised empirical prior is
    simpler, and the [covariance](02_Covariance_Estimation.md) and
    [higher moment](03_Higher_Moment_Estimation.md) pages show it.
=#
using PortfolioOptimisers, PrettyTables
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
mmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
hmmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100*1e4, digits=2))e-4 %" : v
    end
end;

#=
## 1. The data

We load the prices of 20 assets and of five factor funds over the same year. The two tables
print their last rows, and `prices_to_returns` converts both to returns.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(F[(end - 5):end]; formatters = [tsfmt])

rd = prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))

#=
## 2. Prior statistics

The library has many prior models, and this page uses four of them.

1. [`EmpiricalPrior`](@ref) computes the expected returns vector and the covariance matrix from the data.
2. [`FactorPrior`](@ref) computes them with a factor model.
3. [`HighOrderPriorEstimator`](@ref) computes them with the low-order prior estimator you give it. It adds the coskewness, the cokurtosis or both, which it computes from the same returns, centred on their sample mean.
4. [`HighOrderFactorPriorEstimator`](@ref) computes the expected returns vector, the covariance matrix, and the coskewness, the cokurtosis or both, all with a factor model.

An estimator whose name starts with `High` returns a [`HighOrderPrior`](@ref), and the others return a [`LowOrderPrior`](@ref).

A factor model takes one of two regressions, [`StepwiseRegression`](@ref) or [`DimensionReductionRegression`](@ref). Each has several targets, and their docstrings describe them. We build eight prior estimators. The ones with a factor model use the stepwise regression by default, and the dimension reduction regression where the cell names it.
=#

pes = [EmpiricalPrior(), FactorPrior(), FactorPrior(; re = DimensionReductionRegression()),
       HighOrderPriorEstimator(), HighOrderPriorEstimator(; pe = FactorPrior()),
       HighOrderPriorEstimator(; pe = FactorPrior(; re = DimensionReductionRegression())),
       HighOrderFactorPriorEstimator(;),
       HighOrderFactorPriorEstimator(;
                                     pe = FactorPrior(;
                                                      re = DimensionReductionRegression()))]

#=
We compute the prior of each estimator.
=#
prs = prior.(pes, rd)

#=
The plots show what the empirical prior and the factor prior with stepwise regression compute.
The empirical prior is the baseline.
=#

using StatsPlots, GraphRecipes
# The expected returns, the volatilities and the correlations of the empirical prior.
plot_prior(prs[1], rd)
# The same three panels for the factor prior with stepwise regression.
plot_prior(prs[2], rd)
# The factor loadings of that prior, which show the factors that drive each asset.
plot_factor_loadings(prs[2], rd)
# The expected returns of the factors.
plot_factor_mu(prs[2], rd)
# The covariance matrix of the factors.
plot_factor_sigma(prs[2], rd)

#=
We compare the first three priors.

The note under the table prints whether their expected returns, in the `mu` field, are equal within the default tolerance of `≈`.
=#

pretty_table(DataFrame("Assets" => rd.nx, "EmpiricalPrior" => prs[1].mu,
                       "FactorPrior(Step)" => prs[2].mu,
                       "FactorPrior(DimRed)" => prs[3].mu); formatters = [mmtfmt],
             title = "Expected returns",
             source_notes = "prs[1].mu ≈ prs[2].mu ≈ prs[3].mu: $(prs[1].mu ≈ prs[2].mu ≈ prs[3].mu)")

#=
The note prints `true`, so on this data the three priors give the same expected returns. The [expected returns page](01_Expected_Returns_Estimation.md) shows estimators that reduce the noise of expected returns. The covariances, in the `sigma` field, differ. In a factor model the factors carry the common part of the returns, and the noise of each single asset has less effect on the covariance. Each covariance table carries its condition number under it. A lower condition number means that inverting the matrix amplifies its errors less.

A factor prior also stores a sparser Cholesky factor, with better numerical properties than the plain one, in the `chol` field of the prior result. When it is present, the optimisers use it in place of the Cholesky factor of `sigma` in the `SecondOrderCone` constraint of the variance and standard deviation formulations.
=#
using LinearAlgebra

pretty_table(DataFrame([rd.nx prs[1].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "EmpiricalPrior covariance",
             source_notes = "Condition number EmpiricalPrior: $(round(cond(prs[1].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[2].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "FactorPrior(Step) covariance",
             source_notes = "Condition number FactorPrior(Step): $(round(cond(prs[2].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[3].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "FactorPrior(DimRed) covariance",
             source_notes = "Condition number FactorPrior(DimRed): $(round(cond(prs[3].sigma); digits = 3))")

#=
Each plot draws the eigenvalues of one covariance matrix with the Marchenko-Pastur upper bound,
the largest eigenvalue that noise alone would give. The first plot is the empirical prior. Compare
the eigenvalues below the bound in the three plots.
=#

plot_eigenspectrum(prs[1], rd)
# The factor prior with stepwise regression.
plot_eigenspectrum(prs[2], rd)
# The factor prior with dimension reduction regression.
plot_eigenspectrum(prs[3], rd)

#=
Priors 4 to 6 wrap the estimators of priors 1 to 3, so each computes its low-order moments as the prior three places before it does. The loop prints whether the adjusted returns `X`, the `mu` and the `sigma` of prior `i` equal those of prior `i - 3`.
=#
for i in 4:6
    println("prs[$(i-3)].X     == prs[$(i)].X    : $(prs[i-3].X == prs[i].X)")
    println("prs[$(i-3)].mu    == prs[$(i)].mu   : $(prs[i-3].mu == prs[i].mu)")
    println("prs[$(i-3)].sigma == prs[$(i)].sigma: $(prs[i-3].sigma == prs[i].sigma)\n")
end

# The `sk` field holds the coskewness matrix, `V` the negative spectral decomposition of its slices, and `kt` the cokurtosis matrix. We print whether they are equal for priors 4 to 6. The estimator computes them from the returns in `rd`, centred on their sample mean. It uses neither the returns that the factor model rebuilds nor the mean of the low-order prior.
println("prs[4].sk == prs[5].sk == prs[6].sk: $(prs[4].sk == prs[5].sk == prs[6].sk)")
println("prs[4].V  == prs[5].V  == prs[6].V : $(prs[4].V == prs[5].V == prs[6].V)")
println("prs[4].kt == prs[5].kt == prs[6].kt: $(prs[4].kt == prs[5].kt == prs[6].kt)\n")

#=
Next we compare priors 5 to 8. Priors 7 and 8 also use a factor model for the high-order moments.
=#
for i in 5:7
    for j in 6:8
        if i >= j
            continue
        end
        println("prs[$i].mu    == prs[$j].mu   : $(prs[i].mu == prs[j].mu)")
        println("prs[$i].sigma == prs[$j].sigma: $(prs[i].sigma == prs[j].sigma)")
        println("prs[$i].sk    == prs[$j].sk   : $(prs[i].sk == prs[j].sk)")
        println("prs[$i].V     == prs[$j].V    : $(prs[i].V == prs[j].V)")
        println("prs[$i].kt    == prs[$j].kt   : $(prs[i].kt == prs[j].kt)\n")
    end
end

#=
The high-order moments match only for `prs[5]` and `prs[6]`, because neither adjusts them with a factor model. Their low-order moments differ, because they use different regression models. `prs[5]` and `prs[7]` compute their low-order moments with [`StepwiseRegression`](@ref), and `prs[6]` and `prs[8]` with [`DimensionReductionRegression`](@ref), so those two pairs match too. `prs[7]` and `prs[8]` also compute their high-order moments with the regression.

Now we look at the high-order moments themselves. First we make labels for the pairs of assets that index their columns.
=#

nx2 = collect(Iterators.flatten([(nx * "_") .* rd.nx for nx in rd.nx]))

#=
The tables print the coskewness and its negative spectral slices for priors 4, 7 and 8, with the condition number of each under it.
=#
pretty_table(DataFrame([rd.nx prs[4].sk], ["Assets^2 / Assets"; nx2]);
             formatters = [hmmtfmt], title = "HighOrderPriorEstimator coskewness",
             source_notes = "Condition number HighOrderPriorEstimator: $(round(cond(prs[4].sk); digits = 3))")
pretty_table(DataFrame([rd.nx prs[7].sk], ["Assets^2 / Assets"; nx2]);
             formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(Step) coskewness",
             source_notes = "Condition number HighOrderFactorPriorEstimator(Step): $(round(cond(prs[7].sk); digits = 3))")
pretty_table(DataFrame([rd.nx prs[8].sk], ["Assets^2 / Assets"; nx2]);
             formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(DimRed) coskewness",
             source_notes = "Condition number HighOrderFactorPriorEstimator(DimRed): $(round(cond(prs[8].sk); digits = 3))")

pretty_table(DataFrame([rd.nx prs[4].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "HighOrderPriorEstimator negative spectral slices of the coskewness",
             source_notes = "Condition number HighOrderPriorEstimator: $(round(cond(prs[4].V); digits = 3))")
pretty_table(DataFrame([rd.nx prs[7].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(Step) negative spectral slices of the coskewness",
             source_notes = "Condition number HighOrderFactorPriorEstimator(Step): $(round(cond(prs[7].V); digits = 3))")
pretty_table(DataFrame([rd.nx prs[8].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(DimRed) negative spectral slices of the coskewness",
             source_notes = "Condition number HighOrderFactorPriorEstimator(DimRed): $(round(cond(prs[8].V); digits = 3))")

#=
The heatmaps draw the ``N \times N^2`` coskewness matrix, which holds the dependence of third
order between the assets. The first is the empirical high-order prior. A factor model can
change the entries off the diagonal.
=#

plot_coskewness(prs[4], rd)
# The high-order prior with a stepwise factor model.
plot_coskewness(prs[7], rd)

#=
We print the cokurtosis matrix of the same three priors. The [higher moment page](03_Higher_Moment_Estimation.md) shows why the raw cokurtosis is singular. The default processing replaces a matrix that is not positive definite with the nearest correlation matrix, rescaled to the same diagonal, and the condition number under each table shows how close to singular the result is.
=#
pretty_table(DataFrame([nx2 prs[4].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "HighOrderPriorEstimator cokurtosis",
             source_notes = "Condition number HighOrderPriorEstimator: $(round(cond(prs[4].kt); digits = 3))")
pretty_table(DataFrame([nx2 prs[7].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(Step) cokurtosis",
             source_notes = "Condition number HighOrderFactorPriorEstimator(Step): $(round(cond(prs[7].kt); digits = 3))")
pretty_table(DataFrame([nx2 prs[8].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(DimRed) cokurtosis",
             source_notes = "Condition number HighOrderFactorPriorEstimator(DimRed): $(round(cond(prs[8].kt); digits = 3))")

#=
[`plot_cokurtosis`](@ref) draws the eigenvalues of the ``N^2 \times N^2`` cokurtosis matrix
with the Marchenko-Pastur bound. The first plot is the empirical high-order prior.
=#

plot_cokurtosis(prs[4], rd)
# The high-order prior with a stepwise factor model.
plot_cokurtosis(prs[7], rd)

#=
## 3. Comparing optimisations

This section compares how the priors change the efficient frontier. The solver is Clarabel. The cell gives it seven configurations, each with a shorter step than the one before, and the optimiser tries them in order until one solves the problem.
=#
using Clarabel
slv = [Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.7))]
#=
### 3.1 Mean-standard deviation optimisation

We compute a mean-standard deviation efficient frontier of 50 portfolios under the empirical prior and under each of the two factor priors.
=#
opts = [JuMPOptimiser(; pe = prs[1], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = prs[2], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = prs[3], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))))]

mrs = [MeanRisk(; r = StandardDeviation(), obj = MinimumRisk(), opt = opt) for opt in opts]

ress = optimise.(mrs)

#=
Each prior gets a composition plot and a frontier plot.
=#
using StatsPlots, GraphRecipes
# Empirical prior composition.
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "EmpiricalPrior", legend = :outerright))
# Empirical prior frontier.
r = StandardDeviation()
plot_measures(ress[1].w, prs[1]; x = r, y = ExpectedReturn(; rt = ress[1].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "EmpiricalPrior", xlabel = "SD", ylabel = "Arithmetic Return",
              colorbar_title = "\nReturn/Risk Ratio", right_margin = 6Plots.mm)

# Factor prior composition with stepwise regression.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(Step)", legend = :outerright))
# Factor prior frontier with stepwise regression.
r = StandardDeviation()
plot_measures(ress[2].w, prs[2]; x = r, y = ExpectedReturn(; rt = ress[2].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[2].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "FactorPrior(Step)", xlabel = "SD", ylabel = "Arithmetic Return",
              colorbar_title = "\nReturn/Risk Ratio", right_margin = 6Plots.mm)
# Factor prior composition with dimension reduction regression.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(DimRed)",
                                        legend = :outerright))
# Factor prior frontier with dimension reduction regression.
r = StandardDeviation()
plot_measures(ress[3].w, prs[3]; x = r, y = ExpectedReturn(; rt = ress[3].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[3].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "FactorPrior(DimRed)", xlabel = "SD", ylabel = "Arithmetic Return",
              colorbar_title = "\nReturn/Risk Ratio", right_margin = 6Plots.mm)

#=
A frontier holds many portfolios. To compare one portfolio per prior, we compute the maximum risk-adjusted return ratio portfolio under each of the three.
=#
opts = [JuMPOptimiser(; pe = prs[1], slv = slv), JuMPOptimiser(; pe = prs[2], slv = slv),
        JuMPOptimiser(; pe = prs[3], slv = slv)]

mrs = [MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                opt = opt) for opt in opts]

ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "EmpiricalPrior" => ress[1].w,
                       "FactorPrior(Step)" => ress[2].w,
                       "FactorPrior(DimRed)" => ress[3].w); formatters = [resfmt])

# The maximum-ratio portfolios under the empirical prior and the two factor priors.
plot_stacked_bar_composition(ress, rd)

#=
The factor model portfolios are more diversified than the empirical one. The factor model reduces the estimation error of the covariance matrix, and a covariance with less error gives a more diversified portfolio.

### 3.2 Mean-negative skewness optimisation

We repeat the steps of section 3.1 with the negative skewness as the risk measure. The priors are now 4, 7 and 8, the high-order priors that hold the coskewness this measure reads.
=#
opts = [JuMPOptimiser(; pe = prs[4], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = prs[7], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = prs[8], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))))]

mrs = [MeanRisk(; r = NegativeSkewness(), obj = MinimumRisk(), opt = opt) for opt in opts]

ress = optimise.(mrs)

#=
The plots follow the order of section 3.1. The first plot is the composition of the frontier under the empirical high-order prior.
=#
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "HighOrderPriorEstimator",
                                        legend = :outerright))
# Empirical high-order prior frontier.
r = NegativeSkewness()
plot_measures(ress[1].w, prs[4]; x = r, y = ExpectedReturn(; rt = ress[1].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "HighOrderPriorEstimator", xlabel = "NegativeSkewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

# High-order factor prior composition with stepwise regression.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "HighOrderFactorPriorEstimator(Step)",
                                        legend = :outerright))
# High-order factor prior frontier with stepwise regression.
r = NegativeSkewness()
plot_measures(ress[2].w, prs[7]; x = r, y = ExpectedReturn(; rt = ress[2].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[2].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "HighOrderFactorPriorEstimator(Step)", xlabel = "NegativeSkewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)
# High-order factor prior composition with dimension reduction regression.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "HighOrderFactorPriorEstimator(DimRed)",
                                        legend = :outerright))
# High-order factor prior frontier with dimension reduction regression.
r = NegativeSkewness()
plot_measures(ress[3].w, prs[8]; x = r, y = ExpectedReturn(; rt = ress[3].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[3].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "HighOrderFactorPriorEstimator(DimRed)", xlabel = "NegativeSkewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

#=
Then we solve one maximum-ratio portfolio per prior, as in section 3.1.
=#
opts = [JuMPOptimiser(; pe = prs[4], slv = slv), JuMPOptimiser(; pe = prs[7], slv = slv),
        JuMPOptimiser(; pe = prs[8], slv = slv)]

mrs = [MeanRisk(; r = NegativeSkewness(), obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                opt = opt) for opt in opts]

ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "HighOrderPriorEstimator" => ress[1].w,
                       "HighOrderFactorPriorEstimator(Step)" => ress[2].w,
                       "HighOrderFactorPriorEstimator(DimRed)" => ress[3].w);
             formatters = [resfmt])

# The maximum-ratio portfolios with the negative skewness as the risk measure.
plot_stacked_bar_composition(ress, rd)

#=
Here the factor priors give less diversified portfolios than the empirical prior, the opposite of section 3.1. On this data the matrix of the negative spectral slices of the coskewness has a lower condition number under the empirical prior than under the factor priors. The higher moments are more sensitive to noise, and a factor model does not always reduce it.

### 3.3 Mean-kurtosis optimisation

We repeat the steps with the kurtosis as the risk measure.
=#
opts = [JuMPOptimiser(; pe = prs[4], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = prs[7], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = prs[8], slv = slv,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))))]

mrs = [MeanRisk(; r = Kurtosis(), obj = MinimumRisk(), opt = opt) for opt in opts]

ress = optimise.(mrs)

#=
This time every frontier plot measures the kurtosis on the returns of the empirical high-order prior, `prs[4]`. The optimisation reads the cokurtosis matrix of each prior, but a plot reads only the returns and the mean of the prior it is given. The returns of a factor prior are the ones its factor model rebuilds, so we pass `prs[4]` to measure the three frontiers on the same returns.
=#
# Empirical high-order prior composition.
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "HighOrderPriorEstimator",
                                        legend = :outerright))
# Empirical high-order prior frontier.
r = Kurtosis()
plot_measures(ress[1].w, prs[4]; x = r, y = ExpectedReturn(; rt = ress[1].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "HighOrderPriorEstimator", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

# High-order factor prior composition with stepwise regression.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "HighOrderFactorPriorEstimator(Step)",
                                        legend = :outerright))
# High-order factor prior frontier with stepwise regression.
r = Kurtosis()
plot_measures(ress[2].w, prs[4]; x = r, y = ExpectedReturn(; rt = ress[2].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[2].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "HighOrderFactorPriorEstimator(Step)", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)
# High-order factor prior composition with dimension reduction regression.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "HighOrderFactorPriorEstimator(DimRed)",
                                        legend = :outerright))
# High-order factor prior frontier with dimension reduction regression.
r = Kurtosis()
plot_measures(ress[3].w, prs[4]; x = r, y = ExpectedReturn(; rt = ress[3].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[3].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "HighOrderFactorPriorEstimator(DimRed)", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

#=
The section ends with the maximum-ratio portfolios.
=#
opts = [JuMPOptimiser(; pe = prs[4], slv = slv), JuMPOptimiser(; pe = prs[7], slv = slv),
        JuMPOptimiser(; pe = prs[8], slv = slv)]

mrs = [MeanRisk(; r = Kurtosis(), obj = MaximumRatio(; rf = 4.2 / 100 / 252), opt = opt)
       for opt in opts]

ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "HighOrderPriorEstimator" => ress[1].w,
                       "HighOrderFactorPriorEstimator(Step)" => ress[2].w,
                       "HighOrderFactorPriorEstimator(DimRed)" => ress[3].w);
             formatters = [resfmt])

# The maximum-ratio portfolios of the last cell.
plot_stacked_bar_composition(ress, rd)

#=
The kurtosis portfolios follow section 3.1 and not section 3.2. The factor priors give more diversified portfolios than the empirical prior. The [covariance](02_Covariance_Estimation.md) and [higher moment](03_Higher_Moment_Estimation.md) pages show ways to reduce the estimation error of these moments.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Sweep clean (ADR 0014 retrofit): the renamed factor-priors page runs end-to-end with the
#src   factor data, building empirical vs step-wise vs dimension-reduction factor priors and
#src   comparing them across MeanRisk frontiers. Added the "When to reach for this" callout.
#src - OBSERVATION (not a defect): the factor priors reproduce the empirical expected returns
#src   on this data (`prs[1].mu ≈ prs[2].mu ≈ prs[3].mu`), so the factor-model benefit here is
#src   concentrated in the covariance/high-order structure rather than the mean. Worth a sentence
#src   in the prose if the page is revisited. Rolled up to #126.
