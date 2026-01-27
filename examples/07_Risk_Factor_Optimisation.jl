#=
# Example 7: Risk factor optimisation

This example shows how to use factor models to perform optimisations. These reduce the estimation error by modelling asset returns as a function of common risk factors.
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
mipresfmt = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
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
## 1. ReturnsResult data

We will use the same data as the previous example. But we will also load factor data.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

F = TimeArray(CSV.File(joinpath(@__DIR__, "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(F[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X, F)

#=
## 2. Prior statistics

`PortfolioOptimisers.jl` supports a wide range of prior models. Here we will use four of them:

1. [`EmpiricalPrior`](@ref): Computes the expected returns vector and covariance matrix from the empirical data.
2. [`FactorPrior`](@ref): Computes the expected returns vector and covariance matrix using a factor model.
3. [`HighOrderPriorEstimator`](@ref): Computes the expected returns vector and covariance matrix using the low order prior estimator provided, plus the coskewness and/or cokurtosis using the computed expected returns vector.
4. [`HighOrderFactorPriorEstimator`]-(@ref): Computes the expected returns vector and covariance matrix, plus the coskewness and/or cokurtosis using a factor model.

Priors whos names don't start with the prefix `High` return [`LowOrderPrior`](@ref), while those that do return [`HighOrderPrior`](@ref) objects.

We have two different regression models with various targets. We won't explore them all in detail here, see [`StepwiseRegression`](@ref) and [`DimensionReductionRegression`](@ref) for details. First lets define the prior estimators.
=#

pes = [EmpiricalPrior(),#
       FactorPrior(),#
       FactorPrior(; re = DimensionReductionRegression()),#
       HighOrderPriorEstimator(),#
       HighOrderPriorEstimator(; pe = FactorPrior()),#
       HighOrderPriorEstimator(; pe = FactorPrior(; re = DimensionReductionRegression())),#
       HighOrderFactorPriorEstimator(;),#
       HighOrderFactorPriorEstimator(;
                                     pe = FactorPrior(;
                                                      re = DimensionReductionRegression()))]

#=
Now lets compute the prior statistics for each estimator.
=#
prs = prior.(pes, rd)

#=
First lets compare the first three prior results.

The expected returns, found in the `mu` field, do not change much between [`EmpiricalPrior`](@ref) and [`FactorPrior`](@ref). Which illustrates one of the reasons why it's unwise to put much stock on expected returns estimates, since they are highly uncertain and sensitive to noise. We will explore different expected returns estimators, which attempt to improve this drawback in future examples.
=#

pretty_table(DataFrame("Assets" => rd.nx, "EmpiricalPrior" => prs[1].mu,
                       "FactorPrior(Step)" => prs[2].mu,
                       "FactorPrior(DimRed)" => prs[3].mu); formatters = [mmtfmt],
             title = "Expected returns",
             source_notes = "prs[1].mu ≈ prs[1].mu ≈ prs[3].mu: $(prs[1].mu ≈ prs[1].mu ≈ prs[3].mu)")

#=
However, the covariance estimates, found in the `sigma` field, differ significantly more. Factor models tend to produce more stable and robust covariance estimates by capturing the underlying risk factors driving asset returns, while reducing the impact of idiosyncratic noise. We will check the condition number of the covariance matrices to illustrate this. The lower the condition number, the more less noisy and therefore more numerically stable the matrix is.

For the covariance estimation in particular, we can take advantage of a sparser Cholesky decomposition with better numerical properties than the naive version. If present, it is used instead of the cholesky decomposition of `sigma` in the `SecondOrderCone` constraint of the variance and/or standard deviation formulations in traditional optimisations. This special decomposition can be found in the `chol` field of the prior result.
=#
using LinearAlgebra

pretty_table(DataFrame([rd.nx prs[1].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "EmpiricalPrior Covariance",
             source_notes = "Condition number EmpiricalPrior: $(round(cond(prs[1].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[2].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "FactorPrior(Step) Covariance",
             source_notes = "Condition number FactorPrior(Step): $(round(cond(prs[2].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[3].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "FactorPrior(DimRed) Covariance",
             source_notes = "Condition number FactorPrior(DimRed): $(round(cond(prs[3].sigma); digits = 3))")

#=
The next three prior results have the same low order moments and adjusted returns series as the `i-3`'th prior result because they use the same regression model.
=#
for i in 4:6
    println("prs[$(i-3)].X     == prs[$(i)].X    : $(prs[i-3].X == prs[i].X)")
    println("prs[$(i-3)].mu    == prs[$(i)].mu   : $(prs[i-3].mu == prs[i].mu)")
    println("prs[$(i-3)].sigma == prs[$(i)].sigma: $(prs[i-3].sigma == prs[i].sigma)\n")
end

# However, all high order moments for these estimators are identical to each other despite their low order moments being computed differently. This is because they are computed from the prior returns matrix, not the posterior one, as this would be inconsistent. The coskewness matrix is found in the `sk` field, the negative spectral decomposition of its slices is found in the `V` field, and the cokurtosis matrix is found in the `kt` field.
println("prs[4].sk == prs[5].sk == prs[6].sk: $(prs[4].sk == prs[5].sk == prs[6].sk)")
println("prs[4].V  == prs[5].V  == prs[6].V : $(prs[4].V == prs[5].V == prs[6].V)")
println("prs[4].kt == prs[5].kt == prs[6].kt: $(prs[4].kt == prs[5].kt == prs[6].kt)\n")

#=
Now lets compare the last four prior results. Remember the last two also use a factor model for the high order moments
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
As expected, the higher moments are the same only for `prs[5]` and `prs[6]`, since neither of them adjust the higher moments using a factor model. However, their low order moments differ because they use different regression models. The low order moments of `prs[5]` and `prs[7]` use the [`StepwiseRegression`](@ref) model, while `prs[6]` and `prs[8]` use the [`DimensionReductionRegression`](@ref) model, so those match too. Aside from `prs[5]` and `prs[6]`, the higher order moments are computed using regression models.

Lets compare what these higher order moments look like. First lets create the names for the higher order moments.
=#

nx2 = collect(Iterators.flatten([(nx * "_") .* rd.nx for nx in rd.nx]))

#=
Now lets examine what the coskewness and its negative spectral slices look like.
=#
pretty_table(DataFrame([rd.nx prs[4].sk], ["Assets^2 / Assets"; nx2]);
             formatters = [hmmtfmt], title = "HighOrderPriorEstimator Coskewness",
             source_notes = "Condition number HighOrderPriorEstimator: $(round(cond(prs[4].sk); digits = 3))")
pretty_table(DataFrame([rd.nx prs[7].sk], ["Assets^2 / Assets"; nx2]);
             formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(Step) Coskewness",
             source_notes = "Condition number HighOrderFactorPriorEstimator(Step): $(round(cond(prs[7].sk); digits = 3))")
pretty_table(DataFrame([rd.nx prs[8].sk], ["Assets^2 / Assets"; nx2]);
             formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(DimRed) Coskewness",
             source_notes = "Condition number HighOrderFactorPriorEstimator(DimRed): $(round(cond(prs[8].sk); digits = 3))")

pretty_table(DataFrame([rd.nx prs[4].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "HighOrderPriorEstimator Coskewness Negative Spectral Slices",
             source_notes = "Condition number HighOrderPriorEstimator: $(round(cond(prs[4].V); digits = 3))")
pretty_table(DataFrame([rd.nx prs[7].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(Step) Coskewness Negative Spectral Slices",
             source_notes = "Condition number HighOrderFactorPriorEstimator(Step): $(round(cond(prs[7].V); digits = 3))")
pretty_table(DataFrame([rd.nx prs[8].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(DimRed) Coskewness Negative Spectral Slices",
             source_notes = "Condition number HighOrderFactorPriorEstimator(DimRed): $(round(cond(prs[8].V); digits = 3))")

#=
And the cokurtosis.
=#
pretty_table(DataFrame([nx2 prs[4].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "HighOrderPriorEstimator Cokurtosis",
             source_notes = "Condition number HighOrderPriorEstimator: $(round(cond(prs[4].kt); digits = 3))")
pretty_table(DataFrame([nx2 prs[7].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(Step) Cokurtosis",
             source_notes = "Condition number HighOrderFactorPriorEstimator(Step): $(round(cond(prs[7].kt); digits = 3))")
pretty_table(DataFrame([nx2 prs[8].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "HighOrderFactorPriorEstimator(DimRed) Cokurtosis",
             source_notes = "Condition number HighOrderFactorPriorEstimator(DimRed): $(round(cond(prs[8].kt); digits = 3))")

#=
Ideally, the condition numbers of the higher order moments should be lower when using factor models, indicating more stable and robust estimates. However, this is not always the case due to the higher exponentiation involved in their computation.
=#

#=
## 3. Comparing optimisations

We'll now compare the effect of using these different prior estimators in the shape of the efficient frontier. We'll use Clarabel as the solver.
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
### 3.1 Mean-Standard deviation optimisation

First lets examine the mean-standard deviation efficient frontier using the empirical and factor priors. We will compute the efficient fronteir with 50 points for all relevant priors.
=#
## JuMP Optimsiers, we will compute the efficient frontier with 50 points for all of them.
opts = [JuMPOptimiser(; pr = prs[1], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[2], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[3], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)))]

## Mean-Risk estimators using the standard deviation.
mrs = [MeanRisk(; r = StandardDeviation(), obj = MinimumRisk(), opt = opt) for opt in opts]

## Optimise
ress = optimise.(mrs)

#=
Lets plot the efficient frontiers.
=#
using GraphRecipes, StatsPlots
# Empirical prior composition.
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "EmpiricalPrior", legend = :outerright))
# Empirical prior frontier.
r = StandardDeviation()
plot_measures(ress[1].w, prs[1]; x = r, y = ReturnRiskMeasure(; rt = ress[1].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[1].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "EmpiricalPrior", xlabel = "SD", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)

# Factor prior composition with stepwise regression.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(Step)", legend = :outerright))
# Factor prior frontier with stepwise regression.
r = StandardDeviation()
plot_measures(ress[2].w, prs[2]; x = r, y = ReturnRiskMeasure(; rt = ress[2].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[2].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "FactorPrior(Step)", xlabel = "SD", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)
# Factor prior using dimensionality reduction.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(DimRed)",
                                        legend = :outerright))
# Factor prior frontier with stepwise regression.
r = StandardDeviation()
plot_measures(ress[3].w, prs[3]; x = r, y = ReturnRiskMeasure(; rt = ress[3].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[3].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "FactorPrior(DimRed)", xlabel = "SD", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)

#=
Lets optimise the maximum risk-adjusted return ratio of the three to see how a single portfolio differs.
=#
opts = [JuMPOptimiser(; pr = prs[1], slv = slv), JuMPOptimiser(; pr = prs[2], slv = slv),
        JuMPOptimiser(; pr = prs[3], slv = slv)]

## Mean-Risk estimators using the standard deviation.
mrs = [MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                opt = opt) for opt in opts]

## Optimise
ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "EmpiricalPrior" => ress[1].w,
                       "FactorPrior(Step)" => ress[2].w,
                       "FactorPrior(DimRed)" => ress[3].w); formatters = [resfmt])
#=
We can see that the factor model portfolios are more diversified than the empirical one. This is because the factor model reduces estimation error in the covariance matrix, leading to more stable and diversified portfolios.

### 3.2 Mean-NegativeSkewness optimisation

Here we will perform the exact same procedure as before, but using the negative skewness as the risk measure.
=#
## JuMP Optimsiers, we will compute the efficient frontier with 50 points for all of them.
opts = [JuMPOptimiser(; pr = prs[4], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[7], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[8], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)))]

## Mean-Risk estimators using the standard deviation.
mrs = [MeanRisk(; r = NegativeSkewness(), obj = MinimumRisk(), opt = opt) for opt in opts]

## Optimise
ress = optimise.(mrs)

#=
Lets plot the efficient frontiers.

Empirical prior composition.
=#
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "EmpiricalPrior", legend = :outerright))
# Empirical prior frontier.
r = NegativeSkewness()
plot_measures(ress[1].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[1].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[1].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "EmpiricalPrior", xlabel = "NegativeSkewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

# Factor prior composition with stepwise regression.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(Step)", legend = :outerright))
# Factor prior frontier with stepwise regression.
r = NegativeSkewness()
plot_measures(ress[2].w, prs[7]; x = r, y = ReturnRiskMeasure(; rt = ress[2].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[2].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "FactorPrior(Step)", xlabel = "NegativeSkewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
# Factor prior using dimensionality reduction.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(DimRed)",
                                        legend = :outerright))
# Factor prior frontier with stepwise regression.
r = NegativeSkewness()
plot_measures(ress[3].w, prs[8]; x = r, y = ReturnRiskMeasure(; rt = ress[3].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[3].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "FactorPrior(DimRed)", xlabel = "NegativeSkewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

#=
Lets optimise the maximum risk-adjusted return ratio of the three to see how a single portfolio differs.
=#
opts = [JuMPOptimiser(; pr = prs[4], slv = slv), JuMPOptimiser(; pr = prs[7], slv = slv),
        JuMPOptimiser(; pr = prs[8], slv = slv)]

## Mean-Risk estimators using the standard deviation.
mrs = [MeanRisk(; r = NegativeSkewness(), obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                opt = opt) for opt in opts]

## Optimise
ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "EmpiricalPrior" => ress[1].w,
                       "FactorPrior(Step)" => ress[2].w,
                       "FactorPrior(DimRed)" => ress[3].w); formatters = [resfmt])
#=
Here we have the opposite effect to before, this follows from the fact that for this particular scenario the condition number of the matrix of negative spectral slices of the coskewness is actually lower for the empirical prior than for the factor priors. This goes to show that higher order moments are more susceptible to noise, and factor models may not be able to fully mitigate this.

### 3.3 Mean-Kurtosis optimisation

Again we will do the same as before but with the kurtosis.
=#
## JuMP Optimsiers, we will compute the efficient frontier with 50 points for all of them.
opts = [JuMPOptimiser(; pr = prs[4], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[7], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[8], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)))]

## Mean-Risk estimators using the standard deviation.
mrs = [MeanRisk(; r = Kurtosis(), obj = MinimumRisk(), opt = opt) for opt in opts]

## Optimise
ress = optimise.(mrs)

#=
Lets plot the efficient frontiers. However, this time when plotting the frontiers we will use the prior returns because the kurtosis risk measure is not computed from the cokurtosis matrix, but from the returns directly.
=#
# Empirical prior composition.
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "EmpiricalPrior", legend = :outerright))
# Empirical prior frontier.
r = Kurtosis()
plot_measures(ress[1].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[1].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[1].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "EmpiricalPrior", xlabel = "Kurtosis", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)

# Factor prior composition with stepwise regression.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(Step)", legend = :outerright))
# Factor prior frontier with stepwise regression.
r = Kurtosis()
plot_measures(ress[2].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[2].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[2].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "FactorPrior(Step)", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
# Factor prior using dimensionality reduction.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "FactorPrior(DimRed)",
                                        legend = :outerright))
# Factor prior frontier with stepwise regression.
r = Kurtosis()
plot_measures(ress[3].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[3].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[3].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "FactorPrior(DimRed)", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

#=
Lets optimise the maximum risk-adjusted return ratio of the three to see how a single portfolio differs.
=#
opts = [JuMPOptimiser(; pr = prs[4], slv = slv), JuMPOptimiser(; pr = prs[7], slv = slv),
        JuMPOptimiser(; pr = prs[8], slv = slv)]

## Mean-Risk estimators using the standard deviation.
mrs = [MeanRisk(; r = NegativeSkewness(), obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                opt = opt) for opt in opts]

## Optimise
ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "EmpiricalPrior" => ress[1].w,
                       "FactorPrior(Step)" => ress[2].w,
                       "FactorPrior(DimRed)" => ress[3].w); formatters = [resfmt])
#=
These findings are again consistent with the previous result, and reflective of the higher condition numbers of the factor-based higher order statistics. The next example will explore ways of reducing the estimation error.
=#
