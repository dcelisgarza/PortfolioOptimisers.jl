The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/08_Improving_Moment_Estimation.jl"
```

# Example 8: Improving moment estimation

This example will show how to improve the estimation of both low and higher order moments using denoising and sparsification techniques.

````@example 08_Improving_Moment_Estimation
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
nothing #hide
````

## 1. ReturnsResult data

We will use the same data as the previous example. But we will also load factor data.

````@example 08_Improving_Moment_Estimation
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

F = TimeArray(CSV.File(joinpath(@__DIR__, "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(F[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X, F)
````

## 2. Prior statistics

Here we will only use empirical priors but with denoising and sparsification techniques applied.

For denoising we will use [`Denoise`](@ref), and all its associated algorithms [`ShrunkDenoise`](@ref), [`FixedDenoise`](@ref), and [`SpectralDenoise`](@ref), though the last one may not always produce a matrix with a lower condition number.

For sparsification we will use the relationship structure to sparsify the inverse of the matrix using [`LoGo`](@ref) with two different distance matrix similarity measures, [`MaximumDistanceSimilarity`](@ref) and [`ExponentialSimilarity`](@ref).

We will also improve the estimation of the mean return by using a shrunk expected returns estimator [`ShrunkExpectedReturns`](@ref) using [`BayesStein`](@ref) and [`BodnarOkhrinParolya`](@ref) algorithms with all three available expected returns shrinkage targets [`GrandMean`](@ref), [`VolatilityWeighted`](@ref), and [`MeanSquaredError`](@ref).

````@example 08_Improving_Moment_Estimation
pes = [EmpiricalPrior(;),#
       EmpiricalPrior(;
                      me = ShrunkExpectedReturns(;
                                                 alg = BayesStein(;
                                                                  tgt = VolatilityWeighted())),#
                      ce = PortfolioOptimisersCovariance(;
                                                         mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                               dn = Denoise(;
                                                                                                            alg = FixedDenoise())))),#
       EmpiricalPrior(;
                      me = ShrunkExpectedReturns(;
                                                 alg = BayesStein(;
                                                                  tgt = MeanSquaredError())),#
                      ce = PortfolioOptimisersCovariance(;
                                                         mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                               alg = LoGo()))),
       HighOrderPriorEstimator(;
                               pe = EmpiricalPrior(;
                                                   ce = PortfolioOptimisersCovariance(;
                                                                                      mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                                            dn = Denoise(;
                                                                                                                                         alg = ShrunkDenoise(;
                                                                                                                                                             alpha = 0.5)))),
                                                   me = ShrunkExpectedReturns(;
                                                                              alg = BodnarOkhrinParolya()))),
       HighOrderPriorEstimator(;
                               pe = EmpiricalPrior(;
                                                   me = ShrunkExpectedReturns(;
                                                                              alg = BodnarOkhrinParolya(;
                                                                                                        tgt = VolatilityWeighted())),
                                                   ce = PortfolioOptimisersCovariance(;
                                                                                      mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                                            alg = LoGo(),
                                                                                                                            dn = Denoise(;
                                                                                                                                         alg = FixedDenoise())))),
                               ske = Coskewness(;
                                                mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                      dn = Denoise(;
                                                                                                   alg = FixedDenoise()))),
                               kte = Cokurtosis(;
                                                mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                      dn = Denoise(;
                                                                                                   alg = FixedDenoise())))),
       HighOrderPriorEstimator(;
                               pe = EmpiricalPrior(;
                                                   me = ShrunkExpectedReturns(;
                                                                              alg = BodnarOkhrinParolya(;
                                                                                                        tgt = MeanSquaredError())),
                                                   ce = PortfolioOptimisersCovariance(;
                                                                                      mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                                            alg = LoGo(;
                                                                                                                                       sim = ExponentialSimilarity())))),
                               ske = Coskewness(;
                                                mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                      dn = Denoise(),
                                                                                      alg = LoGo())),
                               kte = Cokurtosis(;
                                                mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                      dn = Denoise(),
                                                                                      alg = LoGo())))]
````

Now let's compute the prior statistics for each estimator.

````@example 08_Improving_Moment_Estimation
prs = prior.(pes, rd)
````

### 2.1 Expected returns

First let's view the expected returns.

````@example 08_Improving_Moment_Estimation
pretty_table(DataFrame("Assets" => rd.nx, "Vanilla" => prs[1].mu, "BS(VW)" => prs[2].mu,
                       "BS(MSE)" => prs[3].mu, "BOP(GM)" => prs[4].mu,
                       "BOP(VW)" => prs[5].mu, "BOP(MSE)" => prs[6].mu);
             formatters = [mmtfmt], title = "Expected returns")
````

### 2.2 Covariance matrices

We can now see how the different denoising and sparsification techniques improve the covariance matrix's condition number.

````@example 08_Improving_Moment_Estimation
using LinearAlgebra

pretty_table(DataFrame([rd.nx prs[1].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Covariance: Vanilla",
             source_notes = "Condition number Vanilla: $(round(cond(prs[1].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[2].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Covariance: Fixed denoise",
             source_notes = "Condition number fixed denoise: $(round(cond(prs[2].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[3].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Covariance: LoGo(MaxDist)",
             source_notes = "Condition number LoGo(MaxDist): $(round(cond(prs[3].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[4].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Covariance: Shrunk denoise (0.5)",
             source_notes = "Condition number Shrunk denoise (0.5): $(round(cond(prs[4].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[5].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Covariance: FixedDenoise + LoGo(MaxDist)",
             source_notes = "Condition number FixedDenoise + LoGo(MaxDist): $(round(cond(prs[5].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[6].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Covariance: LoGo(ExpDist)",
             source_notes = "Condition number LoGo(ExpDist): $(round(cond(prs[6].sigma); digits = 3))")
````

### 2.3 Higher order moments

Now let's view how the higher order moments benefit from denoising. The coskewness matrix is not affected because it's not square, but the matrix of negative spectral slices is.

````@example 08_Improving_Moment_Estimation
nx2 = collect(Iterators.flatten([(nx * "_") .* rd.nx for nx in rd.nx]))

pretty_table(DataFrame([rd.nx prs[4].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "Coskewness Negative Spectral Slices: Vanilla",
             source_notes = "Condition number Vanilla: $(round(cond(prs[4].V); digits = 3))")
pretty_table(DataFrame([rd.nx prs[5].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "Coskewness Negative Spectral Slices: FixedDenoise",
             source_notes = "Condition number FixedDenoise: $(round(cond(prs[5].V); digits = 3))")
pretty_table(DataFrame([rd.nx prs[6].V], ["Assets"; rd.nx]); formatters = [hmmtfmt],
             title = "Coskewness Negative Spectral Slices: LoGo(MaxDist)",
             source_notes = "Condition number LoGo(MaxDist): $(round(cond(prs[6].V); digits = 3))")
````

And finally the cokurtosis.

````@example 08_Improving_Moment_Estimation
pretty_table(DataFrame([nx2 prs[4].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "Cokurtosis: Vanilla",
             source_notes = "Condition number Vanilla: $(round(cond(prs[4].kt); digits = 3))")
pretty_table(DataFrame([nx2 prs[5].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "Cokurtosis: FixedDenoise",
             source_notes = "Condition number FixedDenoise: $(round(cond(prs[5].kt); digits = 3))")
pretty_table(DataFrame([nx2 prs[6].kt], ["Assets^2"; nx2]); formatters = [hmmtfmt],
             title = "Cokurtosis: Shrunk denoise (0) + LoGo(MaxDist)",
             source_notes = "Condition number Shrunk denoise (0) + LoGo(MaxDist): $(round(cond(prs[6].kt); digits = 3))")
````

## 3. Comparing optimisations

### 3.1 Mean-Variance optimisation

First let's see how the denoising and sparsification techniques affect the mean-variance optimisation along the efficient frontier.

````@example 08_Improving_Moment_Estimation
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

# JuMP Optimsiers, we will compute the efficient frontier with 50 points for all of them.
opts = [JuMPOptimiser(; pr = prs[1], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[2], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[3], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[4], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[5], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[6], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)))]

# Mean-Risk estimators using the variance.
mrs = [MeanRisk(; obj = MinimumRisk(), opt = opt) for opt in opts]

# Optimise
ress = optimise.(mrs)
````

Let's plot the efficient frontiers.

````@example 08_Improving_Moment_Estimation
using GraphRecipes, StatsPlots

r = Variance()
````

Vanilla sigma and mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Vanilla sigma and mu",
                                        legend = :outerright))
````

Vanilla sigma and mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[1].w, prs[1]; x = r, y = ReturnRiskMeasure(; rt = ress[1].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[1].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Vanilla sigma and mu", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Fixed denoise sigma, BS(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Fixed denoise sigma and BS(VW) mu",
                                        legend = :outerright))
````

Fixed denoise covariance.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[2].w, prs[2]; x = r, y = ReturnRiskMeasure(; rt = ress[2].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[2].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Fixed denoise sigma and BS(VW) mu", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

LoGo(MaxDist) sigma and BS(MSE) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "LoGo(MaxDist) sigma and BS(MSE) mu",
                                        legend = :outerright))
````

LoGo(MaxDist) covariance.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[3].w, prs[3]; x = r, y = ReturnRiskMeasure(; rt = ress[3].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[3].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "LoGo(MaxDist) sigma and BS(MSE) mu", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Shrunk denoise (0.5) sigma and BOP(GM) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[4].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Shrunk denoise (0.5) sigma and BOP(GM) mu",
                                        legend = :outerright))
````

Shrunk denoise (0.5) covariance.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[4].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[4].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[4].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Shrunk denoise (0.5) sigma and BOP(GM) mu", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Fixed denoise + LoGo(MaxDist) sigma and BOP(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[5].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Fixed denoise + LoGo(MaxDist) sigma and BOP(VW) mu",
                                        legend = :outerright))
````

Fixed denoise + LoGo(MaxDist) sigma and BOP(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[5].w, prs[5]; x = r, y = ReturnRiskMeasure(; rt = ress[5].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[5].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Fixed denoise + LoGo(MaxDist) sigma and BOP(VW) mu",
              xlabel = "Variance", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)
````

LoGo(ExpDist) sigma and BOP(MSE) mu prior composition.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[6].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "LoGo(ExpDist) sigma and BOP(MSE) mu",
                                        legend = :outerright))
````

LoGo(ExpDist) sigma and BOP(MSE) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[6].w, prs[6]; x = r, y = ReturnRiskMeasure(; rt = ress[6].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[6].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "LoGo(ExpDist) sigma and BOP(MSE) mu", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

This example is a nice way to show how sensitive moment-based optimisations are to the moment estimation. For now, let's examine how the maximum risk return ratio portfolios differ. An actual analysis would isolate the effect of the covariance and expected returns separately, the point of this example is to show how different ways to improve their estimation and how much the results can be affected by them.

````@example 08_Improving_Moment_Estimation
opts = [JuMPOptimiser(; pr = prs[1], slv = slv), JuMPOptimiser(; pr = prs[2], slv = slv),
        JuMPOptimiser(; pr = prs[3], slv = slv), JuMPOptimiser(; pr = prs[4], slv = slv),
        JuMPOptimiser(; pr = prs[5], slv = slv), JuMPOptimiser(; pr = prs[6], slv = slv)]

# Mean-Risk estimators using the variance.
mrs = [MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 100 / 252), opt = opt) for opt in opts]

# Optimise
ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "Vanilla" => ress[1].w,
                       "Fixed + BS(VW)" => ress[2].w,
                       "LoGo(MaxDist) + BS(MSE)" => ress[3].w,
                       "Shrunk (0.5) + BOP(GM)" => ress[4].w,
                       "Fixed + LoGo(MaxDist) + BOP(VW)" => ress[5].w,
                       "LoGo(ExpDist) + BOP(MSE)" => ress[6].w); formatters = [resfmt])
````

Portfolios that emphasise expected returns are more sensitive to the expected returns estimation. It is important to exercise caution when relying on expected returns in particular. If one thinks about it, it summarises all returns information into a single number per asset, so any error in its estimation can have a large effect on the resulting portfolio. We will finish on showing these effects in higher order moment optimisation.

### 3.2 Mean-NegativeSkewness optimisation

````@example 08_Improving_Moment_Estimation
# JuMP Optimsiers, we will compute the efficient frontier with 50 points for all of them.
opts = [JuMPOptimiser(; pr = prs[4], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[5], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[6], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)))]

r = NegativeSkewness()
# Mean-Risk estimators using the negative skewness.
mrs = [MeanRisk(; r = r, obj = MinimumRisk(), opt = opt) for opt in opts]

# Optimise
ress = optimise.(mrs)
````

Let's plot the efficient frontiers.

Vanilla V and BOP(GM) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Vanilla V and BOP(GM) mu",
                                        legend = :outerright))
````

Vanilla V and BOP(GM) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[1].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[1].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[1].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Vanilla V and BOP(GM) mu", xlabel = "Negative Skewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Fixed Denoise V and BOP(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Fixed Denoise V and BOP(VW) mu",
                                        legend = :outerright))
````

Fixed Denoise V and BOP(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[2].w, prs[5]; x = r, y = ReturnRiskMeasure(; rt = ress[2].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[2].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Fixed Denoise V and BOP(VW) mu", xlabel = "Negative Skewness",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Shrunk(0) Denoise + LoGo(MaxDist) V and BOP(MSE) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Shrunk(0) Denoise + LoGo(MaxDist) V and BOP(MSE) mu",
                                        legend = :outerright))
````

Shrunk(0) Denoise + LoGo(MaxDist) V and BOP(MSE) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[3].w, prs[6]; x = r, y = ReturnRiskMeasure(; rt = ress[3].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[3].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Shrunk(0) Denoise + LoGo(MaxDist) V and BOP(MSE) mu",
              xlabel = "Negative Skewness", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)
````

In this case we can se that in this case, these techniques denoising and sparsification make the evolution of the portfolio composition along the efficient frontier smoother.

Now let's examine how the maximum risk return ratio portfolios differ.

````@example 08_Improving_Moment_Estimation
opts = [JuMPOptimiser(; pr = prs[4], slv = slv), JuMPOptimiser(; pr = prs[5], slv = slv),
        JuMPOptimiser(; pr = prs[6], slv = slv)]

# Mean-Risk estimators using the Negative Skewness.
mrs = [MeanRisk(; r = r, obj = MaximumRatio(; rf = 4.2 / 100 / 252), opt = opt)
       for opt in opts]

# Optimise
ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "Vanilla V + BOP(GM) mu" => ress[1].w,
                       "Fixed Denoise V + BOP(VW) mu" => ress[2].w,
                       "Shrunk(0) Denoise + LoGo(MaxDist) V + BOP(MSE) mu" => ress[3].w);
             formatters = [resfmt])
````

Similarly to the mean-variance case, the more one emphasises the expected returns, the less stable the resulting portfolio. Generally, it's better to use a volatility weighted shrinkage target for the expected returns as that adjusts the expected returns by penalising high volatility and rewarding low volatility assets.

### 3.3 Mean-Kurtosis optimisation

Finally, we will see how the cokurtosis estimation improvements affect the mean-kurtosis optimisation along the efficient frontier.

````@example 08_Improving_Moment_Estimation
# JuMP Optimsiers, we will compute the efficient frontier with 50 points for all of them.
opts = [JuMPOptimiser(; pr = prs[4], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[5], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50))),
        JuMPOptimiser(; pr = prs[6], slv = slv,
                      ret = ArithmeticReturn(; lb = Frontier(; N = 50)))]

r = Kurtosis()
# Mean-Risk estimators using the kurtosis.
mrs = [MeanRisk(; r = r, obj = MinimumRisk(), opt = opt) for opt in opts]

# Optimise
ress = optimise.(mrs)
````

Let's plot the efficient frontiers.

Vanilla kt and BOP(GM) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Vanilla kt and BOP(GM) mu",
                                        legend = :outerright))
````

Vanilla kt and BOP(GM) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[1].w, prs[4]; x = r, y = ReturnRiskMeasure(; rt = ress[1].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[1].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Vanilla kt and BOP(GM) mu", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Fixed Denoise kt and BOP(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Fixed Denoise kt and BOP(VW) mu",
                                        legend = :outerright))
````

Fixed Denoise kt and BOP(VW) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[2].w, prs[5]; x = r, y = ReturnRiskMeasure(; rt = ress[2].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[2].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Fixed Denoise kt and BOP(VW) mu", xlabel = "Kurtosis",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)
````

Shrunk(0) Denoise + LoGo(MaxDist) kt and BOP(MSE) mu.

````@example 08_Improving_Moment_Estimation
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Shrunk(0) Denoise + LoGo(MaxDist) kt and BOP(MSE) mu",
                                        legend = :outerright))
````

Shrunk(0) Denoise + LoGo(MaxDist) kt and BOP(MSE) mu.

````@example 08_Improving_Moment_Estimation
plot_measures(ress[3].w, prs[6]; x = r, y = ReturnRiskMeasure(; rt = ress[3].ret),
              c = ReturnRiskRatioRiskMeasure(; rt = ress[3].ret, rk = r,
                                             rf = 4.2 / 100 / 252),
              title = "Shrunk(0) Denoise + LoGo(MaxDist) kt and BOP(MSE) mu",
              xlabel = "Kurtosis", ylabel = "Arithmetic Return",
              colorbar_title = "\nRisk/Return Ratio", right_margin = 6Plots.mm)
````

Again, the efficient frontier is smoother, but the kurtosis is inherently less stable than the variance and negative skewness, so the improvements are less pronounced.

Finally let's see what the maximum risk return ratio portfolios look like.

````@example 08_Improving_Moment_Estimation
opts = [JuMPOptimiser(; pr = prs[4], slv = slv), JuMPOptimiser(; pr = prs[5], slv = slv),
        JuMPOptimiser(; pr = prs[6], slv = slv)]

# Mean-Risk estimators using the Kurtosis.
mrs = [MeanRisk(; r = r, obj = MaximumRatio(; rf = 4.2 / 100 / 252), opt = opt)
       for opt in opts]

# Optimise
ress = optimise.(mrs)
pretty_table(DataFrame("Assets" => rd.nx, "Vanilla kt + BOP(GM) mu" => ress[1].w,
                       "Fixed Denoise kt + BOP(VW) mu" => ress[2].w,
                       "Shrunk(0) Denoise + LoGo(MaxDist) kt + BOP(MSE) mu" => ress[3].w);
             formatters = [resfmt])
````

Generally, the volatility weighted target for expected returns combined with fixed denoise, performs quite well. That's not to say other methods are not useful. It's worth using different techniques and comparing the results. For example with cross validation, which is as of yet unimplemented in `PortfolioOptimisers.jl`. Even so, it's never an exact science. It's always best to combine techniques, which is why we provide users with the ability to use multiple risk measures like in [`06_Multiple_Risk_Measures`](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/06_Multiple_Risk_Measures). There are also other ways of directly mitigating these instabilities raging:

1. Estimators like [`Stacking`]-(@ref) and [`NearOptimalCentering`]-(@ref).
2. Uncertainty sets for the expected returns and covariance matrices.
3. Logarithmic returns directly in the optimisation.
4. L1 and L2 regularisation.
5. Buy-in threshold constraints.
6. Phylogeny constraints.

These are a few things directly reduce the impact of moment estimation errors on the resulting portfolios, but can be used with other risk measures as well.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
