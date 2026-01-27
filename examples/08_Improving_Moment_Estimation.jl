#=
# Example 8: Improving moment estimation

This example will show how to improve the estimation of both low and higher order moments using denoising and sparsification techniques.
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

Here we will only use empirical priors but with denoising and sparsification techniques applied.

For denoising we will use [`Denoise`](@ref), and all its associated algorithms [`ShrunkDenoise`](@ref), [`FixedDenoise`](@ref), and [`SpectralDenoise`](@ref), though the last one may not always produce a matrix with a lower condition number.

For sparsification we will use the relationship structure to sparsify the inverse of the matrix using [`LoGo`](@ref) with two different distance matrix similarity measures, [`MaximumDistanceSimilarity`](@ref) and [`ExponentialSimilarity`](@ref).

We will also improve the estimation of the mean return by using a shrunk expected returns estimator [`ShrunkExpectedReturns`](@ref) using [`BayesStein`](@ref) and [`BodnarOkhrinParolya`](@ref) algorithms with all three available expected returns shrinkage targets [`GrandMean`](@ref), [`VolatilityWeighted`](@ref), and [`MeanSquaredError`](@ref).
=#
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
                                                                                      alg = LoGo(;
                                                                                                 sim = ExponentialSimilarity()))),
                               kte = Cokurtosis(;
                                                mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                      alg = LoGo(;
                                                                                                 sim = ExponentialSimilarity()))))]
#=
Now lets compute the prior statistics for each estimator.
=#
prs = prior.(pes, rd)

#=
First lets view the expected returns.
=#
pretty_table(DataFrame("Assets" => rd.nx, "Vanilla" => prs[1].mu, "BS(VW)" => prs[2].mu,
                       "BS(MSE)" => prs[3].mu, "BOP(GM)" => prs[4].mu,
                       "BOP(VW)" => prs[5].mu, "BOP(MSE)" => prs[6].mu);
             formatters = [mmtfmt], title = "Expected returns")
#=
We can now see how the different denoising and sparsification techniques improve the covariance matrix's condition number.
=#
using LinearAlgebra

pretty_table(DataFrame([rd.nx prs[1].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Vanilla",
             source_notes = "Condition number Vanilla: $(round(cond(prs[1].sigma); digits = 3))")
pretty_table(DataFrame([rd.nx prs[2].sigma], ["Assets"; rd.nx]); formatters = [mmtfmt],
             title = "Fixed denoise",
             source_notes = "Condition number fixed denoise: $(round(cond(prs[2].sigma); digits = 3))")