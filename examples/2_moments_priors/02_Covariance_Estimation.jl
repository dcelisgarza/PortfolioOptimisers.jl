#=
```@meta
Description = "Covariance estimation in PortfolioOptimisers.jl: denoising and sparsification of the covariance of a short, noisy return sample."
```

# Covariance estimation

Almost every optimiser reads the covariance matrix, and a short window estimates it badly.
With ``N`` assets and a little more than ``N`` observations, the sample covariance is noisy
and ill-conditioned. Mean-variance optimisation inverts it in effect, and the inverse
amplifies the noise. Two families of fixes help.

  - Denoising separates the signal from the noise in the eigenvalues of the matrix.
    [`Denoise`](@ref) takes one of three algorithms for the eigenvalues below the noise
    threshold. [`FixedDenoise`](@ref) replaces them with their mean. [`ShrunkDenoise`](@ref)
    keeps the diagonal of their part of the matrix and shrinks its other entries toward zero.
    [`SpectralDenoise`](@ref) sets them to zero, which leaves a nearly singular matrix, as
    section 2 shows.
  - Sparsification imposes a structure on the inverse. [`LoGo`](@ref) keeps only the entries
    that a network of the assets supports, and it builds the network from a similarity
    measure such as [`MaximumDistanceSimilarity`](@ref) or [`ExponentialSimilarity`](@ref).

You configure both through [`MatrixProcessing`](@ref), the `mp` field of
[`PortfolioOptimisersCovariance`](@ref). A prior takes that estimator in its `ce` field.

!!! tip "When to reach for this"
    Reach for denoising or sparsification when your window is short next to the number of
    assets and you run anything that reads the covariance, such as mean-variance
    optimisation, risk budgeting or clustering. A lower condition number gives a more stable
    inverse, and weights that move less when the data changes a little. Compare condition
    numbers before you settle on a technique. [`SpectralDenoise`](@ref) sets the noise
    eigenvalues to zero and leaves a nearly singular matrix.
=#

using PortfolioOptimisers, PrettyTables, LinearAlgebra

mmtfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v, digits=6))" : v
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
## 1. The data

We load the daily prices of 20 assets over the last 253 trading days, and convert them to
returns.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. Covariance estimators

We build one prior per covariance estimator and change only the `ce` field. The expected
returns stay at the sample mean. We compare the sample covariance with three denoisers and two
LoGo sparsifications.
=#

ces = ["Vanilla" => PortfolioOptimisersCovariance(),
       "FixedDenoise" => PortfolioOptimisersCovariance(;
                                                       mp = MatrixProcessing(;
                                                                             dn = Denoise(;
                                                                                          alg = FixedDenoise()))),
       "ShrunkDenoise" => PortfolioOptimisersCovariance(;
                                                        mp = MatrixProcessing(;
                                                                              dn = Denoise(;
                                                                                           alg = ShrunkDenoise(;
                                                                                                               alpha = 0.5)))),
       "SpectralDenoise" => PortfolioOptimisersCovariance(;
                                                          mp = MatrixProcessing(;
                                                                                dn = Denoise(;
                                                                                             alg = SpectralDenoise()))),
       "LoGo(MaxDist)" =>
           PortfolioOptimisersCovariance(; mp = MatrixProcessing(; alg = LoGo())),
       "LoGo(ExpDist)" => PortfolioOptimisersCovariance(;
                                                        mp = MatrixProcessing(;
                                                                              alg = LoGo(;
                                                                                         sim = ExponentialSimilarity())))]

prs = [k => prior(EmpiricalPrior(; ce = ce), rd) for (k, ce) in ces]

#=
A lower condition number means a better-posed problem. On this data `FixedDenoise` lowers it
the most, and `SpectralDenoise` raises it by many orders of magnitude. Measure it on your own
data before you trust a technique.
=#

pretty_table(DataFrame(; :estimator => [k for (k, _) in prs],
                       Symbol("cond(sigma)") => [cond(p.sigma) for (_, p) in prs]);
             formatters = [mmtfmt], title = "Covariance conditioning by estimator")

#=
## 3. Visualising the eigenspectrum

[`plot_eigenspectrum`](@ref) draws the eigenvalues of a covariance matrix as bars, with the
Marchenko-Pastur upper bound ``\lambda_+`` of that covariance. A bar above the line is larger
than noise alone would give. The denoiser does not use this line. It fits its own threshold to
the eigenvalues of the correlation matrix.
=#

using StatsPlots, GraphRecipes
# The eigenvalues of the sample covariance.
plot_eigenspectrum(prs[1].second, rd)
# The eigenvalues after `FixedDenoise`.
plot_eigenspectrum(prs[2].second, rd)
# The eigenvalues after `ShrunkDenoise`.
plot_eigenspectrum(prs[3].second, rd)
# The eigenvalues after `SpectralDenoise`.
plot_eigenspectrum(prs[4].second, rd)
# The eigenvalues after LoGo sparsification with the maximum distance similarity.
plot_eigenspectrum(prs[5].second, rd)

#=
## 4. Minimum-variance portfolios

The minimum-variance portfolio reads only the covariance, so it isolates the estimator. We solve
it with each prior and compare the weights.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

ress = [k => optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                               opt = JuMPOptimiser(; pe = p, slv = slv))) for (k, p) in prs]

pretty_table(DataFrame(["Assets" => rd.nx; [k => r.w for (k, r) in ress]]);
             formatters = [resfmt],
             title = "Minimum-variance weights by covariance estimator")

#=
We stack the same weights into one bar per covariance estimator.
=#

plot_stacked_bar_composition([r for (_, r) in ress], rd;
                             xticks = (1:length(ress), [k for (k, _) in ress]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Rewritten as a focused covariance page (ADR 0014 ex08 split): expected-returns shrinkage
#src   moved to 01_Expected_Returns_Estimation, higher moments to 03_Higher_Moment_Estimation.
#src   Only the `ce` field varies; me held at the sample mean so the comparison isolates
#src   covariance.
#src - Confirmed the SpectralDenoise caveat empirically: cond(sigma) Vanilla ≈ 177, FixedDenoise
#src   ≈ 87, but SpectralDenoise ≈ 4.6e13 (worse). Kept as a teaching point. All minimum-variance
#src   optimisations solve. Rolled up to #126.
