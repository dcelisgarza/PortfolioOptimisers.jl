#=
```@meta
Description = "Coskewness and cokurtosis estimation in PortfolioOptimisers.jl, the higher moments that skewness and kurtosis risk measures need."
```

# Higher moment estimation

Mean-variance optimisation reads only the first two moments of the returns. Asset returns are
skewed and have fat tails, and risk measures such as [`NegativeSkewness`](@ref) and
[`Kurtosis`](@ref) need estimates of the coskewness and cokurtosis tensors to account for
that. These higher moments are harder to estimate than the covariance. The cokurtosis matrix is
``N^2 \times N^2``, so on a short window it has far more entries than the window has
observations, and it is close to singular. Denoising and sparsification help here, as they do
for the covariance.

[`HighOrderPriorEstimator`](@ref) computes the higher moments. It wraps a prior for the mean
and the covariance, and adds a [`Coskewness`](@ref) estimator in `ske` and a
[`Cokurtosis`](@ref) estimator in `kte`. Each takes the same [`MatrixProcessing`](@ref), with
[`Denoise`](@ref) and [`LoGo`](@ref), that the [covariance page](02_Covariance_Estimation.md)
uses.

!!! tip "When to reach for this"
    Reach for higher moment estimation when you optimise against a risk measure that reads the
    skew or the tails, such as [`NegativeSkewness`](@ref), [`Kurtosis`](@ref) and their
    square-root forms, or when you build a Pareto surface over them. When you do, denoise the
    higher moments. The raw cokurtosis of a short window is numerically singular, and
    denoising is what makes these optimisations well-posed. If your risk measures need no
    tensors, you do not need this page.
=#

using PortfolioOptimisers, PrettyTables, LinearAlgebra

hmmtfmt = (v, i, j) -> begin
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
## 1. ReturnsResult data

We use the same S&P 500 slice as the other examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. High-order priors

We build three high-order priors that differ only in how they process the coskewness and the
cokurtosis. The first keeps the raw estimates. The second applies [`FixedDenoise`](@ref). The
third applies the default [`Denoise`](@ref) and then [`LoGo`](@ref) sparsification. We then
compare the condition numbers of `V`, the matrix of the negative spectral slices of the
coskewness, and of `kt`, the cokurtosis matrix.
=#

hopes = ["Vanilla" => HighOrderPriorEstimator(),
         "Denoise" => HighOrderPriorEstimator(;
                                              ske = Coskewness(;
                                                               mp = MatrixProcessing(;
                                                                                     dn = Denoise(;
                                                                                                  alg = FixedDenoise()))),
                                              kte = Cokurtosis(;
                                                               mp = MatrixProcessing(;
                                                                                     dn = Denoise(;
                                                                                                  alg = FixedDenoise())))),
         "LoGo" => HighOrderPriorEstimator(;
                                           ske = Coskewness(;
                                                            mp = MatrixProcessing(;
                                                                                  dn = Denoise(),
                                                                                  alg = LoGo())),
                                           kte = Cokurtosis(;
                                                            mp = MatrixProcessing(;
                                                                                  dn = Denoise(),
                                                                                  alg = LoGo())))]

prs = [k => prior(pe, rd) for (k, pe) in hopes]

#=
A condition number near ``10^{15}`` means that the matrix is numerically singular, and an
optimisation that reads it is ill-posed. In the table, compare the raw cokurtosis with the
denoised and the sparsified ones.
=#

pretty_table(DataFrame(; :estimator => [k for (k, _) in prs],
                       Symbol("cond(V) coskew") => [cond(p.V) for (_, p) in prs],
                       Symbol("cond(kt) cokurt") => [cond(p.kt) for (_, p) in prs]);
             formatters = [hmmtfmt], title = "High-order moment conditioning")

#=
## 3. Visualising the high-order moments

[`plot_coskewness`](@ref) draws the coskewness matrix as a heatmap, and
[`plot_cokurtosis`](@ref) draws the eigenvalues of the cokurtosis matrix. We draw each for the
raw prior and for the denoised prior. The raw matrices are dense and noisy, and the denoised
ones are cleaner.
=#

using StatsPlots, GraphRecipes
# The coskewness of the raw prior.
plot_coskewness(prs[1].second, rd)
# The coskewness of the denoised prior.
plot_coskewness(prs[2].second, rd)
# The eigenvalues of the raw cokurtosis.
plot_cokurtosis(prs[1].second, rd)
# The eigenvalues of the denoised cokurtosis.
plot_cokurtosis(prs[2].second, rd)

#=
## 4. Minimum skewness and minimum kurtosis portfolios

We minimise two higher moment risk measures, [`NegativeSkewness`](@ref) and
[`Kurtosis`](@ref), with each high-order prior. The raw tensors are close to singular, so with
them the solver works on a degenerate problem. The denoised and sparsified priors give it a
well-posed one.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

ress_sk = [k => optimise(MeanRisk(; r = NegativeSkewness(), obj = MinimumRisk(),
                                  opt = JuMPOptimiser(; pe = p, slv = slv)))
           for (k, p) in prs]
ress_kt = [k => optimise(MeanRisk(; r = Kurtosis(), obj = MinimumRisk(),
                                  opt = JuMPOptimiser(; pe = p, slv = slv)))
           for (k, p) in prs]

pretty_table(DataFrame(["Assets" => rd.nx;
                        ["NSkew $k" => r.w for (k, r) in ress_sk];
                        ["Kurt $k" => r.w for (k, r) in ress_kt]]); formatters = [resfmt],
             title = "Minimum skew / kurtosis weights by prior")

#=
We stack the weights that minimise the negative skewness into one bar per prior.
=#

plot_stacked_bar_composition([r for (_, r) in ress_sk], rd;
                             xticks = (1:length(ress_sk), [k for (k, _) in ress_sk]))

#=
We do the same for the weights that minimise the kurtosis.
=#

plot_stacked_bar_composition([r for (_, r) in ress_kt], rd;
                             xticks = (1:length(ress_kt), [k for (k, _) in ress_kt]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end (split from ex08, focused on the high-order tensors). Headline
#src   result lands cleanly: vanilla cond(kt) ≈ 1.9e15 (numerically singular) vs ≈ 1.8e4 after
#src   FixedDenoise — a textbook motivation for denoising the cokurtosis.
#src - All NegativeSkewness and Kurtosis MinimumRisk optimisations solve to OptimisationSuccess
#src   even with the vanilla near-singular tensor (Clarabel copes here), but the denoised priors
#src   are the well-posed choice. Rolled up to #126.
