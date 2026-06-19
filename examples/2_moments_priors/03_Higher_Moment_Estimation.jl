#=
# Higher moment estimation

Mean–variance optimisation only looks at the first two moments. But asset returns are skewed
and fat-tailed, and risk measures like [`NegativeSkewness`](@ref) and [`Kurtosis`](@ref) need
estimates of the **coskewness** and **cokurtosis** tensors to capture that. These high-order
moments are even harder to estimate than the covariance: the cokurtosis matrix is
``N^2 \times N^2``, so with a short window it is wildly over-parametrised and numerically
near-singular. As with the covariance, **denoising and sparsification** rescue it.

In `PortfolioOptimisers` the high-order moments live in a [`HighOrderPriorEstimator`](@ref),
which wraps a low-order prior and adds a [`Coskewness`](@ref) (`ske`) and [`Cokurtosis`](@ref)
(`kte`) estimator. Each accepts the same [`MatrixProcessing`](@ref) pipeline — [`Denoise`](@ref)
and [`LoGo`](@ref) — that we apply to covariances.

!!! tip "When to reach for this"
    Reach for high-order moment estimation whenever you optimise against a skew- or
    tail-sensitive risk measure ([`NegativeSkewness`](@ref), [`Kurtosis`](@ref), and the
    square-root variants), or build a Pareto surface over them. And reach for *denoised*
    high-order moments essentially always when you do: the raw cokurtosis on a short window is
    numerically singular, so denoising is what makes these optimisations well-posed rather than
    a nicety. If you only use variance/tail measures that need no tensors, skip it.
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

We build three high-order priors that differ only in how the coskewness and cokurtosis are
processed: raw (vanilla), [`FixedDenoise`](@ref)d, and [`LoGo`](@ref)-sparsified. We then
compare the condition numbers of the coskewness negative-spectral-slice matrix `V` and the
cokurtosis matrix `kt`.
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
The condition numbers tell the whole story. The raw cokurtosis is numerically singular (a
condition number of order ``10^{15}``); denoising and sparsification bring it down by many
orders of magnitude, turning an ill-posed optimisation into a stable one.
=#

pretty_table(DataFrame(; :estimator => [k for (k, _) in prs],
                       Symbol("cond(V) coskew") => [cond(p.V) for (_, p) in prs],
                       Symbol("cond(kt) cokurt") => [cond(p.kt) for (_, p) in prs]);
             formatters = [hmmtfmt], title = "High-order moment conditioning")

#=
## 3. Visualising the high-order moments

The coskewness and cokurtosis heatmaps show the denoising at work — the raw matrices are dense
and noisy, the processed ones cleaner and better conditioned.
=#

# Coskewness heatmap: vanilla vs denoised.
using StatsPlots, GraphRecipes
# Coskewness heatmap: denoised.
plot_coskewness(prs[1].second, rd)
# Cokurtosis eigenspectrum: vanilla vs denoised.
plot_coskewness(prs[2].second, rd)
# Cokurtosis eigenspectrum: denoised.
plot_cokurtosis(prs[1].second, rd)
plot_cokurtosis(prs[2].second, rd)

#=
## 4. Why it matters: skew- and tail-aware optimisation

We minimise two high-order risk measures — [`NegativeSkewness`](@ref) and [`Kurtosis`](@ref) —
using each high-order prior. With the raw (near-singular) tensors the solver is working against
a degenerate problem; the denoised and sparsified priors give stable, sensible allocations.
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
The composition plot contrasts the negative skew-minimising portfolios across the three priors.
=#

plot_stacked_bar_composition([r for (_, r) in ress_sk], rd;
                             xticks = (1:length(ress_sk), [k for (k, _) in ress_sk]))

#=
The composition plot contrasts the kurtosis-minimising portfolios across the three priors.
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
