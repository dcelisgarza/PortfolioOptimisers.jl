The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/2_moments_priors/02_Covariance_Estimation.jl"
```

# Covariance estimation

The covariance matrix is the second moment that almost every optimiser depends on, and on a
short window it is badly estimated: with ``N`` assets and only a little more than ``N``
observations, the sample covariance is noisy and ill-conditioned, and inverting it (as
mean–variance implicitly does) amplifies that noise. Two families of fixes help:

- **Denoising** — separate signal from noise in the eigenspectrum. [`Denoise`](@ref) offers
    [`FixedDenoise`](@ref) (collapse the sub-threshold bulk), [`ShrunkDenoise`](@ref) (shrink
    it) and [`SpectralDenoise`](@ref) (zero it) — the last does not always lower the condition
    number, as we will see.
- **Sparsification** — impose a relationship structure on the inverse. [`LoGo`](@ref) keeps
    only the entries justified by the network topology, using a similarity measure such as
    [`MaximumDistanceSimilarity`](@ref) or [`ExponentialSimilarity`](@ref).

Both are configured through the [`MatrixProcessing`](@ref) pipeline on a
[`PortfolioOptimisersCovariance`](@ref), which is the `ce` field of a prior.

!!! tip "When to reach for this"
    Reach for covariance denoising/sparsification whenever your estimation window is short
    relative to the number of assets and you run anything that leans on the covariance —
    mean–variance, risk budgeting, clustering. A lower condition number means a more stable
    inverse and weights that move less when the data wobbles. Compare condition numbers before
    committing to a technique: [`SpectralDenoise`](@ref) in particular can make conditioning
    *worse* on some data.

````@example 02_Covariance_Estimation
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
nothing #hide
````

## 1. ReturnsResult data

We use the same S&P 500 slice as the other examples.

````@example 02_Covariance_Estimation
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
````

## 2. Covariance estimators

We build one prior per covariance estimator, varying **only** the `ce` field (the expected
returns are held at the plain sample mean). We compare the vanilla sample covariance against
three denoisers and two LoGo sparsifications.

````@example 02_Covariance_Estimation
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
````

The condition number is our headline diagnostic — lower is better-posed. FixedDenoise gives
the biggest improvement here, while SpectralDenoise actually makes conditioning dramatically
*worse* on this data, a reminder to always measure rather than assume.

````@example 02_Covariance_Estimation
pretty_table(DataFrame(; :estimator => [k for (k, _) in prs],
                       Symbol("cond(sigma)") => [cond(p.sigma) for (_, p) in prs]);
             formatters = [mmtfmt], title = "Covariance conditioning by estimator")
````

## 3. Visualising the eigenspectrum

[`plot_eigenspectrum`](@ref) shows the Marchenko–Pastur ``\\lambda_+`` threshold: bars above it
carry signal, bars below are noise. Denoising acts on the sub-threshold bulk.

Eigenspectrum: vanilla sample covariance.

````@example 02_Covariance_Estimation
using StatsPlots, GraphRecipes
````

Eigenspectrum: fixed-denoised covariance.

````@example 02_Covariance_Estimation
plot_eigenspectrum(prs[1].second, rd)
````

Eigenspectrum: LoGo(MaxDist) sparsified covariance.

````@example 02_Covariance_Estimation
plot_eigenspectrum(prs[2].second, rd)
plot_eigenspectrum(prs[5].second, rd)
````

## 4. Why it matters: minimum-variance optimisation

Minimum-variance is the optimisation most exposed to covariance conditioning. We solve it with
each prior and compare the weights — better-conditioned estimators produce more stable, less
concentrated allocations.

````@example 02_Covariance_Estimation
using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

ress = [k => optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                               opt = JuMPOptimiser(; pe = p, slv = slv))) for (k, p) in prs]

pretty_table(DataFrame(["Assets" => rd.nx; [k => r.w for (k, r) in ress]]);
             formatters = [resfmt],
             title = "Minimum-variance weights by covariance estimator")
````

The composition plot contrasts the minimum-variance portfolios across estimators.

````@example 02_Covariance_Estimation
plot_stacked_bar_composition([r for (_, r) in ress], rd;
                             xticks = (1:length(ress), [k for (k, _) in ress]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
