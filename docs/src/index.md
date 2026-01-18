```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PortfolioOptimisers.jl"
  text: Sir, this is a casino.
  tagline: 🦍🤝💪 💎🙌 🚀🌕.
  image:
    src: logo.svg
    alt: PortfolioOptimisers
  actions:
    - theme: brand
      text: Getting Started
      link: examples/01_Getting_Started
    - theme: alt
      text: View on Github
      link: https://github.com/dcelisgarza/PortfolioOptimisers.jl
    - theme: alt
      text: API
      link: api/00_Introduction

authors:
  - name: Daniel Celis Garza
    platform: github
    link: https://github.com/dcelisgarza
---

<Authors />
```

```@meta
CurrentModule = PortfolioOptimisers
```

# Welcome to PortfolioOptimisers.jl

[`PortfolioOptimisers.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl) is a package for portfolio optimisation written in Julia.

!!! danger
    
    Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

Portfolio optimisation is the science of either:

  - Minimising risk whilst keeping returns to acceptable levels.
  - Maximising returns whilst keeping risk to acceptable levels.

To some definition of acceptable, and with any number of additional constraints available to the optimisation type.

There exist myriad statistical, pre- and post-processing, optimisations, and constraints that allow one to explore a vast landscape of "optimal" portfolios.

`PortfolioOptimisers.jl` is an attempt at providing as many of these as possible under a single banner. We make extensive use of `Julia`'s type system, module extensions, and multiple dispatch to simplify development and maintenance.

For more information on the package's *vast* feature list, please check out the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction) and [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API_Introduction) docs.

## Caveat emptor

  - `PortfolioOptimisers.jl` is under active development and still in `v0.*.*`. Therefore, breaking changes should be expected with `v0.X.0` releases. All other releases will fall under `v0.X.Y`.
  - The documentation is still under construction.
  - Testing coverage is still under `95 %`. We're mainly missing assertion tests, but some lesser used features are partially or wholly untested.
  - Please feel free to submit issues, discussions and/or PRs regarding missing docs, examples, features, tests, and bugs.

## Installation

`PortfolioOptimisers.jl` is a registered package, so installation is as simple as:

```julia
julia> Pkg.add(PackageSpec(; name = "PortfolioOptimisers"))
using Pkg
```

## Quickstart

The library is quite powerful and extremely flexible. Here is what a very basic end-to-end workflow can look like. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction) contain more thorough explanations and demos. The [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API_Introduction) docs contain toy examples of the many, many features.

First we import the packages we will need for the example.

  - `StatsPlots` and `GraphRecipes` are needed to load the `Plots.jl` extension.
  - `Clarabel` and `HiGHS` are the optimisers we will use.
  - `YFinance` and `TimeSeries` for downloading and preprocessing price data.
  - `PrettyTables` and `DataFrames` for displaying the results.

```@example 0_index
# Import module and plotting extension.
using PortfolioOptimisers, StatsPlots, GraphRecipes
# Import optimisers.
using Clarabel, HiGHS
# Download data.
using YFinance, TimeSeries
# Pretty printing.
using PrettyTables, DataFrames

# Format for pretty tables.
fmt1 = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
fmt2 = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing # hide
```

For illustration purposes, we will use a set of popular meme stocks. We need to download and set the price data in a format `PortfolioOptimisers.jl` can consume.

```@example 0_index
# Function to convert prices to time array.
function stock_price_to_time_array(x)
    # Only get the keys that are not ticker or datetime.
    coln = collect(keys(x))[3:end]
    # Convert the dictionary into a matrix.
    m = hcat([x[k] for k in coln]...)
    return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
end

# Tickers to download. These are popular meme stocks, use something better.
assets = sort!(["SOUN", "RIVN", "GME", "AMC", "SOFI", "ENVX", "ANVS", "LUNR", "EOSE", "SMR",
                "NVAX", "UPST", "ACHR", "RKLB", "MARA", "LGVN", "LCID", "CHPT", "MAXN",
                "BB"])

# Prices date range.
Date_0 = "2024-01-01"
Date_1 = "2025-10-05"

# Download the price data using YFinance.
prices = get_prices.(assets; startdt = Date_0, enddt = Date_1)
prices = stock_price_to_time_array.(prices)
prices = hcat(prices...)
cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
prices = prices[cidx]
TimeSeries.rename!(prices, Symbol.(assets))
pretty_table(prices[(end - 5):end]; formatters = [fmt1])
```

Now we can compute our returns by calling [`prices_to_returns`](@ref).

```@example 0_index
# Compute the returns.
rd = prices_to_returns(prices)
```

`PortfolioOptimisers.jl` uses `JuMP` for handling the optimisation problems, which means it is solver agnostic and therefore does not ship with any pre-installed solver. [`Solver`](@ref) lets us define the optimiser factory, its solver-specific settings, and `JuMP`'s solution acceptance criteria.

```@example 0_index
# Define the continuous solver.
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
             check_sol = (; allow_local = true, allow_almost = true));
nothing # hide
```

`PortfolioOptimisers.jl` implements a number of optimisation types as estimators. All the ones which use mathematical optimisation require a [`JuMPOptimiser`]-(@ref) structure which defines general solver constraints. This structure in turn requires an instance (or vector) of [`Solver`](@ref).

```@example 0_index
opt = JuMPOptimiser(; slv = slv);
nothing # hide
```

Here we will use the traditional Mean-Risk [`MeanRisk`]-(@ref) optimsation estimator, which defaults to the Markowitz optimisation (minimum risk mean-variance optimisation).

```@example 0_index
# Vanilla (Markowitz) mean risk optimisation.
mr = MeanRisk(; opt = opt)
```

As you can see, there are *a lot* of fields in this structure, which correspond to a wide variety of optimisation constraints. We will explore these in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction). For now, we will perform the optimisation via [`optimise`]-(@ref).

```@example 0_index
# Perform the optimisation, res.w contains the optimal weights.
res = optimise(mr, rd)
```

The solution lives in the `sol` field, but the weights can be accessed via the `w` property.

`PortfolioOptimisers.jl` also has the capability to perform finite allocations, which is useful for those of us without infinite money. There are two ways to do so, a greedy algorithm [`GreedyAllocation`]-(@ref) that does not guarantee optimality but is fast and always converges, and a discrete allocation [`DiscreteAllocation`]-(@ref) which uses mixed-integer programming (MIP) and requires a capable solver.

Here we will use the latter.

```@example 0_index
# Define the MIP solver for finite discrete allocation.
mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))

# Discrete finite allocation.
da = DiscreteAllocation(; slv = mip_slv)
```

The discrete allocation minimises the absolute or relative L1- or L2-norm (configurable) between the ideal allocation to the one you can afford plus the leftover cash. As such, it needs to know a few extra things, namely the optimal weights `res.w`, a vector of latest prices `vec(values(prices[end]))`, and available cash which we define to be `4206.90`.

```@example 0_index
# Perform the finite discrete allocation, uses the final asset
# prices, and an available cash amount. This is for us mortals
# without infinite wealth.
mip_res = optimise(da, res.w, vec(values(prices[end])), 4206.90)
```

We can display the results in a table.

```@example 0_index
# View the results.
df = DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
               :opt_weights => res.w, :mip_weights => mip_res.w)
pretty_table(df; formatters = [fmt2])
```

We can also visualise the portfolio using various plotting functions. For example we can plot the portfolio's cumulative returns, in this case compound returns.

```@example 0_index
# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_ptf_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
```

We can plot the histogram of portfolio returns.

```@example 0_index
# Plot histogram of returns.
plot_histogram(mip_res.w, rd.X, slv)
```

We can plot the portfolio drawdowns, in this case compound drawdowns.

```@example 0_index
# Plot compounded drawdowns.
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
```

We can also plot the risk contribution per asset. For this, we must provide an instance of the risk measure we want to use with the appropriate statistics/parameters. We can do this by using the [`factory`](@ref) function (recommended when doing so programmatically), or manually set the quantities ourselves.

```@example 0_index
# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)
```

This awkwardness is due to the fact that `PortfolioOptimisers.jl` tries to decouple the risk measures from optimisation estimators and results. However, the advantage of this approach is that it lets us use multiple different risk measures as part of the risk expression, or as risk limits in optimisations. We explore this further in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction).

!!! info
    
    This section is under active development.

## Features

### Preprocessing

  - Prices to returns [`prices_to_returns`](@ref) and [`ReturnsResult`](@ref)
  - Find complete indices [`find_complete_indices`](@ref)
  - Find uncorrelated indices [`find_uncorrelated_indices`]-(@ref)

### Matrix Processing

  - Positive definite projection [`Posdef`](@ref), [`posdef!`](@ref), [`posdef`](@ref)

  - Denoising [`Denoise`](@ref), [`denoise!`](@ref), [`denoise`](@ref)
    
      + Spectral [`SpectralDenoise`](@ref)
      + Fixed [`FixedDenoise`](@ref)
      + Shrunk [`ShrunkDenoise`](@ref)
  - Detoning [`Detone`](@ref), [`detone!`](@ref), [`detone`](@ref)
  - Matrix processing pipeline [`DenoiseDetoneAlgMatrixProcessing`](@ref), [`matrix_processing!`](@ref), [`matrix_processing`](@ref), [`DenoiseDetoneAlg`](@ref), [`DenoiseAlgDetone`](@ref), [`DetoneDenoiseAlg`](@ref), [`DetoneAlgDenoise`](@ref), [`AlgDenoiseDetone`](@ref), [`AlgDetoneDenoise`](@ref)

### Regression Models

Factor prior models and implied volatility use [`regression`](@ref) in their estimation, which return a [`Regression`](@ref) object.

#### Regression targets

  - Linear model [`LinearModel`](@ref)
  - Generalised linear model [`GeneralisedLinearModel`](@ref)

#### Regression types

  - Stepwise [`StepwiseRegression`](@ref)
    
      + Algorithms
        
          * Forward [`Forward`](@ref)
          * Backward [`Backward`](@ref)
    
      + Selection criteria
        
          * P-value [`PValue`](@ref)
          * Akaike information criteria [`AIC`](@ref)
          * Corrected Akaike information criteria [`AICC`](@ref)
          * Bayesian information criteria [`BIC`](@ref)
          * R-squared [`RSquared`](@ref)
          * Adjusted R-squared criteria [`AdjustedRSquared`](@ref)

  - Dimensional reduction with custom mean and variance estimators [`DimensionReductionRegression`](@ref)
    
      + Dimensional reduction targets
        
          * Principal component [`PCA`](@ref)
          * Probabilistic principal component [`PPCA`](@ref)

### Moment Estimation

#### [Expected Returns](@id readme-expected-returns)

Overloads `Statistics.mean`.

  - Optionally weighted expected returns [`SimpleExpectedReturns`](@ref)

  - Equilibrium expected returns with custom covariance [`EquilibriumExpectedReturns`](@ref)
  - Excess expected returns with custom expected returns estimator [`ExcessExpectedReturns`](@ref)
  - Shrunk expected returns with custom expected returns and custom covariance estimators [`ShrunkExpectedReturns`](@ref)
    
      + Algorithms
        
          * James-Stein [`JamesStein`](@ref)
          * Bayes-Stein [`BayesStein`](@ref)
          * Bodnar-Okhrin-Parolya [`BodnarOkhrinParolya`](@ref)
    
      + Targets: all algorithms can have any of the following targets
        
          * Grand Mean [`GrandMean`](@ref)
          * Volatility Weighted [`VolatilityWeighted`](@ref)
          * Mean Squared Error [`MeanSquaredError`](@ref)
  - Standard deviation expected returns [`StandardDeviationExpectedReturns`]-(@ref)

#### [Variance and Standard Deviation](@id readme-variance)

Overloads `Statistics.var` and `Statistics.std`.

  - Optionally weighted variance with custom expected returns estimator [`SimpleVariance`](@ref)

#### [Covariance and Correlation](@id readme-covariance-correlation)

Overloads `Statistics.cov` and `Statistics.cor`.

  - Optionally weighted covariance with custom covariance estimator [`GeneralCovariance`](@ref)

  - Covariance with custom covariance estimator [`Covariance`](@ref)
    
      + Full covariance [`Full`](@ref)
      + Semi (downside) covariance [`Semi`](@ref)
  - Gerber covariances with custom variance estimator [`GerberCovariance`](@ref)
    
      + Unstandardised algorithms
        
          * Gerber 0 [`Gerber0`](@ref)
          * Gerber 1 [`Gerber1`](@ref)
          * Gerber 2 [`Gerber2`](@ref)
    
      + Standardised algorithms (Z-transforms the data beforehand) with custom expected returns estimator
        
          * Gerber 0 [`StandardisedGerber0`](@ref)
          * Gerber 1 [`StandardisedGerber1`](@ref)
          * Gerber 2 [`StandardisedGerber2`](@ref)
  - Smyth-Broby extension of Gerber covariances with custom expected returns and variance estimators [`SmythBrobyCovariance`](@ref)
    
      + Unstandardised algorithms
        
          * Smyth-Broby 0 [`SmythBroby0`](@ref)
          * Smyth-Broby 1 [`SmythBroby1`](@ref)
          * Smyth-Broby 2 [`SmythBroby2`](@ref)
          * Smyth-Broby-Gerber 0 [`SmythBrobyGerber0`](@ref)
          * Smyth-Broby-Gerber 1 [`SmythBrobyGerber1`](@ref)
          * Smyth-Broby-Gerber 2 [`SmythBrobyGerber2`](@ref)
    
      + Standardised algorithms (Z-transforms the data beforehand)
        
          * Smyth-Broby 0 [`StandardisedSmythBroby0`](@ref)
          * Smyth-Broby 1 [`StandardisedSmythBroby1`](@ref)
          * Smyth-Broby 2 [`StandardisedSmythBroby2`](@ref)
          * Smyth-Broby-Gerber 0 [`StandardisedSmythBrobyGerber0`](@ref)
          * Smyth-Broby-Gerber 1 [`StandardisedSmythBrobyGerber1`](@ref)
          * Smyth-Broby-Gerber 2 [`StandardisedSmythBrobyGerber2`](@ref)
  - Distance covariance with custom distance estimator via [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) [`DistanceCovariance`](@ref)
  - Lower Tail Dependence covariance [`LowerTailDependenceCovariance`](@ref)
  - Rank covariances
    
      + Kendall covariance [`KendallCovariance`](@ref)
      + Spearman covariance [`SpearmanCovariance`](@ref)
  - Mutual information covariance with custom variance estimator and various binning algorithms [`MutualInfoCovariance`](@ref)
    
      + [`AstroPy`](https://docs.astropy.org/en/stable/stats/ref_api.html) provided bins
        
          * Knuth's optimal bin width [`Knuth`](@ref)
          * Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
          * Scott's bin width [`Scott`](@ref)
    
      + Hacine Gharbi Ravier bin width [`HacineGharbiRavier`](@ref)
      + Predefined number of bins
  - Denoised covariance with custom covariance estimator [`DenoiseCovariance`](@ref)
  - Detoned covariance with custom covariance estimator [`DetoneCovariance`](@ref)
  - Custom processed covariance with custom covariance estimator [`ProcessedCovariance`](@ref)
  - Implied volatility with custom covariance and matrix processing estimators, and implied volatility algorithms [`ImpliedVolatility`]-(@ref)
    
      + Premium [`ImpliedVolatilityPremium`]-(@ref)
      + Regression [`ImpliedVolatilityRegression`]-(@ref)
  - Covariance with custom covariance estimator and matrix processing pipeline [`PortfolioOptimisersCovariance`](@ref)
  - Correlation covariance [`CorrelationCovariance`]-(@ref)

#### [Coskewness](@id readme-coskewness)

Implements [`coskewness`](@ref).

  - Coskewness and spectral decomposition of the negative coskewness with custom expected returns estimator and matrix processing pipeline [`Coskewness`](@ref)
    
      + Full coskewness [`Full`](@ref)
      + Semi (downside) coskewness [`Semi`](@ref)

#### [Cokurtosis](@id readme-cokurtosis)

Implements [`cokurtosis`](@ref).

  - Cokurtosis with custom expected returns estimator and matrix processing pipeline [`Cokurtosis`](@ref)
    
      + Full cokurtosis [`Full`](@ref)
      + Semi (downside) cokurtosis [`Semi`](@ref)

### Distance matrices

Implements [`distance`](@ref) and [`cor_and_dist`](@ref).

  - First order distance estimator with custom distance algorithm, and optional exponent [`Distance`](@ref)
  - Second order distance estimator with custom pairwise distance algorithm from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl), custom distance algorithm, and optional exponent [`DistanceDistance`](@ref)

The distance estimators are used together with various distance matrix algorithms.

  - Simple distance [`SimpleDistance`](@ref)

  - Simple absolute distance [`SimpleAbsoluteDistance`](@ref)
  - Logarithmic distance [`LogDistance`](@ref)
  - Correlation distance [`CorrelationDistance`](@ref)
  - Variation of Information distance with various binning algorithms [`VariationInfoDistance`](@ref)
    
      + [`AstroPy`](https://docs.astropy.org/en/stable/stats/ref_api.html) provided bins
        
          * Knuth's optimal bin width [`Knuth`](@ref)
          * Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
          * Scott's bin width [`Scott`](@ref)
    
      + Hacine Gharbi Ravier bin width [`HacineGharbiRavier`](@ref)
      + Predefined number of bins
  - Canonical distance [`CanonicalDistance`](@ref)

### Prior statistics

Many optimisations and constraints use prior statistics computed via [`prior`](@ref).

  - Low order prior [`LowOrderPrior`](@ref)
    
      + Empirical [`EmpiricalPrior`](@ref)
    
      + Factor model [`FactorPrior`](@ref)
      + Black-Litterman
        
          * Vanilla [`BlackLittermanPrior`](@ref)
          * Bayesian [`BayesianBlackLittermanPrior`](@ref)
          * Factor model [`FactorBlackLittermanPrior`](@ref)
          * Augmented [`AugmentedBlackLittermanPrior`](@ref)
      + Entropy pooling [`EntropyPoolingPrior`](@ref)
      + Opinion pooling [`OpinionPoolingPrior`](@ref)

  - High order prior [`HighOrderPrior`](@ref)
    
      + High order [`HighOrderPriorEstimator`](@ref)
      + High order factor model [`HighOrderFactorPriorEstimator`]-(@ref)

### Uncertainty sets

In order to make optimisations more robust to noise and measurement error, it is possible to define uncertainty sets on the expected returns and covariance. These can be used in optimisations which use either of these two quantities. These are implemented via [`ucs`](@ref), [`mu_ucs`](@ref), [`sigma_ucs`](@ref).

`PortfolioOptimisers.jl` implements two types of uncertainty sets.

  - [`BoxUncertaintySet`](@ref) and [`BoxUncertaintySetAlgorithm`](@ref)

  - [`EllipsoidalUncertaintySet`](@ref) and [`EllipsoidalUncertaintySetAlgorithm`](@ref) with various algorithms for computing the scaling parameter via [`k_ucs`](@ref)
    
      + [`NormalKUncertaintyAlgorithm`](@ref)
      + [`GeneralKUncertaintyAlgorithm`](@ref)
      + [`ChiSqKUncertaintyAlgorithm`](@ref)
      + Predefined scaling parameter

It also implements various estimators for the uncertainty sets, the following two can generate box and ellipsoidal sets.

  - Normally distributed returns [`NormalUncertaintySet`](@ref)

  - Bootstrapping via Autoregressive Conditional Heteroskedasticity [`ARCHUncertaintySet`](@ref) via [`arch`](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)
    
      + Circular [`CircularBootstrap`](@ref)
      + Moving [`MovingBootstrap`](@ref)
      + Stationary [`StationaryBootstrap`](@ref)

The following estimator can only generate box sets.

  - [`DeltaUncertaintySet`](@ref)

### Phylogeny

`PortfolioOptimisers.jl` can make use of asset relationships to perform optimisations, define constraints, and compute relatedness characteristics of portfolios.

#### Clustering

Phylogeny constraints and clustering optimisations make use of clustering algorithms via [`ClustersEstimator`](@ref), [`Clusters`](@ref), and [`clusterise`](@ref). Most clustering algorithms come from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

  - Automatic choice of number of clusters via [`OptimalNumberClusters`](@ref) and [`VectorToScalarMeasure`](@ref)
    
      + Second order difference [`SecondOrderDifference`](@ref)
      + Silhouette scores [`SilhouetteScore`](@ref)
      + Predefined number of clusters.

##### Hierarchical

  - Hierarchical clustering [`HClustAlgorithm`](@ref)
  - Direct Bubble Hierarchical Trees [`DBHT`](@ref) and Local Global sparsification of the covariance matrix [`LoGo`](@ref), [`logo!`](@ref), and [`logo`]-(@ref)

##### Non hierachical

Non hierarchical clustering algorithms are incompatible with hierarchical clustering optimisations, but they can be used for phylogeny constraints and [`NestedClustered`]-(@ref) optimisations.

  - K-means clustering [`KMeansAlgorithm`]-(@ref)

#### Networks

##### Adjacency matrices

Adjacency matrices encode asset relationships either with clustering or graph theory via [`phylogeny_matrix`](@ref) and [`PhylogenyResult`](@ref).

  - Network adjacency [`NetworkEstimator`](@ref) with custom tree algorithms, covariance, and distance estimators
    
      + Minimum spanning trees [`KruskalTree`](@ref), [`BoruvkaTree`](@ref), [`PrimTree`](@ref)
    
      + Triangulated Maximally Filtered Graph with various similarity matrix estimators
        
          * Maximum distance similarity [`MaximumDistanceSimilarity`](@ref)
          * Exponential similarity [`ExponentialSimilarity`](@ref)
          * General exponential similarity [`GeneralExponentialSimilarity`](@ref)

  - Clustering adjacency [`ClustersEstimator`](@ref) and [`Clusters`](@ref)

##### Centrality and phylogeny measures

  - Centrality estimator [`CentralityEstimator`](@ref) with custom adjacency matrix estimators (clustering and network) and centrality measures
    
      + Centrality measures
        
          * Betweenness [`BetweennessCentrality`](@ref)
          * Closeness [`ClosenessCentrality`](@ref)
          * Degree [`DegreeCentrality`](@ref)
          * Eigenvector [`EigenvectorCentrality`](@ref)
          * Katz [`KatzCentrality`](@ref)
          * Pagerank [`Pagerank`](@ref)
          * Radiality [`RadialityCentrality`](@ref)
          * Stress [`StressCentrality`](@ref)

  - Centrality vector [`centrality_vector`](@ref)
  - Average centrality [`average_centrality`](@ref)
  - The asset phylogeny score [`asset_phylogeny`](@ref)

### Fees

Fees are a non-negligible aspect of active investing. As such `PortfolioOptimiser.jl` has the ability to account for them in all optimisations but the naïve ones. They can also be used to adjust expected returns calculations via [`calc_fees`](@ref) and [`calc_asset_fees`](@ref).

  - Fees [`FeesEstimator`](@ref) and [`Fees`](@ref)
    
      + Proportional long
      + Proportional short
      + Fixed long
      + Fixed short
      + Turnover [`TurnoverEstimator`](@ref) and [`Turnover`](@ref)

### Portfolio optimisation

Optimisations are implemented via [`optimise`]-(@ref). Optimisations consume an estimator and return a result.

#### Naïve

These return a [`NaiveOptimisationResult`]-(@ref).

  - Inverse Volatility [`InverseVolatility`]-(@ref)
  - Equal Weighted [`EqualWeighted`]-(@ref)
  - Random (Dirichlet) [`RandomWeighted`]-(@ref)

#### Traditional

These optimisations are implemented as `JuMP` problems and make use of [`JuMPOptimiser`]-(@ref), which encodes all supported constraints.

  - Mean-Risk [`MeanRisk`]-(@ref) returns a [`MeanRiskResult`]-(@ref)

  - Factor Risk Contribution [`FactorRiskContribution`]-(@ref) returns a [`FactorRiskContributionResult`]-(@ref)
  - Near Optimal Centering [`NearOptimalCentering`]-(@ref) returns a [`NearOptimalCenteringResult`]-(@ref)
  - Asset and factor risk budgeting [`AssetRiskBudgeting`]-(@ref), [`FactorRiskBudgeting`]-(@ref)
    
      + Risk Budgeting [`RiskBudgeting`]-(@ref) returns a [`RiskBudgetingResult`]-(@ref)
    
      + Relaxed Risk Budgeting [`RelaxedRiskBudgeting`]-(@ref) returns a [`RiskBudgetingResult`]-(@ref)
        
          * Basic [`BasicRelaxedRiskBudgeting`]-(@ref)
          * Regularised [`RegularisedRelaxedRiskBudgeting`]-(@ref)
          * Regularised and penalised [`RegularisedPenalisedRelaxedRiskBudgeting`]-(@ref)

##### Traditional Optimisation Features

  - Objective functions for non risk budgeting optimisations
    
      + Minimum risk [`MinimumRisk`]-(@ref)
      + Maximum utility [`MaximumUtility`]-(@ref)
      + Maximum return over risk ratio [`MaximumRatio`]-(@ref)
      + Maximum return [`MaximumReturn`]-(@ref)
      + Custom objective penalty [`CustomJuMPObjective`]-(@ref)

  - Portfolio returns
    
      + Arithmetic returns [`ArithmeticReturn`]-(@ref)
        
          * Uncertainty set
          * Custom value
    
      + Logarithmic returns [`LogarithmicReturn`]-(@ref)
  - Regularisation penalty
    
      + L1
      + L2
  - Weight bounds
  - Budget
    
      + Long
        
          * Exact
          * Range
    
      + Short
        
          * Exact
          * Range
  - Turnover(s)
  - Tracking(s)
    
      + Returns
        
          * L1-error
          * L2-error
    
      + Risk
        
          * Independent variable
          * Dependent variable
  - Phylogeny
  - Cardinality
    
      + Asset
      + Asset group
      + Set
      + Set group
  - Buy-in threshold
  - N-dimensional Pareto fronts [`Frontier`](@ref)
    
      + Return based
      + Risk based

#### [Clustering](@id readme-clustering-opt)

  - Hierarchical Risk Parity
  - Hierarchical Equal Risk Parity
  - Schur Complementary Hierarchical Risk Parity
  - Nested Clustered

#### Ensemble

  - Stacking

#### Finite allocation

  - Discrete
  - Greedy

### Ordered weights arrays and linear moments

Some risk measures including linear moments may be formulated using ordered weights arrays.

  - Gini Mean Difference.

  - Conditional Value at Risk.
  - Weighted Conditional Value at Risk.
  - Tail Gini.
  - Worst Realisation.
  - Range.
  - Conditional Value at Risk Range.
  - Weighted Conditional Value at Risk Range.
  - Tail Gini Range.
  - Linear Moments Convex Risk Measure: linear moments can be combined using different minimisation targets.
    
      + Normalised Constant Relative Risk Aversion.
      + Minimum Squared Distance.
      + Minimum Sum Squares.

### Tracking

These can be used to track the performance of an index, indicator, or portfolio.

  - Risk measure (experimental).
  - Constraints.

There are four things that can be tracked.

  - Returns via L1 or L2 norm.
    
      + Asset weights.
      + Returns vector.

  - Risk tracking via asset weights.
    
      + Dependent variables (experimental).
      + Independent variables.

### Risk measures

Different optimisations support different risk measures, most measures can also be used to quantify a portfolio's risk-return characteristics.

  - Variance.

  - Risk Contribution Variance.
    
      + Asset risk contribution.
      + Factor risk contribution.
  - Uncertainty set variance.
  - Standard deviation.
  - First lower moment.
  - Second lower moment.
    
      + Semi variance.
      + Semi deviation.
  - Second central moment (historical returns, no covariance matrix).
    
      + Variance.
      + Standard deviation.
  - Mean absolute deviation.
  - Third lower moment (historical returns, no coskewness matrix).
    
      + Standardised (semi skewness).
      + Unstandardised.
  - Fourth lower moment (historical returns, no cokurtosis matrix).
    
      + Standardised (semi kurtosis).
      + Unstandardised.
  - Third central moment (historical returns, no coskewness matrix).
    
      + Standardised (skewness).
      + Unstandardised.
  - Fourth central moment (historical returns, no cokurtosis matrix).
    
      + Standardised (kurtosis).
      + Unstandardised.
  - Square root kurtosis.
    
      + Full.
      + Semi.
  - Negative skewness.
    
      + Full.
      + Semi (experimental).
  - Negative quadratic skewness.
    
      + Full.
      + Semi (experimental).
  - Value at Risk.
  - Conditional Value at Risk.
  - Distributionally Robust Conditional Value at Risk.
  - Entropic Value at Risk.
  - Relativistic Value at Risk.
  - Value at Risk Range.
  - Conditional Value at Risk Range.
  - Distributionally Robust Conditional Value at Risk Range.
  - Entropic Value at Risk Range.
  - Relativistic Value at Risk Range.
  - Drawdown at Risk.
    
      + Absolute (simple returns).
      + Relative (compounded returns).
  - Conditional Drawdown at Risk.
    
      + Absolute (simple returns).
      + Relative (compounded returns).
  - Entropic Drawdown at Risk.
    
      + Absolute (simple returns).
      + Relative (compounded returns).
  - Relativistic Drawdown at Risk.
    
      + Absolute (simple returns).
      + Relative (compounded returns).
  - Ordered Weights Array risk measure.
  - Ordered Weights Array range risk measure.
  - Average Drawdown.
  - Ulcer Index.
  - Maximum Drawdown.
  - Brownian Distance Variance.
  - Worst Realisation.
  - Range.
  - Equal risk.
  - Turnover risk.
  - Tracking risk.
  - Mean return risk.
  - Ratio of measures.

### Portfolio statistics

These are used to summarise a portfolio's risk and return characteristics.

  - Expected returns.
    
      + Arithmetic.
      + Logarithmic.

  - Risk-adjusted return ratio.
    
      + Vanilla.
      + Sharpe ratio information criterion.
  - Risk contribution.
    
      + Asset risk contribution.
      + Factor risk contribution.

### Optimisation constraints

Many of these use the various constraint generation mechanisms mentioned above. These constrain the optimisation so the results meet the user's requirements. Some have specific requirements like a Mixed Integer Programming capable solver, others cannot be used in conjunction with each other, and there exist combinations that make problems infeasible.

  - `JuMP`-based.
    
      + Risk constraints.
        
          * Maximum risk for all supported measures (can be simultaneously provided).
    
      + Return constraints.
        
          * Minimum return.
          * Expected return uncertainty set.
      + Pareto front/surface/hypersurface (efficient frontier 2D, 3D, ND).
        
          * Via risk constraints.
          * Via return constraints.
      + Objective functions.
        
          * Minimum risk.
          * Maximum utility.
          * Maximum risk-adjusted return ratio.
          * Maximum return.
      + Budget constraints.
        
          * Long and/or short budget.
            
              - Exact.
              - Upper and lower bounds.
        
          * Cost budget.
          * Market impact budget.
      + Weight bounds.
      + Linear weights.
      + Cardinality.
        
          * Asset.
          * Set.
      + Group cardinality.
        
          * Asset.
          * Set.
      + Long and short buy-in threshold.
      + Turnover.
      + Fees.
      + Tracking.
      + Phylogeny.
      + Centrality.
      + Regularisation.
        
          * L1.
          * L2.
      + Custom: via subtyping and multiple dispatch.
        
          * Constraint.
          * Objective.
          * Objective penalty.

  - Non-`JuMP`-based.
    
      + Weight bounds.
        
          * Upper.
          * Lower.
    
      + Weight finaliser.
  - Optimisers without a fixed risk measure.
    
      + Scalarisers for multiple simultaneous risk measures.
        
          * Weighted sum.
          * Max risk.
          * LogSumExp.

### Plotting

  - Simple or compound cumulative returns.
    
      + Portfolio.
      + Assets.

  - Portfolio composition.
    
      + Single portfolio.
    
      + Multi portfolio.
        
          * Stacked bar.
          * Stacked area.
  - Risk contribution.
    
      + Asset risk contribution.
      + Factor risk contribution.
  - Asset dendrogram.
  - Asset clusters + optional dendrogram.
  - Simple or compound drawdowns.
  - Portfolio returns histogram + density.
  - 2/3D risk measure scatter plots.
