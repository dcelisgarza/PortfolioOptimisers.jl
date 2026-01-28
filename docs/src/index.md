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

!!! Danger
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
julia> using Pkg

julia> Pkg.add(PackageSpec(; name = "PortfolioOptimisers"))
```

## Quick-start

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
             check_sol = (; allow_local = true, allow_almost = true))
```

`PortfolioOptimisers.jl` implements a number of optimisation types as estimators. All the ones which use mathematical optimisation require a [`JuMPOptimiser`]-(@ref) structure which defines general solver constraints. This structure in turn requires an instance (or vector) of [`Solver`](@ref).

```@example 0_index
opt = JuMPOptimiser(; slv = slv);
nothing # hide
```

Here we will use the traditional Mean-Risk [`MeanRisk`]-(@ref) optimisation estimator, which defaults to the Markowitz optimisation (minimum risk mean-variance optimisation).

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

The discrete allocation minimises the absolute or relative L1- or L2-norm (configurable) between the ideal allocation to the one you can afford plus the leftover cash. As such, it needs to know a few extra things, namely the optimal weights `res.w`, a vector of the latest prices `vec(values(prices[end]))`, and available cash which we define to be `4206.90`.

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

We can also visualise the portfolio using various plotting functions. For example, we can plot the portfolio's cumulative returns, in this case compound returns.

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

Furthermore, we can also plot the risk contribution per asset. For this, we must provide an instance of the risk measure we want to use with the appropriate statistics/parameters. We can do this by using the [`factory`](@ref) function (recommended when doing so programmatically), or manually set the quantities ourselves.

```@example 0_index
# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)
```

This awkwardness is due to the fact that `PortfolioOptimisers.jl` tries to decouple the risk measures from optimisation estimators and results. However, the advantage of this approach is that it lets us use multiple different risk measures as part of the risk expression, or as risk limits in optimisations. We explore this further in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction).

We can also plot the returns' histogram and probability density.

```@example 0_index
plot_histogram(mip_res.w, rd.X, slv)
```

We can also plot the compounded or uncompounded drawdowns, here we plot the former.

```@example 0_index
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
```

There are other kinds of plots which we explore in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction).

## Features

This section is under active development and any [`<name>`]-(@ref) lack docstrings. Some docstrings are also outdated, please refer to [Issue #58](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/58) for details on what docstrings have been completed in the `dev` branch.

### Preprocessing

- Prices to returns [`prices_to_returns`](@ref) and [`ReturnsResult`](@ref)
- Find complete indices [`find_complete_indices`](@ref)
- Find uncorrelated indices [`find_uncorrelated_indices`]-(@ref)

### Matrix processing

- Positive definite projection [`Posdef`](@ref), [`posdef!`](@ref), [`posdef`](@ref)
- ::: details Denoising [`Denoise`](@ref), [`denoise!`](@ref), [`denoise`](@ref)
  - Spectral [`SpectralDenoise`](@ref)
  - Fixed [`FixedDenoise`](@ref)
  - Shrunk [`ShrunkDenoise`](@ref)
- Detoning [`Detone`](@ref), [`detone!`](@ref), [`detone`](@ref)
- Matrix processing pipeline [`DenoiseDetoneAlgMatrixProcessing`](@ref), [`matrix_processing!`](@ref), [`matrix_processing`](@ref), [`DenoiseDetoneAlg`](@ref), [`DenoiseAlgDetone`](@ref), [`DetoneDenoiseAlg`](@ref), [`DetoneAlgDenoise`](@ref), [`AlgDenoiseDetone`](@ref), [`AlgDetoneDenoise`](@ref)

### Regression models

Factor prior models and implied volatility use [`regression`](@ref) in their estimation, which return a [`Regression`](@ref) object.

#### Regression targets

- Linear model [`LinearModel`](@ref)
- Generalised linear model [`GeneralisedLinearModel`](@ref)

#### Regression types

- ::: details Stepwise [`StepwiseRegression`](@ref)
  - ::: details Algorithms
    - Forward [`Forward`](@ref)
    - Backward [`Backward`](@ref)
  - ::: details Selection criteria
    - P-value [`PValue`](@ref)
    - Akaike information criteria [`AIC`](@ref)
    - Corrected Akaike information criteria [`AICC`](@ref)
    - Bayesian information criteria [`BIC`](@ref)
    - R-squared [`RSquared`](@ref)
    - Adjusted R-squared criteria [`AdjustedRSquared`](@ref)
- :::details Dimensional reduction with custom mean and variance estimators [`DimensionReductionRegression`](@ref)
  - Principal component [`PCA`](@ref)
  - Probabilistic principal component [`PPCA`](@ref)

### Moment estimation

#### [Expected returns](@id readme-expected-returns)

Overloads `Statistics.mean`.

- Optionally weighted expected returns [`SimpleExpectedReturns`](@ref)
- Equilibrium expected returns with custom covariance [`EquilibriumExpectedReturns`](@ref)
- Excess expected returns with custom expected returns estimator [`ExcessExpectedReturns`](@ref)
- ::: details Shrunk expected returns with custom expected returns and custom covariance estimators [`ShrunkExpectedReturns`](@ref)
  - ::: details Algorithms
    - James-Stein [`JamesStein`](@ref)
    - Bayes-Stein [`BayesStein`](@ref)
    - Bodnar-Okhrin-Parolya [`BodnarOkhrinParolya`](@ref)
  - ::: details Targets: all algorithms can have any of the following targets
    - Grand Mean [`GrandMean`](@ref)
    - Volatility Weighted [`VolatilityWeighted`](@ref)
    - Mean Squared Error [`MeanSquaredError`](@ref)
- Standard deviation expected returns [`StandardDeviationExpectedReturns`]-(@ref)

#### [Variance and standard deviation](@id readme-variance)

Overloads `Statistics.var` and `Statistics.std`.

- Optionally weighted variance with custom expected returns estimator [`SimpleVariance`](@ref)

#### [Covariance and correlation](@id readme-covariance-correlation)

Overloads `Statistics.cov` and `Statistics.cor`.

- Optionally weighted covariance with custom covariance estimator [`GeneralCovariance`](@ref)
- ::: details Covariance with custom covariance estimator [`Covariance`](@ref)
  - Full [`Full`](@ref)
  - Semi [`Semi`](@ref)
- ::: details Gerber covariances with custom variance estimator [`GerberCovariance`](@ref)
  - ::: details Unstandardised algorithms
    - Gerber 0 [`Gerber0`](@ref)
    - Gerber 1 [`Gerber1`](@ref)
    - Gerber 2 [`Gerber2`](@ref)
  - ::: details Standardised algorithms (Z-transforms the data beforehand) with custom expected returns estimator
    - Gerber 0 [`StandardisedGerber0`](@ref)
    - Gerber 1 [`StandardisedGerber1`](@ref)
    - Gerber 2 [`StandardisedGerber2`](@ref)
- ::: details Smyth-Broby extension of Gerber covariances with custom expected returns and variance estimators [`SmythBrobyCovariance`](@ref)
  - ::: details Unstandardised algorithms
    - Smyth-Broby 0 [`SmythBroby0`](@ref)
    - Smyth-Broby 1 [`SmythBroby1`](@ref)
    - Smyth-Broby 2 [`SmythBroby2`](@ref)
    - Smyth-Broby-Gerber 0 [`SmythBrobyGerber0`](@ref)
    - Smyth-Broby-Gerber 1 [`SmythBrobyGerber1`](@ref)
    - Smyth-Broby-Gerber 2 [`SmythBrobyGerber2`](@ref)
  - ::: details Standardised algorithms (Z-transforms the data beforehand)
    - Smyth-Broby 0 [`StandardisedSmythBroby0`](@ref)
    - Smyth-Broby 1 [`StandardisedSmythBroby1`](@ref)
    - Smyth-Broby 2 [`StandardisedSmythBroby2`](@ref)
    - Smyth-Broby-Gerber 0 [`StandardisedSmythBrobyGerber0`](@ref)
    - Smyth-Broby-Gerber 1 [`StandardisedSmythBrobyGerber1`](@ref)
    - Smyth-Broby-Gerber 2 [`StandardisedSmythBrobyGerber2`](@ref)
- Distance covariance with custom distance estimator via [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) [`DistanceCovariance`](@ref)
- Lower Tail Dependence covariance [`LowerTailDependenceCovariance`](@ref)
- ::: details Rank covariances
  - Kendall covariance [`KendallCovariance`](@ref)
  - Spearman covariance [`SpearmanCovariance`](@ref)
- ::: details Mutual information covariance with custom variance estimator and various binning algorithms [`MutualInfoCovariance`](@ref)
  - ::: details [`AstroPy`](https://docs.astropy.org/en/stable/stats/ref_api.html)-defined bins
    - Knuth's optimal bin width [`Knuth`](@ref)
    - Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
    - Scott's bin width [`Scott`](@ref)
  - Hacine-Gharbi-Ravier bin width [`HacineGharbiRavier`](@ref)
  - Predefined number of bins
- Denoised covariance with custom covariance estimator [`DenoiseCovariance`](@ref)
- Detoned covariance with custom covariance estimator [`DetoneCovariance`](@ref)
- Custom processed covariance with custom covariance estimator [`ProcessedCovariance`](@ref)
- ::: details Implied volatility with custom covariance and matrix processing estimators, and implied volatility algorithms [`ImpliedVolatility`]-(@ref)
  - Premium [`ImpliedVolatilityPremium`]-(@ref)
  - Regression [`ImpliedVolatilityRegression`]-(@ref)
- Covariance with custom covariance estimator and matrix processing pipeline [`PortfolioOptimisersCovariance`](@ref)
- Correlation covariance [`CorrelationCovariance`]-(@ref)

#### [Coskewness](@id readme-coskewness)

Implements [`coskewness`](@ref).

- ::: details Coskewness and spectral decomposition of the negative coskewness with custom expected returns estimator and matrix processing pipeline [`Coskewness`](@ref)
  - Full [`Full`](@ref)
  - Semi [`Semi`](@ref)

#### [Cokurtosis](@id readme-cokurtosis)

Implements [`cokurtosis`](@ref).

- ::: details Cokurtosis with custom expected returns estimator and matrix processing pipeline [`Cokurtosis`](@ref)
  - Full [`Full`](@ref)
  - Semi [`Semi`](@ref)

### Distance matrices

Implements [`distance`](@ref) and [`cor_and_dist`](@ref).

- First order distance estimator with custom distance algorithm, and optional exponent [`Distance`](@ref)
- Second order distance estimator with custom pairwise distance algorithm from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl), custom distance algorithm, and optional exponent [`DistanceDistance`](@ref)

The distance estimators are used together with various distance matrix algorithms.

- Simple distance [`SimpleDistance`](@ref)
- Simple absolute distance [`SimpleAbsoluteDistance`](@ref)
- Logarithmic distance [`LogDistance`](@ref)
- Correlation distance [`CorrelationDistance`](@ref)
- ::: details Variation of Information distance with various binning algorithms [`VariationInfoDistance`](@ref)
  - ::: details [`AstroPy`](https://docs.astropy.org/en/stable/stats/ref_api.html)-defined bins
    - Knuth's optimal bin width [`Knuth`](@ref)
    - Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
    - Scott's bin width [`Scott`](@ref)
  - Hacine-Gharbi-Ravier bin width [`HacineGharbiRavier`](@ref)
  - Predefined number of bins
- Canonical distance [`CanonicalDistance`](@ref)

### Phylogeny

`PortfolioOptimisers.jl` can make use of asset relationships to perform optimisations, define constraints, and compute relatedness characteristics of portfolios.

#### Clustering

Phylogeny constraints and clustering optimisations make use of clustering algorithms via [`ClustersEstimator`](@ref), [`Clusters`](@ref), and [`clusterise`](@ref). Most clustering algorithms come from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

- ::: details Automatic choice of number of clusters via [`OptimalNumberClusters`](@ref) and [`VectorToScalarMeasure`](@ref)
  - Second order difference [`SecondOrderDifference`](@ref)
  - Silhouette scores [`SilhouetteScore`](@ref)
  - Predefined number of clusters.

##### Hierarchical

- Hierarchical clustering [`HClustAlgorithm`](@ref)
- Direct Bubble Hierarchical Trees [`DBHT`](@ref) and Local Global sparsification of the covariance matrix [`LoGo`](@ref), [`logo!`](@ref), and [`logo`]-(@ref)

##### Non-hierarchical

Non-hierarchical clustering algorithms are incompatible with hierarchical clustering optimisations, but they can be used for phylogeny constraints and [`NestedClustered`]-(@ref) optimisations.

- K-means clustering [`KMeansAlgorithm`]-(@ref)

#### Networks

##### Adjacency matrices

Adjacency matrices encode asset relationships either with clustering or graph theory via [`phylogeny_matrix`](@ref) and [`PhylogenyResult`](@ref).

- ::: details Network adjacency [`NetworkEstimator`](@ref) with custom tree algorithms, covariance, and distance estimators
  - Minimum spanning trees [`KruskalTree`](@ref), [`BoruvkaTree`](@ref), [`PrimTree`](@ref)
  - ::: details Triangulated Maximally Filtered Graph with various similarity matrix estimators
    - Maximum distance similarity [`MaximumDistanceSimilarity`](@ref)
    - Exponential similarity [`ExponentialSimilarity`](@ref)
    - General exponential similarity [`GeneralExponentialSimilarity`](@ref)
- Clustering adjacency [`ClustersEstimator`](@ref) and [`Clusters`](@ref)

##### Centrality and phylogeny measures

- ::: details Centrality estimator [`CentralityEstimator`](@ref) with custom adjacency matrix estimators (clustering and network) and centrality measures
  - Betweenness [`BetweennessCentrality`](@ref)
  - Closeness [`ClosenessCentrality`](@ref)
  - Degree [`DegreeCentrality`](@ref)
  - Eigenvector [`EigenvectorCentrality`](@ref)
  - Katz [`KatzCentrality`](@ref)
  - Pagerank [`Pagerank`](@ref)
  - Radiality [`RadialityCentrality`](@ref)
  - Stress [`StressCentrality`](@ref)
- Centrality vector [`centrality_vector`](@ref)
- Average centrality [`average_centrality`](@ref)
- The asset phylogeny score [`asset_phylogeny`](@ref)

### Optimisation constraints

Non clustering optimisers support a wide range of constraints, while naive and clustering optimisers only support weight bounds. Furthermore, entropy pooling prior supports a variety of views constraints. It is therefore important to provide users with the ability to generate constraints manually and/or programmatically. We therefore provide a wide, robust, and extensible range of types such as [`AbstractEstimatorValueAlgorithm`](@ref) and [`UniformValues`](@ref), and functions that make this easy, fast, and safe.

Constraints can be defined via their estimators or directly by their result types. Some using estimators need to map key-value pairs to the asset universe, this is done by defining the assets and asset groups in [`AssetSets`](@ref). Internally, `PortfolioOptimisers.jl` uses all the information and calls [`group_to_val!`](@ref), and [`replace_group_by_assets`](@ref) to produce the appropriate arrays.

- Equation parsing [`parse_equation`](@ref) and [`ParsingResult`](@ref).
- Linear constraints [`linear_constraints`](@ref), [`LinearConstraintEstimator`](@ref), [`PartialLinearConstraint`](@ref), and [`LinearConstraint`](@ref)
- Risk budgeting constraints [`risk_budget_constraints`](@ref), [`RiskBudgetEstimator`](@ref), and [`RiskBudget`](@ref)
- Phylogeny constraints [`phylogeny_constraints`](@ref), [`centrality_constraints`](@ref), [`SemiDefinitePhylogenyEstimator`](@ref), [`SemiDefinitePhylogeny`](@ref), [`IntegerPhylogenyEstimator`](@ref), [`IntegerPhylogeny`](@ref), [`CentralityConstraint`](@ref)
- Weight bounds constraints [`weight_bounds_constraints`](@ref), [`WeightBoundsEstimator`](@ref), [`WeightBounds`](@ref)
- Asset set matrices [`asset_sets_matrix`](@ref) and [`AssetSetsMatrixEstimator`](@ref)
- Threshold constraints [`threshold_constraints`](@ref), [`ThresholdEstimator`](@ref), and [`Threshold`](@ref)

### Prior statistics

Many optimisations and constraints use prior statistics computed via [`prior`](@ref).

- ::: details Low order prior [`LowOrderPrior`](@ref)
  - Empirical [`EmpiricalPrior`](@ref)
  - Factor model [`FactorPrior`](@ref)
  - ::: details Black-Litterman
    - Vanilla [`BlackLittermanPrior`](@ref)
    - Bayesian [`BayesianBlackLittermanPrior`](@ref)
    - Factor model [`FactorBlackLittermanPrior`](@ref)
    - Augmented [`AugmentedBlackLittermanPrior`](@ref)
  - Entropy pooling [`EntropyPoolingPrior`](@ref)
  - Opinion pooling [`OpinionPoolingPrior`](@ref)
- ::: details High order prior [`HighOrderPrior`](@ref)
  - High order [`HighOrderPriorEstimator`](@ref)
  - High order factor model [`HighOrderFactorPriorEstimator`]-(@ref)

### Uncertainty sets

In order to make optimisations more robust to noise and measurement error, it is possible to define uncertainty sets on the expected returns and covariance. These can be used in optimisations which use either of these two quantities. These are implemented via [`ucs`](@ref), [`mu_ucs`](@ref), and [`sigma_ucs`](@ref).

`PortfolioOptimisers.jl` implements two types of uncertainty sets.

- [`BoxUncertaintySet`](@ref) and [`BoxUncertaintySetAlgorithm`](@ref)
- ::: details [`EllipsoidalUncertaintySet`](@ref) and [`EllipsoidalUncertaintySetAlgorithm`](@ref) with various algorithms for computing the scaling parameter via [`k_ucs`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - Predefined scaling parameter

It also implements various estimators for the uncertainty sets, the following two can generate box and ellipsoidal sets.

- Normally distributed returns [`NormalUncertaintySet`](@ref)
- ::: details Bootstrapping via Autoregressive Conditional Heteroscedasticity [`ARCHUncertaintySet`](@ref) via [`arch`](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)
  - Circular [`CircularBootstrap`](@ref)
  - Moving [`MovingBootstrap`](@ref)
  - Stationary [`StationaryBootstrap`](@ref)

The following estimator can only generate box sets.

- [`DeltaUncertaintySet`](@ref)

### [Turnover](@id readme-turnover)

The turnover is defined as the element-wise absolute difference between the vector of current weights and a vector of benchmark weights. It can be used as a constraint, method for fee calculation, and risk measure. These are all implemented using [`turnover_constraints`](@ref), [`TurnoverEstimator`](@ref), and [`Turnover`](@ref).

### Fees

Fees are a non-negligible aspect of active investing. As such `PortfolioOptimiser.jl` has the ability to account for them in all optimisations but the naive ones. They can also be used to adjust expected returns calculations via [`calc_fees`](@ref) and [`calc_asset_fees`](@ref).

- ::: details Fees [`FeesEstimator`](@ref) and [`Fees`](@ref)
  - Proportional long
  - Proportional short
  - Fixed long
  - Fixed short
  - Turnover

### Portfolio returns and drawdowns

Various risk measures and analyses require the computation of simple and cumulative portfolio returns and drawdowns both in aggregate and per-asset. These are computed by [`calc_net_returns`](@ref), [`calc_net_asset_returns`](@ref), [`cumulative_returns`](@ref), [`drawdowns`](@ref).

### [Tracking](@id readme-tracking)

It is often useful to create portfolios that track the performance of an index, indicator, or another portfolio.

- ::: details Tracking error [`tracking_benchmark`](@ref), [`TrackingError`](@ref)
  - Returns tracking [`ReturnsTracking`](@ref)
  - Weights tracking [`WeightsTracking`](@ref)

The error can be computed using different algorithms using [`norm_tracking`](@ref).

- L1-norm [`NOCTracking`](@ref)
- L2-norm [`SOCTracking`](@ref)
- L2-norm squared [`SquaredSOCTracking`](@ref)

It is also possible to track the error in with risk measures [`RiskTrackingError`]-(@ref) using [`WeightsTracking`](@ref), which allows for two approaches.

- Dependent variable tracking [`DependentVariableTracking`](@ref)
- Independent variable tracking [`IndependentVariableTracking`](@ref)

### Risk measures

`PortfolioOptimisers.jl` provides a wide range of risk measures. These are broadly categorised into two types based on the type of optimisations that support them.

#### Risk measures for traditional optimisation

These are all subtypes of [`RiskMeasure`](@ref), and are supported by all optimisation estimators.

- ::: details Variance [`Variance`]
  - ::: details Traditional optimisations also support:
    - Risk contribution
    - ::: details Formulations
      - Quadratic risk expression [`QuadRiskExpr`](@ref)
      - Squared second order cone [`SquaredSOCRiskExpr`](@ref)
- Standard deviation [`StandardDeviation`](@ref)
- Uncertainty set variance [`UncertaintySetVariance`](@ref) (same as variance when used in non-traditional optimisation)
- ::: details Low order moments [`LowOrderMoment`](@ref)
  - First lower moment [`FirstLowerMoment`](@ref)
  - Mean absolute deviation [`MeanAbsoluteDeviation`](@ref)
  - ::: details Second moment [`SecondMoment`](@ref)
    - ::: details Second squared moments
      - Scenario variance [`Full`](@ref)
      - Scenario semi-variance [`Semi`](@ref)
      - ::: details Traditional optimisation formulations
        - Quadratic risk expression [`QuadRiskExpr`](@ref)
        - Squared second order cone [`SquaredSOCRiskExpr`](@ref)
        - Rotated second order cone [`RSOCRiskExpr`](@ref)
    - ::: details Second moments [`SOCRiskExpr`](@ref)
      - Scenario standard deviation [`Full`](@ref)
      - Scenario semi-standard deviation [`Semi`](@ref)
- ::: details Kurtosis [`Kurtosis`](@ref)
  - Actual kurtosis
    - ::: details Full and semi-kurtosis are supported in traditional optimisers via the `kt` field. Risk calculation uses
      - Full [`Full`](@ref)
      - Semi [`Semi`](@ref)
    - ::: details Traditional optimisation formulations
      - Quadratic risk expression [`QuadRiskExpr`](@ref)
      - Squared second order cone [`SquaredSOCRiskExpr`](@ref)
      - Rotated second order cone [`RSOCRiskExpr`](@ref)
  - ::: details Square root kurtosis [`SOCRiskExpr`](@ref)
    - Full [`Full`](@ref)
    - Semi [`Semi`](@ref)
- ::: details Negative skewness [`NegativeSkewness`]-(@ref)
  - ::: details Squared negative skewness
    - ::: details Full and semi-skewness are supported in traditional optimisers via the `sk` and `V` fields. Risk calculation uses
      - Full [`Full`](@ref)
      - Semi [`Semi`](@ref)
    - ::: details Traditional optimisation formulations
      - Quadratic risk expression [`QuadRiskExpr`](@ref)
      - Squared second order cone [`SquaredSOCRiskExpr`](@ref)
    - Square root negative skewness [`SOCRiskExpr`](@ref)
- ::: details Value at Risk [`ValueatRisk`]-(@ref)
  - ::: details Traditional optimisation formulations
    - Exact MIP formulation [`MIPValueatRisk`]-(@ref)
    - Approximate distribution based [`DistributionValueatRisk`]-(@ref)
- ::: details Value at Risk Range [`ValueatRiskRange`]-(@ref)
  - ::: details Traditional optimisation formulations
    - Exact MIP formulation [`MIPValueatRisk`]-(@ref)
    - Approximate distribution based [`DistributionValueatRisk`]-(@ref)
- Drawdown at Risk [`DrawdownatRisk`]-(@ref)
- Conditional Value at Risk [`ConditionalValueatRisk`]-(@ref)
- Distributionally Robust Conditional Value at Risk [`DistributionallyRobustConditionalValueatRisk`]-(@ref) (same as conditional value at risk when used in non-traditional optimisation)
- Conditional Value at Risk Range [`ConditionalValueatRiskRange`]-(@ref)
- Distributionally Robust Conditional Value at Risk Range [`DistributionallyRobustConditionalValueatRiskRange`]-(@ref) (same as conditional value at risk range when used in non-traditional optimisation)
- Conditional Drawdown at Risk [`ConditionalDrawdownatRisk`]-(@ref)
- Distributionally Robust Conditional Drawdown at Risk [`DistributionallyRobustConditionalDrawdownatRisk`]-(@ref)(same as conditional drawdown at risk when used in non-traditional optimisation)
- Entropic Value at Risk [`EntropicValueatRisk`]-(@ref)
- Entropic Value at Risk Range [`EntropicValueatRiskRange`]-(@ref)
- Entropic Drawdown at Risk [`EntropicDrawdownatRisk`]-(@ref)
- Relativistic Value at Risk [`RelativisticValueatRisk`]-(@ref)
- Relativistic Value at Risk Range [`RelativisticValueatRiskRange`]-(@ref)
- Relativistic Drawdown at Risk [`RelativisticDrawdownatRisk`]-(@ref)
- ::: details Ordered Weights Array
  - ::: details Risk measures
    - Ordered Weights Array risk measure [`OrderedWeightsArray`]-(@ref)
    - Ordered Weights Array range risk measure [`OrderedWeightsArrayRange`]-(@ref)
  - ::: details Traditional optimisation formulations
    - Exact [`ExactOrderedWeightsArray`]-(@ref)
    - Approximate [`ApproxOrderedWeightsArray`]-(@ref)
  - ::: details Array functions
    - Gini Mean Difference [`owa_gmd`](@ref)
    - Worst Realisation [`owa_wr`](@ref)
    - Range [`owa_rg`](@ref)
    - Conditional Value at Risk [`owa_cvar`](@ref)
    - Weighted Conditional Value at Risk [`owa_wcvar`](@ref)
    - Conditional Value at Risk Range [`owa_cvarrg`](@ref)
    - Weighted Conditional Value at Risk Range [`owa_wcvarrg`](@ref)
    - Tail Gini [`owa_tg`](@ref)
    - Tail Gini Range [`owa_tgrg`](@ref)
    - ::: details Linear moments (L-moments)
      - Linear Moment [`owa_l_moment`](@ref)
      - ::: details Linear Moment Convex Risk Measure [`owa_l_moment_crm`](@ref)
        - ::: details L-moment combination formulations
          - ::: details Maximum Entropy [`MaximumEntropy`]-(@ref)
            - Exponential Cone Entropy [`ExponentialConeEntropy`]-(@ref)
            - Relative Entropy [`RelativeEntropy`]-(@ref)
          - Minimum Squared Distance [`MinimumSquaredDistance`]-(@ref)
          - Minimum Sum Squares [`MinimumSumSquares`]-(@ref)
- Average Drawdown [`AverageDrawdown`]-(@ref)
- Ulcer Index [`UlcerIndex`]-(@ref)
- Maximum Drawdown [`MaximumDrawdown`]-(@ref)
- ::: details Brownian Distance Variance [`BrownianDistanceVariance`]-(@ref)
  - ::: details Traditional optimisation formulations
    - ::: details Distance matrix constraint formulations
      - Norm one cone Brownian distance variance [`NormOneConeBrownianDistanceVariance`]-(@ref)
      - Inequality Brownian distance variance [`IneqBrownianDistanceVariance`]-(@ref)
    - ::: details Risk formulation
      - Quadratic risk expression [`QuadRiskExpr`](@ref)
      - Rotated second order cone [`RSOCRiskExpr`](@ref)
- Worst Realisation [`WorstRealisation`]-(@ref)
- Range [`Range`]-(@ref)
- Turnover Risk Measure [`TurnoverRiskMeasure`]-(@ref)
- ::: details Tracking Risk Measure [`TrackingRiskMeasure`]-(@ref)
  - L1-norm [`NOCTracking`](@ref)
  - L2-norm [`SOCTracking`](@ref)
  - L2-norm squared [`SquaredSOCTracking`](@ref)
- ::: details Risk Tracking Risk Measure
  - Dependent variable tracking [`DependentVariableTracking`](@ref)
  - Independent variable tracking [`IndependentVariableTracking`](@ref)
- Power Norm Value at Risk [`PowerNormValueatRisk`]-(@ref)
- Power Norm Value at Risk Range [`PowerNormValueatRiskRange`]-(@ref)
- Power Norm Drawdown at Risk [`PowerNormDrawdownatRisk`]-(@ref)

#### Risk measures for hierarchical optimisation

These are all subtypes of [`HierarchicalRiskMeasure`](@ref), and are only supported by hierarchical optimisation estimators.

- ::: details High order moment [`HighOrderMoment`](@ref)
  - Unstandardised third lower moment [`ThirdLowerMoment`](@ref)
  - Standardised third lower moment [`StandardisedHighOrderMoment`](@ref) and [`ThirdLowerMoment`](@ref)
  - ::: details Unstandardised fourth moment [`FourthMoment`](@ref)
    - Full [`Full`](@ref)
    - Semi [`Semi`](@ref)
  - ::: details Standardised fourth moment [`StandardisedHighOrderMoment`](@ref) and [`FourthMoment`](@ref)
    - Full [`Full`](@ref)
    - Semi [`Semi`](@ref)
- Relative Drawdown at Risk [`RelativeDrawdownatRisk`]-(@ref)
- Relative Conditional Drawdown at Risk [`RelativeConditionalDrawdownatRisk`]-(@ref)
- Relative Entropic Drawdown at Risk [`RelativeEntropicDrawdownatRisk`]-(@ref)
- Relative Relativistic Drawdown at Risk [`RelativeRelativisticDrawdownatRisk`]-(@ref)
- Relative Average Drawdown [`RelativeAverageDrawdown`]-(@ref)
- Relative Ulcer Index [`RelativeUlcerIndex`]-(@ref)
- Relative Maximum Drawdown [`RelativeMaximumDrawdown`]-(@ref)
- Relative Power Norm Drawdown at Risk [`RelativePowerNormDrawdownatRisk`]-(@ref)
- Risk Ratio Risk Measure [`RiskRatioRiskMeasure`]-(@ref)
- Equal Risk Measure [`EqualRiskMeasure`]-(@ref)
- Median Absolute Deviation [`MedianAbsoluteDeviation`]-(@ref)

#### Non-optimisation risk measures

These risk measures are unsuitable for optimisation because they can return negative values. However, they can be used for performance metrics.

- Mean Return [`MeanReturn`]-(@ref)
- Third Central Moment [`ThirdCentralMoment`]-@(ref)
- Skewness [`Skewness`]-(@ref)
- Return Risk Measure [`ReturnRiskMeasure`](@ref)
- Return Risk Ratio Risk Measure [`ReturnRiskRatioRiskMeasure`](@ref)

### Performance metrics

- Expected risk [`expected_risk`]-(@ref)
- Number of effective assets [`number_effective_assets`]-(@ref)
- ::: details Risk contribution
  - Asset risk contribution [`risk_contribution`]-(@ref)
  - Factor risk contribution [`factor_risk_contribution`]-(@ref)
- ::: details Expected return [`expected_return`](@ref)
  - Arithmetic [`ArithmeticReturn`]-(@ref)
  - Logarithmic [`LogarithmicReturn`]-(@ref)
- Expected risk-adjusted return ratio [`expected_ratio`](@ref) and [`expected_risk_ret_ratio`](@ref)
- Expected risk-adjusted ratio information criterion [`expected_sric`](@ref) and [`expected_risk_ret_sric`](@ref)
- Brinson performance attribution [`brinson_attribution`](@ref)

### Portfolio optimisation

Optimisations are implemented via [`optimise`]-(@ref). Optimisations consume an estimator and return a result.

#### Naive

These return a [`NaiveOptimisationResult`]-(@ref).

- Inverse Volatility [`InverseVolatility`]-(@ref)
- Equal Weighted [`EqualWeighted`]-(@ref)
- Random (Dirichlet) [`RandomWeighted`]-(@ref)

##### Naive optimisation features

- Weight bounds [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(@ref)
  - ::: details JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(@ref)
    - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(@ref)
    - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(@ref)
    - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(@ref)
    - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(@ref)

#### Traditional

These optimisations are implemented as `JuMP` problems and make use of [`JuMPOptimiser`]-(@ref), which encodes all supported constraints.

##### Objective function optimisations

These optimisations support a variety of objective functions.

- ::: details Objective functions
  - Minimum risk [`MinimumRisk`]-(@ref)
  - Maximum utility [`MaximumUtility`]-(@ref)
  - Maximum return over risk ratio [`MaximumRatio`]-(@ref)
  - Maximum return [`MaximumReturn`]-(@ref)
- ::: details Exclusive to [`MeanRisk`]-(@ref) and [`NearOptimalCentering`]-(@ref)
  - ::: details N-dimensional Pareto fronts [`Frontier`](@ref)
    - Return based
    - Risk based
- ::: details Optimisation estimators
  - Mean-Risk [`MeanRisk`]-(@ref) returns a [`MeanRiskResult`]-(@ref)
  - Near Optimal Centering [`NearOptimalCentering`]-(@ref) returns a [`NearOptimalCenteringResult`]-(@ref)
  - Factor Risk Contribution [`FactorRiskContribution`]-(@ref) returns a [`FactorRiskContributionResult`]-(@ref)

##### Risk budgeting optimisations

These optimisations attempt to achieve weight values according to a risk budget vector. This vector can be provided on a per asset or per factor basis.

- ::: details Budget targets
  - Asset risk budgeting [`AssetRiskBudgeting`]-(@ref)
  - Factor risk budgeting [`FactorRiskBudgeting`]-(@ref)
- ::: details Optimisation estimators
  - Risk Budgeting [`RiskBudgeting`]-(@ref) returns a [`RiskBudgetingResult`]-(@ref)
  - ::: details Relaxed Risk Budgeting [`RelaxedRiskBudgeting`]-(@ref) returns a [`RiskBudgetingResult`]-(@ref)
    - Basic [`BasicRelaxedRiskBudgeting`]-(@ref)
    - Regularised [`RegularisedRelaxedRiskBudgeting`]-(@ref)
    - Regularised and penalised [`RegularisedPenalisedRelaxedRiskBudgeting`]-(@ref)

##### Traditional optimisation features

- Custom objective penalty [`CustomJuMPObjective`]-(@ref)
- Weight bounds [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Budget
  - ::: details Directionality
    - Long
    - Short
  - ::: details Type
    - Exact
    - Range [`BudgetRange`]-(@ref)
- ::: details Threshold [`ThresholdEstimator`](@ref) and [`Threshold`](@ref)
  - ::: details Directionality
    - Long
    - Short
  - ::: details Type
    - Asset
    - Set [`AssetSetsMatrixEstimator`](@ref)
- Linear constraints [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
- Centralit(y/ies) [`CentralityEstimator`](@ref)
- ::: details Cardinality
  - Asset
  - Asset group(s) [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
  - Set(s)
  - Set group(s) [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
- Turnover(s) [`TurnoverEstimator`](@ref) and [`Turnover`](@ref)
- Fees [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Tracking error(s) [`TrackingError`](@ref)
- Phylogen(y/ies) [`IntegerPhylogenyEstimator`](@ref) and [`SemiDefinitePhylogenyEstimator`](@ref)
- ::: details Portfolio returns
  - ::: details Arithmetic returns [`ArithmeticReturn`]-(@ref)
    - Uncertainty set [`BoxUncertaintySet`](@ref), [`BoxUncertaintySetAlgorithm`](@ref), [`EllipsoidalUncertaintySet`](@ref), and [`EllipsoidalUncertaintySetAlgorithm`](@ref)
    - Custom expected returns vector
  - Logarithmic returns [`LogarithmicReturn`]-(@ref)
- ::: details Risk vector scalarisation
  - Weighted sum [`SumScalariser`](@ref)
  - Maximum value [`MaxScalariser`](@ref)
  - Log-sum-exp [`LogSumExpScalariser`](@ref)
- Custom constraint
- Number of effective assets
- ::: details Regularisation penalty
  - L1
  - L2

#### [Clustering optimisation](@id readme-clustering-opt)

Clustering optimisations make use of asset relationships to either minimise the risk exposure by breaking the asset universe into subsets which are hierarchically or individually optimised.

##### Hierarchical clustering optimisation

These optimisations minimise risk by hierarchically splitting the asset universe into subsets, computing the risk of each subset, and combining them according to their hierarchy.

- Hierarchical Risk Parity [`HierarchicalRiskParity`]-(@ref) returns a [`HierarchicalResult`]-(@ref)
- Hierarchical Equal Risk Contribution [`HierarchicalEqualRiskContribution`]-(@ref) returns a [`HierarchicalResult`]-(@ref)

###### Hierarchical clustering optimisation features

- Weight bounds [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Fees [`FeesEstimator`](@ref) and [`Fees`](@ref)
- ::: details Risk vector scalarisation
  - Weighted sum [`SumScalariser`](@ref)
  - Maximum value [`MaxScalariser`](@ref)
  - Log-sum-exp [`LogSumExpScalariser`](@ref)
- ::: details Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(@ref)
  - ::: details JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(@ref)
    - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(@ref)
    - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(@ref)
    - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(@ref)
    - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(@ref)

##### Schur complementary optimisation

Schur complementary hierarchical risk parity provides a bridge between mean variance optimisation and hierarchical risk parity by using an interpolation parameter. It converges to hierarchical risk parity, and approximates mean variance by adjusting this parameter. It uses the Schur complement to adjust the weights of a portfolio according to how much more useful information is gained by assigning more weight to a group of assets.

- Schur Complementary Hierarchical Risk Parity [`SchurComplementHierarchicalRiskParity`]-(@ref) returns a [`SchurComplementHierarchicalRiskParityResult`]-(@ref)

###### Schur complementary optimisation features

- Weight bounds [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Fees [`FeesEstimator`](@ref) and [`Fees`](@ref)
- ::: details Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(@ref)
  - ::: details JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(@ref)
    - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(@ref)
    - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(@ref)
    - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(@ref)
    - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(@ref)

##### Nested clusters optimisation

Nested clustered optimisation breaks the asset universe into smaller subsets and treats every subset as an individual portfolio. Then it creates a synthetic asset out of each portfolio, optimises the portfolio of synthetic assets. The final weights are the inner product between the individual portfolio weights and outer portfolio.

- Nested Clustered [`NestedClustered`]-(@ref) returns a [`NestedClusteredResult`]-(@ref)

##### Clustering optimisation features

- Any features supported by the inner and outer estimators.
- Weight bounds [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(@ref)
  - ::: details JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(@ref)
    - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(@ref)
    - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(@ref)
    - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(@ref)
    - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(@ref)
- Cross validation predictor for the outer estimator

#### Ensemble optimisation

These work similar to the Nested Clustered estimator, only instead of breaking the asset universe into subsets, a list of inner estimators is provided, all of which are optimised, and each result is treated as a synthetic asset from which a synthetic portfolio is created and optimised according to an outer estimator. The final weights are the inner product between the individual portfolio weights and outer portfolio.

- Stacking [`Stacking`]-(@ref) returns a [`StackingResult`]-(@ref)

##### Ensemble optimisation features

- Any features supported by the inner and outer estimators.
- Weight bounds [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Weight finalisers
  - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(@ref)
  - ::: details JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(@ref)
    - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(@ref)
    - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(@ref)
    - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(@ref)
    - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(@ref)
- Cross validation predictor for the outer estimator

#### Finite allocation optimisation

Unlike all other estimators, finite allocation does not yield an "optimal" value, but rather the optimal attainable solution based on a finite amount of capital. They use the result of other estimations, the latest prices, and a cash amount.

- ::: details Discrete (MIP) [`DiscreteAllocation`]-(@ref)
  - ::: details Weight finalisers
    - Iterative Weight Finaliser [`IterativeWeightFinaliser`]-(@ref)
    - ::: details JuMP Weight Finaliser [`JuMPWeightFinaliser`]-(@ref)
      - Relative Error Weight Finaliser [`RelativeErrorWeightFinaliser`]-(@ref)
      - Squared Relative Error Weight Finaliser [`SquaredRelativeErrorWeightFinaliser`]-(@ref)
      - Absolute Error Weight Finaliser [`AbsoluteErrorWeightFinaliser`]-(@ref)
      - Squared Absolute Error Weight Finaliser [`SquaredAbsoluteErrorWeightFinaliser`]-(@ref)
- Greedy [`GreedyAllocation`]

### Plotting

Visualising the results is quite a useful way of summarising the portfolio characteristics or evolution. To this extent we provide a few plotting functions with more to come.

- ::: details Simple or compound cumulative returns.
  - Portfolio [`plot_ptf_cumulative_returns`]-(@ref).
  - Assets [`plot_asset_cumulative_returns`]-(@ref).
- ::: details Portfolio composition.
  - Single portfolio [`plot_composition`]-(@ref).
  - ::: details Multi portfolio.
    - Stacked bar [`plot_stacked_bar_composition`]-(@ref).
    - Stacked area [`plot_stacked_area_composition`]-(@ref).
- ::: details Risk contribution.
  - Asset risk contribution [`plot_risk_contribution`]-(@ref).
  - Factor risk contribution [`plot_factor_risk_contribution`]-(@ref).
- Asset dendrogram [`plot_dendrogram`]-(@ref).
- Asset clusters + optional dendrogram [`plot_clusters`]-(@ref).
- Simple or compound drawdowns [`plot_drawdowns`]-(@ref).
- Portfolio returns histogram + density [`plot_histogram`]-(@ref).
- 2/3D risk measure scatter plots [`plot_measures`]-(@ref).
