# PortfolioOptimisers.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/dev)
[![Test workflow status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl)
[![Docs workflow Status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Build Status](https://api.cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl.svg)](https://cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/dcelisgarza/PortfolioOptimisers.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Welcome to PortfolioOptimisers.jl!

[`PortfolioOptimisers.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl) is a package for portfolio optimisation written in Julia.

> [!CAUTION]
> Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

Portfolio optimisation is the science of reducing investment risk by being clever about how you distribute your money. Ironically, some of the most robust ways to ensure risk is minimised is to distribute your money equally among a portfolio of proven assets. There exist however, a rather large number of methods, risk measures, constraints, prior statistics estimators, etc. Which give a huge number of combinations.

`PortfolioOptimisers.jl` is an attempt at providing as many as possible, and to make it possible to add more by leveraging Julia's type system.

The feature list is *quite large* and under *active development*. New features will be added over time. Check out the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/1_Getting_Started) and [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_Introduction) documentation for details.

Please feel free to file issues and/or start discussions if you have any issues using the library, or if I haven't got to writing docs/examples for something you need. That way I know what to prioritise.

## Quickstart

The library is quite powerful and extremely flexible. Here is what a very basic end-to-end workflow can look like. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/1_Getting_Started) contain more thorough explanations and demos. The [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_Introduction) contains toy examples of the many, many features.

```julia
# Import module and plotting extension.
using PortfolioOptimisers, StatsPlots, GraphRecipes
# Import optimisers.
using Clarabel, HiGHS
# Download data and pretty printing
using YFinance, PrettyTables, TimeSeries, DataFrames

# Format for pretty tables.
fmt1 = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end

fmt2 = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end

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
#=
┌────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  timestamp │    ACHR │     AMC │    ANVS │      BB │    CHPT │    ENVX │    EOSE │     GME │    LCID │    LGVN │    LUNR │    MARA │    MAXN │    NVAX │    RIVN │    RKLB │     SMR │    SOFI │    SOUN │    UPST │
│   DateTime │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 2025-09-26 │    9.28 │    2.89 │    1.97 │    4.96 │   10.84 │   10.09 │   10.12 │   26.42 │   23.96 │   0.799 │   10.08 │   16.13 │    3.53 │    8.55 │   15.59 │   46.26 │    38.0 │   27.98 │   15.94 │   57.35 │
│ 2025-09-29 │    9.65 │     3.0 │    2.04 │     5.0 │   11.05 │    9.97 │   11.17 │   27.21 │   24.11 │    0.77 │   10.26 │   18.66 │    3.55 │    8.57 │   15.25 │   47.01 │   38.16 │   27.55 │   15.68 │   52.74 │
│ 2025-09-30 │    9.58 │     2.9 │    2.07 │    4.88 │   10.92 │    9.97 │   11.39 │   27.28 │   23.79 │    0.75 │   10.52 │   18.26 │    3.35 │    8.67 │   14.68 │   47.91 │    36.0 │   26.42 │   16.08 │    50.8 │
│ 2025-10-01 │    9.81 │    2.95 │    2.13 │    4.79 │   11.63 │   11.11 │   12.37 │   27.69 │  24.295 │   0.742 │   10.61 │   18.61 │    3.58 │     9.5 │   14.61 │   47.97 │   36.61 │   25.76 │   16.15 │   52.13 │
│ 2025-10-02 │   10.18 │    3.15 │    2.23 │    4.75 │   11.32 │   11.65 │   12.36 │   27.22 │    24.1 │   0.765 │   11.22 │   18.79 │    3.78 │    9.55 │   13.53 │   52.47 │   39.51 │   25.97 │   17.84 │   52.88 │
│ 2025-10-03 │   11.57 │    3.06 │    2.22 │     4.5 │   11.94 │   11.92 │    12.6 │   25.38 │   24.77 │   0.788 │   11.44 │   18.82 │     3.6 │    9.46 │   13.65 │   56.16 │   40.12 │   25.24 │   17.85 │   51.96 │
└────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
=#

# Compute the returns.
rd = prices_to_returns(prices)

# Define the continuous solver.
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
             check_sol = (; allow_local = true, allow_almost = true))

# Vanilla (Markowitz) mean risk optimisation.
mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

# Perform the optimisation, res.w contains the optimal weights.
res = optimise(mr, rd)

# Define the MIP solver for finite discrete allocation.
mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))

# Discrete finite allocation.
da = DiscreteAllocation(; slv = mip_slv)

# Perform the finite discrete allocation, uses the final asset 
# prices, and an available cash amount. This is for us mortals 
# without infinite wealth.
mip_res = optimise(da, res.w, vec(values(prices[end])), 4206.90)

# View the results.
df = DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
               :opt_weights => res.w, :mip_weights => mip_res.w)
pretty_table(df; formatters = [fmt2])
#=
┌────────┬─────────┬─────────┬─────────────┬─────────────┐
│ assets │  shares │    cost │ opt_weights │ mip_weights │
│ String │ Float64 │ Float64 │     Float64 │     Float64 │
├────────┼─────────┼─────────┼─────────────┼─────────────┤
│   ACHR │     0.0 │     0.0 │       0.0 % │       0.0 % │
│    AMC │    73.0 │  223.38 │     5.324 % │      5.31 % │
│   ANVS │    22.0 │   48.84 │     1.249 % │     1.161 % │
│     BB │   273.0 │  1228.5 │    29.184 % │    29.203 % │
│   CHPT │    11.0 │  131.34 │     3.002 % │     3.122 % │
│   ENVX │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   EOSE │     8.0 │   100.8 │     2.435 % │     2.396 % │
│    GME │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   LCID │     1.0 │   24.77 │     0.638 % │     0.589 % │
│   LGVN │   325.0 │   256.1 │     6.089 % │     6.088 % │
│   LUNR │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   MARA │     1.0 │   18.82 │     0.613 % │     0.447 % │
│   MAXN │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   NVAX │    28.0 │  264.88 │      6.21 % │     6.297 % │
│   RIVN │    55.0 │  750.75 │    17.897 % │    17.847 % │
│   RKLB │     4.0 │  224.64 │     4.896 % │      5.34 % │
│    SMR │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   SOFI │    37.0 │  933.88 │    22.462 % │      22.2 % │
│   SOUN │     0.0 │     0.0 │       0.0 % │       0.0 % │
│   UPST │     0.0 │     0.0 │       0.0 % │       0.0 % │
└────────┴─────────┴─────────┴─────────────┴─────────────┘
=#

# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_ptf_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
```

![Fig. 1](./docs/src/assets/readme_1.svg)

```julia
# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)
```

![Fig. 2](./docs/src/assets/readme_2.svg)

```julia
# Plot histogram of returns.
plot_histogram(mip_res.w, rd.X, slv)
```

![Fig. 3](./docs/src/assets/readme_3.svg)

```julia
# Plot compounded drawdowns.
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
```

![Fig. 4](./docs/src/assets/readme_4.svg)

## Caveats

### Documentation

  - Mathematical formalism: I've got API documentation for a lot of features, but the mathematical formalisms aren't yet thoroughly explained. It's more of a high level view.
  - Citation needed: I haven't gone over all the citations for the docs because stabilising the API, adding new features, and writing the API docs has taken priority.
  - Docstring examples: some features require set up steps, and I haven't had the patience to do that. Mostly the examples are still mostly for doctesting my implementation of `Base.show` for my types, and showcasing low-hanging fruit of functionality.

### API

  - Unstable: there will likely be breaking changes as I figure out better, more general ways to do things, or better naming conventions.

### Internals

  - Dependencies: some deps are only used for certain small things, I may end up removing them in favour of having just the small bit of functionality the package needs. I'm very open to replacement suggestions.

## Features

The feature list is rather large, so I will attempt to summarise it ~~via interpretative dance~~ as best I can. There are also some experimental features (some tracking risk measures) that I'm not sure how well they'd perform, but they're interesting nonetheless, especially when used in clustering optimisations. Luckily, those haven't been documented yet, so I haven't had to reckon with the consequences of my actions just yet.

Without further ado, here is a summary of the features in this package.

### Price data

Every optimisation but the finite allocation work off of returns data. Some optimisations may use price data in the future.

  - Preprocessing to drop highly correlated and/or incomplete data. These are not well integrated yet, but the functions exist.
  - Computing them, validating and cleaning up data.

### Co-moment matrix processing

Price data is often noisy and follows general macroeconomic trends. Every optimisation model is at risk of overfitting the data. In particular, those which rely on summary statistics (moments) can be overly sensitive to the input data, for example a covariance matrix. It is therefore important to have methods that increase the robustness of their estimation.

  - Positive definite projection.

  - Matrix denoising.
    
      + Spectral, shrunk, fixed.
  - Matrix detoning.

### Moment estimation

Many of these can be used in conjunction. For example, some covariance estimators use expected returns, or variance estimators in their calculation, and some expected returns use the covariance in turn. Also, some accept weight vectors.

  - Expected returns.
    
      + Arithmetic expected returns.
    
      + Shrunk expected returns.
        
          * James-Stein, Bayes-Stein, Bodnar-Okhrin-Parolya. All of them with Grand Mean, Volatility Weighted, Mean Squared Error targets.
      + Equilibrium expected returns.
      + Excess expected returns.

  - Variance.
  - Covariance/Correlation matrix.
    
      + Custom: estimator + processing pipeline.
    
      + Pearson: weighted, unweighted, any `StatsBase.CovarianceEstimator`.
        
          * Full.
          * Semi.
      + Gerber.
        
          * Gerber 0, 1, 2. Standardised and unstandardised.
      + Smyth-Broby.
        
          * Smyth-Broby 0, 1, 2. Standardised and unstandardised.
          * Smyth-Broby-Gerber 0, 1, 2. Standardised and unstandardised.
      + Distance covariance.
      + Lower tail dependence.
      + Kendall.
      + Spearman.
      + Mutual information.
        
          * Predefined, Hacine-Gharbi-Ravier, Knuth, Scott, Freedman-Draconis bin widths.
      + Coskewness.
        
          * Full.
          * Semi.
      + Cokurtosis.
        
          * Full.
          * Semi.
      + Implied volatility.

### Regression Models

Factor models and implied volatility use regression in their estimation.

  - Stepwise.
    
      + Forward and Backward.
        
          * P-value, Corrected and "vanilla" Akaike info, Bayesian info, R-squared, and Adjusted R-squared criteria.

  - Dimensional reduction.
    
      + Principal Component.
      + Probabilistic Principal Component.

### Ordered weights array and Linear moments

  - Ordered weights arrays.
    
      + Gini Mean Difference.
      + Conditional Value at Risk.
      + Weighted Conditional Value at Risk.
      + Tail Gini.
      + Worst Realisation.
      + Range.
      + Conditional Value at Risk Range.
      + Weighted Conditional Value at Risk Range.
      + Tail Gini Range.

  - Linear Moments Convex Risk Measure: linear moments can be combined using different minimisation targets.
    
      + Normalised Constant Relative Risk Aversion.
      + Minimum Squared Distance.
      + Minimum Sum Squares.

### Distance Matrices

Distance matrices are used for clustering. They are related to correlation distances, but all positive and with zero diagonal.

  - Distance: these compare pairwise relationships.
  - Distance of distances: these are computed by applying a distance metric to every pair of columns/rows of the distance matrix. They compare the entire space and often give more stable clusters.

Individual entries can be raised to an integer power and scaled according to whether that power is even or odd. The following methods can be used to compute distance matrices.

  - Simple.

  - Absolute.
  - Logarithmic.
  - Correlation.
  - Variation of Information.
    
      + Predefined, Hacine-Gharbi-Ravier, Knuth, Scott, Freedman-Draconis bin widths.
  - Canonical: depends on the covariance/correlation estimator used.

### Phylogeny

These define asset relationships. They can be used to set constraints on and/or compute the relatedness of assets in a portfolio.

  - Clustering.
    
      + Optimal number of clusters:
        
          * Predefined, Second order difference, Standardised silhouette scores.
    
      + Hierarchical clustering.
      + Direct Bubble Hierarchy Trees.
        
          * Local Global sparsification of the inverse covariance/correlation matrix.

  - Phylogeny matrices.
    
      + Network (Minimum Spanning Tree) adjacency.
      + Clustering adjacency.
  - Centrality vectors and average centrality.
    
      + Betweenness, Closeness, Degree, Eigenvector, Katz, Pagerank, Radiality, Stress centrality measures.
  - Asset phylogeny score.

### Constraint generation

These let users easily manually or programatically define optimisation constraints.

  - Equation parsing.

  - Linear weights.
  - Risk budget.
  - Asset set matrices.
  - Phylogeny.
    
      + Phylogeny matrix.
        
          * Semi definite.
          * Mixed integer programming.
    
      + Centrality.
  - Weight bounds.
  - Buy-in threshold.

### Prior statistics

As previously mentioned, every optimisation but the finite allocation work off of returns data. These returns can be adjusted and summarised using these estimators. Like the moment estimators, these can be mixed in various ways.

  - Empirical.

  - Factor model.
  - High order moments (coskewness and cokurtosis).
  - Black-Litterman.
    
      + Vanilla.
      + Bayesian.
      + Factor model.
      + Augmented.
  - Entropy pooling.
  - Opinion pooling.

### Uncertainty sets

These sets can be used to make some optimisations more robust. Namely, there exist uncertainty sets on expected returns and covariance. They can be used on any optimisation which uses any one of these quantities.

  - Box.
    
      + Delta.
    
      + Normally distributed returns.
      + Autoregressive Conditional Heteroskedasticity.
        
          * Circular, moving, stationary bootstrap.

  - Ellipse uncertainty sets.
    
      + Normally distributed returns.
    
      + Autoregressive Conditional Heteroskedasticity.
        
          * Circular, moving, stationary bootstrap.

### Turnover (Rebalancing)

These penalise moving away from a benchmark vector of weights.

  - Risk measure (experimental).
  - Constraints.
  - Fees.

### Fees

These encode various types of fees, which can be used in portfolio optimisation and analysis.

  - Relative long.
  - Relative short
  - Fixed long.
  - Fixed short.
  - Turnover (rebalance).

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
      + Kelly (Logarithmic).

  - Risk-adjusted return ratio.
    
      + Vanilla.
      + Sharpe ratio information criterion.
  - Risk contribution.
    
      + Asset risk contribution.
      + Factor risk contribution.

### Optimisation

There are many different optimisation methods, each with different characteristics and configurable options, including exclusive constraint types and risk measures. Though all of them have an optional fallback method in case the optimisation fails.

  - Clustering.
    
      + Hierarchical Risk Parity.
      + Hierarchical Equal Risk Contribution.
      + Nested Clustered Optimisation.
      + Schur Complement Hierarchical Risk Parity.

  - `JuMP`-based.
    
      + Mean Risk.
    
      + Factor Risk Contribution.
      + Near Optimal Centering.
      + Risk Budgeting.
        
          * Asset risk budgeting.
          * Factor risk budgeting.
      + Relaxed Risk Budgeting.
  - Stacking.
  - Naive.
    
      + Inverse volatility.
      + Equal weighted.
      + Random weighted.
  - Finite Allocation.
    
      + Discrete.
      + Greedy.

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

## How to Cite

If you use PortfolioOptimisers.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first take a look into our [contributing guide directly on GitHub](docs/src/contribute/1-contributing.md) or the [contributing page on the website](https://dcelisgarza.github.io/PortfolioOptimisers.jl/dev/contribute/1-contributing)
