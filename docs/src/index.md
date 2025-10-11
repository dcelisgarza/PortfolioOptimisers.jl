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
      link: examples/1_Getting_Started
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

# Welcome to PortfolioOptimisers.jl!

[`PortfolioOptimisers.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl) is a package for portfolio optimisation written in Julia.

!!! danger
    
    Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

Portfolio optimisation is the science of reducing investment risk by being clever about how you distribute your money. Ironically, some of the most robust ways to ensure risk is minimised is to distribute your money equally among a portfolio of proven assets. There exist however, a rather large number of methods, risk measures, constraints, prior statistics estimators, etc. Which give a huge number of combinations.

`PortfolioOptimisers.jl` is an attempt at providing as many as possible, and to make it possible to add more by leveraging Julia's type system.

The feature list is quite large and under active development. New features will be added over time. Check out the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/1_Getting_Started) and [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_Introduction) documentation for details.

Please feel free to file issues and/or start discussions if you have any issues using the library, or if I haven't got to writing docs/examples for something you need. That way I know what to prioritise.

## Quickstart

The library is quite powerful and extremely flexible. Here is what a very basic end-to-end workflow can look like. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/1_Getting_Started) contain more thorough explanations and demos. The [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_Introduction) contains toy examples of the many, many features.

````@example 0_index
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
end;
fmt2 = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing # hide
````

We will now download the prices.

````@example 0_index
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
pretty_table(prices[(end - 5):end]; formatters = fmt1)
````

We now have all we need to perform a basic optimisation.

````@example 0_index
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
````

`PortfolioOptimisers.jl` also has the capability to perform finite allocations for those of us without infinite money.

````@example 0_index
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
pretty_table(df; formatters = fmt2)
````

Finally, lets plot some results.

````@example 0_index
# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_ptf_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
````

````@example 0_index
# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)
````

````@example 0_index
# Plot histogram of returns.
plot_histogram(mip_res.w, rd.X, slv)
````

````@example 0_index
# Plot compounded drawdowns.
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
````

## Caveats

### Documentation

  - Mathematical formalism: I've got API documentation for a lot of features, but the mathematical formalisms aren't yet thoroughly explained. It's more of a high level view.
  - Citation needed: I haven't gone over all the citations for the docs because stabilising the API, adding new features, and writing the API docs has taken priority.
  - Docstring examples: some features require set up steps, and I haven't had the patience to do that. Mostly the examples are still mostly for doctesting my implementation of `Base.show` for my types, and showcasing low-hanging fruit of functionality.

### API

  - Unstable: there will likely be breaking changes as I figure out better, more general way to do things, or better naming conventions.

### Internals

  - Dependencies: some deps are only used for certain small things, I may end up removing them in favour of having just the small bit of functionality the package needs. I'm very open to replacement suggestions.

## Features

The feature list is rather large, so I will attempt to summarise it ~~via interpretative dance~~ as best I can. There's also some experimental features (some tracking risk measures) that I'm not sure how well they'd perform, but they're interesting nonetheless, especially when used in clustering optimisations. Luckily, those haven't been documented yet, so I haven't had to reckon with the consequences of my actions just yet.

### Price data

Everything but the finite allocation optimisations work off of returns data. Some optimisations may use price data in the future.

  - Preprocessing to drop highly correlated and/or incomplete data. These are not well integrated yet, but the functions exist.
  - Computing them, validating and cleaning up data.

### Co-moment matrix processing

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
        
          * Full and Semi covariance algorithms.
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
      + Coskewness.
        
          * Full and Semi algorithms.
      + Implied volatility.

### Regression

Factor models and implied volatility work off of regression.

  - Stepwise.
    
      + Forward and Backward.
          * P-value, Corrected and "vanilla" Akaike info, Bayesian info, R-square, and Adjusted R-squared criteria.

  - Dimensional reduction.
    
      + Principal Component.
      + Probabilistic Principal Component.

### Distance

Distance matrices are used for clustering. They are related to correlation distances, but all positive and with zero diagonal.

  - Distance: these compare pairwise relationships.
  - Distance of distances: these are computed by applying a distance metric to every pair of columns/rows of the distance matrix. They compare the entire space and often give more stable clusters.

Individual entries can be raised to an integer power and scaled according to whether that power is even or odd. The following methods can be used to compute distance matrices.

  - Simple.
  - Absolute.
  - Logarithmic.
  - Correlation.
  - Canonical: depends on the covariance/correlation estimator used.

### Phylogeny

  - Clustering.
    
      + Optimal number of clusters:
        
          * Predefined, Second order difference, Standardised silhouette scores.
    
      + Hierarchical clustering.
      + Direct Bubble Hierarchy Trees.
        
          * Local Global sparsification of the inverse covariance/correlation matrix.
  - Phylogeny matrices.

      + Network (MST) adjacency.
      + Clustering adjacency.
  - Centrality vectors and average centrality.
        
      + Betweenness, Closeness, Degree, Eigenvector, Katz, Pagerank, Radiality, Stress centrality measures.
  - Asset phylogeny score.

### Constraint generation

  - Equation parsing: lets users define linear constraints by directly writing the equations.
  - Linear weight constraint generation.
  - Risk budget constraint generation.
  - Asset set matrix generation.
