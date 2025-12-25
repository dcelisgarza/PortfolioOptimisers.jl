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
julia> ]add PortfolioOptimisers
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

`PortfolioOptimisers.jl` implements a number of optimisation types as estimators. All the ones which use mathematical optimisation require a [`JuMPOptimiser`](@ref) structure which defines general solver constraints. This structure in turn requires an instance (or vector) of [`Solver`](@ref).

```@example 0_index
opt = JuMPOptimiser(; slv = slv);
nothing # hide
```

Here we will use the traditional Mean-Risk [`MeanRisk`](@ref) optimsation estimator, which defaults to the Markowitz optimisation (minimum risk mean-variance optimisation).

```@example 0_index
# Vanilla (Markowitz) mean risk optimisation.
mr = MeanRisk(; opt = opt)
```

As you can see, there are *a lot* of fields in this structure, which correspond to a wide variety of optimisation constraints. We will explore these in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction). For now, we will perform the optimisation via [`optimise`](@ref).

```@example 0_index
# Perform the optimisation, res.w contains the optimal weights.
res = optimise(mr, rd)
```

The solution lives in the `sol` field, but the weights can be accessed via the `w` property.

`PortfolioOptimisers.jl` also has the capability to perform finite allocations, which is useful for those of us without infinite money. There are two ways to do so, a greedy algorithm [`GreedyAllocation`](@ref) that does not guarantee optimality but is fast and always converges, and a discrete allocation [`DiscreteAllocation`](@ref) which uses mixed-integer programming (MIP) and requires a capable solver.

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

## Features

### Preprocessing

- Prices to returns.

### Portfolio Optimisation

#### Naïve

- Equal Weighted
- Inverse Volatility
- Random (Dirichlet)

#### Traditional

- Mean-Risk
- Factor Risk Contribution
- Near Optimal Centering
- Risk Budgeting
  - Asset Risk Budgeting
  - Factor Risk Budgeting
- Relaxed Risk Budgeting
  - Asset Relaxed Risk Budgeting
  - Factor Relaxed Risk Budgeting

##### Traditional Optimisation Features

- Objective Functions
  - Minimum Risk
  - Maximum Utility
  - Maximum Return Over Risk Ratio
  - Maximum Return
  - Custom
- Fees
  - Proportional
    - Long
    - Short
  - Fixed
    - Long
    - Short
  - Turnover
- Regularisation
  - L1
  - L2
- Weight Constraints
- Budget Constraints

#### Clustering

- Hierarchical Risk Parity
- Hierarchical Equal Risk Parity
- Schur Complementary Hierarchical Risk Parity
- Nested Clustered

#### Ensemble

- Stacking

#### Finite Allocation

- Discrete
- Greedy

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
      + Denoised.
      + Detoned.
      + Custom algorithm.
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
