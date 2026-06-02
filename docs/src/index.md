```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PortfolioOptimisers.jl"
  text: Quantitative portfolio construction
  tagline: Democratising, demystifying, and derisking investing
  image:
    src: logo.svg
    alt: PortfolioOptimisers
  actions:
    - theme: brand
      text: User Guide
      link: user_guide/00_User_Guide
    - theme: brand
      text: Examples
      link: examples/00_Examples
    - theme: alt
      text: API
      link: api/00_API

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

::: danger

Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

:::

Portfolio optimisation is the science of either:

- Minimising risk whilst keeping returns to acceptable levels.
- Maximising returns whilst keeping risk to acceptable levels.

To some definition of acceptable, and with any number of additional constraints available to the optimisation type.

There exist myriad statistical, pre- and post-processing, optimisations, and constraints that allow one to explore an extensive landscape of "optimal" portfolios.

`PortfolioOptimisers.jl` is an attempt at providing as many of these as possible under a single banner. We make extensive use of `Julia`'s type system, module extensions, and multiple dispatch to simplify development and maintenance.

Please visit the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples) and [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API) for details.

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

## Roadmap

- For a roadmap of planned and desired features in no particular order please refer to Issue [#37](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/37).

- Some docstrings are incomplete and/or outdated, please refer to Issue [#58](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/58) for details on what docstrings have been completed in the `dev` branch.

## Quick-start

The library is quite powerful and extremely flexible. Here is what a very basic end-to-end workflow can look like. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples) contain more thorough explanations and demos. The [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API) docs contain toy examples of the many, many features.

First we import the packages we will need for the example.

- `StatsPlots` and `GraphRecipes` are needed to load the plotting extension.
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

`PortfolioOptimisers.jl` implements a number of optimisation types as estimators. All the ones which use mathematical optimisation require a [`JuMPOptimiser`](@ref) structure which defines general solver constraints. This structure in turn requires an instance (or vector) of [`Solver`](@ref).

```@example 0_index
opt = JuMPOptimiser(; slv = slv);
nothing # hide
```

Here we will use the traditional Mean-Risk [`MeanRisk`](@ref) optimisation estimator, which defaults to the Markowitz optimisation (minimum risk mean-variance optimisation).

```@example 0_index
# Vanilla (Markowitz) mean risk optimisation.
mr = MeanRisk(; opt = opt)
```

As you can see, there are *a lot* of fields in this structure, which correspond to a wide variety of optimisation constraints. We will explore these in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples). For now, we will perform the optimisation via [`optimise`](@ref).

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
plot_histogram(mip_res.w, rd.X; slv = slv)
```

We can also plot the risk contribution per asset. For this, we must provide an instance of the risk measure we want to use with the appropriate statistics/parameters. We can do this by using the [`factory`](@ref) function (recommended when doing so programmatically), or manually set the quantities ourselves.

```@example 0_index
# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx)
```

This awkwardness is due to the fact that `PortfolioOptimisers.jl` tries to decouple the risk measures from optimisation estimators and results. However, the advantage of this approach is that it lets us use multiple different risk measures as part of the risk expression, or as risk limits in optimisations. We explore this further in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples).

There are many other types of plotting functionality in `PortfolioOptimisers.jl`, check out the [Plotting](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/22_Plotting) page of the documentation.
