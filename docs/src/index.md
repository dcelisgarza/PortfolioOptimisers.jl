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

# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)

# Plot histogram of returns.
plot_histogram(mip_res.w, rd.X, slv)

# Plot compounded drawdowns.
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
````
