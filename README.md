# PortfolioOptimisers.jl

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/dev)
[![Test workflow status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl)
[![Lint workflow Status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Build Status](https://api.cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl.svg)](https://cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/dcelisgarza/PortfolioOptimisers.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Welcome to PortfolioOptimisers.jl!

[`PortfolioOptimisers.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl) is a package for portfolio optimisation written in Julia.

!!! danger
    
    Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

The feature list is quite large and under active development. New features will be added over time. Check out the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/1_Getting_Started) and [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_Introduction) documentation for details.

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
pretty_table(prices[(end - 5):end]; formatters = fmt1)
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
pretty_table(df; formatters = fmt2)
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

## How to Cite

If you use PortfolioOptimisers.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first take a look into our [contributing guide directly on GitHub](docs/src/contribute/1-contributing.md) or the [contributing page on the website](https://dcelisgarza.github.io/PortfolioOptimisers.jl/dev/contribute/1-contributing)

* * *

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
