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

## Welcome to PortfolioOptimisers.jl

 [`PortfolioOptimisers.jl`](https://github.com/dcelisgarza/PortfolioOptimisers.jl) is a package for portfolio optimisation written in Julia.

> [!CAUTION]
> Investing conveys real risk, the entire point of portfolio optimisation is to minimise it to tolerable levels. The examples use outdated data and a variety of stocks (including what I consider to be meme stocks) for demonstration purposes only. None of the information in this documentation should be taken as financial advice. Any advice is limited to improving portfolio construction, most of which is common investment and statistical knowledge.

Portfolio optimisation is the science of either:

- Minimising risk whilst keeping returns to acceptable levels.
- Maximising returns whilst keeping risk to acceptable levels.

To some definition of acceptable, and with any number of additional constraints available to the optimisation type.

There exist myriad statistical, pre- and post-processing, optimisations, and constraints that allow one to explore an extensive landscape of "optimal" portfolios.

`PortfolioOptimisers.jl` is an attempt at providing as many of these as possible under a single banner. We make extensive use of `Julia`'s type system, module extensions, and multiple dispatch to simplify development and maintenance.

For more information on the package's *vast* feature list, check out the [documentation](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable) for details.

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

# Compute the returns.
rd = prices_to_returns(prices)

# Define the continuous solver.
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
             check_sol = (; allow_local = true, allow_almost = true))

# `PortfolioOptimisers.jl` implements a number of optimisation types as estimators. All the ones which use mathematical optimisation require a `JuMPOptimiser` structure which defines general solver constraints. This structure in turn requires an instance (or vector) of `Solver`.
opt = JuMPOptimiser(; slv = slv);

# Vanilla (Markowitz) mean risk optimisation, i.e. minimum variance portfolio
mr = MeanRisk(; opt = opt)

# Perform the optimisation, res.w contains the optimal weights.
res = optimise(mr, rd)

# Define the MIP solver for finite discrete allocation.
mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true));

# Discrete finite allocation.
da = DiscreteAllocation(; slv = mip_slv)

# Perform the finite discrete allocation, uses the final asset
# prices, and an available cash amount. This is for us mortals
# without infinite wealth.
mip_res = optimise(da, res.w, vec(values(prices[end])), 4206.90)

df = DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
               :opt_weights => res.w, :mip_weights => mip_res.w)
pretty_table(df; formatters = [fmt2])

# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_ptf_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
```

![Fig. 1](./docs/src/assets/readme_1.svg)

```julia
# Furthermore, we can also plot the risk contribution per asset. For this, we must provide an instance of the risk measure we want to use with the appropriate statistics/parameters. We can do this by using the `factory` function (recommended when doing so programmatically), or manually set the quantities ourselves.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx,
                       percentage = true)

# This awkwardness is due to the fact that `PortfolioOptimisers.jl` tries to decouple the risk measures from optimisation estimators and results. However, the advantage of this approach is that it lets us use multiple different risk measures as part of the risk expression, or as risk limits in optimisations. We explore this further in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples_Introduction).
```

![Fig. 2](./docs/src/assets/readme_2.svg)

```julia
# We can also plot the returns' histogram and probability density.
plot_histogram(mip_res.w, rd.X, slv)
```

![Fig. 3](./docs/src/assets/readme_3.svg)

```julia
# Plot compounded or uncompounded drawdowns. We use the former here.
plot_drawdowns(mip_res.w, rd.X, slv; ts = rd.ts, compound = true)
```

![Fig. 4](./docs/src/assets/readme_4.svg)
