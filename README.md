# PortfolioOptimisers.jl

`PortfolioOptimisers.jl` is a portfolio optimisation (portfolio optimization) library for Julia.
Every component is an immutable estimator you compose, so a prior, a risk measure or a
constraint swaps out without touching the optimiser.

| Category | Badge |
| :--------- | :---- |
| Docs | [![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable) [![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://dcelisgarza.github.io/PortfolioOptimisers.jl/dev) |
| CI | [![Test workflow status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Test.yml?query=branch%3Amain) [![Docs workflow Status](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Docs.yml?query=branch%3Amain) [![Aqua](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Aqua.yml/badge.svg)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/actions/workflows/Aqua.yml) |
| Coverage | [![Coverage](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl) |
| Contribute | [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/CODE_OF_CONDUCT.md) |
| Misc | [![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl) |

<!-- [![Build Status](https://api.cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl.svg)](https://cirrus-ci.com/github/dcelisgarza/PortfolioOptimisers.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME) -->

<!-- [![All Contributors](https://img.shields.io/github/all-contributors/dcelisgarza/PortfolioOptimisers.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors) -->

> [!CAUTION]
> Investing carries real risk, and portfolio optimisation tries to reduce that risk to a level you can accept. The examples use old data and a mix of stocks, some of which I consider meme stocks, to show how the library works. Nothing in this documentation is financial advice. The only advice here is about how to build a portfolio, and most of it is common knowledge in investing and statistics.

## What it does

- The optimisers include mean-risk, risk budgeting and relaxed risk budgeting, near-optimal centering, hierarchical risk parity, hierarchical equal risk contribution, Schur complement, nested clustered optimisation, stacking, subset resampling and the naive portfolios. A discrete or a greedy finite allocation turns the weights into whole shares. See the [optimisers guide](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/02_Optimisers).
- The library has over 50 risk measures. Among them are variance, semi-moments, mean absolute deviation, VaR, CVaR, EVaR and RLVaR, ordered weights arrays, the average, maximum and ulcer drawdowns, worst realisation, range, tracking and turnover measures, skewness and kurtosis. See the [risk measures guide](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/03_Risk_Measures).
- The priors include empirical, factor and high-order priors. You can add views with four Black-Litterman variants and with entropy pooling in Meucci's form or in the general form, and you can combine several priors with opinion pooling. See the [data and priors guide](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/01_Data_and_Priors).
- The moment estimators include the Gerber, Gerber-IQ and Smyth-Broby covariances, distance and mutual-information covariance, a regime-adjusted exponentially weighted covariance, coskewness and cokurtosis. The library can also denoise and detone a covariance matrix. See the [covariance estimation example](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/2_moments_priors/02_Covariance_Estimation).
- The constraints include budget, group, factor exposure, cardinality, turnover, tracking, phylogeny and centrality constraints. The costs are fees and market impact, and you can add your own JuMP expressions. See the [constraints and costs guide](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/04_Constraints_and_Costs).
- You can validate a portfolio with walk-forward and combinatorial cross-validation, tune it with a grid or a randomised hyperparameter search, and join the steps of a fit in a pipeline. See the [validation and tuning guide](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/05_Validation_and_Tuning).

The [capability catalogue](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/capability_catalogue) lists everything the library can do. The docs build generates it from the package, so it cannot fall behind the code.

## Installation

`PortfolioOptimisers.jl` is a registered package. Install it with the package manager:

```julia
julia> using Pkg

julia> Pkg.add(PackageSpec(; name = "PortfolioOptimisers"))
```

## Roadmap

- The [Issues](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/) page tracks bugs, feature requests, plans and work in progress.

- Changes go to the [dev](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev) branch first, and they merge into `main` at a release.

## Quick-start

The example below loads a year of prices, finds the portfolio of minimum variance, turns its weights into whole shares, and plots the result. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples) explain each step at more length, and the docstrings on the [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/00_API/) pages include short examples.

The example needs these packages:

- `StatsPlots` and `GraphRecipes` load the plotting extension.
- `Clarabel` and `HiGHS` are the solvers.
- `CSV`, `TimeSeries` and `DataFrames` load and hold the price data.
- `PrettyTables` prints the tables.

The data is the S&P 500 sample in [`examples/SP500.csv.gz`](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples), which has the daily adjusted close prices of 20 large-cap stocks. We keep the last 253 rows, about one year, so the example runs fast.

```julia
# Import module and plotting extension.
using PortfolioOptimisers, StatsPlots, GraphRecipes
# Import optimisers.
using Clarabel, HiGHS
# Load and preprocess data.
using CSV, TimeSeries, DataFrames
# Pretty printing.
using PrettyTables

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
end

# Load the shipped S&P 500 price data as a TimeArray (run from the repo root).
prices = TimeArray(CSV.File(joinpath("examples", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]

#=
Any price history with a `Date` column and one column per asset works. To pull live
data instead, download it with YFinance and assemble a TimeArray:

    using YFinance, TimeSeries
    function stock_price_to_time_array(x)
        coln = collect(keys(x))[3:end]
        m = hcat([x[k] for k in coln]...)
        return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
    end
    assets = sort!(["AAPL", "AMD", "BAC", "BBY", "CVX", "GE", "HD", "JNJ", "JPM", "KO",
                    "LLY", "MRK", "MSFT", "PEP", "PFE", "PG", "RRC", "UNH", "WMT", "XOM"])
    prices = get_prices.(assets; startdt = "2024-01-01", enddt = "2025-01-01")
    prices = stock_price_to_time_array.(prices)
    prices = hcat(prices...)
    cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
    prices = prices[cidx]
    TimeSeries.rename!(prices, Symbol.(assets))
=#

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
mip_res = optimise(da, FiniteAllocationInput(; w = res.w, prices = vec(values(prices[end])), cash = 4206.90))

df = DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
               :opt_weights => res.w, :mip_weights => mip_res.w)
pretty_table(df; formatters = [fmt2])

# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_portfolio_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
```

![Fig. 1](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/src/assets/readme_1.svg)

```julia
# Furthermore, we can also plot the risk contribution per asset. For this, we must provide an instance of the risk measure we want to use with the appropriate statistics/parameters. We can do this by using the `factory` function (recommended when doing so programmatically), or manually set the quantities ourselves.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx, erc = false)

# This awkwardness is due to the fact that `PortfolioOptimisers.jl` tries to decouple the risk measures from optimisation estimators and results. However, the advantage of this approach is that it lets us use multiple different risk measures as part of the risk expression, or as risk limits in optimisations. We explore this further in the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples).
```

![Fig. 2](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/src/assets/readme_2.svg)

```julia
# We can also plot the returns' histogram and probability density.
plot_histogram(mip_res.w, rd.X; slv = slv)
```

![Fig. 3](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/src/assets/readme_3.svg)

```julia
# Plot compounded or uncompounded drawdowns.
plot_drawdowns(mip_res.w, rd.X; slv = slv, ts = rd.ts, compound = true)
```

![Fig. 4](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/src/assets/readme_4.svg)
