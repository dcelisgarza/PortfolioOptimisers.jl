```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PortfolioOptimisers.jl"
  text: Portfolio optimisation library in Julia
  tagline: Democratising, demystifying, and derisking investing
  image:
    # Root-relative: DocumenterLandingPage remaps a bare filename under `/` into the site's
    # `assets/` directory.
    src: /logo.svg
    alt: PortfolioOptimisers
  actions:
    - theme: brand
      text: Get started
      link: user_guide/00_User_Guide
    - theme: alt
      text: Examples
      link: examples/00_Examples
    - theme: alt
      text: API
      link: 00_API

features:
  - icon: 🔮
    title: Priors and views
    details: Empirical, factor, and high-order priors. Impose views with the Black-Litterman family, entropy pooling, or opinion pooling. Swapping the prior leaves the optimiser untouched.
    link: user_guide/01_Data_and_Priors
    linkText: Data and priors
  - icon: 🧮
    title: Robust moment estimation
    details: Gerber, Gerber-IQ, and Smyth-Broby covariances, mutual information and distance covariance, denoising, detoning, regime adjustment, coskewness, and cokurtosis.
    link: examples/2_moments_priors/02_Covariance_Estimation
    linkText: Covariance estimation
  - icon: 📉
    title: Over 50 risk measures
    details: Variance, semi-moments, mean absolute deviation, VaR, CVaR, EVaR, RLVaR, drawdowns, ordered weights arrays, and tail ranges. Combine several in one objective, and/or use them as a limit.
    link: user_guide/03_Risk_Measures
    linkText: Risk measures
  - icon: ⚖️
    title: An optimiser for every mandate
    details: Mean-risk, risk budgeting, near-optimal centering, hierarchical risk parity, HERC, Schur complement, naïve, and the meta-optimisers that nest, stack, and resample them.
    link: user_guide/02_Optimisers
    linkText: Optimisers
  - icon: 🔗
    title: Constraints and costs
    details: Budget, group, factor exposure, cardinality, turnover, tracking, phylogeny, and centrality constraints, plus fees and market impact. Add your own JuMP expressions.
    link: user_guide/04_Constraints_and_Costs
    linkText: Constraints and costs
  - icon: 🔁
    title: Validation and tuning
    details: Walk-forward and combinatorial cross-validation, grid and randomised hyperparameter search, pipelines, and time-dependent constraints.
    link: user_guide/05_Validation_and_Tuning
    linkText: Validation and tuning
---
```

```@meta
CurrentModule = PortfolioOptimisers
Description = "PortfolioOptimisers.jl is a portfolio optimisation (portfolio optimization) library for Julia, built from composable immutable estimators."
```

# PortfolioOptimisers.jl

`PortfolioOptimisers.jl` is a portfolio optimisation (portfolio optimization) library for Julia.
Every component is an immutable estimator you compose, so a prior, a risk measure or a
constraint swaps out without touching the optimiser.

!!! danger

    Investing carries real risk, and portfolio optimisation tries to reduce that risk to a level you can accept. The examples use old data and a mix of stocks, some of which I consider meme stocks, to show how the library works. Nothing in this documentation is financial advice. The only advice here is about how to build a portfolio, and most of it is common knowledge in investing and statistics.

## Before you start

- `PortfolioOptimisers.jl` is in active development, and its version is still `v0.*.*`. A `v0.X.0` release can break your code, and a `v0.X.Y` release does not. The [migration guide](@ref migration) lists each breaking change and the code to write instead.
- The documentation is not complete.
- Test coverage is below `95 %`. Most of the missing tests are tests of argument checks, and some less-used features have few tests or none.
- Open an issue, a discussion or a pull request for a bug, or for a missing doc, example, feature or test.

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

This section loads a year of prices, finds the portfolio of minimum variance, turns its weights into whole shares, and plots the result. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples) explain each step at more length, and the docstrings on the [API](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/00_API) pages include short examples.

We first load the packages that the example needs.

- `StatsPlots` and `GraphRecipes` load the plotting extension.
- `Clarabel` and `HiGHS` are the solvers.
- `CSV`, `TimeSeries` and `DataFrames` load and hold the price data.
- `PrettyTables` prints the tables.

```@example 0_index
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
end;
nothing # hide
```

The data is the S&P 500 sample that ships with the documentation. It has the daily adjusted close prices of 20 large-cap stocks, and we keep the last 253 rows, about one year, so the example runs fast.

```@example 0_index
# Load the shipped S&P 500 price data as a TimeArray.
prices = TimeArray(CSV.File(joinpath(@__DIR__, "examples", "SP500.csv.gz"));
                   timestamp = :Date)[(end - 252):end]
pretty_table(prices[(end - 5):end]; formatters = [fmt1])
```

!!! tip "Using your own data"
    The file above is a gzipped CSV with a `Date` column and one column per asset, and any price history in that shape works. To use live data, download it with [`YFinance.jl`](https://github.com/eohne/YFinance.jl) and build a `TimeArray`:

    ```julia
    using YFinance, TimeSeries

    # Convert a YFinance price dictionary into a TimeArray.
    function stock_price_to_time_array(x)
        # Only get the keys that are not ticker or datetime.
        coln = collect(keys(x))[3:end]
        # Convert the dictionary into a matrix.
        m = hcat([x[k] for k in coln]...)
        return TimeArray(x["timestamp"], m, Symbol.(coln), x["ticker"])
    end

    assets = sort!(["AAPL", "AMD", "BAC", "BBY", "CVX", "GE", "HD", "JNJ", "JPM", "KO",
                    "LLY", "MRK", "MSFT", "PEP", "PFE", "PG", "RRC", "UNH", "WMT", "XOM"])

    # Download the adjusted close prices and assemble a single TimeArray.
    prices = get_prices.(assets; startdt = "2024-01-01", enddt = "2025-01-01")
    prices = stock_price_to_time_array.(prices)
    prices = hcat(prices...)
    cidx = colnames(prices)[occursin.(r"adj", string.(colnames(prices)))]
    prices = prices[cidx]
    TimeSeries.rename!(prices, Symbol.(assets))
    ```

We compute the returns with [`prices_to_returns`](@ref).

```@example 0_index
# Compute the returns.
rd = prices_to_returns(prices)
```

`PortfolioOptimisers.jl` builds its optimisation problems with `JuMP`, so it works with any solver that `JuMP` supports, and it ships with none. A [`Solver`](@ref) holds the solver's `Optimizer`, its settings, and the solver statuses that the library accepts as a solution.

```@example 0_index
# Define the continuous solver.
slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
             check_sol = (; allow_local = true, allow_almost = true))
```

Every optimiser that solves a mathematical program takes a [`JuMPOptimiser`](@ref), which holds the solvers and the constraints that these optimisers share. A `JuMPOptimiser` takes one [`Solver`](@ref), or a vector of them that it tries in order.

```@example 0_index
opt = JuMPOptimiser(; slv = slv);
nothing # hide
```

We use [`MeanRisk`](@ref). Its defaults minimise the variance, which gives the Markowitz portfolio of minimum risk.

```@example 0_index
# Vanilla (Markowitz) mean risk optimisation.
mr = MeanRisk(; opt = opt)
```

The printout shows the settings of the estimator and of its `JuMPOptimiser`. Most of them are constraints, and the [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples) show how to set them.

We solve the problem with [`optimise`](@ref).

```@example 0_index
# Perform the optimisation, res.w contains the optimal weights.
res = optimise(mr, rd)
```

`res.sol` holds the solution, and `res.w` gives the weights.

The weights are fractions of the portfolio, but you buy whole shares with a fixed amount of cash. A finite allocation turns the weights into numbers of shares. [`GreedyAllocation`](@ref) is fast and always finishes, but its answer is not always the best one. [`DiscreteAllocation`](@ref) solves a mixed-integer program, so it needs a solver that handles one.

We use `DiscreteAllocation`, with `HiGHS` as the mixed-integer solver.

```@example 0_index
# Define the MIP solver for finite discrete allocation.
mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))

# Discrete finite allocation.
da = DiscreteAllocation(; slv = mip_slv)
```

The discrete allocation minimises the distance between the ideal allocation and the one you can afford, plus the leftover cash. Its `wf` keyword picks the distance, which is the L1 or the L2 norm of the absolute or the relative difference. It needs three inputs, which `FiniteAllocationInput` holds: the optimal weights `res.w`, the latest prices `vec(values(prices[end]))`, and the cash, which we set to `4206.90`.

```@example 0_index
# Perform the finite discrete allocation, uses the final asset
# prices, and an available cash amount. This is for us mortals
# without infinite wealth.
mip_res = optimise(da, FiniteAllocationInput(; w = res.w, prices = vec(values(prices[end])), cash = 4206.90))
```

We put the shares, their cost, the optimal weights and the weights of the shares in one table, to compare the two sets of weights.

```@example 0_index
# View the results.
df = DataFrame(:assets => rd.nx, :shares => mip_res.shares, :cost => mip_res.cost,
               :opt_weights => res.w, :mip_weights => mip_res.w)
pretty_table(df; formatters = [fmt2])
```

We plot the compounded cumulative returns of the portfolio of whole shares.

```@example 0_index
# Plot the portfolio cumulative returns of the finite allocation portfolio.
plot_portfolio_cumulative_returns(mip_res.w, rd.X; ts = rd.ts, compound = true)
```

The plot of each asset's risk contribution needs a risk measure with its statistics set, here the covariance matrix of a `Variance`. [`factory`](@ref) builds a copy of the measure that takes its covariance from the prior result `res.pr`. You can also set it by hand, but `factory` is the better choice in code that builds many measures.

The risk measure needs this step because the library keeps the risk measures apart from the optimisers and their results. The same design lets you put several risk measures in one objective, or use a risk measure as a limit in an optimisation. The [examples](https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples) show both.

```@example 0_index
# Plot the risk contribution per asset.
plot_risk_contribution(factory(Variance(), res.pr), mip_res.w, rd.X; nx = rd.nx)
```

The histogram shows the distribution of the portfolio returns.

```@example 0_index
# Plot histogram of returns.
plot_histogram(mip_res.w, rd.X; slv = slv)
```

The drawdown plot shows the compounded drawdowns. Pass `compound = false` for the uncompounded ones.

```@example 0_index
plot_drawdowns(mip_res.w, rd.X; slv = slv, ts = rd.ts, compound = true)
```

The [Plotting](public_api/22_Plotting.md) page lists the other plots.
