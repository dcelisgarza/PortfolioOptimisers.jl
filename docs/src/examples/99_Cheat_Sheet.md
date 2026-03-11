The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/99_Cheat_Sheet.jl"
```

# Cheat Sheet

This is a collection of quickfire tutorials to help you get started with `PortfolioOptimisers.jl` without delving into the examples and/or documentation.

## 1. Downloading data

There are both open and close source providers, in Julia we have [`YFinance.jl`](https://github.com/eohne/YFinance.jl) and [`MarketData.jl`](https://github.com/JuliaFinance/MarketData.jl).

## 2. Computing returns

Usually data is obtained from a provider and the returns have to be computed. `PortfolioOptimisers.jl` has a [`prices_to_returns`](@ref) to do so from price data. It can handle assets, factors and implied volatilities

````@example 99_Cheat_Sheet
# Load data
using PortfolioOptimisers, CSV, TimeSeries
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
