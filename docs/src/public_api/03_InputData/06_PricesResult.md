```@meta
Description = "Prices result, public API of PortfolioOptimisers.jl: AbstractPricesResult, PricesResult, port_opt_view."
```

# Prices result

## Types

A new type of price data subtypes `AbstractPricesResult`, has the fields `X` and `pnl`, and adds a method of `port_opt_view`.

```@docs
PortfolioOptimisers.AbstractPricesResult
PricesResult
```

## Functions

```@docs
port_opt_view(pr::PricesResult, ::Colon, ::Colon)
```
