#=
# Example 9: Regularisation

This example shows one of the simplest ways to imporve the robustness of portfolios, regularisation penalties.
=#
using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
mmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Setting up

We will use the same data as the previous example. But we will also load factor data.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

#=
## 2. Regularised portfolios

### 2.1 L1 Regularisation

The optimal regularisation penalty value depends on the data, the investor preferences, and type of regularisation.
=#
