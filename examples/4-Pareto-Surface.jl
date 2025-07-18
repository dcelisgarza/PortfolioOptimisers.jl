#=
# Example 4: Pareto surface

This example kicks up the complexity a couple of notches. We will introduce a new optimisation estimator, `NearOptimalCentering` optimiser.
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

#=
## 1. Returns data

We will use the same data as the previous example.
=#
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = tsfmt)

## Compute the returns
rd = prices_to_returns(X)

#=
## 2. Pareto surface

The pareto surface is a generalisation of the efficient frontier, in fact, we can even think of hypersurfaces, but that would be difficult to visualise so we will stick to a 2D surface in 3D space.
=#
using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
              check_sol = (; allow_local = true, allow_almost = true))]