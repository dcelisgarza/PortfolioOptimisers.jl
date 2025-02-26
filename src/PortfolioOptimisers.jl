module PortfolioOptimisers

using AverageShiftedHistograms, Clustering, Distances, GLM, JuMP, LinearAlgebra,
      MultivariateStats, NearestCorrelationMatrix, Optim, Graphs, SimpleWeightedGraphs,
      PythonCall, Random, SmartAsserts, SparseArrays, Statistics, StatsBase

# Turn readme into PortfolioOptimisers' docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers

include("./Moments/Moments.jl")

end
