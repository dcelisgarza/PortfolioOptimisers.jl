module PortfolioOptimisers

using ArgCheck, AverageShiftedHistograms, Clustering, Distances, Distributions, FLoops, GLM,
      Impute, InteractiveUtils, Interfaces, JuMP, LinearAlgebra, LogExpFunctions,
      MultivariateStats, NearestCorrelationMatrix, Optim, Graphs, SimpleWeightedGraphs,
      StatsAPI, PythonCall, PrecompileTools, Random, Roots, SparseArrays, Statistics,
      StatsBase, DataFrames, TimeSeries

# Turn readme into PortfolioOptimisers' docs.
@doc let
    path = joinpath(dirname(@__DIR__), "docs/src/index.md")
    include_dependency(path)
    read(path, String)
end PortfolioOptimisers

src_files = String[]
sizehint!(src_files, 140)
for (root, dirs, files) in walkdir(@__DIR__)
    for file in files
        if file == "PortfolioOptimisers.jl"
            continue
        end
        push!(src_files, joinpath(root, file))
    end
end
sort!(src_files)
include.(src_files)

end
