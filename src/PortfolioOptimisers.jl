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

# Utility
include("./LinearAlgebra/LinearAlgebra.jl")

# Fix Non Positive Definite
include("./Moments/MomentsUtil/FixNonPositiveDefiniteMatrices/FNPDM_AbstractTypes.jl")
include("./Moments/MomentsUtil/FixNonPositiveDefiniteMatrices/FNPDM_NoFix.jl")
include("./Moments/MomentsUtil/FixNonPositiveDefiniteMatrices/FNPDM_NearestCorrelationMatrix.jl")

# Detone
include("./Moments/MomentsUtil/Detone/Detone_AbstractTypes.jl")
include("./Moments/MomentsUtil/Detone/NoDetone.jl")
include("./Moments/MomentsUtil/Detone/Detone.jl")

# Denoise
include("./Moments/MomentsUtil/Denoise/Denoise_AbstractTypes.jl")
include("./Moments/MomentsUtil/Denoise/NoDenoise.jl")
include("./Moments/MomentsUtil/Denoise/ShrunkDenoise.jl")
include("./Moments/MomentsUtil/Denoise/SpectralDenoise.jl")
include("./Moments/MomentsUtil/Denoise/FixedDenoise.jl")
include("./Moments/MomentsUtil/Denoise/Denoise.jl")

# Histogram
include("./Moments/MomentsUtil/Histogram/Histogram_AbstractTypes.jl")
include("./Moments/MomentsUtil/Histogram/HistogramAstroPyBins.jl")
include("./Moments/MomentsUtil/Histogram/HistogramHGRBins.jl")
include("./Moments/MomentsUtil/Histogram/HistogramIntegerBins.jl")
include("./Moments/MomentsUtil/Histogram/HistogramInformation.jl")

# Simple Expected Returns
include("./Moments/ExpectedReturns/ExpectedReturns_AbstractTypes.jl")
include("./Moments/ExpectedReturns/SimpleExpectedReturns.jl")

# Pearson Covariance
include("./Moments/Covariance/Covariance_AbstractTypes.jl")
include("./Moments/Covariance/PearsonCovariance/PearsonCovariance_AbstractTypes.jl")
include("./Moments/Covariance/PearsonCovariance/GeneralWeightedCovariance.jl")
include("./Moments/Covariance/PearsonCovariance/FullCovariance.jl")
include("./Moments/Covariance/PearsonCovariance/SemiCovariance.jl")

# Simple Variance
include("./Moments/Covariance/VarianceStd/VarianceStd_AbstractTypes.jl")
include("./Moments/Covariance/VarianceStd/SimpleVarianceStd.jl")

# Rank Covariance
include("./Moments/Covariance/RankCovariance/RankCovariance_AbstractTypes.jl")
include("./Moments/Covariance/RankCovariance/KendallCovariance.jl")
include("./Moments/Covariance/RankCovariance/SpearmanCovariance.jl")

# Distance Covariance
include("./Moments/Covariance/DistanceCovariance.jl")

# LTD Covariance
include("./Moments/Covariance/LTDCovariance.jl")

# Mutual Information Covariance
include("./Moments/Covariance/MutualInfoCovariance.jl")

# Shrunk Expected Returns
include("./Moments/ExpectedReturns/ShrunkExpectedReturnsTargets.jl")

# Gerber Covariance
include("./Moments/Covariance/GerberCovariance/GerberCovariance_AbstractTypes.jl")

## Base Gerber Covariance
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/BaseGerberCovariance_AbstractTypes.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber0Covariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber0NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber1Covariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber1NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber2Covariance.jl")
include("./Moments/Covariance/GerberCovariance/BaseGerberCovariance/Gerber2NormalisedCovariance.jl")

## Smyth Broby Covariance
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBrobyCovariance_AbstractTypes.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBrobyDelta.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby0Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby0NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby1Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby1NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby2Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyCovariance/SmythBroby2NormalisedCovariance.jl")

## Smyth Broby Gerber Covariance
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerberCovariance_AbstractTypes.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber0Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber0NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber1Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber1NormalisedCovariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber2Covariance.jl")
include("./Moments/Covariance/GerberCovariance/SmythBrobyGerberCovariance/SmythBrobyGerber2NormalisedCovariance.jl")

# Distances
include("./Moments/Distance/Distance_AbstractTypes.jl")

## Simple distance
include("./Moments/Distance/SimpleDistance/SimpleDistance.jl")
include("./Moments/Distance/SimpleDistance/SimpleAbsoluteDistance.jl")
include("./Moments/Distance/SimpleDistance/SimpleDistanceDistance.jl")
include("./Moments/Distance/SimpleDistance/SimpleAbsoluteDistanceDistance.jl")

## General distance
include("./Moments/Distance/GeneralDistance/GeneralDistance.jl")
include("./Moments/Distance/GeneralDistance/GeneralAbsoluteDistance.jl")
include("./Moments/Distance/GeneralDistance/GeneralDistanceDistance.jl")
include("./Moments/Distance/GeneralDistance/GeneralAbsoluteDistanceDistance.jl")

## General distance
include("./Moments/Distance/LogDistance/LogDistance.jl")

# Matrix Processing
include("./MatrixProcessing/MatrixProcessing_AbstractTypes.jl")
include("./MatrixProcessing/DefaultMatrixProcessing.jl")

include("./Moments/Covariance/PortfolioOptimisersCovariance.jl")

end
